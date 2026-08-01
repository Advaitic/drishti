# drishti — learnings & glossary

A plain-language guide to what we built and what every number means. Read top to bottom; it
goes from "what is this" → "the policy" → "the uncertainty detectors" → "calibration" → "the live
monitor" → "gotchas." Our actual measured numbers are inline so you can connect concept to result.

---

## 1. What is this, in one breath

**drishti is an observability + uncertainty layer for robot policies.** A robot policy is a neural
network that looks at camera images and outputs motor commands. Like any neural net it sometimes
fails — and a failing robot is dangerous/expensive. Our job: **watch the policy at runtime and flag
(or halt) when it's about to fail**, *before* it does.

The hard part isn't running the policy — it's knowing **when to trust it**. That's uncertainty
quantification (UQ). The whole product is: run the policy, score its uncertainty every step, and
when uncertainty crosses a calibrated threshold → alert / halt.

---

## 2. The setup we're proving it on

- **Robot:** SO-101 — a small 5-joint arm + gripper (6 controllable DoF), simulated in **MuJoCo**
  (a physics engine). We use sim because it's fast, free, and gives us *ground truth*.
- **Task:** pick up a red cube and place it in a tray. The language instruction sent to the policy
  is the fixed string `"Pick up the cube and place it in the box"` — same every time.
- **Policy:** **SmolVLA** — a Vision-Language-Action model (~450M params). It's a small
  vision-language model (it "sees" + "reads the instruction") bolted to an *action expert* (a
  transformer that outputs joint commands). We **finetuned** it on 50 scripted demos so it actually
  does *our* task (~83% success at `nas=1` — **unbacked**, see §3). Frozen vision+language, trained
  only the action expert.
- **Success oracle (the "contract field"):** `cube_in_tray` — a **deterministic geometric check**
  (is the cube inside the tray bounds at low height?). In sim we *know* the cube's exact position,
  so success is ground truth, not a guess. This labels every rollout and is what we validate the
  detectors against.

---

## 3. Action chunking & receding horizon (why it matters for speed AND quality)

SmolVLA predicts a **chunk of 50 actions** at once (≈1.7s of motion). You choose how many to
*execute* before re-predicting — that's **`n_action_steps` (nas)**:

- **nas = 50** → execute the whole chunk open-loop, then re-plan. Cheap (1 inference per 50 steps)
  but *blind* — the arm commits to a 1.7s plan and drifts off it. **Result: 0% success** (it never
  actually grasps).
- **nas = 1** → re-plan every single frame ("full receding horizon"). Constantly corrects against
  the latest image. **Result: ~83% success** — but ~8s/inference on the Mac → painfully slow.
  **This number has no artifact:** it was measured during finetune checkpoint selection and the
  eval log was not retained. Believe the direction (nas=1 ≫ nas=6 ≫ nas=50), not the digits.
- **nas = 6** (what we record with) → execute 6 frames, re-plan. ~6× fewer inferences (practical on
  the Mac, ~15s/rollout) and still re-plans often enough to mostly work (~50% success → a nice
  mix of successes and failures for calibration). This one *is* backed: **10/20** of the recorder
  runs in the bundled store (`data/demo/`, `smolvla_ft@4500 (nas=6)`) succeed.

**Takeaway:** receding horizon (re-plan frequently) is *required* for the policy to work, and it's
also why the live monitor gets a fresh forward pass to score — the two needs line up.

---

## 4. The three uncertainty detectors (SMD / RND / ACE)

Each looks at a *different* part of the system. They're a **cascade** — cheap→expensive, different
failure modes. All three are computed per step.

### SMD — "is the robot's body in a weird pose?"
- **What:** Mahalanobis distance of the **proprioceptive state** (the 6 joint angles) from the
  *normal* (successful-rollout) state distribution.
- **Intuition:** during good rollouts the joints follow a tight, repeatable path. A failing rollout
  drives the arm into joint configurations it never visits when succeeding → large distance.
- **Mahalanobis distance:** like a z-score but for multi-dimensional data — it measures "how many
  standard deviations away, accounting for how the joints co-vary." 0 = dead center of normal.
- **Needs:** the mean + covariance of normal states → so it *requires calibration* to exist.
- **Cost:** nearly free (just the state vector). Our AUC: **0.79**.

### RND — "does the scene look off-manifold?"
- **What:** Random Network Distillation on the **vision embedding** — the pooled SigLIP features of
  **all three cameras concatenated** (960 each → **2880-dim**: up + side + wrist). The hook captures
  every camera per forward, not just the last one, so RND sees the *whole scene* (the workspace
  views that show the cube/tray, not only the wrist close-up).
- **Intuition:** train a small network to mimic a fixed random network *on the embeddings seen
  during successful rollouts*. On a familiar (in-distribution) scene the two match → low error. On
  a weird scene (cube knocked away, gripper empty, arm flailing) they diverge → high error =
  "novelty."
- **Why the embedding, not raw pixels:** the encoder has already thrown away action-irrelevant
  detail, so novelty in embedding-space is sharper than in pixel-space.
- **Whitening is the biggest lever:** the raw SigLIP dims have wildly different scales (and the three
  camera blocks differ too), so RND **standardizes each input dim** (mean/std fit on the success
  embeddings) before scoring. Ablation on a fixed set: 1 camera → AUC 0.58, 2–3 cameras → 0.81, and
  **+ whitening → ~1.0**. More cameras help; standardizing the input is what makes it *excellent*.
- **Needs:** training on normal embeddings → *requires calibration*. This is the first detector that
  reaches *inside* the model (a forward hook grabs the embedding) — the "internals" wedge.
- **Cost:** one tiny network forward — a 2-layer MLP, `2880→128→64`, ReLU; ~377k trained params
  (predictor) + a frozen random twin. Our AUC: **1.0** (whitened, in-sample — see the caveat below).

### ACE — "is the policy itself unsure of its next move?"
- **What:** the spread of the **action head** output (how much the predicted action varies). A
  confident policy commits to one action; an unsure one is diffuse.
- **Intuition:** this is the policy's *own* uncertainty about what to do, read straight off its
  action layer. It's **intrinsic** — computable from a single forward pass with **no calibration**,
  which is why it can show live before you've calibrated anything.
- **Caveat (honest):** for our finetuned SmolVLA this signal is nearly constant (~1.0), so its
  variation is tiny. It still carries some signal on our set (AUC **0.73**), but it is the weakest
  of the three and some of that is small-sample noise. The principled upgrade: sample N action chunks from the same
  observation (different flow-matching noise) and measure their **entropy/disagreement** — confident
  policy → chunks agree, unsure policy → chunks scatter. We have `uq/ace.py: sample_action_chunks` +
  `ace_score` for exactly this; we use the cheap single-forward spread today because sampling N
  chunks costs N× inference (fine on GPU, slow on MPS).

---

## 5. Calibration — turning detectors into a threshold

You can't "score uncertainty" in a vacuum — you need a reference for *normal*. **Calibration =
learn "normal" from the successful rollouts**, then anything far from normal is suspicious.

Steps (`drishti/calibrate.py`):
1. Take the rollouts you marked **successful** (deterministic oracle).
2. Fit SMD (on their states) and RND (on their vision embeddings). Collect ACE. **Crucially, score at
   the SAME cadence the live Monitor runs at** — every frame, with SMD on each frame's state and
   RND/ACE carried forward from the last inference frame. (Calibrating only at inference frames but
   *serving* every frame was a train/serve skew: mid-chunk poses the calibration never saw spiked SMD,
   so every live rollout false-alarmed. Score both the same way.)
3. **Normalize + combine** → one number in [0,1]. Each signal becomes a **capped-z**:
   `clip((x − μ_success)/σ_success, 0, CAP) / CAP` with **`CAP = 32`** — z-score against the *success*
   mean/std, clipped at 32σ and scaled to [0,1]. Then **combine = the *mean* of the three capped-z's**
   (`mean_capped_z`). The rule lives in `drishti/scoring.py` and is imported by calibration, the
   server sweep and the AHA benchmark, so the three cannot drift apart.
   - *Why capped-z, not raw z:* whitened RND has a razor-thin success-std, so its raw z explodes
     ~100× past SMD/ACE. The cap bounds every signal to [0,1] so **no detector dominates** — SMD/ACE
     actually move the combined line.
   - *Why 32 and not 8 (a finding, not a knob):* the ceiling has to sit **above** the z-range that
     real out-of-distribution data reaches. At a cap of 8 both held-out successes *and* failures
     saturate to 1.0 and the separation collapses — **held-out AUC 0.78 at cap 8 vs 0.99 at cap ≥ 32**.
     A too-tight cap doesn't just compress the scale, it destroys the signal.
   - *Why mean, not max:* `max` fires on the single loudest detector, so a lone grazing signal causes
     a false alarm. The **mean is agreement-aware** — a lone graze gets averaged down. In our ablation
     this beat `max` on both AUC (0.986 vs 0.979) *and* false-alarms-at-full-recall (1 vs 2). (Rank /
     conformal normalization was *worse* — it saturates to 1.0 on out-of-range failures and dropped
     AUC to 0.888.)
4. Set the **threshold** = the **90th percentile of success ROLLOUT-PEAKS** (`PCTL_RUN`), not the 95th
   percentile of *steps*. This is **conformal at the rollout level**: a new successful rollout exceeds
   it with prob ~≤10%. (The per-*step* 95th pct looked tight but, over a ~50-step run, 5% of steps
   compounds to ~73% of *runs* grazing at least once — the wrong object to bound.) This single number
   is the **"calibration value"** (0–1 scale; **0.0793** on the bundled dataset).
5. **Drop the oracle artifacts.** A "success" shorter than 20 steps never picked anything up — the
   geometric oracle just found the cube already in the tray where it spawned. Those runs are still
   *scored*, but they are not *fit* on (`--min-steps`, default 20; 7 of the 35 successes in the
   bundle). Refitting with them included (`--min-steps 0`) drops the combined AUC from 0.9955 to 0.94.

### What the validation numbers mean (n = 60 rollouts: 26 failure / 34 success)

Every number below comes out of `docs/results/calibration/report.json`, fitted on the bundled
dataset with a fixed seed. The fit used **28 successful rollouts** (the bundle's 35 successes minus
the 7 short oracle artifacts); **all 60 rollouts that carry embeddings are scored**, failures
included.

- **Combined AUC = 0.9955.** AUC = the probability a *random failure* scores higher than a *random
  success*. Per-detector: SMD **0.79**, **RND 1.00**, ACE **0.73** — RND does the discriminating,
  SMD and ACE mostly keep it honest inside the mean.
- **At the calibrated threshold (0.0793): failures caught 100% (26/26), false alarms 17.6% (6/34).**
  The threshold never looks at the failures — it is the 90th percentile of the *successes'*
  run-peaks — so the catch rate isn't fitted; the false-alarm rate is what the conservative rule
  costs.
- **The separation, on the 0–1 scale:** failure peaks **0.190–0.346**, success peaks **0.015–0.223**.
  A single success peaks at 0.223, above the three lowest failures — that is the irreducible overlap
  on this set. See `assets/figures/calibration_separation.png`.
- **A better operating point than the calibrated default exists.** Slide the threshold to **≈0.15**
  and the same 60 runs give **26/26 failures caught with 1 false alarm in 34 (2.9%)**; it holds until
  ~0.19, where the first failure slips under. The 90th-percentile rule is deliberately conservative;
  `assets/figures/calibration_sweep.png` is what you actually tune against, and the Monitor's slider
  is the live version of that figure.
- **Mean lead = 99.3 steps** on caught failures. Read this carefully: every failure in this set runs
  to the 110-step cap, so a 99-step lead means the score is already over threshold within roughly the
  first ten steps. These are rollouts that go wrong at the *approach* — the detector is catching a
  doomed trajectory early, not calling a late, subtle slip.
- **Honest caveat — the success side is in-sample.** SMD and RND are fitted on the successful
  rollouts and then those same rollouts are scored, so **RND's AUC of 1.00 is in-sample and will
  soften on held-out data**. The failures are never fitted on, but the successes are not held out.
  Treat 0.9955 as an upper bound, and the flywheel — more rollouts → re-calibrate → re-measure — as
  how it gets pressure-tested.

Figures: `assets/figures/calibration_separation.png` (per-rollout peaks vs the threshold),
`calibration_roc.png`, `calibration_sweep.png` (catches and false alarms as the threshold slides).
Reproduce the whole thing — seeded, so it matches digit for digit — with:

```bash
drishti calibrate --demo --auto   # refits and rewrites data/demo/calib/report.json
drishti figures --demo            # redraws the three figures
```

`tests/test_demo_bundle.py` asserts that this reproduces the committed report, so the doc and the
code fail together rather than drifting apart.

> Precision vs recall, in one line: **recall** = "what fraction of failures did we catch?"
> **precision/false-alarm** = "when we alarmed, how often were we right?" Raising the threshold
> trades recall down for precision up. The slider lets you pick the point.

---

## 6. The live Monitor (Stage 2)

`drishti/drivers/live_fleet.py` runs one rollout live, scores each step with the *calibrated*
detectors, computes the combined score, and **flags** the moment it crosses the threshold — but does
**not** halt: it runs to the real end so we can compare the prediction against the deterministic
outcome and show *detection accuracy* (correct catch / false alarm / missed / nominal). It streams
every step + frame to the dashboard over the existing `/v1/ingest` → `/v1/tail` WebSocket, so you
watch it unfold in real time. The threshold is the tunable slider, and under it the Monitor shows a
**sweep readout** — catches / false-alarms / missed across *all* logged rollouts at the current
threshold (`GET /v1/sweep`), so you tune against the whole set, not just the rollout on screen.

---

## 7. The architecture (how data flows)

```
record rollouts ─┐
(rollout_record) │  start_run / step / log_frame      ┌─ dashboard (Logs):
                 ├─►  emit ─► store (sqlite) ─► server ─┤   list + replay + select + calibrate
live fleet ──────┘            + blobs (jpg/npy)  (FastAPI)└─ dashboard (Fleet/Monitor):
(live_fleet)                                              live stream via /v1/tail WebSocket
calibrate.py ─► data/calib/ (detectors + threshold + report)  ◄─ /v1/calibrate, /v1/calib, /v1/fleet/run
```

- **Telemetry spine:** `sdk.py` (start_run/step/log_*) → `emit.py` (async queue) → `store.py`
  (sqlite tables: runs / steps / frames) → `drishti/server/app.py` (FastAPI: query + ingest + WebSocket).
- **Frames** (camera jpgs) + **vision embeddings** (`.npy`) live in `data/obs_blobs/`, referenced
  by path. **Calibration artifacts** in `data/calib/`.

---

## 8. The bundled dataset (`data/demo/`) — run the numbers yourself

Everything in §5 is measured on a dataset that ships with the repo, so none of it has to be taken on
faith. `data/demo/` (~16 MB, the one thing under `data/` that isn't gitignored) is **63 real SO-101
pick-place rollouts from the finetuned SmolVLA**, carrying the per-step proprioceptive states, the
per-step vision embeddings the detectors actually score (float16 to halve the size), camera frames
for six representative rollouts, and the calibration fitted on them. Three of the 63 kept no
embeddings, so **60 are scorable — the 34 success / 26 failure set §5 reports on**.
`MANIFEST.json` records exactly what went in; `data/README.md` says how to regenerate it from a full
local store.

```bash
drishti demo                      # dashboard at :8010, reading the bundle — no GPU, no checkpoint
drishti calibrate --demo --auto   # refit the detectors; reproduces the numbers in §5
drishti figures --demo            # redraw the three calibration figures
```

Calibration is seeded, so the refit reproduces `docs/results/calibration/report.json` exactly, and
`tests/test_demo_bundle.py` fails if it ever stops doing so.

---

## 9. Gotchas we hit (so you don't re-learn them the hard way)

- **`.safetensors` not `.pt`:** `.pt` is Python pickle — it can execute arbitrary code on load and
  is Python-only. `.safetensors` is just a header + raw tensor bytes: safe, fast (memory-mapped),
  language-agnostic. The whole ecosystem moved to it.
- **Camera names:** SmolVLA expects `observation.images.camera1/2/3`; our dataset uses `up/side/wrist`
  → use `--rename_map` at train time.
- **Video decode:** the dataset stores frames as AV1 video; if `torchcodec` fails on an old system
  ffmpeg, use `--dataset.video_backend=pyav` (bundled ffmpeg).
- **Headless GPU rendering:** MuJoCo needs the glvnd EGL loader (`apt install libegl1`) so
  `MUJOCO_GL=egl` finds the NVIDIA driver.
- **Pick the best checkpoint, not the last:** finetuning success climbed then *dipped* at the final
  step (overfit) — checkpoint 4500 beat 6000. Always evaluate checkpoints on held-out positions.
- **Detectors need a reference:** SMD/RND can't produce a number before calibration (they're defined
  *relative to normal*). ACE is intrinsic and shows live. The threshold is a calibration *output*.
