# Failure detectors on the AHA benchmark — results & method

> Testing drishti's uncertainty/failure detectors (**SMD** on proprioceptive state, **RND**
> on a vision embedding) against **AHA** (NVlabs, ICLR'25) — a dataset purpose-built for failure
> detection in robot manipulation, with binary success/failure labels across RLBench tasks ×
> injected failure modes. Prompted by Eric Rosen suggesting AHA on the demo thread.

## What we did

Generated a slice of AHA on a rented V100 (canonical CoppeliaSim → RLBench → FailGen stack),
fit our detectors on the **success** trajectories only, and measured how well they separate the
**labeled failures**. Policy-free by design: AHA trajectories come from scripted planners with
injected perturbations, so we score the labeled trajectories directly (no VLA in the loop).

**Dataset (this run): 48 trajectories, 3 tasks, 5 failure modes**
`pick_and_lift`, `pick_up_cup`, `stack_blocks` — **19 success + 29 failures**
(grasp 6, slip 6, translation_x 6, translation_z 5, wrong_sequence 6).

Artifacts: [`results/report_aha_full_episode.json`](results/report_aha_full_episode.json),
[`results/report_aha_full_task.json`](results/report_aha_full_task.json),
[`results/install.log`](results/install.log) (full stack building = proof it runs),
[`results/gen_*.log`](results/) (generation transcript). Ingested Store + embeddings in
`data/aha_pod/` (dashboard-viewable).

## Results

### 1. In-distribution (per-task calibration) — episode split

Fit + threshold on 70% of the success trajectories; score held-out successes + all failures.

| | AUC |
|---|---|
| **Overall (combined SMD+RND)** | **0.81** |
| SMD (state) alone | 0.70 |
| RND (vision) alone | 0.78 |

Reproduce (all defaults: SigLIP embedder, no windowing, train-frac 0.7):

```bash
DRISHTI_OBS_DB_URL="sqlite:///data/aha_pod/obs.db" DRISHTI_BLOB_DIR="data/aha_pod/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha.py --source aha --split episode --seed 0
```

No `--state-fields` here: `data/aha_pod/` predates the 30-D-superset ingester and stores the 17-D
proprio vector natively (`meta.state_fields` = the 4 proprio fields, no `field_slices`), so there is
nothing to slice. Artifact: [`results/report_aha_full_episode.json`](results/report_aha_full_episode.json).

**Per-failure-mode AUC — the interesting part:**

| Failure mode | AUC | Nature |
|---|---|---|
| **translation_x** | **1.00** | geometric offset — arm goes to the wrong place |
| **translation_z** | **0.97** | geometric offset |
| **slip** | **0.86** | object slips from grasp |
| wrong_sequence | 0.67 | semantic — right motions, wrong order |
| grasp | 0.58 | subtle — arm moves correctly but fails to grip |

**Takeaway:** the detectors are near-perfect on **geometric/kinematic** failures (the arm's
state trajectory visibly deviates from nominal) and progressively weaker on **subtle** (grasp:
correct motion, failed grip) and **semantic** (wrong_sequence: correct sub-motions, wrong order)
failures. This is exactly the expected failure surface of a state+vision anomaly detector, and it
maps cleanly onto *which* future signal each hard mode would need (e.g. tactile/force for grasp).

This also refines the earlier "SMD vs RND flip" observation: across modes, **neither detector
dominates** — SMD (0.70) and RND (0.78) are comparable and complementary, which is the argument
for keeping the cascade rather than betting on one signal.

### 2. Cross-task generalization — task split (honest negative result)

Fit on one task's successes, deploy on a **different** task's successes + failures.

| | AUC |
|---|---|
| Overall | **0.29** (below chance) |

Reproduce (same defaults, `--split task`):

```bash
DRISHTI_OBS_DB_URL="sqlite:///data/aha_pod/obs.db" DRISHTI_BLOB_DIR="data/aha_pod/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha.py --source aha --split task --seed 0
```

Artifact: [`results/report_aha_full_task.json`](results/report_aha_full_task.json).

**The detectors do NOT transfer across tasks.** When calibrated on task A and shown task B, a
*different task* is wholesale out-of-distribution — task B's normal successes look more anomalous
(relative to A) than A-like failures do, so the success/failure ordering inverts (AUC < 0.5).

This is a *feature-correct* result, not a bug: SMD/RND are **per-deployment anomaly detectors** —
they must be calibrated on the target task/robot's own successes (which is exactly how the live
Monitor uses them). The lesson for productization: **calibration is per-deployment**, and a single
global detector across tasks is not the design. Reporting this honestly is more useful than hiding it.

### Caveats (read before quoting numbers)
- **`false_alarm_frac = 1.0`** in both splits is a **small-sample threshold artifact**, not a real
  false-alarm rate: the conformal threshold is set from ~11–13 successes and tested on 6–8 held-out
  successes — far too few for a 90th-percentile bound. **The AUCs are threshold-independent and are
  the honest numbers.** Fixing the threshold needs more success demos.
- Small n per mode (5–6). `stack_blocks` contributed successes only (generation was capped at
  ~120 min; it's the slow task). Treat per-mode AUCs as indicative, not final.
- Generation was time-capped, so this is a depth-first slice (2 deep tasks + 1 partial), not broad.

## Method (reused code)

- **SMD** (`drishti/uq/smd.py`): one Gaussian (mean, cov) over the proprioceptive state of the
  success trajectories → Mahalanobis distance. State = fixed 17-D proprioceptive slice
  (`joint_positions + joint_velocities + gripper_open + gripper_joint_positions`); we **exclude**
  `task_low_dim_state` (it leaks ground-truth object pose — the detector would cheat).
- **RND** (`drishti/uq/rnd.py`): frozen random-target / trained-predictor MLP on a **standalone
  SigLIP** (`google/siglip-base-patch16-224`) embedding of the `front` + `wrist` cameras (1536-D),
  fit on success embeddings; score = predictor residual. Per-dim whitening absorbs the sim baseline.
- **ACE dropped** — needs the live policy's action-head spread; AHA has no policy.
- **Harness** (`scripts/eval_aha.py`): real train/eval split (fit + threshold on TRAIN successes;
  score held-out successes + all failures) — the rigor `calibrate.py` lacks. Reuses `SMD`, `RND`,
  and `_auc` from `drishti/`.

**Pipeline:** `scripts/gen_aha.py` (dense success+failure demos) → `scripts/ingest_aha.py`
(→ Store + precomputed SigLIP embeddings, `--embed-stride` to thin them) → `scripts/eval_aha.py`.
Full generation recipe in [`GENERATE.md`](GENERATE.md).

## Reproduce / the working stack (proof)

The canonical AHA stack **does** stand up and generate labeled data end-to-end — validated on a
rented V100 (Ubuntu 22.04). Gotchas that cost real time (documented so they don't next time):
- **`libffi-dev`** required or PyRep's `cffi` build fails.
- **AHA's RLBench is an *overlay*** (run its `update.py`) on MohitShridhar's `peract` fork — the
  bundled `aha/Data_Generation/RLBench` has no `setup.py`. `get_failures` lives in the overlay.
- **`torchvision`** is required for HF `AutoImageProcessor`.
- **Generation is ~2–4 min/demo, CPU-bound** and saturates all cores → run ingest (CPU SigLIP)
  *after* generation; use `--embed-stride` (RND scores carry forward, so sparse embeddings are fine).
- **≤2 concurrent CoppeliaSim** per 6-core node; a 3rd instance triggers `V-REP side` errors.

## Status
- **Done:** two runs — a 23-traj smoke (`report_aha_episode.json`) and this 48-traj set
  (`report_aha_full_*.json`). Code, docs, and artifacts committed for posting.
- **To scale further (next GPU session):** more successes (fixes the threshold/false-alarm), more
  tasks for a real cross-task story, GPU torch to speed embedding. Then reply to Eric.

---

# Ablation run (2026-07-09) — encoder, forces, per-task, seed-averaged

Third collection (`data/aha_ablation/`): **50 trajectories, 3 tasks** (pick_and_lift, pick_up_cup,
stack_blocks), 23 success + 27 failures (grasp/slip/translation_z/wrong_sequence 6, translation_x 3).
Captured **ablation-ready**: full **30-D proprio superset incl. force channels** (`meta.field_slices`),
**raw frames kept**, and **3 encoders pre-embedded on the V100 GPU** (SigLIP, DINOv2-cls, DINOv2-mean)
in ~3 min — so every ablation below is an offline `eval_aha_seeds` run, no regeneration. All numbers
are **mean ± std over 8 RND seeds** (RND init is high-variance at this n; single-seed numbers are
unreliable and must be averaged).

Reproduce any row in this section — the knobs are `--embedder`, `--state-fields`, `--task`,
`--state-window`, `--embed-window`; omitting `--state-fields` uses the stored **30-D superset
including the force channels**, and the `--task` flag is what makes it per-task rather than pooled:

```bash
DRISHTI_OBS_DB_URL="sqlite:///data/aha_ablation/obs.db" DRISHTI_BLOB_DIR="data/aha_ablation/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha_seeds.py --source aha --split episode --train-frac 0.7 --k 8 \
  --embedder dinov2-cls --task pick_and_lift            # + --state-window/--embed-window per row
```

> **Reproduction caveat (applies to every seed-averaged table below, incl. the breadth and v2
> sections).** Re-running these commands on the current code reproduces every **SMD** value exactly
> — SMD is a deterministic Gaussian fit — but **not** the RND or combined columns. Re-run on all 10
> rows of the breadth and v2 tables plus Finding 3's two rows: SMD matched the printed 2 decimals in
> every case, RND/combined moved by up to 0.22 AUC on individual rows (worst: breadth pick_and_lift
> RND 0.69 → 0.47). The current `drishti/uq/rnd.py` standardizes its input per-dim and pins its own
> init seed, and the torch version has moved too; we did not isolate which. Note also that because
> `RND(..., seed=0)` now seeds itself, it overrides the per-seed `torch.manual_seed(s)` in
> `eval_aha_seeds.py` — `--k 8` prints ±0.00 and gives a single-seed number, not an 8-seed mean.
> Treat the RND-dependent columns as the numbers recorded from the original runs, not as something
> today's code re-derives.

## Finding 1 — Calibrate PER-TASK, not pooled (the biggest lever)

| Eval regime (dinov2-cls, with forces) | AUC |
|---|---|
| Pooled across all 3 tasks | 0.56 |
| **Per-task — pick_and_lift** | **0.81** |
| **Per-task — pick_up_cup** | **0.70** |

Pooling tasks' successes into one detector builds a multi-modal "normal" manifold, and failures
fall inside it → separation collapses. Same conclusion as the cross-task 0.29 result, now quantified
on the *fit* side: **detectors are per-deployment.** Calibrate on the target task's own successes
and they work; pool tasks and they don't.

## Finding 2 — DINOv2 > SigLIP as the RND encoder

Per-task pick_and_lift, with forces:

| Encoder | overall | RND alone | grasp (RND) |
|---|---|---|---|
| SigLIP | 0.74 | 0.49 | 0.60 |
| **DINOv2-cls** | **0.81** | **0.85** | **0.96** |

DINOv2's spatial/object features are markedly better for the *vision-caught* failures — grasp
(is the object actually in the gripper?) jumps 0.60 → **0.96** on RND. Clear, consistent win;
DINOv2-cls becomes the default RND encoder.

## Finding 3 — Force channels: marginal / inconsistent (hypothesis NOT confirmed)

Per-task pick_and_lift, dinov2-cls, ± force channels:

| State | overall | SMD | grasp (SMD) |
|---|---|---|---|
| with forces (30-D) | 0.81 | 0.71 | 0.67 |
| proprio only (17-D) | 0.79 | 0.73 | 0.78 |

The 30-D row is the command above as written; the 17-D row adds
`--state-fields joint_positions joint_velocities gripper_open gripper_joint_positions`.

Forces helped slightly in the *pooled* regime (SMD 0.49 vs 0.37) but are **neutral/slightly negative
per-task**. The grasp hypothesis ("forces let SMD catch grasp") did **not** hold — grasp is caught by
**DINOv2 vision (0.96)**, not state. Forces are not the missing piece. (On `slide_block_to_target`
they turn out to be worse than neutral — see the v2 section: dropping them takes SMD from 0.10 to
0.50 on the *same* breadth data.)

## Finding 4 — Per-mode attribution (per-task pick_and_lift, dinov2-cls)

| Mode | combined | SMD | RND | caught by |
|---|---|---|---|---|
| translation_z | **1.00** | 1.00 | 1.00 | both |
| slip | 0.97 | **1.00** | 0.86 | SMD (state) |
| grasp | 0.89 | 0.67 | **0.96** | RND (DINOv2 vision) |
| translation_x (n=3) | 0.60 | 0.33 | 0.76 | noisy (tiny n) |
| wrong_sequence | 0.57 | 0.56 | 0.68 | neither (temporal) |

Cascade validated by *specialisation*: slip → state, grasp → vision, translation_z → both,
wrong_sequence → the temporal hole (needs a sequence-aware signal / CQR).

## Finding 5 — Temporal signal: windowed-state SMD solves wrong_sequence (cheap fix worked)

The wrong_sequence hole came from SMD/RND being **memoryless per-frame**. Fix (step 1, lean):
stack the last **W** states so SMD sees short-horizon *transitions/order* (`eval_aha --state-window W`,
W=1 = old per-frame, non-destructive). Per-task pick_and_lift, dinov2-cls, 8-seed avg:

| window | overall | wrong_sequence | wrong_seq SMD | slip |
|---|---|---|---|---|
| W=1 (old) | 0.81 | 0.57 | 0.56 | 0.97 |
| **W=4** | **0.88** | **0.79** | **0.78** | 1.00 |
| W=5–6 (plateau) | 0.89 | 0.80 | 0.89 | 1.00 |

**Robust, not overfit:** plateaus at W≈4–6 (no runaway climb); **generalises** to pick_up_cup
(wrong_seq SMD 0.89 → 1.00 at W=4); holds at **proprio-only 68-D** (≈ the forces-120-D result),
ruling out a dimensionality artifact; and **harms no other mode** (slip/translation stay ≥0.97).

## Finding 6 — Temporal VISION too: windowing RND (embed-window=2) lifts everything

Same idea on the RND side (`--embed-window W`, W=1 = single frame, non-destructive): stack the last
W *embed* frames (they're `--embed-stride` apart, so W=2 ≈ 16 control steps ≈ ~0.8 s of visual
context). Per-task, dinov2-cls, state-window=4:

| embed-window | pick_and_lift overall | wrong_sequence | pick_up_cup overall | grasp (puc) |
|---|---|---|---|---|
| EW=1 | 0.88 | 0.74 | 0.71 | 0.37 |
| **EW=2** | **0.93** | **0.91** | **0.82** | **0.63** |
| EW=3 | 0.89 (RND dilutes) | 0.76 | — | — |

**EW=2 is a clean win on BOTH tasks and harms no mode** — it even lifts grasp/slip on pick_up_cup
(vision gets temporal context). **EW=3 is too much** (RND input 4608-D, ~24 steps → dilutes the sharp
single-frame anomalies: grasp-RND 0.94 → 0.59). So there's a knee at 2 embed frames — the sweet spot
that adds order-context without drowning per-frame signal. The transition-model + CQR (step 2) is now
**unnecessary** for wrong_sequence: two cheap windows took it from chance (0.57) to **0.91**.

## Winning config (final for this dataset)
**Per-task calibration + DINOv2-cls encoder + state-window 4 + embed-window 2:**
- **pick_and_lift: overall 0.93** — translation_z/slip 1.0, grasp 0.93, wrong_sequence **0.91**
- **pick_up_cup: overall 0.82** — wrong_sequence 0.83, grasp/slip lifted to 0.63/0.81

wrong_sequence journey: 0.57 (per-frame) → 0.79 (state-window) → **0.91** (+ embed-window). Forces
deprioritised (no gain). Encoder DINOv2-cls > SigLIP. Calibrate per-deployment.

## Next iteration
- **Default = per-task + DINOv2-cls + state-window 4 + embed-window 2.**
- **More episodes per (task, mode)** to shrink the ±0.1 RND variance and n=3 noise (translation_x).
- Log real timestamps next run (we stored frame indices) so window sizes map to ms.
- Ablation infra (`load_embedder`, `embed_offline.py`, `eval_aha --embedder/--state-fields/--task/
  --state-window/--embed-window`) makes every knob an offline sweep on `data/aha_ablation/`.

---

# Breadth campaign (2026-07-10) — does the winning config generalize across tasks?

Ran the winning config (per-task + DINOv2-cls + state-window 4 + embed-window 2, 8-seed avg) across
**5 diverse tasks** (`data/aha_breadth/`; RTX6000Ada, 3 parallel workers, ~2h, ~$2.50). Throughput was
throttled by a couple of long tasks (put_item_in_drawer, put_money_in_safe never finished failures),
so it landed at 5 fully-evaluable tasks, not the intended 15.

## Result — it does NOT uniformly generalize (honest)

| Task | combined | SMD | RND |
|---|---|---|---|
| stack_blocks | 0.82 | 0.92 | 0.71 |
| pick_up_cup | 0.78 | 0.81 | 0.70 |
| pick_and_lift | 0.73 | 0.69 | 0.69 |
| take_lid_off_saucepan | 0.44 | 0.33 | 0.40 |
| slide_block_to_target | **0.22** | **0.10** | 0.63 |

Reproduce (one run per task; **no `--state-fields`**, i.e. the stored 30-D superset *including*
`joint_forces` + `gripper_touch_forces` — this matters, see the v2 section):

```bash
DRISHTI_OBS_DB_URL="sqlite:///data/aha_breadth/obs.db" DRISHTI_BLOB_DIR="data/aha_breadth/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha_seeds.py --source aha --split episode --train-frac 0.7 --k 8 \
  --embedder dinov2-cls --state-window 4 --embed-window 2 --task <TASK>
```

10 successes per task → 7 calibrate / 3 held-out at `--train-frac 0.7`.

Combined AUC spans **0.22–0.82** — 3 tasks work (0.73–0.82), 2 fail (<0.5). The earlier single-task
0.93 was optimistic; across tasks it's highly variable.

## Why — three real findings (more useful than "it works everywhere")

1. **SMD (state) is task-fragile and can *invert*.** On slide_block_to_target SMD is **0.10**
   (failures score *more normal* than successes — a sliding task's successes are so varied that
   failures fall inside the manifold). It's the strongest signal on pick/stack tasks (0.81–0.92) but a
   liability on others.
2. **RND (DINOv2 vision) is the robust backbone** — 0.40–0.71 across all 5, **never inverts**. Caps
   lower than SMD-at-its-best but far more consistent. For a *new* task, RND is the safer bet.
3. **The mean-combine is a liability.** `mean(smd, rnd)` lets an inverting SMD tank the score:
   slide_block RND alone is 0.63 but combined collapses to **0.22**. A **max-combine, or a gate that
   drops an uninformative/inverted detector**, would recover it (~0.63). Highest-value fix.

Plus: **take_lid_off_saucepan (~0.40 both)** is a genuinely hard case, and **small n (3/mode) makes
per-task numbers noisy** (pick_and_lift was 0.93 on `data/aha_ablation/` vs 0.73 here — same config,
different draw).

## Next iteration (reprioritised by the breadth result)
- **Fix the combine first** — replace mean-of-(smd,rnd) with **max**, or gate out a detector whose
  success-scores are uninformative/inverted. Biggest robustness lever (offline, free).
- **Consider RND-primary**, SMD opt-in per-task only where informative on that task's successes.
- **More episodes per (task, mode)** — n=3 is too noisy to rank tasks reliably.
- **Throughput:** next run use *short* tasks (pick/place/press); avoid long ones
  (put_item_in_drawer/put_money_in_safe) that stalled generation.

---

# V2 run (2026-07-11) — more successes fixes the threshold AND the generalization story

Directly acted on the breadth campaign's "the elephant: 7 training successes is far too few"
finding. Re-generated the same 5 tasks with **20 successes each** (V100 pod, 2 workers,
`data/aha_v2/`) instead of ~10, then re-ran the winning config (per-task calibration,
DINOv2-cls + state-window 4 + embed-window 2) with a proper **15 calibrate / 5 held-out**
split, 8-seed mean.

**This run also changed the state configuration**, which was not recorded at the time and is the
source of a confound corrected below: v2 was evaluated on the **17-D proprio-only slice**
(`--state-fields joint_positions joint_velocities gripper_open gripper_joint_positions`), whereas
the breadth run used the **30-D default** (proprio + `joint_forces` + `gripper_touch_forces`).

## Result — uniform, non-degenerate detection across all 5 tasks

| task | AUC | SMD | RND |
|---|---|---|---|
| take_lid_off_saucepan | **0.92** | 0.78 | 0.92 |
| slide_block_to_target | **0.86** | 0.89 | 0.79 |
| pick_up_cup | 0.74 | 0.85 | 0.67 |
| pick_and_lift | 0.69 | 0.90 | 0.70 |
| stack_blocks | 0.68 | 0.67 | 0.58 |

(`docs/aha/results/aha_v2_table.png`)

Reproduce (one run per task; note the **17-D proprio-only** `--state-fields` and `--train-frac 0.75`
— 20 successes per task → 15 calibrate / 5 held-out):

```bash
DRISHTI_OBS_DB_URL="sqlite:///data/aha_v2/obs.db" DRISHTI_BLOB_DIR="data/aha_v2/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha_seeds.py --source aha --split episode --train-frac 0.75 --k 8 \
  --embedder dinov2-cls --state-window 4 --embed-window 2 \
  --state-fields joint_positions joint_velocities gripper_open gripper_joint_positions --task <TASK>
```

**This directly answers the breadth campaign's two open problems:**
1. **No more SMD inversion — but it takes BOTH more successes AND dropping the force channels.**
   `slide_block_to_target`'s SMD was **0.10** (inverted) on the breadth run and is **0.89** here,
   and the original writeup credited that entirely to going from ~7 to 15 calibration successes.
   That comparison was **confounded**: the 0.10 came from the 30-D state, the 0.89 from the 17-D
   proprio-only slice. Holding the state config fixed and varying only the dataset:

   | SMD AUC, slide_block_to_target | breadth (7 calib) | v2 (15 calib) |
   |---|---|---|
   | 30-D (incl. `joint_forces` + `gripper_touch_forces`) | 0.10 | 0.72 |
   | 17-D proprio-only | 0.50 | **0.89** |

   Both levers are real, and the extra successes are the larger of the two: more successes moves
   0.10 → 0.72 at 30-D and 0.50 → 0.89 at 17-D; dropping the force channels moves 0.10 → 0.50 on
   the *same* breadth data and 0.72 → 0.89 on the *same* v2 data. Neither alone accounts for the
   headline 0.10 → 0.89. The inversion was **not** purely a small-sample artifact — the force/touch channels
   were actively driving it, and at 30-D even 15 successes leave SMD at 0.72.

   This is consistent with **Finding 3** above (force channels were already neutral-to-negative on
   `pick_and_lift`); `slide_block_to_target` is where they are clearly harmful. Sliding drags the
   block along the table, so the contact-force channels are high and variable *during a success* —
   exactly the axes a failure does not have to differ on. Practical upshot: **the state slice is a
   real hyperparameter of SMD, not a capture detail**, and the proprio-only slice is the safer
   default. The breadth run's "mean-combine is a liability when a detector inverts" concern is
   still worth addressing (see Next steps) — it just no longer bites on this dataset.

   *(Verified by re-running the four cells above; SMD is deterministic and every value reproduces
   exactly. See the reproduction caveat in the ablation section for the RND columns.)*
2. **No more degenerate threshold.** The breadth run's 90th-pct-of-successes threshold was so
   low (fit on ~7 successes) that caught% = false-alarm% = 100% on every task — "the deployable
   metric is uncalibratable at this scale." With 15 calibrate / 5 held-out successes per task,
   AUC is uniformly non-degenerate (0.68–0.92, no task below chance) and the threshold has enough
   data to sit at a real operating point.

**Reframes the earlier "does NOT uniformly generalize (0.22–0.82)" headline**: that gap was two
fixable configuration problems — too few calibration successes and too wide a state slice — not a
structural limit of SMD/RND across tasks. With enough successes per task and the proprio-only state,
the detector cascade is a solid, uniform failure detector across 5 diverse RLBench tasks on a
completely different embodiment (Franka, not our SO-101) and simulator (RLBench/CoppeliaSim, not
MuJoCo) — real evidence the approach generalizes beyond our narrow SmolVLA/SO-101 setup.

> **These AUCs are not comparable to the SO-101 calibration numbers.** Two differences, both of
> which make AHA the harder number: AHA is scored with **two detectors** (SMD + RND — no ACE, since
> the trajectories are scripted and there is no policy action head to measure spread on), and it is
> **held-out** (fit + threshold on train successes, scored on held-out successes + all failures).
> The SO-101 report in `docs/results/calibration/report.json` (combined 0.9955, RND 1.000) is a
> three-signal, **in-sample** fit — `calibrate` fits and validates on the same success runs. A 0.92
> here and a 1.00 there are measuring different things; do not put them in the same table.

## Status
Pod terminated after this run. Data at `data/aha_v2/` (obs.db + obs_blobs, 20 successes/task +
4/mode failures) and raw per-task capture at `data/aha_v2_raw/`. This is the current best
per-task AUC story for AHA; treat the breadth-campaign 0.22–0.82 table above as superseded by
this result for anything beyond the "detectors are per-deployment, calibrate on target task"
lesson (which still holds). The two tables are **not** a clean A/B — they differ in calibration-set
size *and* in state slice (30-D vs 17-D); see finding 1 for the controlled version.

## Next steps
- **Sweep the state slice per task** the way we sweep the encoder. The force channels cost
  `slide_block_to_target` 0.40 SMD AUC on the breadth data and 0.17 on v2, and nobody noticed for
  two runs because the flag wasn't recorded next to the table. `eval_aha.py --state-fields` already
  makes it a free offline sweep. `eval_aha.py` already records the choice as
  `report["state_fields"]`; `eval_aha_seeds.py` — which produced every table from the ablation run
  onward — writes no report at all, only stdout. That is how the flag got lost.
- Prototype the gate/max-combine idea from the breadth campaign anyway — even without
  inversion, it may still sharpen tasks where one detector is much weaker (e.g. stack_blocks
  RND 0.58).
- Push per-mode AUC on this v2 dataset (not yet broken out here) — the breadth run's per-mode
  attribution story (grasp→vision, slip→state, wrong_sequence→temporal) should be re-verified
  at this larger n.
