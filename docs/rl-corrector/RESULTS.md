# rl-corrector — results & proof

The positive record: milestones hit, with the numbers and where the raw proof lives. Pair with
`DEBUG_LOG.md` (the errors) for the full story. Artifacts in `results/`.

---

> **Checkpoint availability.** Every `aryanmadhavverma/*` model referenced below was pushed to the
> HuggingFace Hub as a **private** repo, and they are still private (checked: the API returns 401).
> Treat the HF ids as a record of what was trained, not as something you can pull today. The
> reproducible artifacts that *are* public live in `results/` here, and the CPU stub server
> (`drishti.sim.stub_server`) lets you exercise the loop without any checkpoint.

## Which number belongs to which model (read this first)

Several success rates appear below and in the other docs. They come from **different models, different
demo sets, and different execution horizons** — they are not comparable to each other. This table is
the key.

| Success rate | Model | Task / set | Execution config | Source |
|---|---|---|---|---|
| ~83% | SmolVLA finetune (the main drishti policy) | SO-101 sim pick-place | `n_action_steps=1` (re-plan every frame) | `docs/LEARNINGS.md` |
| ~50% | same SmolVLA checkpoint | SO-101 sim pick-place | `n_action_steps=6` (the recording config) | `docs/LEARNINGS.md` |
| 11/16 = **68.75%** | MolmoAct2 finetune, 50 demos | 16-episode calibration recording | buffered, `n_action_steps=4` | `results/logs/rec_calib.log` |
| 3/8 = **37.5%** | same MolmoAct2 checkpoint | the fixed 8 seed-0 eval positions | buffered, `n_action_steps=4` | `results/logs/eval_ft.log`, `results/logs/nas_ab.log` |
| 59/120 = **49%** | MolmoAct2 phase-captioned finetunes, 150 demos (frozen + LoRA pooled) | 10 cube positions × trials, phase-caption inference | buffered, `n_action_steps=4` | no retained log — see the language-diversity section |

The 68.75% and the 37.5% are **the same weights at the same execution horizon** — the gap is which
episodes were run, not a config change. Nothing here is a *grasp* rate: grasp and task success differ
(`eval_ft.log` ends `success 3/8 | grasped 2/8`).

---

## Milestone progression (the arc)

| Stage | What | Result | Proof |
|---|---|---|---|
| Stage 0 | `verifiers` env + rollout server + UQ, built local | pytest green on CPU/Mac (15 tests at the time; 22 now) | `tests/`, `environments/vla-corrector/` |
| Stage 1 (R4) | MolmoAct2 joint convention on real hardware | ✅ correct (arm moves toward cube) | memory `project_rl_corrector` |
| Stage 1 (R9) | Zero-shot grasp on sim frames | ❌ approaches (2 cm) but never grasps | — |
| Finetune | LeRobot action-expert finetune, 50 demos | loss **0.881 → 0.019** (1500 steps, ~1 h A100) | `results/logs/ft_full.log` |
| Diagnosis | Teacher-forced: did it learn? | ✅ learned; joint-frame bug on j1/j2 | `results/logs/tf_check.log` |
| Eval | Frame-corrected closed-loop | **3/8 success, 2 grasps** | `results/logs/eval_ft.log`, `results/eval_frames/` |
| Calibration | Buffered VLA (nas=4), 16-episode recording | **11/16 = 68.75% task success** | `results/logs/rec_calib.log` |
| Calibration | UQ threshold + success/failure AUC | **combined AUC 1.0** — *in-sample*, RND-only (RND 1.0, SMD .52→.78, ACE weak) | `results/calib/` |
| Stage 2 | corrective prompting help? | partial pass — `procedural` +12%, situation-dependent | `results/logs/prompt_sweep.log` |
| RL smoke | loop on live VLA → verifiable reward | **6/6 spike→correct→resume→reward, mean 0.50** | `results/logs/rlsmoke.log` |
| **GRPO (Stage 3)** | prime-rl trains the Qwen3-VL corrector | **signal path proven: reward→advantages→LoRA update** | `results/logs/smoke.log`, `modest.log` |

---

## The finetune worked (loss curve)

Action-expert-only finetune of MolmoAct2-SO100_101 on 50 sim pick-place demos:

```
step    5   loss 0.881
step  750   loss 0.029
step 2000   loss 0.019   (7.6 epochs, batch 16)
```

Clean convergence, no divergence. Full log: `results/logs/ft_full.log`.

---

## The model learned — the eval failure was a frame bug, not training

Teacher-forced (feed training obs, compare predicted vs true demo action):

```
MEAN ABS ERR per joint (deg): [ 0.73  38.19  91.98  0.00  0.00  5.50 ]
                                pan   lift   elbow  wristF wristR grip
```

4 of 6 joints near-perfect; only shoulder_lift + elbow_flex off — the two joints with the
sign/offset transform. Proof the weights are good and the bug is in the action-frame handling. Full
per-frame pairs in `results/logs/tf_check.log`.

---

## Task success, step by step (each fix moved the number)

These are **task-success** rates (cube ends in the tray, per the geometric oracle), not grasp rates.
The two differ: `eval_ft.log` ends `success 3/8 | grasped 2/8`.

```
zero-shot MolmoAct2      : 0 grasps  (approaches to 2 cm, never closes)
finetuned, wrong frame   : 1/8 task success, 0 grasps
finetuned, frame-fixed   : 3/8 = 37.5% task success, 2/8 grasped   (empirical j1/j2 correction; buffered, nas=4)
finetuned, 16-ep recording: 11/16 = 68.75% task success            (buffered, nas=4)
```

Buffering is what rescued the **recorder** path, which had defaulted to `n_action_steps=1`: 0/2 at
nas=1 → 11/16 at nas=4 (DEBUG_LOG Phase 4). The 3/8 eval was *already* buffered at nas=4 — `nas_ab.log`
reproduces exactly 3/8 at nas=4 on those same 8 seed-0 positions — so the 3/8 → 11/16 gap is a
**different episode set**, not a config change. Either way this is a **working VLA with a
success/failure mix** — exactly what the corrective loop needs (successes to calibrate "normal,"
failures for the corrector to fix).

Sample eval frames: `results/eval_frames/` (32 frames of a rollout).

---

## Calibration — the "aha" separation test

Does the calibrated SMD+RND+ACE uncertainty separate successes from failures? **Yes — but RND does all
the work.** (16-episode set, 10 success / 6 fail, `results/calib/report.json`.)

- Task-success rate in the calibration set: **10/16** (buffered, nas=4)
- **Combined AUC (fail vs success): 1.0** · threshold 0.107 · caught 100% · false-alarm 10%
- **Per-detector AUC: RND 1.0 · SMD 0.52 · ACE 0.43**
- Lead time: ~127 steps (of 130 max → flags from ~frame 0)

> **Read the 1.0 correctly.** It is **in-sample** — RND is fit on the same 10 successes it is then
> scored against — **single-detector** (RND alone carries it; SMD 0.52 and ACE 0.43 add nothing), and
> n=16. It is **not comparable** to the held-out cross-embodiment AUCs in `docs/aha/` (0.68–0.92 across
> RLBench tasks), nor to the held-out SO-101 + SmolVLA calibration in `docs/results/calibration/`. Every
> "AUC 1.0" in this file carries this caveat.

### What this means (honest read)
- **RND (vision-embedding novelty via SigLIP) perfectly separates** success from failure and is
  **cheap** (a tens-of-ms SigLIP embed, NOT a VLM forward). This is the corrector's trigger.
- **ACE (0.43) is below chance — it contributes nothing** as implemented (I reduce each sampled chunk
  to its FIRST action, `chunk[:,0,:]`, instead of using the full horizon). Since ACE sampling is ~75%
  of the per-frame compute (3 VLM forwards at ace-b=3), and RND alone gives AUC 1.0, **dropping ACE
  makes the loop ~4× faster** with no loss of separation. Speed win + correctness fix in one.
- **Caveats:** AUC 1.0 is **in-sample** (RND fit on the 10 successes) + small-N → it will soften on
  held-out data. The ~127-step lead means RND flags from ~frame 0 — likely keying on **initial cube
  position** (failures cluster on harder/edge placements that look OOD immediately), i.e. detecting
  *difficulty* more than *in-progress failure*. Worth knowing for how we frame the corrector.
- Full-chunk ACE (use the whole `(n,T,D)` chunk, not just step 0) is a free future lever if we want a
  policy-intrinsic signal that doesn't depend on the vision embedding — but RND suffices for now.

---

## Stage 2 — does corrective prompting help? (the gate)

Ran each of 6 instruction variants as the full-episode task over the same 8 positions calibration used
(so `base` sanity-checks the driver). `results/logs/prompt_sweep.log`.

```
base           3/8 (0.38)      procedural     4/8 (0.50)  Δ=+0.12  ← best
red            3/8 (=base)     locate         2/8 (0.25)  Δ=-0.12  ← worst
firm/descend   3/8 (=base)
```

**Partial pass — the mechanism works, modestly, and situation-dependently:**
- ~~The VLA **is** steerable by language~~ (**superseded** — see the retraction below and "Steerability
  tests"): prompts change outcomes per-position (pos3: `base` fails but all 5 corrections succeed; pos7:
  only `procedural` succeeds; pos0: `base` succeeds but corrections break it).

  > **RETRACTED (2026-07-09):** this bullet originally read "the steerability probe measured
  > **0.3–9.2°** of action change per re-instruction — non-zero but small." Those per-episode `probeΔ`
  > values come from `results/logs/stage2_scripted.log`, produced by the **first Stage-2 driver, which
  > is broken**: it called `vla.predict` two extra times per spike to take the measurement, and
  > `predict` **pops the execution queue**, so the probe desynced the arm's committed trajectory
  > (DEBUG_LOG Phase 5). The deltas it reported are that desync, not language sensitivity. The clean
  > successor probe — which measures at a *frozen* state without consuming the execution buffer —
  > reports a **grand mean of 0.02°/joint** (`results/steerability_tests/tests.log`). **0.02, not
  > 0.3–9.2, is the number**, and the whole conclusion of this project rests on it. See "Steerability
  > tests" below. The per-position outcomes in this section stand as recorded (they came from the
  > rewritten full-episode prompt sweep, not the broken driver) — but with a frozen-state delta of
  > 0.02°/joint, they are consistent with VLA flow-sampling noise and are **not** evidence of
  > steerability.
- A **procedural** instruction ("lower onto the cube, close to grasp, lift, place") nets **+12%**.
- **No fixed prompt wins everywhere** — same prompt helps some positions, hurts others. This is precisely
  the gap a *trained, per-state* corrector (GRPO) is meant to fill; a fixed prompt structurally can't.
- Hard/edge cube positions fail under every prompt — a VLA-capability ceiling, not a correction-policy one.
- Caveat: the VLA was finetuned on ONE fixed string, so novel instructions are mildly OOD; a corrector
  that stays close to the training phrasing (like `procedural`) does best.

## Detector findings (why RND wins, SMD/ACE don't)

- **SMD is real but sample-dependent: 0.52 then 0.78 across two 10/16-success runs.** The 0.78 run
  **matches your SmolVLA ~0.8** — so SMD *does* work; the earlier 0.52 was just a weaker-separation draw
  (which failures you happen to get). Why it's borderline here at all: our failures don't drive the arm to
  grossly abnormal joint poses — the arm does roughly the right motion but **fails to secure the cube
  (gripper stays open ~52° vs ~24° closed)**. That gripper signal + shoulder-pan (failures cluster at
  harder off-center positions, sep 2.17) exist at the frame level but get **diluted** in the full-6-joint
  Mahalanobis (high-variance elbow/lift that don't separate) and the max-per-run aggregation. Vision (RND)
  directly sees "cube not grasped"; proprio (SMD) only sometimes.
- **ACE: switched N-chunk sampling → single-forward action-spread** (SmolVLA's `action_head_std` analog:
  per-dim std of the executed chunk over its horizon, captured for FREE during the buffered `predict`, zero
  extra forwards). AUC dropped 0.43 → **0.167** — the action-spread is a *weak, inverted* failure signal
  for this task (it tracks motion decisiveness, not failure). So ACE isn't a useful detector here either —
  but it's now free, and **RND carries the combined AUC to 1.0 regardless**. Net: dropped ~75% of the
  per-frame compute (no N× sampling) at no cost to separation. `results/logs/facalib.log`.

### Fast-ACE calibration (final, 10/16 task success)
Combined AUC **1.0** · threshold 0.109 · caught 100% · false-alarm 10% · **SMD 0.78 · RND 1.0 · ACE 0.17**.
Same caveat as above: **in-sample and RND-only**, n=16 — not a held-out number.

## RL-environment smoke test — the loop works on the live VLA

`rl_env_smoke.py` drives `EpisodeManager` (the core the rollout server + verifiers env wrap) end-to-end on
the finetuned VLA: **start → RND spike → inject a corrective instruction → resume → verifiable reward.**

```
6/6 episodes fired a spike · correction injected · reward returned
rewards: [0, 0, 1, 1, 1, 0]  → mean 0.50
```

Every stage of the loop is functional with the real VLA + calibrated detectors, and the reward has the
**variance GRPO needs** (mix of 0/1). What remains for actual RL *tuning*: wrap this in the verifiers
`MultiTurnEnv` over HTTP (Stage-0-tested with a stub) and stand up the **prime-rl GRPO trainer** (2 GPUs:
vLLM inference serving Qwen3-VL + FSDP/LoRA trainer). The env + reward path is proven; the trainer is the
remaining integration.

## Stage 3 — prime-rl GRPO: the signal path, proven end-to-end

Stood up the whole trainer on a **datacrunch 2× A100-80GB** pod: the finetuned-VLA rollout server (GPU1
:8100) + prime-rl (vLLM **Qwen3-VL-4B** corrector on GPU0 + FSDP/LoRA trainer on GPU1 + orchestrator running
our `vla-corrector` verifiers env). It **runs end-to-end** — sim reward → GRPO grouping/advantages → LoRA
trainer steps.

- **Smoke** (`smoke.toml`, 3 steps): all 3 steps executed, reward flowed from the sim into the trainer — but
  uniform reward (every episode failed) → `Trainable 0/4` (zero advantage) → no *meaningful* update. (The
  prime-rl multimodal example warns about exactly this "uniform early steps" case.)
- **Modest run** (`grpo_qwen3vl_4b.toml`, 20 steps, group 8, **shaping=true**): the continuous
  uncertainty-reduction shaping created within-group reward variance →

```
Step 0 | Reward -0.1490 | Trainable 8/9  (88.9%)
Step 1 | Reward -0.1490 | Trainable 13/15 (86.7%)
Step 2 | Reward -0.1486 | Trainable 6/9  (66.7%)
Step 3 | Reward -0.1494 | Trainable 7/7  (100%)
```

**66–100% of rollouts trainable → non-zero advantages → the corrector's LoRA weights are actually updated
from the sim reward.** That's the milestone: **reward → advantages → LoRA weight update**, on a real VLM
corrector over the real MolmoAct2-in-sim loop.

**What it doesn't yet do: *learn* (raise success).** Reward variance is thin — the VLA fails most episodes and
RND-uncertainty barely drops with corrections (matches the Stage-2 finding: corrections are modest and
situational). Next levers: curriculum/easier episodes where the VLA sometimes succeeds, a denser reward, or a
LoRA-VLM VLA finetune to lift the baseline. Getting the loop *running* was the goal here; making it *climb* is
the follow-on.

## H200 single-GPU session (dense reward + curriculum) — validated a lot, hit a 1-GPU wall

No 2× GPU nodes were available, so this session ran everything **colocated on one H200-141GB**. Outcome is
honest and mixed:

**✅ Validated (the pipeline is real):**
- Dense **staged-progress reward** + **curriculum sampler** built, unit-tested (10 green), and running in the
  live env (`vla-corrector ready: num_tasks=64`).
- **Recalibration** on the current 1536-dim embedding pipeline: **RND AUC 1.0**, threshold 0.083 (the old
  2880-dim calib was stale — see DEBUG_LOG Phase 7). Again **in-sample and single-detector** — RND is fit
  on the same successes it scores; not comparable to the held-out AHA numbers.
- **Frontier profile** ran (48 baseline episodes): the finetuned VLA is fairly *binary* across the grid —
  only **1 cube position** (`[0.275, 0.0]`) fell in the 0.2–0.8 "flippable" band. A real finding: limited
  headroom for a corrector, and a finer grid / more-capable VLA is the lever.
- **Colocated single-GPU serving proven**: MolmoAct2 rollout server + Qwen3-VL-4B vLLM + trainer all on one
  card (~88 GB/141 GB); the verifiers env drove **spike→correct→resume→reward** with the corrector invoked
  (`Turns 4.0`) and the orchestrator computing non-zero advantages (`Trainable 4/4`).
- Checkpoint + calib pushed to **HF** (`aryanmadhavverma/molmoact2-so101-ft`) → pod is disposable.

**❌ Blocked (no NEW training milestone this session):** sustained GRPO training. On 1 GPU we're forced into
prime-rl's **manual 3-process split**, which deadlocks at ~step 3 — the trainer starts (torchrun fix) but
hangs `Waiting for run 0 to be created`; the orchestrator/trainer **run-registration handshake** that the
combined `rl` launcher provides isn't reproduced by hand. Tried batch 8→4 and `max_off_policy_steps` 0→16;
identical wall each time. `outputs/weights/` stayed empty → **the LoRA was never actually updated this
session.** (The earlier 2×A100 session — combined `rl` — is where real LoRA updates were proven.)

**➡️ Clear next step:** a **2-GPU node** + `uv run rl @ config` (the proven combined entrypoint that wires the
coordination). Do NOT hand-split on 1 GPU. Everything else (dense reward, curriculum, calib, colocated
serving) is ready to plug in. Pod terminated; validation logs in `results/h200_run/`.

## A100 2-GPU VISION run (2026-07-09) — the loop works end-to-end; doesn't climb yet

Went back to the **proven combined path** on a **2× A100-80GB** node (datacrunch, $3.58/hr): our rollout
server on GPU1 + **one `uv run rl @ grpo_qwen3vl_4b.toml`** command (inference GPU0 + trainer GPU1 +
orchestrator). Checkpoint + calib **pulled from HF in minutes** (no upload saga). Dry-run validated the
config first (free).

**✅ THE INFRA MILESTONE (what broke on 1 GPU) IS PROVEN ON THE VISION CORRECTOR:**
- **40 GRPO steps, zero deadlock** — the combined launcher owns the trainer↔orchestrator run-registration
  handshake that the 1-GPU manual split couldn't. Steps ~30–60 s each.
- **Qwen3-VL-4B corrector, `include_image=true`** — actually seeing the spike frames.
- **`Max Off-Policy 1–2`** every step → the trainer is genuinely consuming batches + updating the LoRA
  (async pipelining working). **The adapter trained + saved** → HF `aryanmadhavverma/qwen3vl-so101-corrector`
  (66 MB).

**❌ HONEST NEGATIVE — it trains but does NOT climb:**
```
Step 1  Reward -0.0594   Step 20 Reward -0.0594   Step 40 Reward -0.0577   (flat across all 40)
```
`Trainable` was **non-zero on 39 of the 40 steps** — 14 distinct values ranging from 2/6 (33%) up to
8/8 (100%), with **one step at 0/8** (step 33, no gradient at all) — so GRPO had a gradient on nearly
every step from dense-reward/shaping variance, but average reward never rose. Full per-step table:
`results/a100_vision_run/training_curve.txt`. The corrector's interventions don't flip episodes to success with this VLA + the narrow
1-position curriculum (matches the Stage-2 "corrections are weak/situational" finding). **The loop learns
*something* each step; there's just no real success signal to climb.** Levers for a climb (next iteration):
a more language-steerable VLA (refinetune on varied instructions), a reward that credits real
progress-to-success (not just shaping noise), and a wider curriculum with more flippable positions (the
frontier was only 1 — the VLA is too binary).

**Net:** the *"train a vision corrector over a VLA via verifiers + prime-rl"* pipeline is fully working and
reproducible on Prime 2-GPU compute. Making it *improve the policy* is the open research problem, not an
infra one.

## Steerability tests (2026-07-09) — WHY it didn't climb, settled by two independent tests

Before concluding "text correction doesn't work," we ran two cheap 1-GPU tests (no training) that isolate the
three explanations we'd conflated. **They agree decisively: the finetuned VLA is deaf to language.**

**Test #2 — mechanistic (do the ACTIONS change?):** at frozen states, sample the VLA under OPPOSITE
instructions ("move left" vs "move right", "close gripper" vs "keep open") and measure the action delta,
averaging out flow-noise. Result: **grand-mean 0.02°/joint** — essentially zero. The VLA produces the same
action regardless of the instruction.

**Test #1 — behavioral (does SUCCESS change?):** baseline vs a hand-crafted correction injected at the first
spike, over 6 positions × 2 trials. Result: **baseline 4/12 (33%); best correction 5/12 (42%, Δ +0.08 = one
extra success, within noise); two corrections Δ +0.00.** Corrections do not recover failures.

**Conclusion:** not a reward/curriculum/RL problem — **no reward can steer a deaf actuator.** Root cause: we
finetuned MolmoAct2 on ONE fixed instruction string, collapsing its language-conditioning for this task.
Raw logs: `results/steerability_tests/tests.log`. Drivers: `drishti/drivers/{steerability_probe,correction_ab}.py`.

**Then we tested the BASE (zero-shot) MolmoAct2 too** (`steerability_probe --policy base`, fine + task-level
pairs) — to check whether refinetuning could *restore* a lost steerability:

| policy | FINE (corrections) | TASK (whole goals) |
|---|---|---|
| base MolmoAct2 (zero-shot) | **0.01** | **0.01** |
| our finetuned | 0.02 | — |

**The base ignores even whole-task switches** ("pick up the cube" vs "push it off the table" → 0.01). So the
"our finetune collapsed a real steerability" story is **NOT supported** — there's nothing to restore, and
**refinetuning on varied instructions has no evidence behind it.**

**Honest state (two confounds, can't cleanly blame the architecture):** in OUR sim both VLAs ignore language —
decisive that the corrector can't work as-is. But we can't say "MolmoAct2 is architecturally deaf": the
finetuned result is confounded by our one-string finetune, and the base result is confounded by our sim being
**OOD to the base** (it approaches the cube 6.7→2.5cm but never grasps, so its ~0 delta could be
OOD-confusion, not deafness). Definitive test to distinguish: base MolmoAct2 on its *native* inputs (Ai2's
eval), not our sim. Raw: `results/steerability_tests/base_probe.log`.

**The pivot this points to:** drop LANGUAGE correction; use **action-level correction** — when uncertainty
spikes, SMD already gives the direction back onto the success-manifold, so the corrector emits a corrective
*action/state* instead of words. This sidesteps "is the VLA steerable by language" entirely (it doesn't need
to hear language). That's the more defensible direction now. (Also on the table for Prime: **continuous-space
policy RL** in prime-rl — see PRIME_RL_CONSTRAINTS.md — which would let the VLA itself be RL-explored
directly in the sim, no language corrector needed.)

## Language-diversity finetune (2026-07-21/22) — tested the root-cause fix, honest negative

> **⚠ NO RAW LOGS FOR THIS SECTION.** Everything from here to the end of the success-rate follow-up —
> both probe tables and the whole n=120 success sweep — was run on a Prime Intellect pod that was
> **terminated without the logs being pulled**; `results/success_eval/` is empty and there is no
> committed artifact for these numbers. They are transcribed from the live session. Unlike every other
> number in this file, **a reader cannot check them**. The two trained checkpoints themselves are on HF
> (`molmoact2-so101-ft-phasecap-frozen`, `molmoact2-so101-ft-phasecap-lora`) and the drivers are in the
> repo (`drishti/drivers/{steerability_probe,phase_caption_probe,success_eval}.py`), so the runs are
> **reproducible** — but they have not been re-run.

Direct follow-up to the steerability conclusion above: outside researchers suggested the VLA
ignored language because it was finetuned on **one static instruction string** applied to every
training frame — vision+proprio alone determined every action, so there was no gradient pressure
to ever attend to language. Tested the fix: relabel the scripted demos with **3 sub-goal captions
that change over the course of each episode** ("reach toward the cube and grasp it" → "lift the
cube and carry it to the tray" → "place the cube in the tray and let go"), sized to the model's
16-step action-chunk conditioning granularity, on a grown 150-episode demo set (up 3x from 50).
Fixed a real, separate data-quality bug along the way: the demo generator's cube-position range let
33/150 (originally 50) episodes spawn the cube visually overlapping the tray and 7/50 spawn already
inside the success-oracle's tolerance box — tightened `X_RANGE` in `drishti/sim/record.py` and confirmed
0/150 affected in the regenerated set.

Ran **two finetunes** on this identical phase-captioned dataset, isolating a second variable the
user proposed testing: whether the VLM backbone needs to be trainable at all, not just the action
expert. `aryanmadhavverma/molmoact2-so101-ft-phasecap-frozen` (`train_action_expert_only=true`,
matches the original recipe exactly — VLM 100% frozen) and
`aryanmadhavverma/molmoact2-so101-ft-phasecap-lora` (`enable_lora_vlm=true` +
`enable_lora_action_expert=false` — LoRA adapters on the VLM's attention/MLP layers, full
finetune on the action expert). Both: 1500 steps, batch 16, identical loss trajectories
(0.08→0.03→0.02→0.015ish by step 800), Prime Intellect crusoecloud A100-80GB, $6.04 total,
~1hr (frozen) + ~2h12m (LoRA — backprop now traverses the VLM's internals, ~2.2x slower/step
as predicted).

**Result — clean negative on both configs, on both test types:**

| checkpoint | steerability_probe FINE | steerability_probe TASK | own-trained-vocabulary probe |
|---|---|---|---|
| frozen VLM | 0.00 | 0.01 | 0.01 |
| LoRA-VLM | 0.01 | 0.01 | 0.00 |
| *original baseline (for reference)* | *0.02* | — | — |
| *base zero-shot (for reference)* | *0.01* | *0.01* | — |

Both checkpoints land right on top of the original "near-deaf" baseline and the untrained
zero-shot base model — nowhere near the code's own >2 "near-deaf floor" threshold, let alone the
>10 "clearly steerable" bar. Ran a second, more targeted probe (`phase_caption_probe.py`) that
swaps between the model's **own trained captions** (not the generic held-out FINE/TASK phrasing)
at a frozen state, specifically to rule out "doesn't generalize to novel phrasing" as a confound —
same result, both checkpoints ~0.00-0.01 even on their own training vocabulary. Neither more data
diversity nor an unfrozen, LoRA-adaptable backbone produced any language steerability.

**Why (the sharper diagnosis this result points to):** even though the caption now varies *over
time* within an episode, each caption's time window still maps to a unique, fully
vision-determined state — the model can still solve the task perfectly from vision+proprio alone,
so language remains completely redundant with vision, exactly as before. Temporal variation alone
doesn't create the thing that forces attention to language: **the same visual state needing a
different correct action depending on the instruction.** This reframes the earlier-deferred v2
idea (two placement targets, so the same cube position requires a different correct action
depending on which tray the instruction names) from "a nice-to-have stretch goal" to "probably the
actual necessary condition" — worth prioritizing over further temporal-only caption schemes.

**Not run:** `correction_ab.py` (live mid-trajectory steering) and the RL-corrector-loop
re-run — both gated on the cheap steerability probes showing a real signal first, per plan, and
neither did. No point spending GPU time on the live-steering path when the static probe already
shows zero action-language coupling to steer with.

### Success-rate follow-up (2026-07-22) — the audit gap, closed; still a null

The steerability probe only measures action delta at a *frozen* state under *contradictory*
instructions. It cannot see a language effect that requires the caption to *agree* with the
visual phase and to accumulate over a full rollout — so we ran closed-loop task success too
(`drishti/drivers/success_eval.py`), driving each checkpoint with its **own trained phase
captions injected at the geometry-appropriate moment** (approach/descend→"reach and grasp",
lift/carry→"lift and carry", place→"place and release"), vs the static string. A6000, $2.46.

An initial n=20/condition sweep *hinted* at a real effect (phase captions +15-20 pts over static
on both checkpoints). **It did not survive a bigger sample.** Pooled n=60/condition (10 cube
positions, seed 0):

| checkpoint | phase-caption inference | static-string inference | Δ (same weights) |
|---|---|---|---|
| frozen-phasecap | 31/60 (52%) | 26/60 (43%) | +8 pts (~1.4σ) |
| lora-phasecap | 28/60 (47%) | 27/60 (45%) | +2 pts (noise) |
| **pooled both** | **59/120 (49%)** | **53/120 (44%)** | **+5 pts (~0.8σ)** |

**+5 pts at 0.8σ = not significant.** The lora model shows essentially zero effect. The initial
n=20 signal was small-sample luck that regressed to the mean — so **feeding the trained captions
at inference does not reliably change task success**, confirming (not overturning) the frozen-state
probe from two independent angles. Honest note on process: the n=20 result was flagged at the time
as ~2σ / "could soften," and it did; chasing it to n=120 rather than stopping at the flattering
number is what turned a false positive into a clean null.

Separately, on whether the 150-episode models beat the *original* 50-episode checkpoint: **this
comparison does not hold up, and the "vs 30%" it was originally stated against was wrong.** The
annotated models score ~47–52% (10 cube positions from the *tightened* range, buffered nas=4). The
original checkpoint, also buffered at nas=4, measured **3/8 = 37.5%** on the 8 seed-0 eval positions
(`results/logs/eval_ft.log` — the cited source; it never said 30%) and **11/16 = 68.75%** on the
16-episode recording (`results/logs/rec_calib.log`). So depending on which of the original's two
measurements you pick, the annotated models look either better or worse — and the cube-position range
was *changed between them* anyway (`X_RANGE` was tightened for the 150-episode set), so the position
distributions are not the same. **Neither "more data helped" nor "annotation helped" is supported by
this file.** What is supported: within the annotated models, captioning barely moves the number.

### Audit of the whole pipeline (2026-07-22) — what was and wasn't done right

A full skeptical audit (probe validity, data pipeline, RL env) found the core experiment was set
up correctly — the steerability probe genuinely threads the instruction into the transformer as
re-tokenized text; **Run B's LoRA genuinely trains the language pathway** (regex matches the LLM
decoder's `att_proj`/`attn_out`/`ff_proj`/`ff_out`, verified against the HF module tree — so
"LoRA-VLM didn't help" is a valid test, not a plumbing no-op, which *strengthens* the negative);
captions land on the right frames; base behavior was preserved, not destroyed. Real secondary
flaws worth recording:
- **Chunk-boundary label noise**: with `chunk_size=16` and macro-phases of 28/18/17 frames,
  ~48% of training examples pair a caption with a 16-step action target that partly belongs to a
  different phase; ~22% have the *majority* of the target in the next phase. The `phase_captions.py`
  docstring's rationale ("phases ≥ chunk_size prevents decorrelation") is wrong — it only protects
  each phase's *first* frame. Secondary confound, not the root cause (captions are redundant with
  vision regardless).
- **Undertrained vs the original**: 2.5 epochs (150 ep, 1500 steps) vs the original's 7.6 (50 ep).
  Loss still converged (~0.015), likely minor.
- **RL-env design flaws (the earlier GRPO loop, for the record)**: the RND uncertainty spike fires
  at ~frame 4 on a *start* state (it keys on cube-position difficulty, not in-progress failure), so
  the corrector is asked to fix a failure that hasn't happened; immediate re-spiking likely
  truncates episodes; and GRPO's within-group advantage is **zero by construction** if the VLA
  ignores text (all within-group reward variance is VLA sampling noise), which makes the "dense
  progress fixes flat reward" claim invalid, not just weak. The corrective *prompts* themselves were
  well-posed — not the problem. These would have suppressed signal independently even if language
  were a lever.

Cost for the training + probe part of this round: **$7.43** ($6.04 training + $0.48 + $0.91 eval, two
eval pods — the first was under-provisioned on disk for holding both the base checkpoint and a
finetuned checkpoint simultaneously, terminated and recreated with more disk). This figure **predates
and excludes** the $2.46 A6000 success-eval pod above; see the accounting note below.

## Status & cost

**GPU spend — this section is the single accounting for the rl-corrector work.** Other docs quote
different figures; where they disagree, this section is the one to read. (`DEBUG_LOG.md`'s "$5–6" is a
*running* snapshot taken mid-arc, not a final total. Any round figure in the blog is a summary of the
per-round numbers below, not a separate measurement.)

**Scope:** rl-corrector pods only — the MolmoAct2 finetunes, calibration recordings, prompt/steerability
probes, the GRPO runs, and the phase-caption follow-up. **Excludes** the AHA/RLBench generation pods
(`docs/aha/`), the SmolVLA work, and any local/Mac compute.

**What was actually recorded**, per run, as noted from the provider dashboard at the time:

| Round | Figure quoted in this doc |
|---|---|
| Original RL-corrector arc (finetune → calibration → Stage 2 → GRPO smoke/modest → H200 → A100 vision run) | ≈ **$7** |
| Language-diversity finetunes + probes | **$7.43** ($6.04 training + $0.48 + $0.91, two eval pods) |
| Success-rate follow-up (A6000) | **$2.46** |

**No billing export or cost artifact was retained**, so none of these can be checked against a receipt,
and they **do not reconcile to a clean total** — the $7.43 "whole round" line above was written before
the $2.46 pod and does not include it. Treat them as approximate per-run costs, not an audited sum.
Reference rates that *are* recorded: massedcompute A100-80GB $1.23/hr, datacrunch 2× A100-80GB
$3.58/hr. **All pods terminated** — no ongoing billing.

- Finetuned checkpoint (11 GB) backed up at `data/molmoact2_so101_ft_ckpt/` — re-deployable without retraining.

## Artifact index (`results/`)

- `logs/ft_full.log` — the full finetune training log (loss curve).
- `logs/ft_val.log`, `logs/ft_install*.log` — training validation + pod setup logs.
- `logs/tf_check.log` — teacher-forced diagnostic (the frame-bug smoking gun).
- `logs/eval_ft.log` — frame-corrected closed-loop eval (3/8).
- `logs/rec_calib.log` — 16-episode buffered recording (11/16).
- `logs/rec_validate.log` — early 2-episode timing validation.
- `eval_frames/` — rendered rollout frames.
- `calib/` — calibration report + detector artifacts (pulled when ready).
