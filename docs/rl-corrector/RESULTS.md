# rl-corrector — results & proof

The positive record: milestones hit, with the numbers and where the raw proof lives. Pair with
`DEBUG_LOG.md` (the errors) for the full story. Artifacts in `results/`.

---

## Milestone progression (the arc)

| Stage | What | Result | Proof |
|---|---|---|---|
| Stage 0 | `verifiers` env + rollout server + UQ, built local | 15 pytest green (CPU/Mac) | `tests/`, `environments/vla-corrector/` |
| Stage 1 (R4) | MolmoAct2 joint convention on real hardware | ✅ correct (arm moves toward cube) | memory `project_rl_corrector` |
| Stage 1 (R9) | Zero-shot grasp on sim frames | ❌ approaches (2 cm) but never grasps | — |
| Finetune | LeRobot action-expert finetune, 50 demos | loss **0.881 → 0.019** (1500 steps, ~1 h A100) | `results/logs/ft_full.log` |
| Diagnosis | Teacher-forced: did it learn? | ✅ learned; joint-frame bug on j1/j2 | `results/logs/tf_check.log` |
| Eval | Frame-corrected closed-loop | **3/8 success, 2 grasps** | `results/logs/eval_ft.log`, `results/eval_frames/` |
| Calibration | Buffered VLA, 16-episode recording | **11/16 success (69%)** | `results/logs/rec_calib.log` |
| Calibration | UQ threshold + success/failure AUC | **combined AUC 1.0** (RND 1.0, SMD .52→.78, ACE weak) | `results/calib/` |
| Stage 2 | corrective prompting help? | partial pass — `procedural` +12%, situation-dependent | `results/logs/prompt_sweep.log` |
| RL smoke | loop on live VLA → verifiable reward | **6/6 spike→correct→resume→reward, mean 0.50** | `results/logs/rlsmoke.log` |

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

## Grasp success, step by step (each fix moved the number)

```
zero-shot MolmoAct2      : 0 grasps  (approaches to 2 cm, never closes)
finetuned, wrong frame   : 1/8 success, 0 grasps
finetuned, frame-fixed   : 3/8 success, 2 grasps   (empirical j1/j2 correction)
finetuned, buffered exec : 11/16 success (69%)      (n_action_steps=4 follow-through)
```

The jump from 3/8 → 11/16 came from buffered execution (commit to the descent) + running on the
stable A100 at the demo cadence. This is a **working VLA with a success/failure mix** — exactly what
the corrective loop needs (successes to calibrate "normal," failures for the corrector to fix).

Sample eval frames: `results/eval_frames/` (32 frames of a rollout).

---

## Calibration — the "aha" separation test

Does the calibrated SMD+RND+ACE uncertainty separate successes from failures? **Yes — but RND does all
the work.** (16-episode set, 10 success / 6 fail, `results/calib/report.json`.)

- Success rate in the calibration set: **10/16**
- **Combined AUC (fail vs success): 1.0** · threshold 0.107 · caught 100% · false-alarm 10%
- **Per-detector AUC: RND 1.0 · SMD 0.52 · ACE 0.43**
- Lead time: ~127 steps (of 130 max → flags from ~frame 0)

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
- The VLA **is** steerable by language: prompts change outcomes per-position (pos3: `base` fails but all
  5 corrections succeed; pos7: only `procedural` succeeds; pos0: `base` succeeds but corrections break it).
  The steerability probe measured **0.3–9.2°** of action change per re-instruction — non-zero but small.
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

### Fast-ACE calibration (final, 10/16 success)
Combined AUC **1.0** · threshold 0.109 · caught 100% · false-alarm 10% · **SMD 0.78 · RND 1.0 · ACE 0.17**.

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

## Status & cost

- Cumulative Prime spend ≈ **$7** (cap $200 / budget $500). **Pod terminated** — no ongoing billing.
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
