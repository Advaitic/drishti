# AHA failure-detection benchmark — consolidated result

What we tested, how the data was made, what's captured, what we plugged into what, and the
per-task detection numbers. All offline-reproducible on `data/aha_breadth/` (+ `data/aha_ablation/`,
`data/aha_v2/`).

> **Update (2026-07-11, v2 run):** the breadth-campaign result below (0.22–0.82 AUC, degenerate
> 100%/100% threshold) was a **configuration artifact on two axes**. Re-running the same 5 tasks with
> 20 successes each (15 calibrate / 5 held-out, `data/aha_v2/`) *and* on the 17-D proprio-only state
> instead of the 30-D default gives a **uniform, non-degenerate 0.68–0.92 AUC across all 5 tasks** —
> no SMD inversion, no degenerate threshold. The two changes are confounded in the headline
> comparison and both matter: on `slide_block_to_target`, SMD goes 0.10 → 0.72 from the extra
> successes alone (holding 30-D), and 0.10 → 0.50 from dropping the force channels alone (holding
> the breadth data); together 0.10 → 0.89. Full writeup + the controlled 2×2:
> `docs/aha/RESULTS.md` ("V2 run" section) + `docs/aha/results/aha_v2_table.png`. Treat the
> per-task numbers in this document as **superseded** by the v2 result; the qualitative lessons
> below (per-task calibration, DINOv2 > SigLIP, windowing fixes wrong_sequence, force channels are
> not the missing piece) still hold.

## Read this first — what AHA does and does NOT test

**AHA measures detection of injected perturbations, not anticipation of a policy failing.** This is
the single most important caveat in the study and it bounds every number below.

Our detectors' founding intuition is **anticipatory**: a live policy hits OOD input → gets uncertain
→ that OOD-ness *precedes and causes* the failure, so we catch it early (this is what **ACE**, the
action-head spread, measures). **AHA tests none of that** — its failures are *scripted perturbations*
of a motion planner (clean execution to a wrong waypoint), with **no policy in the loop** (hence ACE
dropped). So AHA is a **detection** benchmark ("is this trajectory diverging from the success
manifold?"), not a **prediction** one ("will the model fail, before it does?").

That's still valid and useful — SMD/RND measure distance-from-success and fire wherever the trajectory
diverges, injected or not. Three caveats:
- The "there'll be an anomaly at the error" logic holds only for failures with a **state/vision
  footprint** (translation, slip). Failures that are anomaly-invisible per-frame (**wrong_sequence** —
  valid poses, wrong order) show nothing at the error until you look **temporally** (why the window
  helped). Semantic failures generally need a non-per-frame signal.
- AHA's geometric perturbations are often **bigger/cleaner** anomalies than a real policy's subtle
  drift → AHA may be **optimistic** on easy modes, **realistic** on hard ones. "0.8 on AHA" ≠ "0.8
  catching SmolVLA's real failures."
- **AHA AUCs are not comparable to the SO-101 calibration AUCs**, in either direction. AHA is
  **two-detector** (SMD + RND; no ACE) and **held-out** (fit and threshold on train successes, score
  held-out successes + all failures). The SO-101 numbers in `docs/results/calibration/report.json`
  (combined 0.9955, RND 1.000) are **three-signal** and **in-sample** — `calibrate` fits and
  validates on the same success runs. Different measurements; never put them in one table.

To test the **anticipatory** half, we'd run our actual policy (SmolVLA / MolmoAct) and catch *its*
organic failures — a separate experiment (ACE + a model in the loop).

## Simulation & embodiment

- **Simulator:** RLBench (task suite) running on **CoppeliaSim v4.1** (physics + OpenGL render) via
  PyRep. Failures are injected by **FailGen** (NVlabs' AHA framework), which procedurally perturbs
  a successful scripted demo (wrong grasp pose, translation offset, slip, out-of-order waypoints…)
  to produce a labeled *failure* trajectory of the same task.
- **Embodiment:** **Franka Emika Panda** — 7-DOF arm + parallel-jaw gripper (RLBench's default robot).
  Not our SO-101; a different, more "prod-like" arm, which is part of the point (stress-test beyond
  our narrow setup).

## How the data is generated

For each task we call FailGen's wrapper: `get_success()` for clean demos and `get_failure()` (one
mode enabled at a time) for labeled failures. Config trimmed to **RGB only, front + wrist cameras,
256²**. Modes generated = the 4 that trigger reliably: **grasp, slip, translation_z, wrong_sequence**
(of AHA's 11 — we skipped retry-hungry ones like wrong_object/rotation that stalled throughput).

## What's captured (per timestep) and what we store

RLBench's `Observation` carries a lot; we deliberately store a subset and derive embeddings:

- **State (30-D proprioceptive superset):** joint_positions(7) + joint_velocities(7) +
  **joint_forces(7)** + gripper_open(1) + gripper_joint_positions(2) + **gripper_touch_forces(6)**.
  A `field_slices` map lets any subset be sliced offline (e.g. ±forces).
  **Excluded on purpose:** `task_low_dim_state` (ground-truth object poses) — it would let the
  detector "cheat" instead of detecting anomalies.
- **Vision:** front + wrist **RGB frames** (kept, every 8th step) → embedded with **3 frozen
  encoders** (SigLIP, DINOv2-cls, DINOv2-mean), stored as `embed_<f>_<encoder>.npy`.
- **Labels:** binary success/failure + the `failure_type` per failure.

## What we plug into what (the detectors)

- **SMD** ← the state vector (optionally **windowed W=4** = last 4 states, for temporal/order signal)
  → one Gaussian fit on success states, Mahalanobis distance = anomaly score.
- **RND** ← the **DINOv2-cls** embedding of front+wrist (optionally **windowed EW=2**) → frozen
  random-target / trained-predictor MLP, prediction error = anomaly score.
- **ACE dropped** here — it needs a live policy's action-head spread; AHA trajectories are scripted.
- **Combine:** mean of capped-z (each detector normalized on the *success* distribution). Threshold
  = 90th percentile of success run-peaks. *(Production also includes ACE as a 3rd signal.)*
- **Calibration is per-task:** each task is fit + thresholded on its *own* successes (this is how a
  deployment would run — you know the task, load its profile).

## Tasks checked + counts

5 diverse tasks (pick/lift, cup-pick, slide, stack, lid-removal). Per task: **~10 successes**
(7 train / 3 held-out) + **8–12 failures** across the 4 modes.

## Results — per-task detection (winning config: DINOv2-cls + W4 + EW2, 8-seed AUC)

| Task | AUC (comb) | SMD | RND | which detector | caught% | false-alarm% |
|---|---|---|---|---|---|---|
| stack_blocks | **0.82** | 0.92 | 0.71 | both | 100 | 100 |
| pick_up_cup | **0.78** | 0.81 | 0.70 | both | 100 | 100 |
| pick_and_lift | 0.73 | 0.69 | 0.69 | both | 100 | 100 |
| take_lid_off_saucepan | 0.44 | 0.33 | 0.40 | neither (hard) | 100 | 100 |
| slide_block_to_target | **0.22** | 0.10↯ | 0.63 | RND only (SMD inverts) | 100 | 100 |

Reproduce (one run per task; **no `--state-fields`** — this table is the stored **30-D** state
*including* `joint_forces` + `gripper_touch_forces`, which is load-bearing for the slide_block row):

```bash
DRISHTI_OBS_DB_URL="sqlite:///data/aha_breadth/obs.db" DRISHTI_BLOB_DIR="data/aha_breadth/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha_seeds.py --source aha --split episode --train-frac 0.7 --k 8 \
  --embedder dinov2-cls --state-window 4 --embed-window 2 --task <TASK>
```

(See `RESULTS.md` for the reproduction caveat: SMD reproduces exactly on current code, the
RND-dependent columns do not.)

**Tasks with good detection:** stack_blocks, pick_up_cup, pick_and_lift (0.73–0.82) — pick/place/stack
tasks where the arm's joint configuration is discriminative (SMD strong) and the object state is
visible (RND solid).
**Where it breaks:** slide_block_to_target (SMD *inverts* to 0.10; RND still 0.63 but the
mean-combine drags the total to 0.22) and take_lid_off_saucepan (both detectors ~chance — genuinely
hard). We first read the inversion as "a sliding task's successes are so varied that failures fall
inside the manifold" — the v2 run shows the **force/touch channels** in this 30-D state are the
bigger cause: drop them and SMD goes 0.10 → 0.50 on this same data.

## The elephant: 7 training successes is far too few

- **AUC** (ranking) is the only trustworthy metric here, and even it is noisy: pick_and_lift was
  **0.93** on `data/aha_ablation/` vs **0.73** here — same config, different 10-success draw.
- **Threshold-based detection is degenerate:** caught = **100%** *and* false-alarm = **100%** on every
  task. With 7 train successes the 90th-pctl threshold sits so low everything crosses it → no usable
  operating point. **The deployable metric is uncalibratable at this scale.**
- **We have more successes stashed** (same 30-D+DINOv2 format): merging `aha_ablation` + `aha_breadth`
  gives **pick_and_lift 20, pick_up_cup 20, stack_blocks 13** successes — the immediate free lever to
  test whether more successes tighten the AUC and rescue the threshold.

## Headline findings

1. **Does not uniformly generalize** — per-task AUC spans 0.22–0.82.
2. **RND (DINOv2 vision) is the robust backbone** (0.40–0.71, never inverts); **SMD is task-fragile**
   and can invert (slide tasks).
3. **The mean-combine is a liability** when a detector inverts — a per-deployment **gate** (drop an
   uninformative/inverted detector, detectable from success calibration) would recover slide_block
   (~0.22 → ~0.63) without raising false alarms.
4. **Need far more successes per task** — both for AUC stability and for a *usable threshold*.
5. **SMD's state slice is a hyperparameter, not a capture detail** (added after the v2 run). This
   table is 30-D-with-forces; the proprio-only 17-D slice is worth **+0.40 SMD AUC on slide_block**
   on this exact data. Every table must therefore record its `--state-fields`, which this one
   originally did not.

## Immediate next steps (all offline/free on saved data)
- Merge datasets → 20 successes for pick_and_lift/pick_up_cup; re-run and check AUC + threshold.
- Prototype the **gate combine** and re-run the per-task table.
- Then generate more successes per task (short tasks only, for throughput) to get a real
  threshold and tighter per-task numbers.
