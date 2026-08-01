# Project log — what we tried, what happened

A short, high-level index across drishti's active initiatives. Each section is a few bullets;
for the full trail (logs, numbers, gotchas) follow the link to the detailed doc. Where a number
here and a number in a detail doc disagree, **the detail doc wins** — it is the one that names its
artifact.

---

## The platform itself — does calibrated uncertainty separate success from failure? (the baseline result)

The thing the two studies below are built on. SO-101 pick-place in MuJoCo with the finetuned
SmolVLA policy, SMD+RND+ACE scored per frame: **combined AUC 0.9955 over 60 rollouts** (34 success /
26 failure, fit on 28 successful runs), 100% of failures caught at 17.6% false alarms, mean lead
99.3 steps. Regenerate deterministically from the bundled dataset with `drishti calibrate --demo
--auto`; `tests/test_demo_bundle.py` asserts it reproduces.
Detail: [`docs/results/calibration/report.json`](results/calibration/report.json).

---

## rl-corrector — RL-trained corrective VLM over a VLA (concluded, pivoted)

Goal: train a Qwen3-VL "corrector" via GRPO that watches a MolmoAct2 VLA doing SO-101 pick-place
in sim, and when calibrated uncertainty (SMD/RND/ACE) spikes, injects a corrective instruction.

- Built the full stack: MolmoAct2 finetuned on 50 sim demos (loss 0.881→0.019, **11/16 = 69% task
  success** on the 16-episode calibration recording under buffered execution, `n_action_steps=4` —
  task success, not grasp rate; the same weights score 3/8 on the fixed 8-position eval set),
  calibrated UQ (combined AUC 1.0 — but *in-sample*, RND-only, n=16), a custom `verifiers` multi-turn
  env, and self-hosted `prime-rl` GRPO.
- **Infra milestone proven end-to-end**: reward → GRPO advantages → LoRA weight update, on a
  real Qwen3-VL-4B corrector over the live MolmoAct2-in-sim loop (2×A100 pod, 40 steps, adapter
  trained + pushed to HF).
- **Honest negative**: reward stayed flat across every run. Two decisive follow-up tests proved
  the finetuned (and base) VLA is essentially deaf to language — opposite instructions move the
  predicted action by 0.02°/joint (finetuned) and 0.01°/joint (base, zero-shot).
- **Root cause (confirmed)**: finetuned on ONE fixed instruction string, applied identically to
  every training frame — vision+proprio alone determined every action, so there was no gradient
  pressure to ever attend to language.
- **Spun out**: `environments/vla-corrector` generalized into a public, bring-your-own-VLM Prime
  environment with a reference rollout server + bundled UQ.
- Detail: [`docs/rl-corrector/RESULTS.md`](rl-corrector/RESULTS.md),
  [`docs/rl-corrector/DEBUG_LOG.md`](rl-corrector/DEBUG_LOG.md).

## AHA failure-detection benchmark — does SMD/RND generalize beyond SO-101? (active)

Goal: test whether drishti's failure detectors (SMD on proprioception, RND on vision embedding)
hold up on NVlabs' AHA (RLBench + FailGen injected failures) — a different embodiment (Franka,
not SO-101), simulator (CoppeliaSim, not MuJoCo), and generation process (procedural failure
injection, not organic policy failure).

- Built the full offline-reproducible pipeline: `scripts/gen_aha.py` → `ingest_aha.py` →
  `eval_aha.py`/`eval_aha_seeds.py`, ablation-ready (per-task calibration, encoder choice,
  state/embed windowing all offline sweeps).
- Key findings along the way: **calibrate per-task, not pooled** (biggest lever); **DINOv2 >
  SigLIP** for RND; **temporal windowing fixes the "wrong_sequence" blind spot** (0.57 → 0.91);
  touch/force channels are **not** the missing piece for grasp detection (vision is).
- Breadth campaign (5 tasks, ~10 successes/task — 7 calibrate / 3 held-out) found uneven detection
  (AUC 0.22–0.82) and a degenerate threshold — **mostly a data-starvation artifact**, confirmed by
  the v2 run.
- **Current best result (v2, 2026-07-11)**: 20 successes/task (15 calibrate / 5 held-out),
  **uniform 0.68–0.92 AUC across all 5 tasks**, no inversions, no degenerate threshold — real
  evidence the detector approach generalizes across embodiment/sim.
- Detail: [`docs/aha/SUMMARY.md`](aha/SUMMARY.md), [`docs/aha/RESULTS.md`](aha/RESULTS.md).

## Language-diversity finetune — does time-varying language fix the steerability gap? (concluded, negative)

Goal: the rl-corrector pivot's root-cause fix, tested. Relabeled the existing scripted SO-101
demos with 3 sub-goal captions that change over the course of each episode ("reach toward the
cube and grasp it" → "lift the cube and carry it to the tray" → "place the cube in the tray and
let go"), instead of one static sentence, on a grown 150-episode set (also fixed a real
data-quality bug found along the way: 33/150 demos had the cube spawning visually overlapping
the tray, 7/50 already inside the success box — tightened the position range, 0/150 affected
after). **Every number in this section was recorded from a terminated pod; the raw logs were not
retained** — see the warning at the top of the RESULTS.md section. The checkpoints and drivers are
in the repo, so the runs are reproducible, but a reader cannot check these figures against an
artifact.

- Tested **two** finetunes on the identical dataset: frozen-VLM/action-expert-only (matches the
  original recipe exactly) and LoRA-on-VLM + full action-expert finetune (the user's proposal —
  isolates whether the backbone needs to be trainable at all, not just captioned data). $6.04,
  Prime Intellect A100-80GB, ~1h (frozen) + ~2h12m (LoRA).
- **Result: clean negative on both configs.** Both land at 0.00-0.01°/joint action delta under
  contrasting instructions — indistinguishable from the original 0.02 baseline and the untrained
  zero-shot base model. Ran a second, more targeted probe swapping the model's *own trained
  captions* (not held-out phrasing) to rule out "doesn't generalize to novel wording" as a
  confound — same near-zero result either way.
- **Success-rate follow-up (the audit gap, closed).** Steerability probe only sees frozen-state
  action delta; ran closed-loop success too, driving each model with its own phase captions at
  the right moments vs the static string. An n=20 sweep hinted at +15-20 pts for captions, but it
  **did not survive** — at n=60/condition per checkpoint (n=120/condition pooled) it collapsed to
  +5 pts at ~0.8σ (lora ~0), i.e. noise. Confirms the negative from a second angle. The annotated
  150-ep models do beat the original 50-ep model (~47-52% vs 30% on comparable positions), but
  that's the 3× data, not the captioning.
- **Full pipeline audit** confirmed the experiment was set up right (probe threads language;
  Run B's LoRA genuinely trains the LLM's language layers — verified against the module tree — so
  the negative is real, not a plumbing no-op). Secondary flaws recorded: ~22-48% chunk-boundary
  caption/action label noise (and a wrong docstring rationale, now fixed); 2.5 vs 7.6 training
  epochs; and the earlier GRPO loop's spike-timing + zero-group-variance design flaws.
- **Sharper diagnosis**: temporal caption variation alone doesn't create the thing that forces
  attention to language — each caption's time window still maps to a unique, fully
  vision-determined state, so language stays completely redundant with vision. The earlier
  deferred "two placement targets" idea (same cube position, different correct action depending
  on the named tray) looks like the actual necessary condition, not just a stretch goal.
- Not run (correctly gated by the plan): live mid-trajectory correction (`correction_ab.py`) and
  the RL-corrector-loop re-run — both need a real steerability signal first to be worth the GPU
  spend, and neither probe found one.
- Detail: [`docs/rl-corrector/RESULTS.md`](rl-corrector/RESULTS.md) ("Language-diversity
  finetune" section).
