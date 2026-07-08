# rl-corrector — corrective-VLM RL over a VLA worker

Train a **corrective VLM** (Qwen3-VL) that supervises a frozen **VLA worker** (MolmoAct2) doing SO-101
pick-place in a MuJoCo sim. When the VLA's calibrated **uncertainty** spikes, the corrector sees the
failure frames + telemetry and emits a corrective instruction; **verifiable reward = sim success**;
train the corrector with **GRPO** on self-hosted **prime-rl**.

This is the entry point. Detail lives in:
- **[RESULTS.md](RESULTS.md)** — milestones + numbers (the positive record).
- **[DEBUG_LOG.md](DEBUG_LOG.md)** — every error, dead-end, and fix (6 phases, ~30 gotchas).
- **[../rl-corrector.md](../rl-corrector.md)** — the original engineering design/plan.
- **[../LEARNINGS.md](../LEARNINGS.md)** — the failure-detection foundation (SMD/RND/ACE, calibration).
- `results/` — raw proof (logs, calib report, eval frames, the trained GRPO LoRA adapter).

---

## Where this came from

Built on the **failure-detection foundation** (the `platform` branch): a MuJoCo SO-101 pick-place + a
live **uncertainty cascade** — **SMD** (proprio Mahalanobis) + **RND** (vision-embedding novelty) +
**ACE** (action-chunk spread) — calibrated so uncertainty separates success from failure, with a
telemetry spine + dashboard. That calibrated uncertainty signal is the asset this experiment builds on:
it's what decides *when* to call the corrector. `rl-corrector` branched cleanly off `platform`.

---

## The arc (all done + committed on `rl-corrector`)

| Stage | Result |
|---|---|
| Finetune the VLA | MolmoAct2 action-expert finetune → grasps **0 → ~69%** (joint-frame bug found + fixed) |
| Calibration | RND separates success/fail — **combined AUC 1.0**, threshold ~0.09 |
| Stage 2 (does prompting help?) | Partial pass — a *procedural* instruction nets **+12%**, situation-dependent |
| RL-env smoke | Full loop on the live VLA: spike→correct→resume→**reward**, 6/6, mean 0.50 |
| **Stage 3 — prime-rl GRPO** | **Signal path PROVEN: sim reward → GRPO advantages → LoRA weight update** on the Qwen3-VL-4B corrector (modest run: 66–100% trainable rollouts/step) |

**Honest state: the loop RUNS and UPDATES the corrector's weights, but doesn't yet raise task
success.** Reward stays ~flat because the VLA fails most episodes and the within-group reward variance
comes from the shaping term (uncertainty-reduction), not from the corrector making the VLA succeed more.
Getting the loop running end-to-end was the goal; making it *climb* is the open frontier.

---

## Architecture (the pieces + where they live)

```
                         corrector (Qwen3-VL-4B + LoRA, trained by GRPO)
                                        ▲ corrective instruction   │ reward (sim success)
   verifiers env (vla-corrector) ───────┘                          ▼
   environments/vla-corrector/vla_corrector.py   ── HTTP :8100 ──►  rollout server
   (thin httpx client; MultiTurnEnv)                                drishti/sim/rollout_server.py
                                                                    (SO101Env + finetuned VLA + UQ)
   prime-rl orchestrates: vLLM inference (GPU0) + FSDP/LoRA trainer (GPU1) + this env
   configs/rl/{smoke,grpo_qwen3vl_4b}.toml
```

- **VLA adapter** `drishti/policy_molmoact2_lerobot.py` — finetuned MolmoAct2 as a VLA (buffered
  receding horizon nas=4, joint-frame correction, fast single-forward ACE).
- **Rollout server** `drishti/sim/rollout_server.py` — `EpisodeManager` (spike→correct→resume→reward)
  behind FastAPI; launch via `:build_default_app --factory`.
- **Detectors + calibration** `drishti/uq/{smd,rnd,ace,embed_frames}.py`, `drishti/calibrate.py`.
- **verifiers env** `environments/vla-corrector/vla_corrector.py` — the RL environment.
- **prime-rl configs** `configs/rl/*.toml`.

---

## What's next (roadmap — turn "it trains" into "it improves the policy")

1. **Reward variance from real success, not just shaping.** The core gap. Options:
   - **Curriculum** — start from cube positions where the VLA *sometimes* succeeds (so corrections can
     flip outcomes → reward variance from success, not the shaping proxy).
   - **Denser / better reward** — the current `uq_reduction` shaping barely moves (RND fires from frame
     ~0 and doesn't drop with corrections). Shape on gripper→cube distance, grasp events, or partial
     progress instead.
2. **Raise the VLA baseline** so there's headroom for corrections to help:
   - **LoRA-VLM VLA finetune** (`--policy.enable_lora_vlm=true`, action expert fully trained — Ai2's
     recipe) — adapts perception, likely lifts success above ~69%.
   - **Clean the joint-frame fix** — rebuild the finetune dataset in the model frame (identity
     transform) instead of the empirical `env_j1=90-pred, env_j2=pred+90` patch.
3. **Corrector quality** — the VLA was finetuned on one fixed instruction, so it's only weakly
   language-steerable (Stage-2: 0.3–9.2° action change). Consider finetuning the VLA with *varied*
   instructions so corrections have more leverage.
4. **Scale the GRPO run** — once reward climbs, more steps + parallel rollout pods (rollout throughput
   is the bottleneck, not $).

---

## Reproduce

- **Failure-detection path** (HF zero-shot VLA): `scripts/pod_setup.sh`.
- **Finetuned-VLA + GRPO path**: a 2-GPU **direct-SSH** pod (datacrunch/massedcompute — NOT crusoe,
  which is proxied-SSH-only). Build a LeRobot venv (see DEBUG_LOG Phase 2/6: `uv sync --python 3.12
  --extra molmoact2 --extra dataset`, transformers 5.x), install prime-rl from source (the working
  install is the nested `prime-rl/prime-rl/`), install the env (`uv pip install -e
  environments/vla-corrector`), launch the rollout server (`:build_default_app --factory`,
  `DRISHTI_LEROBOT_CKPT=…`, `UNCERTAINTY=calib`), then `uv run rl @ configs/rl/smoke.toml`.
- The 11 GB finetuned checkpoint is backed up at `data/molmoact2_so101_ft_ckpt/` (gitignored).
