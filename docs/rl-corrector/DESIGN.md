# RL experiment: training a corrective VLM over a MolmoAct2 VLA worker

Train a **corrective VLM** (Qwen3-VL) that watches a frozen **VLA worker** (MolmoAct2-SO100_101) do
pick-and-place in a MuJoCo SO-101 sim. When the VLA's calibrated **uncertainty** spikes, the corrector
gets the failure context (telemetry + spike frames) and emits a corrective prompt; the **verifiable
reward** is task success. RL via GRPO on self-hosted `prime-rl`.

## Pieces

| file | role |
|---|---|
| `drishti/sim/rollout_server.py` | pod-side FastAPI: `EpisodeManager` (sim + VLA + UQ) behind `/episode/*`. `AceOnlyScorer` (pre-calibration) / `CalibScorer` (post). |
| `drishti/policy_molmoact2.py` | MolmoAct2 VLA adapter; `env_state_to_vla`/`vla_action_to_env` hold the R4 joint-convention fix (no IK adapter — 6-D joint pose feeds `SO101Env.step`). |
| `drishti/sim/stub_vla.py` | deterministic CPU stand-in for MolmoAct2 (Stage-0 tests). |
| `drishti/uq/embed_frames.py` | standalone frozen SigLIP → RND embedding source (keeps RND model-agnostic). |
| `drishti/drivers/rollout_record_molmoact2.py` | Stage-1 recorder for calibration + the `--smoke` R4 gate. |
| `drishti/calibrate.py` | (reused) fits SMD+RND, sets the 90th-pct-of-success-run-peaks threshold. |
| `environments/vla-corrector/` | the verifiers MultiTurnEnv (thin HTTP client of the rollout server). |
| `configs/rl/{smoke,grpo_qwen3vl_4b}.toml` | prime-rl configs. |
| `scripts/pod_up.sh` | rent a 4090 + launch the rollout server. |

## Stages (cost-gated; Prime 4090 ≈ $0.32/hr, A100 ≈ $0.79/hr)

**Stage 0 — local, $0 (DONE).** Rollout server + verifiers env built and tested on a stub VLA.
```bash
pytest -q                                                                        # 22 passed
```

**Stage 1 — rent 1×4090, ~$1–3.** Bring up MolmoAct2 and clear the two empirical gates.
```bash
bash scripts/pod_up.sh                                   # provision + serve (edit REPO_URL first)
python -m drishti.drivers.rollout_record_molmoact2 --smoke   # R4 gate: arm must move TOWARD the cube
python -m drishti.drivers.rollout_record_molmoact2 --episodes 30   # R9 gate: baseline grasp rate on sim frames
python -m drishti.calibrate --auto                       # fit the uncertainty threshold; check report.json AUC
```
- **R4** (joint convention): if the arm moves the wrong way, flip `JOINT_SIGNS`/`JOINT_OFFSETS` in
  `policy_molmoact2.py`.
- **R9** (visual OOD): MolmoAct2 was trained on *real* SO-101 frames; if zero-shot grasp on our synthetic
  renders is too weak, tweak sim visuals/task before spending on training.

**Stage 2 — eval, ~$2–5.** Prompted (untrained) corrector vs. baseline → Δsuccess. Proceed only if
corrections measurably help.
```bash
UNCERTAINTY=calib uvicorn "drishti.sim.rollout_server:build_default_app" --factory --port 8100   # now use the calibrated threshold
prime eval run vla-corrector -m Qwen/Qwen3-VL-4B-Instruct -a '{"rollout_url":"http://POD:8100","max_turns":1}'  # baseline
prime eval run vla-corrector -m Qwen/Qwen3-VL-4B-Instruct -a '{"rollout_url":"http://POD:8100","max_turns":4}'  # prompted
```

**Stage 3a — SMOKE TRAIN, ~$3–6.** Prove reward → advantages → LoRA update.
```bash
uv run rl @ configs/rl/smoke.toml
```

**Stage 3b — MODEST REAL RUN, budget-capped (~$150–200 ceiling; expect ≪).**
```bash
uv run rl @ configs/rl/grpo_qwen3vl_4b.toml
```

## Uncertainty consistency (train == serve)

The three UQ signals are computed identically in the recorder, `calibrate.py`, and the live server:
**SMD** on proprio state (degrees), **RND** on the SigLIP embedding of `up`+`side`, **ACE** = action-sample
entropy (`ace_score`, `value_range=(-3.2,3.2)`). The recorder logs ACE as `action_head_std=[ace]` so
`calibrate._act_spread` reads it; the server recomputes ACE the same way → no train/serve skew.
