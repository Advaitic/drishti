# vla-corrector

> **🚧 DRAFT / RESEARCH ARTIFACT.** This is an early experiment shared for discussion, not a finished,
> plug-and-play env. **What works:** the full loop runs end-to-end (custom multi-turn env + self-hosted
> prime-rl, a vision corrector over a robot VLA-in-sim, 40 GRPO steps, no deadlock). **What we found (the
> honest negative):** the corrector's language instructions don't improve task success, because the VLA
> ignores them — steerability probes give ~0 action-change under *opposite* instructions for **both** our
> finetuned VLA (0.02) and the zero-shot base (0.01, even on whole-task switches). So the RL machinery works;
> the language channel to this VLA doesn't. See the drishti repo's `docs/rl-corrector/RESULTS.md` for the full
> story + graphs, and `PRIME_HOSTED_VLM_LIMITATION.md` for the open questions. Layer-1 (sim + VLA + detectors)
> is self-hosted — see "Set it up" below.

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment that RL-trains a **corrective
policy** to supervise a frozen **VLA worker** doing robot pick-and-place in a MuJoCo sim. When the VLA's
calibrated uncertainty spikes, the corrector sees the failure context (telemetry + spike frames) and emits a
short corrective instruction; the **verifiable reward is task success in sim**. Works with a text corrector
(reads the telemetry) or a vision corrector like Qwen3-VL (also sees the frames).

## The VLA worker

The reference worker is **MolmoAct2-SO100_101** — a ~5B Molmo2 VLM + a flow-matching action expert —
finetuned on SO-101 pick-place, grasping ~69% of the time (a real success/failure mix, which is what the
corrective loop needs). Its checkpoint + calibration live on HF (`aryanmadhavverma/molmoact2-so101-ft`). But
the env is **worker-agnostic** — it trains a corrector over *any* action model you expose behind the
rollout-server API. The uncertainty signal is a small **SMD (proprio) + RND (vision) + ACE (action)** detector
cascade, calibrated so uncertainty separates success from failure (RND AUC ~1.0); the spike decides *when* to
call the corrector.

## ⚠️ You must self-host the VLA (this env is a thin HTTP client)

This package carries **only** `verifiers` + `httpx` + `datasets` — no torch, no MuJoCo, no GPU. The heavy
half (MuJoCo + MolmoAct2 + the detector cascade) runs behind a small HTTP **rollout server**
(`/episode/start|run|correct|end`) that **you host on a GPU** — your own machine or a rented pod — and point
this env at via the `rollout_url` arg. That's the one requirement: **bring your own GPU-hosted VLA server.**

## Set it up with Prime Intellect

1. **Rent a GPU pod** and start the rollout server on it (MolmoAct2 + sim + calibrated UQ):
   ```bash
   prime pods create --gpu-type A100_80GB --gpu-count 1     # any single GPU with ~24GB+ works
   # on the pod: pull the checkpoint from HF, then launch the server (see the drishti repo's
   # scripts/pod_h200_setup.sh + scripts/pod_run_2gpu.sh — it does exactly this)
   #   uvicorn drishti.sim.rollout_server:build_default_app --factory --host 0.0.0.0 --port 8100
   ```
   Note the pod's IP → your `rollout_url` is `http://<pod-ip>:8100`.

2. **Install + point this env at it:**
   ```bash
   prime env install <owner>/vla-corrector          # or: uv pip install -e environments/vla-corrector
   prime eval run vla-corrector -m Qwen/Qwen3-VL-4B-Instruct \
     -a '{"rollout_url":"http://<pod-ip>:8100","n_episodes":32,"include_image":true}'
   ```

3. **RL-train the corrector** with self-hosted `prime-rl` (2 GPUs — the combined `uv run rl` launcher). Put
   the env in your config's `[[orchestrator.train.env]]` with the args below. A worked 2-GPU config +
   launch script live in the drishti repo (`configs/rl/grpo_qwen3vl_4b.toml`, `scripts/pod_run_2gpu.sh`).

> Hosted Training note: today Hosted Training is text-only, so a *vision* corrector must be self-hosted (2×
> GPU). A **text** corrector (`include_image=false`, e.g. Qwen3.5-4B) reads the telemetry report and could run
> on Hosted Training, with the rollout server still on your own GPU pod.

## `load_environment(...)` args

| arg | default | meaning |
|---|---|---|
| `rollout_url` | `http://localhost:8100` | your hosted VLA rollout server |
| `include_image` | `true` | send spike frames as image parts (vision corrector) vs text-only |
| `dense` | `true` | add dense **staged-progress** reward (reached<grasped<lifted<placed) |
| `curriculum` | `None` | list of frontier cube positions to sample from (where corrections can flip outcomes) |
| `shaping` | `true` | add uncertainty-reduction shaping reward |
| `n_episodes`, `max_turns` | 64, 4 | dataset size; interventions per episode |

## Reward (`Rubric`)

`reward_success` (1.0 on cube-in-tray) `+ 0.3·progress` (if `dense`) `− 0.05·interventions`
`+ 0.1·uq_reduction` (if `shaping`). Computed here from `state["success"]`, never trusting the pod.

One rollout = one episode; each model turn = one corrective intervention (`setup_state` seeds the first
failure report, `env_response` injects the correction and resumes to the next spike or terminal outcome).
