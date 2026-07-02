# vla-corrector

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment that trains a **corrective
policy** to supervise a frozen VLA worker (MolmoAct2-SO100_101) doing pick-and-place in a MuJoCo SO-101
sim. When the VLA's calibrated uncertainty spikes, the corrector sees the failure context (telemetry +
spike frames) and emits a short corrective instruction; the **verifiable reward** is task success in sim.

This package is a *thin HTTP client* of the drishti **rollout server** (`drishti/sim/rollout_server.py`),
which runs the heavy half (MuJoCo + MolmoAct2 + the SMD/RND/ACE uncertainty cascade) on a GPU pod. The env
itself carries only `verifiers` + `httpx` + `datasets`.

## Shape (verifiers 0.1.14)

- One rollout = one episode. `setup_state` starts the episode and runs the VLA to the first uncertainty
  spike, seeding that failure report into the prompt. Each model turn is a corrective instruction;
  `env_response` injects it and resumes the VLA to the next spike or a terminal outcome (stop via the
  `final_env_response` sentinel). One turn == one intervention.
- **Reward** (`Rubric`): `reward_success` (1.0 on `_cube_in_tray`) − `0.05·interventions`
  (+ optional `reward_uq_reduction` shaping). Computed here from `state["success"]`, never trusting the
  pod's advisory reward.
- Multimodal-ready: with `include_image=true` the report is sent as OpenAI `image_url` parts (the spike
  frames) for a Qwen3-VL corrector.

## Run

```bash
# point at a running rollout server (see scripts/pod_up.sh)
prime env install environments/vla-corrector          # or: uv pip install -e environments/vla-corrector
prime eval run vla-corrector -m Qwen/Qwen3-VL-4B-Instruct \
  -a '{"rollout_url":"http://ROLLOUT_POD_HOST:8100","n_episodes":32}'
```

`load_environment(rollout_url, n_episodes, max_turns, shaping, include_image, ...)` is the entrypoint.
