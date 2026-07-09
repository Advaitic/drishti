# vla-corrector

Draft / research. RL-train an LLM to **correct a black-box action policy** (a VLA) toward task success in a
sim. Bring your own policy and sim behind a small HTTP contract; the env trains the corrector.

Status: the loop runs end-to-end (custom multi-turn env + self-hosted prime-rl, vision corrector over a robot
VLA-in-sim, 40 GRPO steps). Honest finding so far: with our SO-101 VLA the corrector's *language* doesn't move
task success — steerability probes show ~0 action change under opposite instructions. So the machinery works;
whether language correction helps depends on the policy. Details + graphs in the drishti repo
(`docs/rl-corrector/RESULTS.md`).

## Overview

One rollout = one episode. `setup_state` starts the episode, runs the policy until its uncertainty spikes,
and shows the corrector that failure report. Each corrector turn is one short instruction; `env_response`
injects it, resumes the policy to the next spike or a terminal outcome. Reward = task success (plus optional
shaping). Multimodal: with `include_image=true` the spike frames are sent as image parts (for a VLM corrector
like Qwen3-VL); otherwise the corrector reads the text telemetry only.

## Bring your own policy

The env is a thin HTTP client — no torch, no sim, no GPU. To run the flow with *your* policy:

1. **Host a rollout server** that runs your policy in your sim and exposes `/episode/{start,run,correct,end}`
   (contract below). A minimal, annotated skeleton is included: `rollout_server_reference.py`.
2. **Calibrate an uncertainty trigger** on successful trajectories so the server knows *when* to call the
   corrector. A compact, policy-agnostic detector is included: `uq.py` (SMD over proprioceptive state + RND
   over vision embeddings, both fit on your successful rollouts; vision via a HuggingFace SigLIP encoder).
3. **Point the env at your server** via the `rollout_url` arg.

Reference `MolmoAct2` worker + calibration: HF `aryanmadhavverma/molmoact2-so101-ft`.

## Rollout-server contract

- `POST /episode/start {seed, cube_xy, task}` -> `{episode_id}`
- `POST /episode/run {episode_id, max_frames}` -> a **report**; `status` is `spike` (pause for a correction),
  `success`, or `timeout`.
- `POST /episode/correct {episode_id, prompt}` -> applies the instruction to the policy.
- `POST /episode/end {episode_id}`.

A report carries: `status`, `success`, `progress`, `failure_report` (text the corrector reads), `frames_b64`
(spike frames), `uncertainty` (per-detector scores + combined + threshold), and `geometry`.

## Uncertainty detection (`uq.py`)

Policy-agnostic. Fit on your successful rollouts, then score any step:

```python
from uq import Calibrator
cal = Calibrator().fit(success_states, success_frames)   # states: (N,D) proprio; frames: list of RGB arrays
u = cal.score(state, frame)                              # -> {smd, rnd, combined, spike: bool}
```

SMD is a Mahalanobis distance from the mean successful pose; RND is a random-network-distillation residual on
SigLIP embeddings (low on familiar views, high on novel ones). Both need only your successes — no labels.

## Setup and eval

```bash
prime env install <owner>/vla-corrector          # or: uv pip install -e .
# with your rollout server running at <url>:
prime eval run vla-corrector -m Qwen/Qwen3-VL-4B-Instruct -n 20 -r 2 \
  -a '{"rollout_url":"http://<host>:8100","include_image":true}'
```

RL training (self-hosted prime-rl, 2 GPUs): put the env in `[[orchestrator.train.env]]` with the args below;
a worked config lives in the drishti repo (`configs/rl/grpo_qwen3vl_4b.toml`). Hosted Training is text-only
today, so a *vision* corrector must be self-hosted; a *text* corrector (`include_image=false`) can run hosted.

## Environment arguments

| arg | default | meaning |
|---|---|---|
| `rollout_url` | `http://localhost:8100` | your rollout server |
| `include_image` | `true` | send spike frames (vision corrector) vs text-only |
| `dense` | `true` | add staged-progress reward (reached < grasped < lifted < placed) |
| `curriculum` | `None` | cube positions to sample from (where corrections can flip outcomes) |
| `shaping` | `true` | add uncertainty-reduction reward |
| `n_episodes` | `64` | dataset size |
| `max_turns` | `4` | interventions per episode |

## Reward and metrics

`reward = success + 0.3·progress (dense) − 0.05·interventions + 0.1·uq_reduction (shaping)`, computed here
from `state["success"]`. Metrics: `reward`, `success`, `progress` (0..1 staged), `n_interventions`.
