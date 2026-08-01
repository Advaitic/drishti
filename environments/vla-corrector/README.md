# vla-corrector

Draft / research. RL-train an LLM to **correct a black-box action policy** (a VLA) toward task success in a
sim. Bring your own policy and sim behind a small HTTP contract; the env trains the corrector.

Status: the loop runs end-to-end (custom multi-turn env + self-hosted prime-rl, vision corrector over a robot
VLA-in-sim, 40 GRPO steps). Honest finding so far: with the reference SO-101 VLA the corrector's *language*
doesn't move task success — steerability probes show ~0 action change under opposite instructions. So the
machinery works; whether language correction helps depends on the policy.

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

**Starter policy:** the checkpoint these results came from is a finetuned `MolmoAct2-SO100_101`
(SO-101 pick-place, **11/16 = 68.75% task success** over a 16-episode recording with buffered
execution, `n_action_steps=4`), pushed to HF as `aryanmadhavverma/molmoact2-so101-ft` along with its
calibration. **It is currently a private repo** — verified, it returns 401 — so you cannot pull it
today. Use the CPU stub below to exercise the loop, or bring your own policy.

**No checkpoint? Run the loop CPU-only.** drishti ships a stub rollout server that speaks the same contract
with a deterministic stand-in policy instead of a VLA:

```bash
uvicorn drishti.sim.stub_server:app --port 8100
```

Good for exercising the contract and your client code; the rewards are not science. Note: on macOS MuJoCo
rendering inside the async event loop stalls across successive HTTP requests — drive `EpisodeManager`
directly there, or run the server on Linux with `MUJOCO_GL=egl`.

## Rollout-server contract

- `POST /episode/start {seed, cube_xy, task}` -> `{episode_id}`
- `POST /episode/run {episode_id, max_frames}` -> a **report**; `status` is `spike` (pause for a correction),
  `success`, or `timeout`.
- `POST /episode/correct {episode_id, prompt}` -> applies the instruction to the policy.
- `POST /episode/end {episode_id}`.

A report carries:

| field | meaning |
|---|---|
| `status` | `spike`, `success`, or `timeout` |
| `success` | task succeeded |
| `done` | episode is over — true for `success`/`timeout`, false for `spike` |
| `progress` | 0..1 staged progress |
| `failure_report` | the text the corrector reads |
| `frames_b64` | spike frames |
| `uncertainty` | per-detector scores + combined + threshold |
| `geometry` | gripper / object / target positions |

`done` is what the env reads to end the rollout; if you omit it the env falls back to
`status in ("success", "timeout")`, but send it — the fallback is a courtesy, not the contract.

## Uncertainty detection (`uq.py`)

Policy-agnostic. Fit on your successful rollouts, then score any step:

```python
from uq import Calibrator
cal = Calibrator().fit(success_states, success_frames, trajectory_ids)  # states: (N,D) proprio;
                                                                       # frames: per-step tuple of RGB arrays
u = cal.score(state, frames)   # -> {smd, rnd, combined, threshold, spike: bool}
```

SMD is a Mahalanobis distance from the mean successful pose; RND is a random-network-distillation residual on
SigLIP embeddings (low on familiar views, high on novel ones). Both need only your successes — no labels.

**Combine rule.** Both signals are z-scored against the successful fit, clipped to a ceiling of `CAP = 32`,
scaled to 0..1, and **averaged** — not maxed. `max` fires on whichever detector is loudest, so one detector
grazing its threshold is a false alarm; averaging bounded scores is agreement-aware. The cap matters
independently: after whitening, RND's spread over successes is tiny, so its raw z runs ~100× past SMD's and
would decide every score on its own.

**Threshold.** The `90`th percentile of **per-trajectory peaks**, not of per-step scores. What you want to
bound is how often a good *trajectory* trips the alarm; bounding steps at the 95th percentile sounds tight
until you notice that 5% of steps over a ~50-step trajectory means ~73% of trajectories graze at least once.
Pass `trajectory_ids` to `fit` so steps can be grouped — without it every step is its own trajectory, which
reproduces the looser per-step behaviour.

(Both of these changed recently; earlier revisions of this file used max-of-z at the 95th percentile of
steps. Refit any stored calibrator.)

## Setup and eval

```bash
prime env install <owner>/vla-corrector          # or: uv pip install -e .
# with your rollout server running at <url>:
prime eval run vla-corrector -m Qwen/Qwen3-VL-4B-Instruct -n 20 -r 2 \
  -a '{"rollout_url":"http://<host>:8100","include_image":true}'
```

RL training (self-hosted prime-rl, 2 GPUs): put the env in `[[orchestrator.train.env]]` with the args below.
Hosted Training is text-only today, so a *vision* corrector must be self-hosted; a *text* corrector
(`include_image=false`) can run hosted.

Note: eval and CI require a reachable `rollout_url`. With no server the env can't produce episodes and
`prime eval` fails with a connection error — the env carries the corrector loop, not a policy of its own.
The CPU-only stub server above is enough to satisfy it.

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
