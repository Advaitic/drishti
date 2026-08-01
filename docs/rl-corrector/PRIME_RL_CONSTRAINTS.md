# Training a vision corrector on Prime Intellect — the VLM gap, our workaround, and a request

*Audience: the Prime Intellect team (Hosted Training + Environments Hub) and our own record.*
*Verified against the docs 2026-07-09: `docs.primeintellect.ai/hosted-training/models-and-pricing`,
`/tutorials-environments/create`, `/prime-rl/configuration`.*

## What we're building

An RL loop that trains a **corrective VLM** to supervise a **frozen VLA worker** (MolmoAct2) doing robot
pick-place in a MuJoCo sim. When the VLA's calibrated uncertainty (SMD/RND/ACE) spikes, the corrector sees
the failure state and emits a corrective instruction; **verifiable reward = sim success**; train the
corrector with **GRPO** on **prime-rl**, wrapped as a **`verifiers` environment** (`vla-corrector`). Built
entirely on the Prime Intellect stack (verifiers + prime-rl + compute).

## Why we ended up trying a TEXT corrector (two friction points)

**1. Hosted Training is text-only — no VL/multimodal models.** `prime train models` lists 14 models
(Qwen3.5 0.8B–35B, Qwen3.6, Llama-3.2, Nemotron-3, gpt-oss, …), **all text**. Our corrector was designed as
**Qwen3-VL-4B** so it can literally *see* the robot's failure frames. There is no hosted path to train a
vision corrector today.

**2. Self-hosting prime-rl for a vision model on a single GPU deadlocks.** No 2-GPU nodes were available when
we needed one, so we rented a single (large) H200 and hand-launched prime-rl's three processes
(inference + trainer + orchestrator) colocated on the one GPU. That reliably deadlocks at ~step 3: the
trainer starts (under `torchrun`) but hangs `Waiting for run 0 to be created`, because the manual split
doesn't reproduce the **trainer↔orchestrator run-registration handshake** that the combined `uv run rl`
launcher performs. That launcher requires **2 disjoint GPUs** (`num_infer_gpus + num_train_gpus`, and it
errors if the total exceeds physical GPUs) — so on one GPU there is no clean path. We validated *everything
else* (rollouts, reward, dense staged-progress reward, curriculum, calibration RND AUC 1.0, colocated
serving, the corrector being invoked) — only the sustained training step was blocked. Full trace:
`DEBUG_LOG.md` Phase 7.

## The experiment this motivates: does a TEXT corrector even need vision?

Our failure report is **rich structured telemetry**, not just an image: the uncertainty scores + a geometry
block — **gripper→cube distance, grasping yes/no, gripper/cube/tray positions, and the phase**. A plausible
hypothesis: for *this* task, explicit geometry ("gripper 4 cm from cube, not grasping") may be **more** useful
to a corrector than raw pixels — vision could be redundant. So running the corrector as **Qwen3.5-4B (text)**
on Hosted Training, fed the text report with images off, is a genuinely useful experiment:

- **If it works** → vision isn't required here; we get a fully-managed, cheap, dashboard-tracked training loop.
- **If it doesn't** → concrete evidence that the corrector needs to *see*, which is exactly the case for the
  request below.

## The request to Prime Intellect

**Please enable VL / multimodal models (e.g. Qwen3-VL) on Hosted Training.** It would let vision correctors —
and more generally any "VLM-supervises-a-VLA" or image-in RL loop — be trained fully-managed, without
self-hosting a 2-GPU prime-rl deployment. Our `vla-corrector` env is a ready-made, non-trivial multimodal
use case to validate it against.

## The bigger request: continuous-space (robot) policy RL in prime-rl

Everything above trains a **token/language policy** (an LLM) — prime-rl + verifiers + GRPO are built for
discrete token sequences. Our whole "corrector" detour exists *because* the thing we can RL-train is a
language model, not the robot. But our steerability tests show the language channel to the VLA is dead (both
base and finetuned MolmoAct2 ignore instructions in our setup) — which points at the real want:

**Let prime-rl RL-train continuous-action policies directly** — a VLA / robot policy that outputs continuous
actions, RL-explored *inside* a verifiers-style environment (the sim), with verifiable reward = task success.
Then you don't need a language corrector at all: **the environment becomes a simulation space where the VLA
itself keeps exploring and improving** — the natural home for embodied RL.

We understand this is a significant overhaul, not a config flag — it means continuous-control RL (PPO/SAC-style
over an action distribution), not GRPO-over-tokens; a different rollout/advantage structure; and the policy
being the *actor in the sim* rather than an LLM scoring text. But it would open Prime Intellect's RL stack to
**robotics / embodied agents**, a domain the current token-only design can't reach. Our SO-101 sim + reward +
detectors are a concrete, ready testbed to prototype it against. Worth the team's thought — happy to help
prototype.

## The env as a reusable Hub abstraction (and its one documented limitation)

Our `vla-corrector` verifiers env is **generic**: it trains a corrector over *any* VLA worker exposed via a
small HTTP rollout-server API (`/episode/start|run|correct|end`). It's a clean Environments Hub contribution
— **"train a corrective policy over a black-box action model via verifiable reward."**

**Documented limitation (goes in the Hub README):** the env is a **thin HTTP client** — it carries no
GPU/torch/sim itself. The user must **host the VLA rollout server themselves** (their own machine or a GPU
pod) and pass its URL as the `rollout_url` arg, because the worker (a multi-billion-param policy + a physics
sim) needs a GPU that Hosted env-servers (CPU containers) don't provide. This is a general and legitimate
pattern for Prime's env model — env-servers make outbound HTTP calls to a user-hosted service — and it's the
right separation: **Prime manages the RL (trainer + inference + orchestration); the user hosts the worker.**

Config the env accepts (all optional except `rollout_url`):
`rollout_url` (the hosted VLA server), `dense` (staged-progress reward), `curriculum` (frontier cube
positions), `shaping` (uncertainty-reduction reward), `include_image` (send spike frames — set **false** for a
text corrector on Hosted Training, **true** for a self-hosted VL corrector), `n_episodes`, `max_turns`.
