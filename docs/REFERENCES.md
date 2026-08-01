# References

drishti is an engineering synthesis. Almost nothing in it is a new method — the value is in wiring
published detectors into one runtime layer, calibrating them honestly, and reporting what happened
including when it didn't work. This page says where each idea came from, and where in the code it
ended up, so you can go read the source of anything you want to argue with.

Papers are grouped by what they contribute. A ★ marks the ones that directly shaped the
implementation rather than the thinking.

---

## The architecture we actually built

**★ FIPER — Failure Prediction at Runtime for Generative Robot Policies**
Römer, Kobras, Worbis, Schoellig. NeurIPS 2025. [arXiv:2510.09459](https://arxiv.org/abs/2510.09459)

The closest thing to a blueprint for this repo, and the reason its three parts look the way they do:
random network distillation on policy embeddings for out-of-distribution input, an entropy measure
over generated actions for action-side uncertainty, and — the load-bearing idea — **conformal
calibration on successful rollouts only**, so you never need labelled failures to set a threshold.
→ [`drishti/uq/rnd.py`](../drishti/uq/rnd.py), [`drishti/uq/ace.py`](../drishti/uq/ace.py),
[`drishti/calibrate.py`](../drishti/calibrate.py).

FIPER AND-combines its signals with sliding-window-max aggregation. We landed on the mean of
capped-z over per-rollout peaks instead, after `max` produced false alarms from a single grazing
detector; the reasoning and the ablation are in [`drishti/scoring.py`](../drishti/scoring.py).

**★ Random Network Distillation** — Burda, Edwards, Storkey, Klimov. ICLR 2019.
[arXiv:1810.12894](https://arxiv.org/abs/1810.12894)

The novelty estimator itself, from exploration in RL. A frozen random target network and a predictor
trained to imitate it: prediction error is high exactly where the predictor has seen little data. We
run it on the pooled vision embedding rather than raw observations, because the encoder has already
discarded action-irrelevant detail. → [`drishti/uq/rnd.py`](../drishti/uq/rnd.py).

**★ ReconVLA** — [arXiv:2604.16677](https://arxiv.org/abs/2604.16677)

Where the state-space Mahalanobis detector comes from: fit one Gaussian to the proprioceptive states
of successful rollouts, score by covariance-corrected distance. Zero model access, zero training,
and a surprisingly strong baseline for its cost. → [`drishti/uq/smd.py`](../drishti/uq/smd.py).

**★ FAIL-Detect — Can We Detect Failures Without Failure Data?**
Xu et al. (Toyota Research Institute). RSS 2025. [arXiv:2503.08558](https://arxiv.org/abs/2503.08558) ·
[code](https://github.com/CXU-TRI/FAIL-Detect)

Frames runtime failure detection as *sequential* out-of-distribution detection, distils policy
signals to scalars, and applies conformal prediction for a statistical guarantee. The direct
competitor to FIPER, and the reason our threshold is a per-rollout conformal bound rather than a
tuned constant.

**★ AHA** — NVlabs. [aha-vlm.github.io](https://aha-vlm.github.io/)

The cross-embodiment benchmark in [study 2](aha/SUMMARY.md): RLBench tasks on a Franka Panda with
failures injected procedurally by FailGen. We use it as a *transfer* test — different arm, different
simulator, different failure-generation process — not as the VLM failure-explainer it was built to
be. Its limits as a benchmark for our claim are spelled out in [`docs/aha/SUMMARY.md`](aha/SUMMARY.md).

---

## Failure detection on deployed policies — the surrounding literature

**SAFE — Multitask Failure Detection for Vision-Language-Action Models**
Gu et al. (Toyota Research Institute). NeurIPS 2025. [arXiv:2506.09937](https://arxiv.org/abs/2506.09937) ·
[code](https://github.com/vla-safe/SAFE)

A small head on the VLA's last-layer hidden state, with a failure manifold that generalizes *across
tasks*. The strongest argument that the signal we want is already present inside the policy and just
needs reading out — which is the premise of [`drishti/hooks.py`](../drishti/hooks.py). The obvious
next experiment for this repo is benchmarking a SAFE-style learned head against our
training-free detectors on the same rollouts.

**Sentinel — Unpacking Failure Modes of Generative Policies**
Agia et al. CoRL 2024. [arXiv:2410.04640](https://arxiv.org/abs/2410.04640)

Splits runtime monitoring into two genuinely different problems: erratic behaviour, caught
statistically from temporal action inconsistency, and task-progression failure, caught by a VLM.
Our SMD/RND/ACE cascade covers the first; we have nothing for the second, and the AHA
`wrong_sequence` result — where per-frame detectors see nothing until you look temporally — is the
same lesson arriving from another direction.

**I-FailSense** — [arXiv:2509.16072](https://arxiv.org/abs/2509.16072)

LoRA-tune a VLM, freeze it, then train cheap binary heads on its internal language representations.
The tiered-cost argument: run something cheap every frame and the expensive model only on a trigger.
That is exactly the shape of our cascade, and the reason RND at ~2 ms a frame is preferable to a VLM
judge even when the judge is more accurate.

**FailSafe — Reasoning and Recovery from Failures in Vision-Language-Action Models**
Lin et al. IROS 2026. [arXiv:2510.01642](https://arxiv.org/abs/2510.01642)

The external-assistant pattern: a frozen base VLA plus a finetuned VLM that emits executable
Δ-action recoveries. This is the design our [rl-corrector study](rl-corrector/README.md) attempted
in *language* space rather than action space — and the reason our negative result matters. If the
VLA cannot be steered by an instruction, corrections have to be delivered as actions, the way
FailSafe does it. That is our main takeaway from getting it wrong.

**Diff-DAgger** — [arXiv:2410.14868](https://arxiv.org/abs/2410.14868) ·
**Guardian + FailCoT** — [arXiv:2512.01946](https://arxiv.org/abs/2512.01946) ·
**Gemini Robotics-ER 1.5** — [arXiv:2510.03342](https://arxiv.org/abs/2510.03342)

Uncertainty from the diffusion training objective itself; a large-scale failure-reasoning corpus
with a VLM judge; an off-the-shelf VLM with task-progress and success-detection APIs. All three are
plausible ceilings to measure a cheap detector against, and none is used here yet.

**Shifting Uncertainty to Critical Moments** — [arXiv:2603.18342](https://arxiv.org/abs/2603.18342)

The averaging trap: mean-pooling an uncertainty trace washes out short transients. Max-pooling over
a sliding window with motion-aware weighting recovers them. Directly relevant to us — we take a
per-rollout *peak*, which is the crude version of this, and a windowed variant is the obvious next
improvement.

---

## Why the corrector failed — steerability and mechanistic control

The [rl-corrector study](rl-corrector/README.md) concluded that our VLA is effectively deaf to
language. These are the papers that frame what to do about it.

**Häon et al. — Mechanistic Interpretability for Steering VLAs.** CoRL 2025.
[arXiv:2509.00328](https://arxiv.org/abs/2509.00328) — first mech-interp work on π0 and OpenVLA;
sparse feed-forward directions for concepts like "up" and "slow" turn out to be causally steerable.
If language is not a usable control channel, activations might be.

**DrVLA — sparse autoencoders on π0.5.** [arXiv:2603.19183](https://arxiv.org/abs/2603.19183) ·
[project](https://drvla.github.io/) — distinguishes generalizable motion-primitive features from
episode-specific memorization, and finds that supervised finetuning on small data amplifies the
memorization. A sharp explanation for why our 50-demo finetune keyed on vision alone.

**VLS — Vision-Language Steering.** [arXiv:2602.03973](https://arxiv.org/abs/2602.03973) —
training-free: a VLM-synthesized reward gradient steers the diffusion sampler at inference.

**DSRL — Diffusion Steering via Latent-Space RL.** Wagenmaker, Levine et al. CoRL 2025 (oral).
[arXiv:2506.15799](https://arxiv.org/abs/2506.15799) — RL over the latent noise of a diffusion
policy, black-box with the base frozen. The natural successor to what we tried: steer in
action/latent space, where the policy demonstrably listens, instead of in language space, where it
demonstrably does not.

**TACO — Steering VLAs as Anti-Exploration.** [arXiv:2512.02834](https://arxiv.org/abs/2512.02834) —
a pseudo-count estimator that down-weights low-density action chunks at inference. Note the
symmetry: this is our RND novelty score used as a *controller* rather than as an alarm.

---

## Runtime monitoring as a product, not a paper

**Sirius / Sirius-Fleet** — [arXiv:2410.22689](https://arxiv.org/abs/2410.22689) ·
[code](https://github.com/UT-Austin-RPL/sirius-fleet)

The closest published analogue to the loop this repo is a piece of: a visual world model, a failure
predictor, an OOD detector, and an anomaly threshold that adapts as the fleet improves. Read this
before assuming any of the dashboard is novel.

**Fleet-DAgger** — [arXiv:2206.14349](https://arxiv.org/abs/2206.14349) — introduces Return on Human
Effort, the right metric for "was the alarm worth interrupting a person for", which is the metric a
deployed version of this would be judged on.

**HG-DAgger** — [arXiv:1810.02890](https://arxiv.org/abs/1810.02890) — human-gated DAgger with an
uncertainty-thresholded safety layer. The ancestor of every "detect, then hand over" system.

**AutoEval** — [arXiv:2503.24278](https://arxiv.org/abs/2503.24278) ·
[code](https://github.com/zhouzypaul/auto_eval) — round-the-clock autonomous policy evaluation with a
learned success classifier and autonomous scene reset. Relevant because our biggest limitation is
that the success oracle is a geometric check only available in simulation.

**Causal Confusion in Imitation Learning** — de Haan et al.
[arXiv:1905.11979](https://arxiv.org/abs/1905.11979) — interventions cluster at visually distinctive
states, so the policy learns the wrong abstraction. The standard failure mode of any
detect-then-correct-then-retrain loop, including the one this repo is building toward.

---

## The models and stacks used here

| | |
|---|---|
| **SmolVLA** | The policy behind the headline detection result. [Model card](https://huggingface.co/lerobot/smolvla_base) · [paper](https://arxiv.org/abs/2506.01844) |
| **MolmoAct** (Ai2) | The VLA in the rl-corrector study. [allenai.org](https://allenai.org) |
| **LeRobot** | Policy loading, `LeRobotDataset`, the finetuning path. [github](https://github.com/huggingface/lerobot) |
| **MuJoCo** | The physics engine. [mujoco.org](https://mujoco.org) |
| **SigLIP** / **DINOv2** | The frozen encoders RND scores on. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) · [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| **verifiers** | The RL environment interface. [github](https://github.com/willccbb/verifiers) |
| **prime-rl** | Self-hosted GRPO for the corrector. [github](https://github.com/PrimeIntellect-ai/prime-rl) |
| **RLBench** / **FailGen** | The AHA study's simulator and failure injector. [RLBench](https://github.com/stepjam/RLBench) |
| **SO-ARM100 / SO-101** | The arm. [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100) |

---

## What we'd read next

Ordered by how much it would change what is in this repo:

1. **SAFE**, and benchmark a learned head against these training-free detectors on the same 60
   rollouts. It is the cheapest experiment with the highest chance of moving the number.
2. **Shifting Uncertainty to Critical Moments**, for windowed aggregation — the AHA
   `wrong_sequence` result says we need it.
3. **DSRL** and **TACO**, for correction in action space instead of language space, which is where
   the rl-corrector study says the leverage actually is.
4. **AutoEval**, for a success oracle that survives contact with a real robot.
