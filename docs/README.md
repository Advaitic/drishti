# Documentation

Three things live in here: an explanation of how the failure detector works, the write-ups of the
studies we ran with it, and the trail of what went wrong along the way. The trail is deliberate —
the debug logs are more useful to someone rebuilding this than the results are.

## Start here

| | |
|---|---|
| [**LEARNINGS.md**](LEARNINGS.md) | **Read this first.** Plain-language guide to the whole system: what a VLA is, what each detector measures, what calibration does, what every number means. Written for someone who has not seen the code. |
| [**PROJECT_LOG.md**](PROJECT_LOG.md) | What we tried, in order, and how each thing turned out — including the two that didn't work. A few bullets per initiative, with links to the detail. |
| [**REFERENCES.md**](REFERENCES.md) | The papers this is built on, what each one contributed, and where it shows up in the code. |
| [**ROADMAP.md**](ROADMAP.md) | Where this goes next, and the build-vs-buy calls behind it (LeRobot as a dependency, Rerun for visualization, MuJoCo over Isaac). |

## The result

| | |
|---|---|
| [**results/calibration/report.json**](results/calibration/report.json) | The canonical calibration on the SO-101 + SmolVLA set. Every headline number in the README comes from this file, and `drishti calibrate --demo --auto` regenerates it byte-for-byte. |
| [`../assets/figures/`](../assets/figures) | Separation, ROC and threshold-sweep figures, regenerated from that report by `drishti figures --demo`. |

## Study 1 — rl-corrector: training a corrective VLM over a VLA

A corrector that watches a VLA and re-instructs it when uncertainty spikes, trained with GRPO
against verifiable sim success. The infrastructure worked end to end. The reward never moved, and
finding out why produced the most useful thing in this repo.

| | |
|---|---|
| [rl-corrector/README.md](rl-corrector/README.md) | Entry point — the arc, the architecture, what's next |
| [rl-corrector/RESULTS.md](rl-corrector/RESULTS.md) | Milestones and numbers, with the artifact behind each |
| [rl-corrector/DEBUG_LOG.md](rl-corrector/DEBUG_LOG.md) | Every error and dead end, in six phases. The most reusable document here |
| [rl-corrector/DESIGN.md](rl-corrector/DESIGN.md) | The original engineering plan, kept for the record |
| [rl-corrector/PRIME_RL_CONSTRAINTS.md](rl-corrector/PRIME_RL_CONSTRAINTS.md) | Limits we hit in hosted training and single-GPU prime-rl |
| [rl-corrector/blog/vla-steerability.html](rl-corrector/blog/vla-steerability.html) | The write-up of the negative result, as a standalone page |
| [rl-corrector/results/](rl-corrector/results) | Raw proof: training curves, pod logs, eval frames, the trained LoRA adapter |

## Study 2 — AHA: do the detectors transfer off our own robot?

The same SMD + RND detectors on a Franka Panda in RLBench/CoppeliaSim, against procedurally injected
failures. Different arm, different simulator, different failure process.

| | |
|---|---|
| [aha/SUMMARY.md](aha/SUMMARY.md) | The consolidated result and the honest note on what AHA does *not* test |
| [aha/RESULTS.md](aha/RESULTS.md) | Full per-task tables, ablations, and the reproduction commands |
| [aha/GENERATE.md](aha/GENERATE.md) | How to regenerate the dataset from NVlabs/AHA yourself |
| [aha/BREADTH_PLAN.md](aha/BREADTH_PLAN.md) | The plan for the multi-task campaign, kept for provenance |

## A note on the numbers

Every quantitative claim in these documents should either point at an artifact in the repo or say
explicitly that it doesn't. Where a result was superseded we marked it superseded and left it in,
because the way a result goes wrong is usually more transferable than the result. If you find a
number that doesn't reproduce, [that is the most valuable issue you can file](../CONTRIBUTING.md).
