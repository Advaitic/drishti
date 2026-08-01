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

## Everything that is runnable

`drishti demo` / `serve` / `calibrate` / `record` / `figures` are the maintained entry points. The
rest are the experiments themselves — kept because they are what produced the results above, not
because they are polished. Each is a `python -m` away and each explains itself in its docstring.

| Runs the system | |
|---|---|
| `drishti.sim.so101_env` | load the SO-101 scene, render up/side/wrist |
| `drishti.sim.scripted_pick` | one scripted top-down pick-and-place, prints success |
| `drishti.sim.record` | record scripted demos to `data/demos/` (`drishti record`) |
| `drishti.drivers.rollout_record` | drive a finetuned policy and log rollouts into the store |
| `drishti.drivers.live_fleet` | one live rollout, scored by the calibrated Monitor as it runs |
| `drishti.drivers.replay` | open-loop replay of a LeRobot episode through an instrumented policy |

| The rl-corrector experiments | |
|---|---|
| `drishti.sim.stub_server` | CPU-only rollout server — try the RL loop with no checkpoint |
| `drishti.drivers.rollout_record_molmoact2` | the MolmoAct2 recorder, with the R4/R9 smoke gates |
| `drishti.drivers.steerability_probe` | *is the VLA language-steerable?* — the test that settled it |
| `drishti.drivers.phase_caption_probe` | the follow-up: does it respond to its **own** trained vocabulary? |
| `drishti.drivers.eval_prompt_sweep` | does a corrective instruction change task success? |
| `drishti.drivers.correction_ab` | does a *good* correction, injected mid-trajectory, help? |
| `drishti.drivers.rl_env_smoke` | the full corrector loop on a live VLA, minus GRPO |
| `drishti.drivers.success_eval` | closed-loop task success for a phase-captioned checkpoint |
| `drishti.drivers.frontier_profile` | find the cube positions where the VLA sometimes succeeds |

| Research tooling (`scripts/`) | |
|---|---|
| `gen_aha.py` → `ingest_aha.py` → `eval_aha.py` / `eval_aha_seeds.py` | the AHA benchmark pipeline |
| `embed_offline.py` | re-embed stored frames with a different encoder |
| `build_lerobot_dataset.py` | scripted demos → `LeRobotDataset` for finetuning |
| `eval_policy.py` | checkpoint shootout on the sim task |
| `export_demo_bundle.py` | regenerate `data/demo/` from a full local store |
| `make_gif.py`, `watch_demos.py` | look at what the arm actually did |
| `finetune.sh`, `pod_*.sh` | the GPU-pod recipes, kept as the record of what we ran |

## A note on the numbers

Every quantitative claim in these documents should either point at an artifact in the repo or say
explicitly that it doesn't. Where a result was superseded we marked it superseded and left it in,
because the way a result goes wrong is usually more transferable than the result. If you find a
number that doesn't reproduce, [that is the most valuable issue you can file](../CONTRIBUTING.md).
