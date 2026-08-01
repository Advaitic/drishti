# Roadmap

What exists today is [LEARNINGS.md](LEARNINGS.md): a MuJoCo SO-101 pick-place, a SMD/RND/ACE
uncertainty cascade with calibration, a telemetry spine and a dashboard, plus two studies. This page
is about where it goes and — more usefully — the build-vs-buy calls we made along the way after
reading the current tooling landscape. **None of what follows is implemented.** It is a decision
record, published because the reasoning is the part worth reusing.

## Near term, inside this repo

Ordered by how much each would change what the repo claims:

1. **A learned failure head, benchmarked against the training-free detectors.** SAFE-style: a small
   head on the VLA's last-layer hidden state, evaluated on the same 60 rollouts. This is the
   cheapest experiment with the best chance of beating the current numbers, and it would tell us how
   much the training-free framing is costing.
2. **Windowed aggregation.** We take a per-rollout peak, which is the crudest possible temporal
   summary. Sliding-window max with motion-aware weighting is the published fix, and the AHA
   `wrong_sequence` result — where per-frame detectors see nothing until you look temporally — says
   we need it.
3. **Held-out calibration as the default.** Today RND is fitted on the same successes it is then
   scored against, so the headline AUC is in-sample. A train/held-out split inside `calibrate.py`
   would make the reported number honest by construction instead of by footnote.
4. **Correction in action space, not language space.** The [rl-corrector
   study](rl-corrector/README.md) established that our VLA does not listen to instructions. DSRL and
   TACO steer in latent/action space, where it demonstrably does. That is the shape the next attempt
   should take.
5. **A success oracle that survives a real robot.** Everything downstream of "we know which rollouts
   failed" currently rests on a geometric check against ground-truth cube position — available only
   in simulation. AutoEval-style learned success classification is the unlock for anything physical.

## Real hardware

A physical bimanual SO-101 rig — two leader and two follower arms over USB, teleoperated, for real
data collection — is the natural next testbed, and it is a solved problem: don't build custom
teleop or calibration code. LeRobot has first-class native support (`bi_so_follower` /
`bi_so_leader`):

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=left_follower
lerobot-teleoperate --robot.type=bi_so_follower --teleop.type=bi_so_leader ...
lerobot-record --robot.type=bi_so_follower ... --dataset.repo_id=you/task --dataset.num_episodes=50
lerobot-train --dataset.repo_id=you/task --policy.type=act --steps=60000
```

The rig becomes a second backend alongside the MuJoCo sim, feeding the same
`sdk.py` → `store.py` → dashboard pipeline drishti already has.

One design consequence worth writing down now: **don't hardcode camera count, camera names, or a
single fixed-arm action space anywhere in the UQ or hooks layer.** The hook already pools and
concatenates whatever cameras it sees, which is the right default; keep it that way.

## Build vs buy

### LeRobot: a dependency, not a fork — and where the line is

Use LeRobot as a real Python dependency (as drishti already does for `SmolVLAPolicy` and
`LeRobotDataset`) for everything up to "produce a trained policy artifact and a well-formed
dataset." Verified by reading their source, not just the docs:

- **Embodiments are a genuine plugin system.** `so101_follower` / `bi_so_leader` and friends register
  via `@RobotConfig.register_subclass(...)`, and a third-party package named `lerobot_robot_*` or
  `lerobot_teleoperator_*` is auto-discovered at CLI startup — **no modification to LeRobot's own
  source required** ([Bring Your Own Hardware](https://huggingface.co/docs/lerobot/integrate_hardware)).
  Precedent: [lerobot-robot-xarm](https://github.com/SpesRobotics/lerobot-robot-xarm).
- **So are policy architectures**, including genuinely novel ones rather than wrappers around
  published VLAs. Ship a `lerobot_policy_*` package (`PreTrainedConfig` + `PreTrainedPolicy`
  subclasses) and `lerobot-train --policy.type=my_policy` picks it up
  ([Bring Your Own Policies](https://huggingface.co/docs/lerobot/bring_your_own_policies)). No forced
  transformer backbone, no forced action chunking, no forced normalization; non-standard training
  loops go through an optional `update()` hook. Precedent:
  [DiTFlow Policy](https://github.com/danielsanjosepro/lerobot_policy_ditflow). Registering here
  costs nothing and buys `lerobot-train`/`-eval`, checkpoint and Hub push, and the `LeRobotDataset`
  dataloader/normalization/stats machinery.
  - One constraint to design around: policy batches are flat dicts keyed by fixed constants
    (`OBS_STATE`, `OBS_IMAGES`, `OBS_LANGUAGE`, `ACTION`). Depth, point clouds and LiDAR have no
    obvious slot — fold depth into `OBS_IMAGES` as an extra channel, or extend the constants
    deliberately in your own plugin package once an architecture actually needs them as model input.
- **`LeRobotDataset` is the format to standardize on** — not because the storage format is special,
  but because it is what every pretrained checkpoint's finetuning code expects (ACT, Diffusion,
  SmolVLA, π0/π0.5, GR00T) and what tens of thousands of community datasets on the Hub use.

**Where LeRobot's involvement should stop: production control and policy serving.** The
async-inference serving path has a hardcoded `SUPPORTED_POLICIES` list with no registration
mechanism — unlike the training-time registry above — so any custom or proprietary policy hits a
wall there ([issue #2619](https://github.com/huggingface/lerobot/issues/2619) shows another team
hitting the identical problem with XVLA). drishti's own drivers sidestep this entirely by calling
`policy.select_action()` in-process in their own loop, and that is the right pattern to keep.

The regime that decides it: if the control loop tolerates a slow path — a network-reachable
inference server, tens of milliseconds of tolerance, a receding-horizon VLA chunk every few hundred
milliseconds, which is roughly drishti's current setup — LeRobot's stack is fine. The moment you
need genuinely tight closed-loop control (sub-millisecond cycles, high-DOF hands, hard jitter
bounds), that is the point to build a real-time runtime *underneath* LeRobot's dataset and training
side, not as a replacement for it. There is a real counterpoint: ROS2 solves most hard robotics
problems for free, and a custom stack can cost a small team time-to-market it cannot afford.

One thing that generalizes regardless: **telemetry must be asynchronous and structurally separate
from the control loop**, never bolted on, or it steals control-loop cycles and silently drops
samples. That is why [`drishti/emit.py`](../drishti/emit.py) is a background queue and
`run.step()` returns in microseconds.

### Visualization: adopt Rerun, keep the UQ panel custom

[Rerun](https://rerun.io) (MIT/Apache-2.0, self-hostable) should replace drishti's own
storage-and-viz layer. What would go away: the dashboard's replay UI, the `steps`/`frames` SQLite
tables, the JPEG blob directory, the live-tail WebSocket. Rerun's `.rrd` format handles multi-rate
multimodal logging natively, with time-synced scrubbing across cameras, joints and scalars.

What stays entirely custom: `hooks.py`, `uq/*`, `calibrate.py`. None of that is visualization. Log
the SMD/RND/ACE scores, the combined score and the threshold as Rerun `Scalars` on the same
timeline, and build the failure-detection panel through the Blueprint API next to Rerun's native
camera and 3D views.

It is not a competing dataset format either — Rerun ships a native LeRobot loader, so store training
episodes as `LeRobotDataset` and point Rerun at them for review. LeRobot already defaults to Rerun
for its own visualization (`lerobot-record --display_data=true`), so this is one pipe, not two
standards.

Sensor modalities, with the caveats that matter:

- **RGB + depth — adopt now.** The `DepthImage` archetype plus a `Pinhole` on the same entity path
  auto-back-projects to 3D. No turnkey ROS2 bridge; write the subscribe-and-log glue, and bound
  retention for long sessions.
- **LiDAR — adopt with an explicit point budget.** Works via `Points3D`, but the viewer re-uploads
  the whole cloud every frame with no diffing; it degrades above ~1M points and has a hard ceiling
  around 2M. This is a [multi-year open issue](https://github.com/rerun-io/rerun/issues/1136), so
  treat it as a durable constraint. Decimate before logging, cap retained history, and use the
  native desktop viewer rather than the WASM one.
- **Gaussian splats — out of this stack.** No splat archetype exists, deliberately. Do
  reconstruction offline with a dedicated renderer; at most surface a coarse `Points3D` proxy.

### Simulation: MuJoCo stays; Isaac is a later decision

Keep MuJoCo/MJCF as the authoring format. Two questions get conflated here:

- **Porting assets between engines** (MJCF ↔ USD): Isaac has an official importer, but actuators,
  contacts and materials routinely break in conversion and there is no mature path back. Fine for a
  one-time import, not a live bridge.
- **Live visualization of a real rig** driven by real telemetry: not a MuJoCo-vs-Isaac question at
  all. Log a URDF once and animate it from live joint data — Rerun's job, covered above.

Isaac's actual inflection point is GPU-parallel RL at scale, photorealistic domain randomization, or
a true multi-robot fleet — none of which apply to a static rig doing imitation learning. Adopting it
sooner is a large infrastructure tax for capability not yet needed.

## Summary

| Layer | Decision |
|---|---|
| Embodiment abstraction (`Robot` / `Teleoperator`) | Keep LeRobot's plugin contracts |
| Dataset format | Keep `LeRobotDataset` |
| Policy architectures + training loop | Keep LeRobot's; register via a `lerobot_policy_*` plugin |
| Query, visualization, dataset review | Adopt Rerun; custom Blueprint panel for UQ and threshold |
| Live telemetry (RGB + depth) | Adopt Rerun now |
| Live telemetry (LiDAR) | Adopt with an explicit decimation policy |
| Gaussian splats | Separate offline pipeline, not in this stack |
| Sim authoring | Keep MuJoCo / MJCF |
| Production control loop and policy serving | Build custom; don't use LeRobot's async-inference server |
| UQ, calibration, failure detection | Entirely custom — the actual differentiated layer |
| Training compute dispatch, edge inference | Deferred |
