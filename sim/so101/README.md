# SO-101 MuJoCo description

The arm model this repo simulates. Vendored rather than pulled at runtime so a clone can run the
sim with no network and no extra install — MuJoCo reads these files directly.

| File | Origin | Modified? |
|---|---|---|
| `assets/*.stl` (13 meshes) | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), Apache-2.0 | no — byte-identical copies |
| `so101_new_calib.xml` | generated from the same CAD by [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) (MIT) | **yes** — added the `up`/`side`/`wrist` cameras, retuned sts3215 position gains |
| `joints_properties.xml` | upstream sts3215 servo parameters (damping, friction, backlash, force range) | no |
| `scene.xml` | upstream bare arm-on-a-floor scene | no — kept for provenance; nothing in this repo loads it |
| `task_scene.xml` | **ours** | the pick-place scene: table, red cube, tray, cameras, contact tuning |

Full attribution and licensing in the repository [NOTICE](../../NOTICE).

## What `task_scene.xml` adds, and why it looks the way it does

`task_scene.xml` is the scene everything actually runs against (`drishti/sim/so101_env.py` loads it
by default). Two decisions in it are worth knowing about, because both were arrived at the hard way:

**The camera placements are copied from a real dataset, not chosen for looks.** `up` is a diagonal
over-the-shoulder view and `side` is a low front view because that is where the cameras sit in
[`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace). A
policy pretrained on real SO-101 footage is being asked to act on synthetic renders, and every
degree of viewpoint mismatch is distribution shift you pay for at inference. Side-by-side reference
frames are in [`assets/real/`](../../assets/real).

**The contact solver is tuned for grasp stability, following [ggand0/pick-101](https://github.com/ggand0/pick-101).**
An elliptic friction cone, `noslip_iterations`, a high `impratio`, stiff `solref`/`solimp`, and small
finger-pad collision boxes. Without these the cube wobbles out of a closed gripper and no policy —
scripted or learned — can complete the task; the failures you would then be detecting would be the
simulator's, not the policy's.

## Poking at it

```bash
python -m drishti.sim.so101_env      # load the scene, render up/side/wrist to data/so101_smoke/
python -m drishti.sim.scripted_pick  # run one scripted pick-and-place, print success
```
