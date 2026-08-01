# Real SO-101 reference frames

Two frames from [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)
(Apache-2.0) — a real SO-101 arm doing pick-and-place on a white tabletop with a red brick and an
orange tray.

They are here as the target, not as data. The sim cameras in
[`sim/so101/task_scene.xml`](../../sim/so101/task_scene.xml) were positioned to match these two
views, and the tabletop, brick and tray colours were matched to them, because the policies used here
were pretrained on real SO-101 footage and every bit of visual mismatch is distribution shift the
policy has to absorb before it can even attempt the task. Put `so101_up.png` next to
[`../sim/camera_up.png`](../sim/camera_up.png) to see how close it got.

| File | View |
|---|---|
| `so101_up.png` | diagonal over-the-shoulder top view |
| `so101_side.png` | low front view |
