"""The task the SO-101 sim runs, in one place.

The instruction string used to be redefined in six modules with three different spellings, and the
capitalisation drifted: the VLA was finetuned on "Pick up the cube and place it in the box" while
the RL rollout server drove it with the lowercase variant. For a policy whose language pathway is
weak to begin with (see docs/rl-corrector/RESULTS.md), an off-training instruction is exactly the
kind of confound you do not want under an experiment. So there is one constant, and everything
imports it.
"""

from __future__ import annotations

import os

# Must match the `task` column of the finetuning dataset verbatim, capital P included.
DEFAULT_TASK = os.environ.get("DRISHTI_TASK", "Pick up the cube and place it in the box")

# Cameras the sim renders. `up` and `side` watch the workspace; `wrist` rides the moving jaw.
CAMERAS = ("up", "side", "wrist")
