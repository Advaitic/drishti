"""Per-frame sub-goal captions for the scripted SO-101 pick-place demos.

The scripted trajectory (`drishti/sim/scripted_pick.py::run_episode`) always runs the same
8-phase, fixed-duration waypoint sequence, so the frame index at which a phase starts/ends
is deterministic and identical across every recorded episode (`rec_every=18` subsampling of
the hardcoded step counts 200,120,150,150,160,120,80,80 -> 63 recorded frames/episode).

Captions are grouped into 3 MACRO-phases (not the raw 8) because MolmoAct2/LeRobot condition
language once per `chunk_size`-step action chunk (currently 16), not per timestep. Grouping
reduces (does NOT eliminate) caption/action decorrelation across phase boundaries.

CAVEAT (audit 2026-07-22): sizing each macro-phase >= chunk_size only guarantees the phase's
FIRST frame gets a clean in-phase 16-step target. Every frame in the last 15 of a phase still
has its action horizon bleed into the next phase. Concretely, with phases of 28/18/17 frames:
~48% of training examples cross a boundary and ~22% have the MAJORITY of their action target in
the next phase (label noise). This is a secondary confound; the dominant issue is that captions
are redundant with vision (all demos run one fixed script), so no granularity fixes steerability.
"""

from __future__ import annotations

import numpy as np

# (first_frame_inclusive, last_frame_inclusive, paraphrases)
MACRO_PHASES = [
    (0, 27, [
        "reach toward the cube and grasp it",
        "move to the cube and pick it up",
        "approach the cube, lower onto it, and close the gripper",
    ]),
    (28, 45, [
        "lift the cube and carry it to the tray",
        "raise the cube and move it toward the box",
        "pick the cube up and bring it over the tray",
    ]),
    (46, 62, [
        "place the cube in the tray and let go",
        "lower the cube into the box and release it",
        "set the cube down in the tray and open the gripper",
    ]),
]


def phase_for(frame_idx: int) -> int:
    """Return the macro-phase index (0,1,2) a given frame index falls in."""
    for i, (start, end, _) in enumerate(MACRO_PHASES):
        if start <= frame_idx <= end:
            return i
    raise ValueError(f"frame_idx {frame_idx} outside all macro-phases (max {MACRO_PHASES[-1][1]})")


def caption_for(frame_idx: int, rng: np.random.Generator | None = None) -> str:
    """Return a (randomly paraphrased) caption for the macro-phase containing `frame_idx`."""
    _, _, paraphrases = MACRO_PHASES[phase_for(frame_idx)]
    if rng is None:
        return paraphrases[0]
    return paraphrases[int(rng.integers(len(paraphrases)))]
