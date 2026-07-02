"""The failure-report composer is deterministic and surfaces the fields a corrector needs."""

from __future__ import annotations

from drishti.sim.rollout_server import _compose_report


def _rep(dominant="ace", grasping=False, interventions=1):
    return {
        "frame": 37, "phase": "grasp", "task": "pick up the cube and place it in the box",
        "n_interventions": interventions, "vla_reasoning": "hunting for the cube",
        "geometry": {"gripper_tcp": [0.31, 0.01, 0.06], "cube_pos": [0.30, 0.0, 0.02],
                     "tray_xy": [0.34, 0.02], "cube_to_tray_dx": -0.04, "cube_to_tray_dy": -0.02,
                     "gripper_to_cube": 0.032, "grasping": grasping},
        "uncertainty": {"smd": 0.20, "rnd": 0.19, "ace": 0.55, "combined": 0.56,
                        "threshold": 0.42, "dominant": dominant, "margin": 0.14},
    }


def test_compose_report_fields():
    txt = _compose_report(_rep())
    assert txt.startswith("[UNCERTAINTY SPIKE @ frame 37 | phase=grasp]")
    assert "dominant: ACE" in txt
    assert "combined=0.56" in txt and "threshold 0.42" in txt
    assert "interventions so far: 1" in txt
    assert "grasping = no" in txt
    assert "hunting for the cube" in txt
    assert txt.rstrip().endswith("No preamble.")


def test_compose_report_dominant_meaning_changes():
    assert "action samples disagree" in _compose_report(_rep(dominant="ace"))
    assert "camera view is unfamiliar" in _compose_report(_rep(dominant="rnd"))
    assert "pose is unusual" in _compose_report(_rep(dominant="smd"))
    assert "grasping = yes" in _compose_report(_rep(grasping=True))
