"""Generate DENSE success + failure demos with rlbench-failgen (runs on the generation pod).

The stock AHA example (`ex_custom_data_generator.py`) sets `save_keyframes_only=True`, which
writes only REFLECT-style keyframe PNGs — NO per-step state, NO `low_dim_obs.pkl`. Our
detectors need dense per-timestep trajectories, so this driver runs the wrapper with
`save_keyframes_only=False` (→ RLBench `save_demo`, which writes `low_dim_obs.pkl` + per-camera
`<cam>_rgb/<t>.png`) and produces BOTH classes into one tree that `ingest_aha.py` reads:

    <save_path>/<task>/success/episodes/episode<N>/         # get_success(), failures disabled
    <save_path>/<task>/<fail_type>/episodes/episode<N>/      # get_failure(), one mode enabled

Run inside the failgen conda env (CoppeliaSim + PyRep + RLBench + failgen importable), headless
via xvfb. See docs/aha/GENERATE.md for the environment + the RGB-only / 2-camera config trim.

    xvfb-run -a -s "-screen 0 1400x900x24" python scripts/gen_aha.py \
        --tasks pick_and_lift stack_blocks put_item_in_drawer \
        --episodes 10 --save-path /data/aha_out
"""

from __future__ import annotations

import argparse

from failgen.env_wrapper import FailGenEnvWrapper

ALL_FAILS = [
    "grasp", "slip", "rotation_x", "rotation_y", "rotation_z",
    "translation_x", "translation_y", "translation_z",
    "no_rotation", "wrong_sequence", "wrong_object",
]


def _enable_only(wrapper, fail_type):
    """Enable exactly one failure mode (or none if fail_type is None). Returns True if present."""
    found = False
    for fo in wrapper.manager._failures:
        on = (fail_type is not None and fo.failure_type == fail_type)
        fo.set_enabled(on)
        found = found or on
    return found or fail_type is None


def gen_task(task, failtypes, episodes, max_tries, save_path, fail_episodes=None):
    fail_episodes = episodes if fail_episodes is None else fail_episodes
    w = FailGenEnvWrapper(task_name=task, headless=True, save_data=True,
                          save_keyframes_only=False)
    # both success and failure demos must land under one root that ingest_aha.py globs;
    # save_failure_ext writes under config.data.save_path, so point that at our out dir.
    w.config.data.save_path = save_path

    # --- successes: all failures disabled, save under <task>/success/ ---
    _enable_only(w, None)
    n_ok = 0
    for i in range(episodes):
        w.reset()
        demo = w.get_success()
        if demo is not None:
            w.save_failure_ext(i, "success", demo)   # wp=-1 -> <task>/success/episodes/episodeN
            n_ok += 1
    print(f"[{task}] success: {n_ok}/{episodes}", flush=True)

    # --- failures: one mode at a time ---
    for ft in failtypes:
        if not _enable_only(w, ft):
            print(f"[{task}] {ft}: not available for this task, skipping", flush=True)
            continue
        n_f = 0
        for i in range(fail_episodes):
            w.reset()
            attempts = max_tries
            while attempts > 0:
                demo, success = w.get_failure()
                if demo is not None and not success:
                    w.save_failure_ext(i, ft, demo)   # -> <task>/<ft>/episodes/episodeN
                    n_f += 1
                    break
                attempts -= 1
        print(f"[{task}] {ft}: {n_f}/{episodes}", flush=True)

    w.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--failtypes", nargs="+", default=ALL_FAILS)
    ap.add_argument("--episodes", type=int, default=10, help="success demos per task")
    ap.add_argument("--fail-episodes", type=int, default=None, help="failure demos per mode (default=episodes)")
    ap.add_argument("--max-tries", type=int, default=5)
    ap.add_argument("--save-path", required=True)
    args = ap.parse_args()
    for task in args.tasks:
        try:
            gen_task(task, args.failtypes, args.episodes, args.max_tries, args.save_path, args.fail_episodes)
        except Exception as e:                      # a flaky task must not kill the whole batch
            import traceback
            print(f"[{task}] ERROR: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
