"""`drishti` — one entry point for the things you actually run.

    drishti demo                  # dashboard on :8010, against the bundled rollouts
    drishti serve                 # dashboard on :8010, against your own data/
    drishti calibrate --auto      # fit the detectors on the successful rollouts, write the report
    drishti record --episodes 20  # drive the scripted pick-place in sim and log it
    drishti figures               # regenerate the calibration figures from the current report

`demo` exists because the interesting part of this project is a *calibrated* detector, and a
calibration needs recorded rollouts. Rather than ask a newcomer to obtain a policy checkpoint and a
GPU before they can see anything, the repo ships ~60 real rollouts (data/demo/) and `demo` points
everything at them. It is the same code path as `serve`; only the data location differs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
DEFAULT_PORT = 8010


def _use_demo_data() -> None:
    """Point the store, blob dir and calibration at the bundled dataset."""
    if not (DEMO / "obs.db").exists():
        sys.exit(f"no demo bundle at {DEMO} — this checkout is missing data/demo/")
    os.environ.setdefault("DRISHTI_OBS_DB_URL", f"sqlite:///{DEMO / 'obs.db'}")
    os.environ.setdefault("DRISHTI_BLOB_DIR", str(DEMO / "blobs"))
    os.environ.setdefault("DRISHTI_CALIB_DIR", str(DEMO / "calib"))


def _serve(host: str, port: int) -> None:
    import uvicorn

    print(f"drishti → http://{host}:{port}", flush=True)
    print(f"  store {os.environ.get('DRISHTI_OBS_DB_URL', 'sqlite:///data/obs.db')}", flush=True)
    uvicorn.run("drishti.server.app:app", host=host, port=port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="drishti", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, help_ in (("demo", "serve the dashboard against the bundled rollouts"),
                        ("serve", "serve the dashboard against your own data/")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--host", default="127.0.0.1",
                       help="bind address. Keep it on localhost: the API runs subprocesses on "
                            "request and has no authentication.")
        p.add_argument("--port", type=int, default=DEFAULT_PORT)

    p = sub.add_parser("calibrate", help="fit the detectors and write the validation report")
    p.add_argument("--runs", nargs="*", default=None, help="run ids (default: every success)")
    p.add_argument("--auto", action="store_true", help="calibrate on ALL successful runs")
    p.add_argument("--min-steps", type=int, default=None,
                   help="ignore 'successes' shorter than this (oracle artifacts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", action="store_true", help="use the bundled rollouts")

    p = sub.add_parser("record", help="run the scripted pick-place in sim and log rollouts")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", default="data/demos")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("figures", help="regenerate the calibration figures")
    p.add_argument("--out", default="assets/figures")
    p.add_argument("--demo", action="store_true", help="use the bundled rollouts")

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        _use_demo_data()
        _serve(args.host, args.port)
    elif args.cmd == "serve":
        _serve(args.host, args.port)
    elif args.cmd == "calibrate":
        if args.demo:
            _use_demo_data()
            os.environ["DRISHTI_CALIB_DIR"] = str(DEMO / "calib")
        from drishti import calibrate as C

        kw = {} if args.min_steps is None else {"min_steps": args.min_steps}
        C.calibrate(run_ids=args.runs, auto=args.auto or not args.runs, seed=args.seed, **kw)
    elif args.cmd == "record":
        from drishti.sim.record import main as record_main

        record_main(["--episodes", str(args.episodes), "--out", args.out,
                     "--seed", str(args.seed), "--headless"])
    elif args.cmd == "figures":
        if args.demo:
            _use_demo_data()
        from drishti.figures import main as figures_main

        sys.argv = ["drishti-figures", "--out", args.out]
        figures_main()


if __name__ == "__main__":
    main()
