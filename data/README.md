# data/

Everything here is generated, and everything here is gitignored — **except `data/demo/`**, which is
the small bundled dataset that makes the repo runnable without a GPU, a policy checkpoint, or a
download.

## `data/demo/` — the bundled rollouts (committed, ~16 MB)

60 real SO-101 pick-place rollouts from a finetuned SmolVLA (34 succeeded, 26 failed), with the
per-step proprioceptive states, the per-step vision embeddings the detectors score, camera frames
for six representative rollouts, and the calibration fitted on them.

```bash
drishti demo                  # dashboard at http://127.0.0.1:8010, reading this bundle
drishti calibrate --demo --auto   # refit the detectors; reproduces docs/results/calibration/report.json
```

Regenerate it from a full local store with:

```bash
python -m scripts.export_demo_bundle --db data/obs.db --blobs data/obs_blobs --out data/demo
```

`MANIFEST.json` records exactly what went in. Embeddings are stored float16 to halve the size;
calibration casts back to float32 on load. Frames are kept only for the longest three successes and
three failures, subsampled — they exist to be looked at, and they are 90% of the bytes.

## Everything else (generated, ignored)

| Path | What produces it |
|---|---|
| `obs.db`, `obs_blobs/` | the telemetry store and its blobs — written by any recorder or by `drishti serve` |
| `calib/` | `drishti calibrate` — fitted detectors, the threshold, and the validation report |
| `demos/` | `drishti record` — scripted pick-place episodes as `.npz` |
| `lerobot/` | `scripts/build_lerobot_dataset.py` — the demos converted to a LeRobotDataset for finetuning |
| `policy/` | a finetuned policy checkpoint you supply |
| `aha_*/` | `scripts/gen_aha.py` + `scripts/ingest_aha.py` — the AHA/RLBench study data (see [docs/aha/GENERATE.md](../docs/aha/GENERATE.md)) |

The AHA data in particular is derived from [NVlabs/AHA](https://github.com/NVlabs/AHA) and RLBench
and is not ours to redistribute — generate it yourself from those upstreams.
