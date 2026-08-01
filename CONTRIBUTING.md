# Contributing

This is a research codebase with a small team behind it, so be warned about what that means: we are
unlikely to merge large feature PRs, and we are unlikely to be fast. What we will always read
carefully is anything that says a claim in here is wrong.

## The most useful thing you can send us

**A number that doesn't reproduce.** Every quantitative claim in this repo is supposed to point at a
committed artifact, and the headline calibration is supposed to regenerate byte-for-byte:

```bash
pip install -e '.[uq]'
drishti calibrate --demo --auto     # should match docs/results/calibration/report.json
pytest -q
```

If that diverges on your machine, open an issue with your OS, Python version, `pip freeze`, and the
report you got. That is a real finding — either our seeding isn't as deterministic as we think, or
a dependency changed under us.

**A claim that isn't backed.** If a document states a number and you can't find the artifact behind
it, say so. We have already found and marked several of these ourselves; there are probably more.

**A failure mode the detectors miss.** If you run this on your own policy and it doesn't separate
success from failure, we want to know what your setup looks like. That is more informative than
another task where it works.

## Also welcome

- Bug reports with a reproduction.
- Documentation fixes, especially broken links, stale paths, and commands that no longer run.
- Small, self-contained improvements to the detectors or the calibration, **with a before/after
  number on the bundled dataset** so the effect is measurable.

## Please ask first

- New detectors, new benchmarks, new dependencies.
- Anything that changes the combine rule in [`drishti/scoring.py`](drishti/scoring.py) — every
  published number moves if that changes, so it needs a deliberate decision and a re-run.
- Restructuring. The layout is the way it is on purpose; open an issue and make the case.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
pytest -q                     # no network, no model weights, no GPU
```

House style, such as it is:

- **Comments explain why, not what.** Most of the non-obvious code here exists because something
  simpler was tried and failed; say which, and say what it cost. `drishti/scoring.py` is the model.
- **Never state a number without an artifact.** If the artifact was lost, say that explicitly in the
  doc rather than dropping the number or hedging it into vagueness.
- **Keep negatives.** Superseded results stay in, marked superseded. The way a result goes wrong
  transfers better than the result.
- Line length ~100. Follow the surrounding file.

## Code of conduct

Be decent. Assume good faith. Technical disagreement is welcome and personal attacks are not; we'll
say something once and then stop engaging.

## License

Contributions are accepted under [Apache-2.0](LICENSE), the same license as the project.
