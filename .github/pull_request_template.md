**What this changes, and why.**

**If it touches the detectors, calibration, or `drishti/scoring.py`:** the before/after numbers on
the bundled dataset (`drishti calibrate --demo --auto`). Every published number moves when that code
does, so the effect has to be measurable.

**Checklist**

- [ ] `pytest -q` passes
- [ ] Any claim added to the docs points at a committed artifact
