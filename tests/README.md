# Tests

`test_archive.py` covers manifest completeness, structure, recomputed metrics, figure labels/data, and parsing. `test_camera_ready_controlled.py` tests retry-augmented versus original-first-capture estimands, sign conventions, clustered resampling, source accounting, corrections, timing invariants, and portable output. `test_camera_ready_deployment.py` tests orphan-response retention, no-ID handling, conflicting-ID failure, exact 77,228-row ledger replay, and B300 23/25 selection; `test_camera_ready_deployment_stdlib.py` supplies equivalent smoke coverage without a pytest installation.

Run every included test with `python3 -m pytest -q -p no:cacheprovider tests`. The smoke invokes the three standard-library suites directly and does not import the pytest-only file.
