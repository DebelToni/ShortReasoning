# Environments

- `smoke/requirements.txt` is intentionally empty: integrity checks, headline analysis replay, and standard-library verification tests need no package download.
- `tests/requirements.txt` pins the complete pytest runner stack for all included tests.
- `figures/requirements.txt` pins the tested Matplotlib stack used by `scripts/make_archive_figures.py`.
- `acquisition/requirements.txt` pins direct optional SWE-bench acquisition dependencies. Fresh hosted acquisition additionally needs provider credentials; LiveCodeBench needs its evaluator checkout; SWE-bench needs Docker and public benchmark assets.

The tested interpreter is CPython 3.14.3. Retained Python syntax requires 3.11 or newer.
