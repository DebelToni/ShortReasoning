# Reproduction entry points

The none/low DeepSeek self-compaction runners, Sol judging protocol, and offline analysis are documented in [the compressor-effort diagnostic](../results/20260915-deepseek-compressor-effort-audit-v1/README.md).

Offline headline recomputation is performed by `run_archive_smoke.py`. Individual deterministic commands are:

```bash
python3 scripts/analyze_camera_ready_controlled.py --root . \
  --input results/20260804-six-model-common-slice-infrastructure-corrected-v2 \
  --output ~/tmp/controlled-replay
python3 results/20260910-camera-ready-deployment-audit-v2/analyze.py analyze \
  --input results/20260910-camera-ready-deployment-audit-v2/inputs.json \
  --ledger results/20260910-camera-ready-deployment-audit-v2/response-ledger.jsonl.gz \
  --output ~/tmp/deployment-replay.json
python3 results/20260910-camera-ready-deployment-audit-v2/b300_selection_transform.py \
  --input results/20260910-camera-ready-deployment-audit-v2/inputs.json \
  --output ~/tmp/b300-selections.json
python3 scripts/make_archive_figures.py --output-dir ~/tmp/short-reasoning-figures
```

For the controls and LiveCodeBench, copy their result directory under `~/tmp` and invoke `analyze_history_controls.py <copy> --replace` or `analyze_livecodebench_frozen.py <copy> --replace`; the smoke automates both without touching archived captures.

Fresh acquisition is optional and makes paid/provider calls. Discover the frozen interfaces without credentials or network access:

```bash
python3 scripts/run_frozen_experiment.py --help
python3 scripts/run_four_model_replication.py --help
python3 scripts/run_history_controls.py --help
python3 scripts/run_history_tiers.py --help
python3 scripts/run_livecodebench_frozen.py --help
PYTHONPATH=src python3 scripts/run_swebench_mini_arm.py --help
```

Use `manifests/tasks/tasks_v2.json` with the controlled runners and `manifests/tasks/livecodebench_v6_screen.json` with the LiveCodeBench runner. Exact first-stage templates are:

```bash
python3 scripts/run_frozen_experiment.py init --run-dir ~/tmp/fresh-controlled \
  --tasks-file manifests/tasks/tasks_v2.json --models all --tasks all \
  --source-seed 81000 --continuation-seed 91000 --replicates 3
python3 scripts/run_four_model_replication.py init --run-dir ~/tmp/fresh-four-model \
  --config configs/four_model_replication_v1.json
python3 scripts/run_history_controls.py init --run-dir ~/tmp/fresh-controls \
  --parent-run ~/tmp/fresh-controlled --replicates 3
python3 scripts/run_history_tiers.py init --run-dir ~/tmp/fresh-tiers \
  --parent-run ~/tmp/fresh-controlled --replicates 3
python3 scripts/run_livecodebench_frozen.py init --run-dir ~/tmp/fresh-livecodebench \
  --manifest manifests/tasks/livecodebench_v6_screen.json \
  --evaluator-repo ~/tmp/LiveCodeBench --source-seed 310000 \
  --continuation-seed 410000 --replicates 1
PYTHONPATH=src python3 scripts/run_swebench_mini_arm.py \
  --root ~/tmp/swebench-work --dataset ~/tmp/swebench-verified.parquet \
  --config configs/swebench_verified_mini_kimi-k2.6_normal_v8.yaml \
  --run-dir ~/tmp/swebench-run --run-id fresh-kimi-normal \
  --max-new-terminal-tasks 50 --max-new-attempts 50
```

Proceed to each interface's `acquire`, `generate`, `continue`, `finalize`, or `audit` stages as shown by `--help`; those stages can make paid calls. The SWE-bench dataset and LiveCodeBench evaluator are third-party inputs and are intentionally not redistributed. The extension runners default to the frozen JSON files under `configs/`; SWE-bench V7/V8 arm YAMLs and matrix contracts are all supplied under `configs/`. LiveCodeBench also accepts `LIVECODEBENCH_REPOSITORY`; the SWE runner accepts `MINI_SWE_AGENT` and `SWE_BENCH_PYTHON` for external executables. No acquisition command is run by archive verification.
