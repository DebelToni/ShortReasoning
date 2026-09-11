#!/usr/bin/env python3
"""Run the pinned LiveCodeBench multi-turn frozen-history adaptation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import (  # noqa: E402
    BudgetFloorReached,
    load_dotenv,
    remaining_openrouter_credits,
)
from short_reasoning.livecodebench import (  # noqa: E402
    fetch_dataset,
    load_manifest,
    select_records,
)
from short_reasoning.livecodebench_frozen import (  # noqa: E402
    acquire_source,
    continue_source,
    task_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "acquire", "continue", "all"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifests" / "tasks" / "livecodebench_v6_screen.json",
    )
    parser.add_argument("--evaluator-repo", type=Path, default=Path(os.environ.get("LIVECODEBENCH_REPOSITORY", "LiveCodeBench")))
    parser.add_argument("--models", default="deepseek-v4-flash,laguna-s-2.1")
    parser.add_argument("--items", default="all")
    parser.add_argument("--source-seed", type=int, default=310000)
    parser.add_argument("--continuation-seed", type=int, default=410000)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--min-balance", type=float, default=2.0)
    parser.add_argument("--call-reserve", type=float, default=0.10)
    return parser.parse_args()


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def select_models(config: dict[str, Any], requested: str) -> dict[str, Any]:
    names = requested.split(",")
    missing = set(names) - set(config)
    if missing:
        raise SystemExit(f"unknown models: {sorted(missing)}")
    return {name: config[name] for name in names}


def select_items(records: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
    if requested == "all":
        return records
    wanted = set(requested.split(","))
    selected = [record for record in records if record["question_id"] in wanted]
    missing = wanted - {record["question_id"] for record in selected}
    if missing:
        raise SystemExit(f"unknown items: {sorted(missing)}")
    return selected


def evaluator_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def initialize(
    run_dir: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    models: dict[str, Any],
    rewriter: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    actual_commit = evaluator_commit(args.evaluator_repo)
    if actual_commit != manifest["official_evaluator_commit"]:
        raise SystemExit(
            f"LiveCodeBench evaluator commit {actual_commit} != pinned {manifest['official_evaluator_commit']}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    item_ids = [record["question_id"] for record in records]
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["models"] != list(models) or prior["question_ids"] != item_ids:
            raise SystemExit("run directory cohort differs from request")
        return
    atomic_json(
        path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "livecodebench-v6-frozen-reasoning-history-v1",
            "conditions": ["clean", "rewritten"],
            "models": list(models),
            "question_ids": item_ids,
            "source_seed": args.source_seed,
            "minimum_openrouter_balance_usd": args.min_balance,
            "maximum_paid_call_reserve_usd": args.call_reserve,
            "initial_openrouter_balance_usd": remaining_openrouter_credits(),
            "dataset": {
                key: manifest[key]
                for key in (
                    "dataset_repo",
                    "dataset_file",
                    "dataset_revision",
                    "dataset_sha256",
                    "release",
                    "selection_rule",
                )
            },
            "evaluator_repo": str(args.evaluator_repo.resolve()),
            "evaluator_commit": actual_commit,
        },
    )
    atomic_json(run_dir / "models.snapshot.json", {"targets": models, "rewriter": rewriter})
    atomic_json(
        run_dir / "items.snapshot.json",
        [
            {
                key: record[key]
                for key in (
                    "question_id",
                    "question_title",
                    "platform",
                    "difficulty",
                    "contest_id",
                    "contest_date",
                )
            }
            for record in records
        ],
    )


def source_name(model_name: str, record: dict[str, Any]) -> str:
    return f"{model_name}__{record['platform']}-{record['question_id']}.json"


def case_name(model_name: str, record: dict[str, Any], replicate: int, seed: int) -> str:
    return (
        f"{model_name}__{record['platform']}-{record['question_id']}__"
        f"rep-{replicate:02d}__seed-{seed}.json"
    )


def acquire(
    run_dir: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    models: dict[str, Any],
    rewriter: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    directory = run_dir / "sources"
    directory.mkdir(exist_ok=True)
    provenance = {
        "dataset_repo": manifest["dataset_repo"],
        "dataset_file": manifest["dataset_file"],
        "dataset_revision": manifest["dataset_revision"],
        "dataset_sha256": manifest["dataset_sha256"],
    }
    for model_index, (model_name, model) in enumerate(models.items()):
        for item_index, record in enumerate(records):
            path = directory / source_name(model_name, record)
            if path.exists():
                continue
            seed = args.source_seed + model_index * 100000 + item_index * 100
            print(
                f"acquire model={model_name} item={record['question_id']} seed={seed}",
                flush=True,
            )
            try:
                result = acquire_source(
                    copy.deepcopy(record),
                    provenance,
                    model_name,
                    copy.deepcopy(model),
                    copy.deepcopy(rewriter),
                    seed,
                )
            except BudgetFloorReached as exc:
                result = {
                    "task_id": task_id(record),
                    "question_id": record["question_id"],
                    "model_name": model_name,
                    "source_seed": seed,
                    "status": "budget_floor",
                    "error": str(exc),
                }
            except Exception as exc:
                result = {
                    "task_id": task_id(record),
                    "question_id": record["question_id"],
                    "model_name": model_name,
                    "source_seed": seed,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            atomic_json(path, result)
            print(f"finish status={result['status']}", flush=True)
    sources = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    atomic_json(
        run_dir / "acquisition-summary.json",
        {
            "sources": len(sources),
            "status_counts": count_statuses(sources),
            "selected_hashes": [
                source["source_sha256"]
                for source in sources
                if source.get("status") == "selected"
            ],
        },
        replace=True,
    )
    return sources


def continue_run(
    run_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_id = {record["question_id"]: record for record in records}
    source_paths = sorted((run_dir / "sources").glob("*.json"))
    sources = [json.loads(path.read_text()) for path in source_paths]
    sources = [source for source in sources if source.get("status") == "selected"]
    directory = run_dir / "continuations"
    directory.mkdir(exist_ok=True)
    model_order = {
        name: index
        for index, name in enumerate(json.loads((run_dir / "run.json").read_text())["models"])
    }
    item_order = {record["question_id"]: index for index, record in enumerate(records)}
    for source in sources:
        record = records_by_id[source["question_id"]]
        for replicate in range(1, args.replicates + 1):
            seed = (
                args.continuation_seed
                + model_order[source["model_name"]] * 1000000
                + item_order[source["question_id"]] * 1000
                + (replicate - 1) * 10
            )
            path = directory / case_name(source["model_name"], record, replicate, seed)
            if path.exists():
                continue
            print(
                f"continue model={source['model_name']} item={source['question_id']} "
                f"rep={replicate} seed={seed}",
                flush=True,
            )
            try:
                result = continue_source(
                    copy.deepcopy(record),
                    copy.deepcopy(source),
                    args.evaluator_repo,
                    seed,
                    replicate,
                )
            except BudgetFloorReached as exc:
                result = {
                    "task_id": source["task_id"],
                    "question_id": source["question_id"],
                    "model_name": source["model_name"],
                    "source_sha256": source["source_sha256"],
                    "continuation_seed": seed,
                    "replicate": replicate,
                    "status": "budget_floor",
                    "error": str(exc),
                }
            except Exception as exc:
                result = {
                    "task_id": source["task_id"],
                    "question_id": source["question_id"],
                    "model_name": source["model_name"],
                    "source_sha256": source["source_sha256"],
                    "continuation_seed": seed,
                    "replicate": replicate,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            atomic_json(path, result)
            print(f"finish status={result['status']}", flush=True)
    cases = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    atomic_json(
        run_dir / "continuation-summary.json",
        {
            "requested_replicates": args.replicates,
            "cases": len(cases),
            "status_counts": count_statuses(cases),
        },
        replace=True,
    )
    return cases


def count_statuses(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record.get("status", "missing")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["OPENROUTER_MIN_BALANCE_USD"] = str(args.min_balance)
    os.environ["OPENROUTER_MAX_CALL_COST_USD"] = str(args.call_reserve)
    manifest = load_manifest(args.manifest)
    records = select_items(select_records(fetch_dataset(manifest), manifest), args.items)
    config = json.loads((ROOT / "configs" / "models.json").read_text())
    models = select_models(config["targets"], args.models)
    run_dir = args.run_dir.resolve()
    initialize(run_dir, args, manifest, models, config["rewriter"], records)
    if args.mode in {"acquire", "all"}:
        acquire(run_dir, args, manifest, models, config["rewriter"], records)
    if args.mode in {"continue", "all"}:
        continue_run(run_dir, args, records)
    atomic_json(
        run_dir / "balance.json",
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "remaining_usd": remaining_openrouter_credits(),
            "floor_usd": args.min_balance,
        },
        replace=True,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
