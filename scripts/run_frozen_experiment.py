#!/usr/bin/env python3
"""Acquire immutable model-specific history forks, then fan out paired continuations."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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
    load_tasks,
    remaining_openrouter_credits,
)
from short_reasoning.frozen import (  # noqa: E402
    acquire_frozen_source,
    continue_frozen_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "acquire", "continue", "all"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tasks-file", type=Path, default=ROOT / "manifests" / "tasks" / "tasks_v2.json")
    parser.add_argument("--models", default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--source-seed", type=int, default=81000)
    parser.add_argument("--continuation-seed", type=int, default=91000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-balance", type=float, default=2.0)
    parser.add_argument("--call-reserve", type=float, default=0.10)
    return parser.parse_args()


def select_models(config: dict[str, Any], requested: str) -> dict[str, Any]:
    if requested == "all":
        return config
    wanted = requested.split(",")
    missing = set(wanted) - set(config)
    if missing:
        raise SystemExit(f"unknown models: {sorted(missing)}")
    return {name: config[name] for name in wanted}


def select_tasks(tasks: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
    if requested == "all":
        return tasks
    wanted = set(requested.split(","))
    selected = [task for task in tasks if task["id"] in wanted]
    missing = wanted - {task["id"] for task in selected}
    if missing:
        raise SystemExit(f"unknown tasks: {sorted(missing)}")
    return selected


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def repository_state() -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    return {
        "commit": output("git", "rev-parse", "HEAD"),
        "branch": output("git", "branch", "--show-current"),
        "dirty": bool(output("git", "status", "--porcelain")),
    }


def source_filename(model_name: str, task_id: str) -> str:
    return f"{model_name}__{task_id}.json"


def continuation_filename(model_name: str, task_id: str, replicate: int, seed: int) -> str:
    return f"{model_name}__{task_id}__rep-{replicate:02d}__seed-{seed}.json"


def initialize_run(
    run_dir: Path,
    args: argparse.Namespace,
    models: dict[str, Any],
    rewriter: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = run_dir / "run.json"
    if metadata.exists():
        existing = json.loads(metadata.read_text())
        if existing["models"] != list(models) or existing["tasks"] != [task["id"] for task in tasks]:
            raise SystemExit("run directory model/task cohort differs from the requested cohort")
        return
    atomic_json(
        metadata,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "frozen-reasoning-history-fork-v2",
            "conditions": ["clean", "rewritten"],
            "instruction_condition": False,
            "models": list(models),
            "tasks": [task["id"] for task in tasks],
            "tasks_file": str(args.tasks_file.resolve().relative_to(ROOT)),
            "source_seed": args.source_seed,
            "minimum_openrouter_balance_usd": args.min_balance,
            "maximum_paid_call_reserve_usd": args.call_reserve,
            "repository": repository_state(),
            "initial_openrouter_balance_usd": remaining_openrouter_credits(),
        },
    )
    atomic_json(run_dir / "models.snapshot.json", {"targets": models, "rewriter": rewriter})
    atomic_json(run_dir / "tasks.snapshot.json", tasks)


def acquire(
    run_dir: Path,
    args: argparse.Namespace,
    models: dict[str, Any],
    rewriter: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(exist_ok=True)
    jobs = []
    for model_index, (model_name, model) in enumerate(models.items()):
        for task_index, task in enumerate(tasks):
            path = sources_dir / source_filename(model_name, task["id"])
            if path.exists():
                continue
            seed = args.source_seed + model_index * 10000 + task_index * 100
            jobs.append((task, model_name, model, seed, path))

    def execute(job: tuple[Any, ...]) -> dict[str, Any]:
        task, model_name, model, seed, path = job
        print(f"acquire start model={model_name} task={task['id']} seed={seed}", flush=True)
        try:
            result = acquire_frozen_source(
                copy.deepcopy(task),
                model_name,
                copy.deepcopy(model),
                copy.deepcopy(rewriter),
                seed,
            )
        except BudgetFloorReached as exc:
            result = {
                "task_id": task["id"],
                "model_name": model_name,
                "source_seed": seed,
                "status": "budget_floor",
                "error": str(exc),
            }
        except Exception as exc:
            result = {
                "task_id": task["id"],
                "model_name": model_name,
                "source_seed": seed,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        atomic_json(path, result)
        print(f"acquire finish model={model_name} task={task['id']} status={result['status']}", flush=True)
        return result

    if args.workers == 1:
        for job in jobs:
            execute(job)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(execute, jobs))
    sources = [json.loads(path.read_text()) for path in sorted(sources_dir.glob("*.json"))]
    atomic_json(
        run_dir / "acquisition-summary.json",
        {
            "sources": len(sources),
            "status_counts": count_statuses(sources),
            "selected_hashes": [
                source["source_sha256"] for source in sources if source.get("status") == "selected"
            ],
        },
        replace=True,
    )
    return sources


def continue_sources(
    run_dir: Path,
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks_by_id = {task["id"]: task for task in tasks}
    sources = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "sources").glob("*.json"))
    ]
    selected = [source for source in sources if source.get("status") == "selected"]
    cases_dir = run_dir / "continuations"
    cases_dir.mkdir(exist_ok=True)
    model_order = {name: index for index, name in enumerate(json.loads((run_dir / "run.json").read_text())["models"])}
    task_order = {task["id"]: index for index, task in enumerate(tasks)}
    jobs = []
    for source in selected:
        model_index = model_order[source["model_name"]]
        task_index = task_order[source["task_id"]]
        for replicate in range(1, args.replicates + 1):
            seed = (
                args.continuation_seed
                + model_index * 100000
                + task_index * 1000
                + (replicate - 1) * 10
            )
            path = cases_dir / continuation_filename(
                source["model_name"], source["task_id"], replicate, seed
            )
            if path.exists():
                continue
            jobs.append((source, tasks_by_id[source["task_id"]], replicate, seed, path))

    def execute(job: tuple[Any, ...]) -> dict[str, Any]:
        source, task, replicate, seed, path = job
        print(
            f"continue start model={source['model_name']} task={task['id']} rep={replicate} seed={seed}",
            flush=True,
        )
        try:
            result = continue_frozen_source(
                copy.deepcopy(task), copy.deepcopy(source), seed, replicate
            )
        except BudgetFloorReached as exc:
            result = {
                "task_id": task["id"],
                "model_name": source["model_name"],
                "source_sha256": source["source_sha256"],
                "continuation_seed": seed,
                "replicate": replicate,
                "status": "budget_floor",
                "error": str(exc),
            }
        except Exception as exc:
            result = {
                "task_id": task["id"],
                "model_name": source["model_name"],
                "source_sha256": source["source_sha256"],
                "continuation_seed": seed,
                "replicate": replicate,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        atomic_json(path, result)
        print(
            f"continue finish model={source['model_name']} task={task['id']} rep={replicate} status={result['status']}",
            flush=True,
        )
        return result

    if args.workers == 1:
        for job in jobs:
            execute(job)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(execute, jobs))
    cases = [json.loads(path.read_text()) for path in sorted(cases_dir.glob("*.json"))]
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
    model_config = json.loads((ROOT / "configs" / "models.json").read_text())
    models = select_models(model_config["targets"], args.models)
    tasks = select_tasks(load_tasks(args.tasks_file), args.tasks)
    run_dir = args.run_dir.resolve()
    initialize_run(run_dir, args, models, model_config["rewriter"], tasks)
    if args.mode in {"acquire", "all"}:
        acquire(run_dir, args, models, model_config["rewriter"], tasks)
    if args.mode in {"continue", "all"}:
        continue_sources(run_dir, args, tasks)
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
