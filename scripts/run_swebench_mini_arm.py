#!/usr/bin/env python3
"""Run one Verified Mini model/compaction arm through infrastructure completion."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from short_reasoning import remaining_openrouter_credits

MODEL_TERMINAL_STATUSES = {
    "LimitsExceeded",
    "RepeatedFormatError",
    "Submitted",
    "TimeExceeded",
}
FREE_RESET_PATTERN = re.compile(
    r'"X-RateLimit-Reset"\s*:\s*"?(\d+)'
)


def free_daily_reset_at(exception: str) -> float | None:
    """Parse OpenRouter's UTC reset epoch from its free daily-limit error."""
    if "free-models-per-day" not in exception:
        return None
    match = FREE_RESET_PATTERN.search(exception)
    if match is None:
        return None
    reset = float(match.group(1))
    return reset / 1000 if reset > 100_000_000_000 else reset


def select_pending(pending: set[str], batch_size: int) -> set[str]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return set(sorted(pending)[:batch_size])


def task_reserve_available(
    remaining: float, floor: float, task_reserve: float
) -> bool:
    if task_reserve < 0:
        raise ValueError("task_reserve must be nonnegative")
    return remaining - task_reserve > floor


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_subset(source: Path, destination: Path, ids: set[str]) -> None:
    table = pq.read_table(source)
    subset = table.filter([value.as_py() in ids for value in table["instance_id"]])
    if subset.num_rows != len(ids):
        raise RuntimeError(f"subset mismatch: expected {len(ids)}, got {subset.num_rows}")
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, destination / "test.parquet")


def trajectory_records(inference: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in inference.glob("*/*.traj.json"):
        trajectory = json.loads(path.read_text())
        records[trajectory["instance_id"]] = {"path": path, "trajectory": trajectory}
    return records


def run_attempt(
    root: Path,
    dataset: Path,
    config: Path,
    run_dir: Path,
    attempt: int,
    pending: set[str],
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    attempt_dir = run_dir / "attempts" / f"attempt-{attempt:03d}"
    subset = attempt_dir / "dataset"
    inference = attempt_dir / "inference"
    write_subset(dataset / "test.parquet", subset, pending)
    inference.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["SHORT_REASONING_DURABLE_JOURNAL_ROOT"] = str(run_dir / "journals")
    environment["SHORT_REASONING_DURABLE_SESSION"] = f"attempt-{attempt:03d}"
    command = [
        os.environ.get("MINI_SWE_AGENT", "mini-extra"),
        "swebench",
        "--subset",
        str(subset),
        "--split",
        "test",
        "--output",
        str(inference),
        "--workers",
        str(min(workers, len(pending))),
        "--config",
        "swebench.yaml",
        "--config",
        str(config),
    ]
    with (attempt_dir / "runner.log").open("wb") as log:
        process = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    predictions_path = inference / "preds.json"
    predictions = json.loads(predictions_path.read_text()) if predictions_path.exists() else {}
    write_json(
        attempt_dir / "attempt.json",
        {
            "attempt": attempt,
            "command": command,
            "pending_ids": sorted(pending),
            "returncode": process.returncode,
        },
    )
    return trajectory_records(inference), predictions


def summarize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    responses = [
        message["extra"]["response"]
        for message in trajectory.get("messages", [])
        if message.get("extra", {}).get("response")
    ]
    compaction = trajectory.get("info", {}).get("compaction", {})
    return {
        "exit_status": trajectory["info"]["exit_status"],
        "exception_str": trajectory["info"].get("exception_str"),
        "target_calls": len(responses),
        "target_cost_usd": sum(
            float(response.get("usage", {}).get("cost") or 0) for response in responses
        ),
        "target_providers": sorted(
            {response.get("provider") for response in responses if response.get("provider")}
        ),
        "target_models": sorted(
            {response.get("model") for response in responses if response.get("model")}
        ),
        "compaction_unique_blocks": int(compaction.get("unique_blocks", 0)),
        "compaction_cost_usd": float(
            compaction.get("provider_reported_cost_usd", 0) or 0
        ),
        "compaction_events": compaction.get("events", []),
    }


def recover_existing_state(
    run_dir: Path, all_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    """Reconstruct first terminal predictions so interrupted arms can resume."""
    accepted: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in all_ids
    }
    last_attempt = 0
    for attempt_dir in sorted((run_dir / "attempts").glob("attempt-*")):
        try:
            attempt = int(attempt_dir.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        last_attempt = max(last_attempt, attempt)
        metadata_path = attempt_dir / "attempt.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            pending = set(metadata.get("pending_ids", [])) & all_ids
        else:
            subset_path = attempt_dir / "dataset" / "test.parquet"
            if not subset_path.exists():
                continue
            subset = pq.read_table(subset_path)
            pending = {
                value.as_py() for value in subset["instance_id"]
            } & all_ids
        inference = attempt_dir / "inference"
        records = trajectory_records(inference)
        predictions_path = inference / "preds.json"
        predictions = (
            json.loads(predictions_path.read_text())
            if predictions_path.exists()
            else {}
        )
        for instance_id in sorted(pending):
            record = records.get(instance_id)
            if record is None:
                attempts[instance_id].append(
                    {
                        "attempt": attempt,
                        "exit_status": "missing_trajectory",
                    }
                )
                continue
            summary = summarize_trajectory(record["trajectory"])
            summary["attempt"] = attempt
            summary["trajectory"] = str(
                record["path"].relative_to(run_dir)
            )
            attempts[instance_id].append(summary)
            if instance_id in accepted:
                continue
            if summary["exit_status"] not in MODEL_TERMINAL_STATUSES:
                continue
            prediction = predictions.get(instance_id)
            if not isinstance(prediction, dict):
                continue
            accepted[instance_id] = {
                "attempt": attempt,
                "prediction": prediction,
                "trajectory": summary["trajectory"],
                "summary": summary,
            }
    return accepted, attempts, last_attempt


def evaluate(dataset: Path, predictions: Path, run_dir: Path, run_id: str) -> Path:
    evaluation = run_dir / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    command = [
        os.environ.get("SWE_BENCH_PYTHON", "python"),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset / "test.json"),
        "--split",
        "test",
        "--predictions_path",
        str(predictions),
        "--max_workers",
        os.environ.get("SWE_MATRIX_EVAL_WORKERS", "2"),
        "--run_id",
        run_id,
        "--cache_level",
        "instance",
        "--clean",
        "false",
        "--report_dir",
        str(evaluation),
    ]
    environment = os.environ.copy()
    environment["HOME"] = "/tmp/shortreasoning-docker-home"
    environment["DOCKER_CONFIG"] = "/tmp/shortreasoning-docker-home/.docker"
    with (evaluation / "runner.log").open("wb") as log:
        subprocess.run(
            command,
            cwd=evaluation,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reports = list(evaluation.glob("*.json"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one evaluation report, found {reports}")
    return reports[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-task-reserve-usd", type=float, default=0)
    parser.add_argument("--max-new-terminal-tasks", type=int, default=0)
    parser.add_argument("--max-new-attempts", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    completed_path = args.run_dir / "result.json"
    if completed_path.exists():
        print(completed_path.read_text(), end="")
        return
    shutil.copy2(args.config, args.run_dir / "config.yaml")
    shutil.copy2(args.dataset / "test.parquet", args.run_dir / "dataset.parquet")
    shutil.copy2(args.dataset / "test.json", args.run_dir / "dataset.json")
    table = pq.read_table(args.dataset / "test.parquet")
    all_ids = {value.as_py() for value in table["instance_id"]}
    accepted, attempts, attempt = recover_existing_state(args.run_dir, all_ids)
    accepted_at_start = len(accepted)
    attempt_at_start = attempt

    while set(accepted) != all_ids:
        pending = all_ids - set(accepted)
        scheduled = select_pending(pending, args.batch_size)
        if args.min_task_reserve_usd:
            remaining = remaining_openrouter_credits()
            floor = float(os.environ.get("OPENROUTER_MIN_BALANCE_USD", "2.0"))
            if not task_reserve_available(
                remaining, floor, args.min_task_reserve_usd
            ):
                write_json(
                    args.run_dir / "checkpoint.json",
                    {
                        "status": "paused_before_task_for_reserve",
                        "accepted": len(accepted),
                        "pending": len(pending),
                        "remaining_balance_usd": remaining,
                        "balance_floor_usd": floor,
                        "task_reserve_usd": args.min_task_reserve_usd,
                    },
                )
                return
        attempt += 1
        records, predictions = run_attempt(
            args.root,
            args.dataset,
            args.config,
            args.run_dir,
            attempt,
            scheduled,
            args.workers,
        )
        progress = False
        budget_floor = False
        free_reset_at = 0.0
        for instance_id in sorted(scheduled):
            record = records.get(instance_id)
            if record is None:
                attempts[instance_id].append(
                    {"attempt": attempt, "exit_status": "missing_trajectory"}
                )
                continue
            trajectory = record["trajectory"]
            summary = summarize_trajectory(trajectory)
            summary["attempt"] = attempt
            summary["trajectory"] = str(record["path"].relative_to(args.run_dir))
            attempts[instance_id].append(summary)
            status = summary["exit_status"]
            exception = str(summary.get("exception_str") or "")
            lowered_exception = exception.lower()
            if status == "BudgetFloorReached" or (
                "balance" in lowered_exception and "floor" in lowered_exception
            ):
                budget_floor = True
            parsed_reset = free_daily_reset_at(exception)
            if parsed_reset is not None:
                free_reset_at = max(free_reset_at, parsed_reset)
            if status not in MODEL_TERMINAL_STATUSES:
                continue
            prediction = predictions.get(instance_id)
            if not isinstance(prediction, dict):
                continue
            accepted[instance_id] = {
                "attempt": attempt,
                "prediction": prediction,
                "trajectory": summary["trajectory"],
                "summary": summary,
            }
            progress = True
        write_json(
            args.run_dir / "progress.json",
            {
                "attempt": attempt,
                "accepted": sorted(accepted),
                "pending": sorted(all_ids - set(accepted)),
                "attempts": attempts,
            },
        )
        if set(accepted) == all_ids:
            break
        if (
            args.max_new_terminal_tasks
            and len(accepted) - accepted_at_start
            >= args.max_new_terminal_tasks
        ):
            write_json(
                args.run_dir / "checkpoint.json",
                {
                    "status": "task_boundary_checkpoint",
                    "accepted": len(accepted),
                    "pending": len(all_ids - set(accepted)),
                    "new_terminal_tasks": len(accepted) - accepted_at_start,
                    "last_attempt": attempt,
                },
            )
            return
        if (
            args.max_new_attempts
            and attempt - attempt_at_start >= args.max_new_attempts
        ):
            write_json(
                args.run_dir / "checkpoint.json",
                {
                    "status": "attempt_boundary_checkpoint",
                    "accepted": len(accepted),
                    "pending": len(all_ids - set(accepted)),
                    "new_terminal_tasks": len(accepted) - accepted_at_start,
                    "new_attempts": attempt - attempt_at_start,
                    "last_attempt": attempt,
                    "free_reset_at": free_reset_at or None,
                },
            )
            return
        if free_reset_at > time.time():
            time.sleep(free_reset_at - time.time() + 2.0)
        elif budget_floor:
            time.sleep(300)
        elif not progress:
            time.sleep(30)

    merged = {instance_id: accepted[instance_id]["prediction"] for instance_id in sorted(accepted)}
    inference = args.run_dir / "inference"
    inference.mkdir(exist_ok=True)
    write_json(inference / "preds.json", merged)
    with (inference / "preds.jsonl").open("w") as handle:
        for instance_id in sorted(merged):
            handle.write(json.dumps(merged[instance_id], ensure_ascii=False) + "\n")
    report_path = evaluate(
        args.dataset,
        inference / "preds.jsonl",
        args.run_dir,
        args.run_id,
    )
    report = json.loads(report_path.read_text())
    selected = [accepted[instance_id]["summary"] for instance_id in sorted(accepted)]
    all_summaries = [
        summary
        for instance_attempts in attempts.values()
        for summary in instance_attempts
        if "target_calls" in summary
    ]
    events = [
        event for item in all_summaries for event in item["compaction_events"]
    ]
    result = {
        "status": "complete",
        "tasks": len(all_ids),
        "nonempty_patches": sum(
            bool(value["model_patch"].strip()) for value in merged.values()
        ),
        "resolved": report["resolved_instances"],
        "unresolved": report["unresolved_instances"],
        "selected_target_calls": sum(item["target_calls"] for item in selected),
        "acquisition_target_calls": sum(
            item["target_calls"] for item in all_summaries
        ),
        "target_cost_usd": sum(item["target_cost_usd"] for item in all_summaries),
        "compaction_unique_blocks": sum(
            item["compaction_unique_blocks"] for item in all_summaries
        ),
        "compaction_cost_usd": sum(
            item["compaction_cost_usd"] for item in all_summaries
        ),
        "compaction_raw_words": sum(event["raw_words"] for event in events),
        "compaction_state_words": sum(event["state_words"] for event in events),
        "compaction_events_with_missing_identifiers": sum(
            bool(event["missing_identifiers"]) for event in events
        ),
        "compaction_events_with_missing_numbers": sum(
            bool(event["missing_numeric_literals"]) for event in events
        ),
        "selected_target_providers": sorted(
            {provider for item in selected for provider in item["target_providers"]}
        ),
        "all_target_providers": sorted(
            {
                provider
                for item in all_summaries
                for provider in item["target_providers"]
            }
        ),
        "attempt_count": attempt,
        "attempts_per_item": {
            instance_id: len(attempts[instance_id]) for instance_id in sorted(all_ids)
        },
        "official_report": str(report_path.relative_to(args.run_dir)),
    }
    write_json(args.run_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
