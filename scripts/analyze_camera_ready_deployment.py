#!/usr/bin/env python3
"""Portable camera-ready deployment and latency audit.

``collect`` reads immutable local captures and emits privacy-minimized compact inputs
plus a gzip JSONL response ledger. ``analyze`` reads only those two derived files.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ARMS = ("normal", "luna-compact", "self-compact")
TERMINAL = {"LimitsExceeded", "RepeatedFormatError", "Submitted", "TimeExceeded"}
RUN_PREFIXES = {
    "deepseek-v4-flash": "deepseek-v4-flash",
    "minimax-m3": "minimax-m3",
    "nemotron-3-ultra": None,
    "kimi-k2.6": None,
}
SELECTED_FIELDS = (
    "assistant_responses", "response_ids", "input_tokens", "output_tokens",
    "reasoning_tokens", "tool_calls",
)
B300_METRICS = (
    "complete_task_seconds", "summed_request_seconds",
    "summed_generation_phase_seconds", "tool_execution_seconds",
    "future_reasoning_tokens", "total_output_tokens", "turns",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def hashed_id(value: str) -> str:
    return sha256_bytes(value.encode())


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_mean_interval(differences: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(differences)
    draws = [
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(10_000)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def trajectory_response_ids(path: Path) -> list[str]:
    output = []
    for message in load(path).get("messages", []):
        response = (message.get("extra") or {}).get("response")
        if response and response.get("id"):
            output.append(str(response["id"]))
    return output


def all_trajectory_linkage(roots: Iterable[Path]) -> tuple[dict[str, set[str]], dict[str, str]]:
    response_tasks: dict[str, set[str]] = defaultdict(set)
    attempt_tasks: dict[str, str] = {}
    ambiguous_attempts: set[str] = set()
    for root in roots:
        for path in root.glob("**/*.traj.json"):
            parts = path.parts
            task = path.parent.name
            attempt = next((part for part in parts if re.fullmatch(r"attempt-\d+", part)), None)
            if attempt:
                if attempt in attempt_tasks and attempt_tasks[attempt] != task:
                    ambiguous_attempts.add(attempt)
                else:
                    attempt_tasks[attempt] = task
            for response_id in trajectory_response_ids(path):
                response_tasks[response_id].add(task)
    for attempt in ambiguous_attempts:
        attempt_tasks.pop(attempt, None)
    return response_tasks, attempt_tasks


def selected_paths(run: Path) -> dict[str, Path]:
    progress = load(run / "progress.json")
    output = {}
    for instance, attempts in progress["attempts"].items():
        selected = next(
            row for row in attempts
            if row.get("exit_status") in TERMINAL and row.get("trajectory")
        )
        output[instance] = run / selected["trajectory"]
    if len(output) != 50:
        raise ValueError(f"expected 50 selected trajectories, found {len(output)}")
    return output


def deepseek_normal_selected_paths(original: Path, recovery: Path, status: Path) -> dict[str, Path]:
    output = {}
    for row in load(original / "item-statuses.json"):
        if row["trajectory_status"] == "Submitted":
            matches = list((original / "inference" / row["instance_id"]).glob("*.traj.json"))
            if len(matches) != 1:
                raise ValueError((row["instance_id"], matches))
            output[row["instance_id"]] = matches[0]
    for row in load(status):
        task, attempt = row["instance_id"], int(row["completed_attempt"])
        matches = list((recovery / "attempts" / f"attempt-{attempt:03d}" / "inference" / task).glob("*.traj.json"))
        if len(matches) != 1:
            raise ValueError((task, matches))
        output[task] = matches[0]
    if len(output) != 50:
        raise ValueError(f"expected 50 DeepSeek Normal trajectories, found {len(output)}")
    return output


def selected_id_map(paths: dict[str, Path]) -> dict[str, str]:
    output = {}
    for task, path in paths.items():
        for response_id in trajectory_response_ids(path):
            if response_id in output and output[response_id] != task:
                raise ValueError(f"selected response appears under two tasks: {response_id}")
            output[response_id] = task
    return output


def trajectory_metrics(path: Path) -> dict[str, Any]:
    data = load(path)
    assistants = [row for row in data.get("messages", []) if row.get("role") == "assistant"]
    responses = [
        (row.get("extra") or {}).get("response") for row in assistants
        if (row.get("extra") or {}).get("response")
    ]
    usages = [(row or {}).get("usage") or {} for row in responses]
    ids = [str(row["id"]) for row in responses if row.get("id")]
    return {
        "assistant_responses": len(assistants),
        "response_ids": len(set(ids)),
        "input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usages),
        "output_tokens": sum(int(row.get("completion_tokens") or 0) for row in usages),
        "reasoning_tokens": sum(int((row.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) for row in usages),
        "tool_calls": sum(len(row.get("tool_calls") or []) for row in assistants),
        "exit_status": (data.get("info") or {}).get("exit_status"),
    }


def selected_metrics(paths: dict[str, Path]) -> dict[str, Any]:
    tasks = {task: trajectory_metrics(path) for task, path in sorted(paths.items())}
    totals = {field: sum(int(row[field]) for row in tasks.values()) for field in SELECTED_FIELDS}
    totals |= {
        "tasks": len(tasks),
        "exit_statuses": dict(sorted(Counter(str(row["exit_status"]) for row in tasks.values()).items())),
    }
    return {"totals": totals, "tasks": tasks, "task_metrics_sha256": canonical_sha256(tasks)}


def official_outcomes(path: Path) -> dict[str, Any]:
    report = load(path)
    task_ids = sorted(
        set(report.get("completed_ids", [])) | set(report.get("submitted_ids", []))
        | set(report.get("unresolved_ids", [])) | set(report.get("error_ids", []))
        | set(report.get("empty_patch_ids", []))
    )
    if len(task_ids) != 50:
        raise ValueError(f"official report has {len(task_ids)} task IDs")
    resolved = set(report["resolved_ids"])
    return {
        "tasks": {task: task in resolved for task in task_ids},
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(set(task_ids) - resolved),
        "official_counts": {key: int(report.get(key, 0)) for key in (
            "total_instances", "submitted_instances", "completed_instances",
            "resolved_instances", "unresolved_instances", "empty_patch_instances", "error_instances",
        )},
    }


def prediction_bindings(path: Path, official: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = load(path)
    output = []
    for task in sorted(official["tasks"]):
        row = predictions[task]
        patch = str(row.get("model_patch") or "")
        output.append({
            "task_id": task,
            "model_name": row.get("model_name_or_path"),
            "patch_sha256": sha256_bytes(patch.encode()),
            "patch_bytes": len(patch.encode()),
            "official_resolved": official["tasks"][task],
        })
    return output


def journal_paths(root: Path) -> Iterable[Path]:
    yield from sorted((root / "journals").glob("**/attempt-*.json"))


def journal_linkage(path: Path, root: Path, attempt_tasks: dict[str, str]) -> dict[str, Any]:
    relative = path.relative_to(root)
    attempt = next((part for part in relative.parts if re.fullmatch(r"attempt-\d+", part)), None)
    call = next((part for part in relative.parts if re.fullmatch(r"call-\d+", part)), None)
    return {
        "acquisition_attempt": attempt,
        "call": call,
        "transport_attempt": path.stem,
        "task_id_from_attempt": attempt_tasks.get(attempt) if attempt else None,
    }


def minimal_usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_input_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        "reported_cost_usd": float(usage.get("cost") or 0),
    }


def collect_response_ledger(
    model: str,
    arm: str,
    roots: list[Path],
    selected: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response_tasks, attempt_tasks = all_trajectory_linkage(roots)
    trajectory_ids = set(response_tasks)
    selected_ids = selected_id_map(selected)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for root_number, root in enumerate(roots, 1):
        for path in journal_paths(root):
            is_compressor = "compressor" in path.parts
            role = "compressor" if is_compressor else "target"
            try:
                journal_bytes = path.read_bytes()
                journal = json.loads(journal_bytes)
            except (OSError, ValueError):
                continue
            status_counts[f"{role}:{journal.get('status')}"] += 1
            if journal.get("status") != "complete" or not journal.get("raw_response_base64"):
                continue
            try:
                raw = base64.b64decode(journal["raw_response_base64"])
                response = json.loads(raw)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            response_id = str(response["id"]) if response.get("id") else None
            raw_sha = str(journal.get("raw_response_sha256") or sha256_bytes(raw))
            journal_sha = sha256_bytes(journal_bytes)
            choices = response.get("choices")
            kind = "model_response" if isinstance(choices, list) and len(choices) > 0 else "error_body"
            if response_id:
                dedup_value = f"id:{response_id}"
            elif kind == "model_response":
                dedup_value = f"source:{raw_sha}"
            else:
                dedup_value = f"journal:{journal_sha}"
            key = (role, dedup_value)
            task_candidates = sorted(response_tasks.get(response_id, set())) if response_id else []
            linkage = journal_linkage(path, root, attempt_tasks)
            task = selected_ids.get(response_id) if response_id else None
            if not task and len(task_candidates) == 1:
                task = task_candidates[0]
            if not task:
                task = linkage["task_id_from_attempt"]
            record = {
                "model": model,
                "arm": arm,
                "role": role,
                "kind": kind,
                "response_id_sha256": hashed_id(response_id) if response_id else None,
                "dedup_basis": (
                    "response_id" if response_id else
                    "raw_response_sha256" if kind == "model_response" else
                    "journal_sha256"
                ),
                "source_sha256": journal_sha,
                "raw_response_sha256": raw_sha if not response_id else None,
                "wall_seconds": float(journal.get("wall_seconds") or 0),
                "provider": response.get("provider"),
                "response_model": response.get("model"),
                "finish_reasons": [row.get("finish_reason") for row in choices] if kind == "model_response" else [],
                "usage": minimal_usage(response),
                "in_any_trajectory": bool(response_id and response_id in trajectory_ids),
                "in_selected_trajectory": bool(response_id and response_id in selected_ids),
                "task_id": task,
                "task_linkage": (
                    "selected_response" if response_id in selected_ids else
                    "trajectory_response" if task_candidates else
                    "attempt_directory" if linkage["task_id_from_attempt"] else "unavailable"
                ) if response_id else ("attempt_directory" if linkage["task_id_from_attempt"] else "unavailable"),
                "source_root": root_number,
                **{key: value for key, value in linkage.items() if key != "task_id_from_attempt"},
            }
            comparable = {key: value for key, value in record.items() if key not in {"source_sha256", "source_root", "acquisition_attempt", "call", "transport_attempt", "task_id", "task_linkage"}}
            if key in by_key:
                previous = by_key[key]
                previous_comparable = {name: value for name, value in previous.items() if name not in {"source_sha256", "source_root", "acquisition_attempt", "call", "transport_attempt", "task_id", "task_linkage"}}
                if previous_comparable != comparable:
                    raise ValueError(f"conflicting duplicate {model}/{arm}/{key}")
                continue
            by_key[key] = record
            records.append(record)
    journal_ids = {
        record["response_id_sha256"] for record in records
        if record["kind"] == "model_response" and record["response_id_sha256"]
    }
    missing = sorted(hashed_id(value) for value in trajectory_ids if hashed_id(value) not in journal_ids)
    for response_hash in missing:
        records.append({
            "model": model, "arm": arm, "role": "target", "kind": "trajectory_only",
            "response_id_sha256": response_hash, "dedup_basis": "response_id",
            "source_sha256": None, "raw_response_sha256": None,
            "wall_seconds": None, "provider": None, "response_model": None,
            "finish_reasons": [], "usage": None, "in_any_trajectory": True,
            "in_selected_trajectory": False, "task_id": None,
            "task_linkage": "trajectory_response", "source_root": None,
            "acquisition_attempt": None, "call": None, "transport_attempt": None,
        })
    return records, {
        "journal_statuses": dict(sorted(status_counts.items())),
        "trajectory_response_ids": len(trajectory_ids),
        "trajectory_only_response_ids": len(missing),
        "successful_journal_responses": sum(row["kind"] == "model_response" for row in records),
        "successful_journal_responses_not_in_trajectory": sum(row["kind"] == "model_response" and not row["in_any_trajectory"] for row in records),
        "successful_no_id_responses": sum(row["kind"] == "model_response" and row["response_id_sha256"] is None for row in records),
        "returned_error_bodies": sum(row["kind"] == "error_body" for row in records),
    }


def write_ledger(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="response-ledger.jsonl", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in sorted(records, key=lambda value: (
                value["model"], value["arm"], value["role"], value["kind"],
                value.get("response_id_sha256") or value.get("source_sha256") or "",
            )):
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def evaluation_bindings(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for patch_file in root.glob("**/patch.sha256"):
        declared_hash = patch_file.read_text().strip()
        patch_path = patch_file.with_name("patch.diff")
        patch_hash = sha256_file(patch_path) if patch_path.exists() else declared_hash
        reports = list(patch_file.parent.glob("**/report.json"))
        output[patch_hash] = {
            "resolved": None,
            "report_sha256": None,
            "patch_binding_sha256": sha256_file(patch_file),
            "declared_patch_sha256": declared_hash,
            "declared_patch_hash_matches": declared_hash == patch_hash,
        }
        if reports:
            report_path = reports[0]
            report = load(report_path)
            task_row = report.get("sphinx-doc__sphinx-10466", report)
            output[patch_hash] |= {
                "resolved": bool(task_row["resolved"]),
                "report_sha256": sha256_file(report_path),
            }
    return output


def compact_b300_arm(source: dict[str, Any], binding: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {"status": source.get("status"), "error": source.get("error")}
    for metric in B300_METRICS:
        output[metric] = source.get(metric)
    patch = source.get("patch_sha256")
    output["patch_sha256"] = patch
    output["evaluation"] = binding.get(patch) if patch else None
    return output


def collect_b300_original(root: Path, evaluation: Path) -> list[dict[str, Any]]:
    bindings = evaluation_bindings(evaluation)
    output = []
    for directory in sorted((root / "blocks").iterdir()):
        if not directory.is_dir():
            continue
        path = directory / "retained-result.json" if (directory / "retained-result.json").exists() else directory / "result.json"
        source = load(path)
        output.append({
            "block": int(source["spec"]["block"]),
            "regime": source["spec"]["regime"],
            "source_sha256": sha256_file(path),
            "arms": {arm: compact_b300_arm(source["arms"][arm], bindings) for arm in ("normal", "self", "luna")},
        })
    return output


def collect_b300_retries(root: Path, evaluation: Path) -> list[dict[str, Any]]:
    bindings = evaluation_bindings(evaluation)
    output = []
    for path in sorted(root.glob("retry-block-*.json")):
        source = load(path)
        result = source["result"]
        block, arm = int(source["block"]), source["arm"]
        parent_dir = next((root / "blocks").glob(f"block-{block:02d}-*"))
        parent_path = parent_dir / "retained-result.json" if (parent_dir / "retained-result.json").exists() else parent_dir / "result.json"
        record = compact_b300_arm(result, bindings)
        if record["evaluation"] and record["evaluation"]["resolved"] is None:
            log_path = root / "retry-normal-evaluation.log"
            if not log_path.exists():
                raise ValueError("retry evaluation report absent and evaluator log unavailable")
            log = log_path.read_text(errors="replace")
            if "Instances resolved: 1" not in log or "Instances with errors: 0" not in log:
                raise ValueError("retry evaluator log does not establish success")
            record["evaluation"] |= {
                "resolved": True,
                "evaluation_log_sha256": sha256_file(log_path),
                "binding_limitation": "official report file unavailable; bound to evaluator summary log",
            }
        output.append({
            "block": block,
            "arm": arm,
            "parent_source_sha256": sha256_file(parent_path),
            "retry_source_sha256": sha256_file(path),
            "record": record,
        })
    return output


def b300_selections(original: list[dict[str, Any]], retries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Construct the 23 contemporaneous triples and retrospective 25-row sensitivity."""
    primary = [row for row in original if all(row["arms"][arm]["status"] == "complete" for arm in ("normal", "self", "luna"))]
    augmented = copy.deepcopy(original)
    index = {row["block"]: row for row in augmented}
    for retry in retries:
        parent = index[retry["block"]]
        if parent["source_sha256"] != retry["parent_source_sha256"]:
            raise ValueError("B300 retry parent hash mismatch")
        if parent["arms"][retry["arm"]]["status"] == "complete":
            raise ValueError("B300 retry would replace a successful arm")
        parent["arms"][retry["arm"]] = retry["record"] | {
            "retrospective_replacement": True,
            "retry_source_sha256": retry["retry_source_sha256"],
        }
    if len(primary) != 23 or len(augmented) != 25 or not all(all(row["arms"][arm]["status"] == "complete" for arm in ("normal", "self", "luna")) for row in augmented):
        raise ValueError("unexpected B300 selection sizes")
    return {"primary_23_contemporaneous": primary, "retrospective_25_retry_augmented": augmented}


def parse_provenance(entries: list[str]) -> dict[str, dict[str, Any]]:
    output = {}
    for entry in entries:
        label, raw = entry.split("=", 1)
        path = Path(raw)
        output[label] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return output


def collect(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    roots = {
        "deepseek-v4-flash": args.deepseek_root,
        "minimax-m3": args.minimax_root,
        "nemotron-3-ultra": args.nemotron_root,
        "kimi-k2.6": args.kimi_root,
    }
    source_hashes = parse_provenance(args.provenance)
    pricing_file = load(args.nemotron_pricing)
    pricing = pricing_file["rates_usd_per_million_tokens"]
    models: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []

    normal_official = official_outcomes(args.deepseek_normal_report)
    normal_selected_paths = deepseek_normal_selected_paths(args.deepseek_normal_original, args.deepseek_normal_recovery, args.deepseek_recovery_status)
    normal_ledger, normal_reconciliation = collect_response_ledger(
        "deepseek-v4-flash", "normal",
        [args.deepseek_normal_original, args.deepseek_normal_recovery], normal_selected_paths,
    )
    ledger.extend(normal_ledger)
    models["deepseek-v4-flash"] = {"arms": {"normal": {
        "official": normal_official,
        "selected": selected_metrics(normal_selected_paths),
        "prediction_bindings": prediction_bindings(args.deepseek_normal_predictions, normal_official),
        "reconciliation": normal_reconciliation,
        "compaction": {"raw_words": 0, "compact_words": 0},
    }}}
    for label, path in (
        ("deepseek-normal-official-report", args.deepseek_normal_report),
        ("deepseek-normal-selected-predictions", args.deepseek_normal_predictions),
        ("deepseek-normal-recovery-status", args.deepseek_recovery_status),
        ("nemotron-pricing", args.nemotron_pricing),
    ):
        source_hashes[label] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    for model, matrix_root in roots.items():
        prefix = RUN_PREFIXES[model]
        for arm in ARMS:
            if model == "deepseek-v4-flash" and arm == "normal":
                continue
            run = matrix_root / "runs" / (f"{prefix}__{arm}" if prefix else arm)
            result_path = run / "result.json"
            result = load(result_path)
            report_path = run / result["official_report"]
            predictions_path = run / "inference" / "preds.json"
            selected = selected_paths(run)
            arm_ledger, reconciliation = collect_response_ledger(model, arm, [run], selected)
            ledger.extend(arm_ledger)
            official = official_outcomes(report_path)
            models.setdefault(model, {"arms": {}})["arms"][arm] = {
                "official": official,
                "selected": selected_metrics(selected),
                "prediction_bindings": prediction_bindings(predictions_path, official),
                "reconciliation": reconciliation,
                "compaction": {
                    "raw_words": int(result.get("compaction_raw_words") or 0),
                    "compact_words": int(result.get("compaction_state_words") or 0),
                },
                "source_counters": {
                    "acquisition_target_calls": int(result["acquisition_target_calls"]),
                    "selected_target_calls": int(result["selected_target_calls"]),
                    "reported_compaction_cost_usd": float(result.get("compaction_cost_usd") or 0),
                },
            }
            for label, path in (("result", result_path), ("progress", run / "progress.json"), ("official-report", report_path), ("selected-predictions", predictions_path)):
                source_hashes[f"{model}__{arm}__{label}"] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    b300 = {
        "task_id": "sphinx-doc__sphinx-10466",
        "original_10": collect_b300_original(args.b300_ten_root, args.b300_ten_evaluation),
        "overload_original_25": collect_b300_original(args.b300_overload_root, args.b300_overload_evaluation),
        "overload_retries": collect_b300_retries(args.b300_overload_root, args.b300_overload_evaluation),
    }
    selections = b300_selections(b300["overload_original_25"], b300["overload_retries"])
    b300["selection_counts"] = {key: len(value) for key, value in selections.items()}
    hosted = load(args.hosted_analysis)
    hosted_rows = [{key: row[key] for key in (
        "source_key", "replicate", "clean_cost_usd", "rewritten_cost_usd", "rewrite_cost_usd",
        "clean_wall_seconds", "rewritten_wall_seconds", "rewrite_wall_seconds",
    )} for row in hosted["rows"]]

    write_ledger(args.ledger_output, ledger)
    inputs = {
        "schema_version": 2,
        "contracts": {
            "primary_response": "status=complete JSON body with one or more choices; all finish reasons including length",
            "error_body": "status=complete returned JSON body without choices; excluded from primary response spend/time and reported separately",
            "deduplication": "response ID when present; otherwise raw-response SHA-256; conflicting ID duplicates fail",
            "request_duration": "successful model-response client duration; excludes tools/orchestration",
            "bootstrap": "paired task resampling conditional on fixed 50-task cohort and captured run",
        },
        "deployment": {"models": models, "pricing": {
            "nemotron_target_common_tariff": "DeepInfra",
            "rates_usd_per_million_tokens": pricing,
            "pricing_file_sha256": sha256_file(args.nemotron_pricing),
        }},
        "hosted_matched_forks": {"rows": hosted_rows},
        "b300_latency": b300,
        "ledger": {
            "relative_path": args.ledger_output.name,
            "sha256": sha256_file(args.ledger_output),
            "records": len(ledger),
            "format": "gzip JSON Lines",
        },
        "provenance": {"sources": dict(sorted(source_hashes.items())), "note": "Logical labels replace machine-specific paths."},
    }
    inputs["numeric_payload_sha256"] = canonical_sha256({key: inputs[key] for key in ("deployment", "hosted_matched_forks", "b300_latency", "ledger")})
    return inputs, ledger


def usage_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"responses": len(records), "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "reported_cost_usd": 0.0, "wall_seconds": 0.0}
    for record in records:
        usage = record["usage"]
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
            output[key] += int(usage[key])
        output["reported_cost_usd"] += float(usage["reported_cost_usd"])
        output["wall_seconds"] += float(record["wall_seconds"])
    return output


def tariff_cost(usage: dict[str, Any], rates: dict[str, float]) -> float:
    prompt, cached, output = usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"]
    return ((prompt - cached) * rates["uncached_input"] + cached * rates["cache_read_input"] + output * rates["output"]) / 1_000_000


def paired_outcomes(label: str, normal: dict, treatment: dict) -> dict[str, Any]:
    n, t = normal["official"]["tasks"], treatment["official"]["tasks"]
    if set(n) != set(t) or len(n) != 50:
        raise ValueError(f"{label}: unmatched official tasks")
    tasks = sorted(n)
    differences = [int(t[key]) - int(n[key]) for key in tasks]
    gains = [key for key in tasks if t[key] and not n[key]]
    losses = [key for key in tasks if n[key] and not t[key]]
    seed = int(hashlib.sha256(f"20260910:{label}".encode()).hexdigest()[:8], 16)
    return {
        "normal_resolved": sum(n.values()), "treatment_resolved": sum(t.values()),
        "accuracy_change_percentage_points": 100 * statistics.fmean(differences),
        "paired_task_bootstrap_95_interval_percentage_points": [100 * value for value in bootstrap_mean_interval(differences, seed)],
        "discordance": {"compact_only": len(gains), "normal_only": len(losses), "compact_only_ids": gains, "normal_only_ids": losses},
        "conditioning": "fixed assigned task cohort and one captured run per arm; no seed/provider-population inference",
        "noninferiority_claim": False,
    }


def analyze_deployment(data: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    rates = data["deployment"]["pricing"]["rates_usd_per_million_tokens"]["DeepInfra"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        grouped[(row["model"], row["arm"])].append(row)
    output = {"models": {}, "accuracy_change_range_percentage_points": None}
    changes = []
    for model, model_data in data["deployment"]["models"].items():
        arms_out = {}
        for arm in ARMS:
            source = model_data["arms"][arm]
            records = grouped[(model, arm)]
            target = [row for row in records if row["role"] == "target" and row["kind"] == "model_response"]
            compressor = [row for row in records if row["role"] == "compressor" and row["kind"] == "model_response"]
            errors = [row for row in records if row["kind"] == "error_body"]
            trajectory_only = [row for row in records if row["kind"] == "trajectory_only"]
            target_totals, compressor_totals = usage_totals(target), usage_totals(compressor)
            target_cost = tariff_cost(target_totals, rates) if model == "nemotron-3-ultra" else target_totals["reported_cost_usd"]
            spend_provenance = "reconstructed_common_deepinfra_tariff" if model == "nemotron-3-ultra" else "captured_provider_reported_usage"
            selected = source["selected"]["totals"]
            arms_out[arm] = {
                "official_resolved": sum(source["official"]["tasks"].values()),
                "official_task_ids": sorted(source["official"]["tasks"]),
                "official_resolved_ids": source["official"]["resolved_ids"],
                "selected_trajectory": {key: selected[key] for key in SELECTED_FIELDS} | {"tasks": selected["tasks"], "exit_statuses": selected["exit_statuses"]},
                "prediction_bindings": source["prediction_bindings"],
                "successful_response_ledger": {"target": target_totals, "compressor": compressor_totals},
                "all_acquisition_spend": {
                    "target_usd": target_cost, "compressor_usd": compressor_totals["reported_cost_usd"],
                    "total_usd": target_cost + compressor_totals["reported_cost_usd"], "provenance": spend_provenance,
                },
                "successful_request_duration": {
                    "target_seconds": target_totals["wall_seconds"], "compressor_seconds": compressor_totals["wall_seconds"],
                    "inclusive_seconds": target_totals["wall_seconds"] + compressor_totals["wall_seconds"],
                    "scope": "completed JSON model responses with choices; excludes tools/orchestration",
                },
                "returned_error_bodies": {
                    role: {
                        "records": sum(row["role"] == role for row in errors),
                        "wall_seconds": sum(float(row["wall_seconds"]) for row in errors if row["role"] == role),
                        "reported_cost_usd": sum(float((row["usage"] or {}).get("reported_cost_usd") or 0) for row in errors if row["role"] == role),
                    }
                    for role in ("target", "compressor")
                } | {
                    "all": {
                        "records": len(errors),
                        "wall_seconds": sum(float(row["wall_seconds"]) for row in errors),
                        "reported_cost_usd": sum(float((row["usage"] or {}).get("reported_cost_usd") or 0) for row in errors),
                    }
                },
                "journal_trajectory_reconciliation": {
                    "target_success_not_in_any_trajectory": sum(row["role"] == "target" and not row["in_any_trajectory"] for row in target),
                    "target_success_in_any_trajectory": sum(row["role"] == "target" and row["in_any_trajectory"] for row in target),
                    "target_trajectory_only": sum(row["role"] == "target" for row in trajectory_only),
                    "compressor_success_not_in_trajectory": len(compressor),
                    "successful_no_id_records": sum(row["response_id_sha256"] is None for row in target + compressor),
                    "target_unlinked_task_records": sum(row["task_id"] is None for row in target),
                    "compressor_unlinked_task_records": sum(row["task_id"] is None for row in compressor),
                },
                "compaction_words": source["compaction"],
                "source_counters": source.get("source_counters"),
            }
        contrasts = {}
        for arm in ("luna-compact", "self-compact"):
            paired = paired_outcomes(f"{model}:{arm}", model_data["arms"]["normal"], model_data["arms"][arm])
            changes.append(paired["accuracy_change_percentage_points"])
            normal, treatment = arms_out["normal"], arms_out[arm]
            paired["all_acquisition_spend_change_percent"] = percent_change(treatment["all_acquisition_spend"]["total_usd"], normal["all_acquisition_spend"]["total_usd"])
            paired["target_request_duration_change_percent"] = percent_change(treatment["successful_request_duration"]["target_seconds"], normal["successful_request_duration"]["target_seconds"])
            paired["inclusive_request_duration_change_percent"] = percent_change(treatment["successful_request_duration"]["inclusive_seconds"], normal["successful_request_duration"]["inclusive_seconds"])
            paired["selected_trajectory_changes_percent"] = {key: percent_change(treatment["selected_trajectory"][key], normal["selected_trajectory"][key]) for key in SELECTED_FIELDS if normal["selected_trajectory"][key]}
            contrasts[f"{arm}_vs_normal"] = paired
        output["models"][model] = {"arms": arms_out, "contrasts": contrasts}
    output["accuracy_change_range_percentage_points"] = [min(changes), max(changes)]
    kimi = output["models"]["kimi-k2.6"]["arms"]
    source_counts = {arm: kimi[arm]["source_counters"]["selected_target_calls"] for arm in ARMS}
    trajectory_counts = {arm: kimi[arm]["selected_trajectory"]["assistant_responses"] for arm in ARMS}
    output["kimi_response_counter_reconciliation"] = {
        "result_level_selected_attempt_response_counters": source_counts,
        "serialized_selected_trajectory_assistant_responses": trajectory_counts,
        "luna_result_counter_change_percent": percent_change(source_counts["luna-compact"], source_counts["normal"]),
        "luna_trajectory_response_change_percent": percent_change(trajectory_counts["luna-compact"], trajectory_counts["normal"]),
        "luna_tool_call_change_percent": percent_change(kimi["luna-compact"]["selected_trajectory"]["tool_calls"], kimi["normal"]["selected_trajectory"]["tool_calls"]),
    }
    return output


def hosted_amortization(data: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["hosted_matched_forks"]["rows"]:
        grouped[row["source_key"]].append(row)
    output = {"sources": len(grouped), "replicates_per_source": sorted({len(rows) for rows in grouped.values()}), "metrics": {}}
    for metric, suffix in (("wall", "seconds"), ("cost", "usd")):
        detail = []
        for source, rows in sorted(grouped.items()):
            overhead = float(rows[0][f"rewrite_{metric}_{suffix}"])
            saving = statistics.fmean(float(row[f"clean_{metric}_{suffix}"]) - float(row[f"rewritten_{metric}_{suffix}"]) for row in rows)
            detail.append({"source_key": source, f"rewrite_overhead_{suffix}": overhead, f"mean_marginal_saving_{suffix}": saving, "continuous_break_even_continuations": overhead / saving if saving > 0 else None})
        finite = [row["continuous_break_even_continuations"] for row in detail if row["continuous_break_even_continuations"] is not None]
        output["metrics"][metric] = {
            "sources_with_positive_marginal_saving": len(finite),
            "sources_without_positive_marginal_saving": len(detail) - len(finite),
            "median_continuous_break_even_continuations_among_positive": statistics.median(finite),
            "range_continuous_break_even_continuations_among_positive": [min(finite), max(finite)],
            "sources_breaking_even_by_three_continuations": sum(value <= 3 for value in finite),
            "source_detail": detail,
        }
    output["claim_boundary"] = "Time and money break-even differ and are source-conditional; no universal 2-3-continuation claim."
    return output


def latency_summary(rows: list[dict[str, Any]], selection: str) -> dict[str, Any]:
    output = {"selection": selection, "rows": len(rows), "regimes": {}}
    for regime in sorted({row["regime"] for row in rows}):
        group = [row for row in rows if row["regime"] == regime]
        arm_output = {}
        for arm in ("self", "luna"):
            changes = [percent_change(float(row["arms"][arm]["complete_task_seconds"]), float(row["arms"]["normal"]["complete_task_seconds"])) for row in group]
            arm_output[arm] = {
                "normal_median_complete_task_seconds": statistics.median(float(row["arms"]["normal"]["complete_task_seconds"]) for row in group),
                "treatment_median_complete_task_seconds": statistics.median(float(row["arms"][arm]["complete_task_seconds"]) for row in group),
                "median_paired_complete_task_change_percent": statistics.median(changes),
                "faster_equal_slower": [sum(value < 0 for value in changes), sum(value == 0 for value in changes), sum(value > 0 for value in changes)],
            }
        output["regimes"][regime] = {"pairs": len(group), "arms": arm_output}
    return output


def analyze(data: dict[str, Any], ledger: list[dict[str, Any]], ledger_path: Path | None = None) -> dict[str, Any]:
    expected = canonical_sha256({key: data[key] for key in ("deployment", "hosted_matched_forks", "b300_latency", "ledger")})
    if expected != data["numeric_payload_sha256"]:
        raise ValueError("compact input digest mismatch")
    if ledger_path and sha256_file(ledger_path) != data["ledger"]["sha256"]:
        raise ValueError("response ledger digest mismatch")
    selections = b300_selections(data["b300_latency"]["overload_original_25"], data["b300_latency"]["overload_retries"])
    return {
        "schema_version": 2,
        "deployment": analyze_deployment(data, ledger),
        "hosted_matched_amortization": hosted_amortization(data),
        "b300_latency": {
            "task_id": data["b300_latency"]["task_id"],
            "original_10": latency_summary(data["b300_latency"]["original_10"], "ten contemporaneous three-arm blocks"),
            "overload_primary_23": latency_summary(selections["primary_23_contemporaneous"], "23 complete contemporaneous three-arm blocks; failed blocks 2 and 17 excluded"),
            "overload_retrospective_25": latency_summary(selections["retrospective_25_retry_augmented"], "25 rows after noncontemporaneous single-arm replacements for Normal block 2 and Self block 17"),
            "original_failed_arm_records": [
                {"block": row["block"], "source_sha256": row["source_sha256"], "failed_arms": {arm: values for arm, values in row["arms"].items() if values["status"] != "complete"}}
                for row in data["b300_latency"]["overload_original_25"] if any(values["status"] != "complete" for values in row["arms"].values())
            ],
            "retry_bindings": data["b300_latency"]["overload_retries"],
            "scope": "marginal full postfork task time after compact history is available; one pinned task; excludes rewrite and setup",
        },
        "statistical_scope": "Paired task intervals are conditional on the fixed cohort and captured runs; no noninferiority or seed/provider-population inference.",
        "input_numeric_payload_sha256": data["numeric_payload_sha256"],
        "response_ledger_sha256": data["ledger"]["sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    for name in ("deepseek", "minimax", "nemotron", "kimi"):
        collect_parser.add_argument(f"--{name}-root", type=Path, required=True)
    for name in ("deepseek-normal-original", "deepseek-normal-recovery", "deepseek-recovery-status", "deepseek-normal-report", "deepseek-normal-predictions", "nemotron-pricing", "hosted-analysis", "b300-ten-root", "b300-ten-evaluation", "b300-overload-root", "b300-overload-evaluation"):
        collect_parser.add_argument(f"--{name}", type=Path, required=True)
    collect_parser.add_argument("--provenance", action="append", default=[], metavar="LABEL=FILE")
    collect_parser.add_argument("--ledger-output", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--input", type=Path, required=True)
    analyze_parser.add_argument("--ledger", type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        inputs, _ = collect(args)
        dump(args.output, inputs)
    else:
        ledger_path = args.ledger or (args.input.parent / load(args.input)["ledger"]["relative_path"])
        data = load(args.input)
        dump(args.output, analyze(data, read_ledger(ledger_path), ledger_path))


if __name__ == "__main__":
    main()
