#!/usr/bin/env python3
"""Durable two-model extension of the frozen 12-task paired-fork screen."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import (  # noqa: E402
    load_dotenv,
    load_tasks,
    reasoning_text,
    remaining_openrouter_credits,
    rewrite_fidelity,
    strip_fence,
    strict_json_loads,
)
from short_reasoning.durable import (  # noqa: E402
    DurableChat,
    canonical_hash,
    write_json_atomic,
)
from short_reasoning.frozen import (  # noqa: E402
    acquire_frozen_source,
    continue_frozen_source,
    frozen_source_hash,
    reasoning_only_fork_audit,
)

DEFAULT_CONFIG = ROOT / "configs" / "four_model_replication_v1.json"
CODE_PATHS = (
    Path("scripts/run_four_model_replication.py"),
    Path("scripts/analyze_frozen_run.py"),
    Path("src/short_reasoning/__init__.py"),
    Path("src/short_reasoning/durable.py"),
    Path("src/short_reasoning/frozen.py"),
)
REVIEW_KEYS = {
    "verdict",
    "critical_or_material_issue",
    "future_result_leakage",
    "requested_action_preserved",
    "explanation",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_provider(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def load_config(path: Path) -> dict[str, Any]:
    config = strict_json_loads(path.read_bytes())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unknown replication config schema")
    if config.get("status") != "frozen_before_paid_behavioral_calls":
        raise ValueError("replication config is not frozen")
    if config.get("actual_human_review_performed") is not False:
        raise ValueError("human-review status must remain explicit")
    if float(config.get("minimum_balance_usd", -1)) != 2.0:
        raise ValueError("the approved OpenRouter floor must remain $2")
    if int(config.get("replicates", 0)) != 3:
        raise ValueError("the exact extension requires three replicates")
    model_names = set(config.get("models", {}))
    if not model_names or model_names != set(config.get("eligibility", {})):
        raise ValueError("models and delivery-eligibility bindings must match")
    for name, model in config["models"].items():
        if model["provider"].get("allow_fallbacks") is not False:
            raise ValueError(f"{name}: fallback must remain disabled")
        if len(model["provider"].get("only", [])) != 1:
            raise ValueError(f"{name}: exact provider is missing")
        if not model.get("expected_provider"):
            raise ValueError(f"{name}: expected provider is missing")
    tasks_path = ROOT / config["tasks_file"]
    if not tasks_path.is_file() or len(load_tasks(tasks_path)) != 12:
        raise ValueError("the controlled 12-task cohort changed")
    predecessor = config.get("diagnostic_predecessor")
    if predecessor:
        source = ROOT / predecessor["source"]
        if file_sha256(source) != predecessor["source_sha256"]:
            raise ValueError("diagnostic predecessor hash changed")
    for name, binding in config["eligibility"].items():
        source = ROOT / binding["source"]
        if file_sha256(source) != binding["source_sha256"]:
            raise ValueError(f"{name}: eligibility analysis hash changed")
        analysis = strict_json_loads(source.read_bytes())
        promoted = {
            item["route_id"]
            for item in analysis.get("promoted_configurations", [])
            if item.get("promotion_gate_passed")
        }
        if binding["route_id"] not in promoted:
            raise ValueError(f"{name}: route is not delivery-eligible")
    return config


def executable_paths(config_path: Path, config: dict[str, Any]) -> tuple[Path, ...]:
    relative_config = config_path.resolve().relative_to(ROOT)
    predecessor_paths = (
        (Path(config["diagnostic_predecessor"]["source"]),)
        if config.get("diagnostic_predecessor") else ()
    )
    return (
        relative_config,
        Path(config["tasks_file"]),
        *(Path(binding["source"]) for binding in config["eligibility"].values()),
        *predecessor_paths,
        *CODE_PATHS,
    )


def ensure_inputs_at_head(paths: tuple[Path, ...]) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for relative in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *map(str, paths)], cwd=ROOT
    ).returncode:
        raise RuntimeError("commit replication executable inputs before initialization")
    return commit


def source_key(model_name: str, task_id: str) -> str:
    return f"{model_name}__{task_id}"


def planned_jobs(config: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    source_jobs = []
    continuation_jobs = []
    for model_index, model_name in enumerate(config["models"]):
        for task_index, task in enumerate(tasks):
            key = source_key(model_name, task["id"])
            source_seed = config["source_seed"] + model_index * 10000 + task_index * 100
            source_jobs.append(
                {
                    "job_id": key,
                    "model_name": model_name,
                    "task_id": task["id"],
                    "source_seed": source_seed,
                }
            )
            for replicate in range(1, config["replicates"] + 1):
                continuation_seed = (
                    config["continuation_seed"]
                    + model_index * 100000
                    + task_index * 1000
                    + (replicate - 1) * 10
                )
                continuation_jobs.append(
                    {
                        "job_id": f"{key}__rep-{replicate:02d}",
                        "source_job_id": key,
                        "model_name": model_name,
                        "task_id": task["id"],
                        "replicate": replicate,
                        "continuation_seed": continuation_seed,
                    }
                )
    return {"sources": source_jobs, "continuations": continuation_jobs}


def initialize(run_dir: Path, config_path: Path) -> None:
    if run_dir.exists():
        raise RuntimeError(f"refusing existing run directory: {run_dir}")
    config = load_config(config_path)
    tasks = load_tasks(ROOT / config["tasks_file"])
    paths = executable_paths(config_path, config)
    commit = ensure_inputs_at_head(paths)
    jobs = planned_jobs(config, tasks)
    load_dotenv(ROOT / ".env")
    initial_balance = remaining_openrouter_credits()
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "config.snapshot.json", config)
    write_json_atomic(run_dir / "tasks.snapshot.json", tasks)
    write_json_atomic(run_dir / "jobs.json", jobs)
    write_json_atomic(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "initialized_before_paid_behavioral_calls",
            "git_commit": commit,
            "config_sha256": canonical_hash(config),
            "tasks_sha256": canonical_hash(tasks),
            "jobs_sha256": canonical_hash(jobs),
            "code_manifest": {str(path): file_sha256(ROOT / path) for path in paths},
            "initial_openrouter_balance_usd": initial_balance,
        },
    )


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if (run_dir / "SHA256SUMS").exists():
        raise RuntimeError("replication run is frozen")
    run = strict_json_loads((run_dir / "run.json").read_bytes())
    config = strict_json_loads((run_dir / "config.snapshot.json").read_bytes())
    tasks = strict_json_loads((run_dir / "tasks.snapshot.json").read_bytes())
    jobs = strict_json_loads((run_dir / "jobs.json").read_bytes())
    if canonical_hash(config) != run["config_sha256"]:
        raise RuntimeError("replication config snapshot changed")
    if canonical_hash(tasks) != run["tasks_sha256"]:
        raise RuntimeError("replication task snapshot changed")
    if canonical_hash(jobs) != run["jobs_sha256"]:
        raise RuntimeError("replication job plan changed")
    for relative, expected in run["code_manifest"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"replication executable input changed: {relative}")
    return config, tasks, jobs, run


def validate_route(
    chat: DurableChat, expected_by_model: dict[str, str]
) -> Callable[[dict[str, Any]], tuple[dict[str, Any], float]]:
    def post(payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        response, wall = chat(payload)
        expected_provider = expected_by_model[payload["model"]]
        returned_model = response.get("model")
        model_match = isinstance(returned_model, str) and (
            returned_model == payload["model"]
            or returned_model.startswith(payload["model"] + "-")
        )
        provider_match = normalized_provider(response.get("provider")) == normalized_provider(
            expected_provider
        )
        if not model_match or not provider_match:
            raise RuntimeError(
                f"route mismatch: model={returned_model!r}, provider={response.get('provider')!r}"
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("response does not contain exactly one choice")
        return response, wall

    return post


def configure_budget(config: dict[str, Any]) -> None:
    load_dotenv(ROOT / ".env")
    os.environ["OPENROUTER_MIN_BALANCE_USD"] = str(config["minimum_balance_usd"])
    os.environ["OPENROUTER_MAX_CALL_COST_USD"] = str(
        config["maximum_call_reserve_usd"]
    )


def error_record(job: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **job,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_parallel(
    jobs: list[dict[str, Any]], workers: int, execute: Callable[[dict[str, Any]], None]
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def acquire(run_dir: Path) -> None:
    config, tasks, jobs, _run = load_run(run_dir)
    configure_budget(config)
    tasks_by_id = {task["id"]: task for task in tasks}
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(exist_ok=True)
    expected = {
        profile["model"]: profile["expected_provider"]
        for profile in config["models"].values()
    }
    expected[config["rewriter"]["model"]] = config["rewriter"]["expected_provider"]

    def execute(job: dict[str, Any]) -> None:
        path = sources_dir / f"{job['job_id']}.json"
        if path.exists():
            return
        print(f"acquire start {job['job_id']}", flush=True)
        chat = DurableChat(
            run_dir / "journals" / "acquire" / job["job_id"],
            max_returned_retries=config["max_returned_retries"],
        )
        try:
            result = acquire_frozen_source(
                copy.deepcopy(tasks_by_id[job["task_id"]]),
                job["model_name"],
                copy.deepcopy(config["models"][job["model_name"]]),
                copy.deepcopy(config["rewriter"]),
                job["source_seed"],
                post_chat_fn=validate_route(chat, expected),
            )
        except Exception as exc:
            result = error_record(job, exc)
        write_json_atomic(path, result)
        print(f"acquire finish {job['job_id']} status={result['status']}", flush=True)

    pending = [
        job for job in jobs["sources"] if not (sources_dir / f"{job['job_id']}.json").exists()
    ]
    run_parallel(pending, config["workers"], execute)
    records = [
        strict_json_loads((sources_dir / f"{job['job_id']}.json").read_bytes())
        for job in jobs["sources"]
    ]
    write_json_atomic(
        run_dir / "acquisition-summary.json",
        {
            "requested_sources": len(jobs["sources"]),
            "status_counts": count_statuses(records),
            "remaining_openrouter_balance_usd": remaining_openrouter_credits(),
        },
    )


def review_payload(config: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    message = shared["response"]["choices"][0]["message"]
    raw = reasoning_text(message)
    candidate = shared["rewrite"]["rewritten"]
    requested_tool = message["tool_calls"][0]["function"]
    rubric = {
        "raw_reasoning": raw,
        "candidate_state": candidate,
        "requested_tool": requested_tool,
    }
    return {
        "model": config["reviewer"]["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Audit one compact reasoning-state rewrite before its pending tool result exists. "
                    "Pass only if it preserves every decision-relevant fact, number, uncertainty, "
                    "conclusion, rejected alternative, evidence-status distinction, and intended next "
                    "action from the raw reasoning; adds no unsupported fact; and neither states nor "
                    "implies that the requested tool result has occurred. The tool call is separately "
                    "preserved and is supplied only to check action consistency. Return only one JSON "
                    "object with exactly these keys: verdict ('pass' or 'fail'), "
                    "critical_or_material_issue (boolean), future_result_leakage (boolean), "
                    "requested_action_preserved (boolean), explanation (short string)."
                ),
            },
            {"role": "user", "content": json.dumps(rubric, ensure_ascii=False)},
        ],
        "reasoning": {
            "effort": config["reviewer"]["reasoning_effort"],
            "exclude": True,
        },
        "include_reasoning": False,
        "provider": copy.deepcopy(config["reviewer"]["provider"]),
    }


def parse_review(response: dict[str, Any]) -> dict[str, Any]:
    content = response["choices"][0]["message"].get("content") or ""
    value = strict_json_loads(strip_fence(content))
    if not isinstance(value, dict) or set(value) != REVIEW_KEYS:
        raise ValueError("semantic review has the wrong schema")
    if value["verdict"] not in {"pass", "fail"}:
        raise ValueError("semantic review verdict is invalid")
    for key in (
        "critical_or_material_issue",
        "future_result_leakage",
        "requested_action_preserved",
    ):
        if not isinstance(value[key], bool):
            raise ValueError(f"semantic review {key} must be boolean")
    if not isinstance(value["explanation"], str):
        raise ValueError("semantic review explanation must be a string")
    value["passed"] = (
        value["verdict"] == "pass"
        and not value["critical_or_material_issue"]
        and not value["future_result_leakage"]
        and value["requested_action_preserved"]
    )
    return value


def audit_source(
    run_dir: Path,
    config: dict[str, Any],
    job: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if source.get("status") != "selected":
        return {**job, "status": "source_not_selected", "eligible": False}
    clean = source["histories"]["clean"]
    rewritten = source["histories"]["rewritten"]
    recomputed = frozen_source_hash(
        source["task_id"], source["model_name"], source["model"], clean, rewritten
    )
    fork = reasoning_only_fork_audit(clean, rewritten)
    expected = {config["reviewer"]["model"]: config["reviewer"]["expected_provider"]}
    reviews = []
    mechanical_pass = True
    raw_words = 0
    rewritten_words = 0
    for turn_index, shared in enumerate(source["shared"], start=1):
        message = shared["response"]["choices"][0]["message"]
        raw = reasoning_text(message)
        rewrite = shared["rewrite"]
        candidate = rewrite["rewritten"]
        raw_words += len(raw.split())
        rewritten_words += len(candidate.split())
        requested_tool_matches = rewrite["requested_tool"] == message["tool_calls"][0]["function"]
        fidelity = rewrite_fidelity(raw, candidate)
        state_mechanical_pass = (
            requested_tool_matches
            and rewrite["fidelity"]["all_candidate_ids_preserved"]
            and rewrite["fidelity"]["all_hard_thresholds_preserved"]
        )
        mechanical_pass = mechanical_pass and state_mechanical_pass
        review: dict[str, Any] | None = None
        if not rewrite.get("identity"):
            chat = DurableChat(
                run_dir / "journals" / "audit" / job["job_id"] / f"turn-{turn_index:02d}",
                max_returned_retries=config["max_returned_retries"],
            )
            try:
                response, wall = validate_route(chat, expected)(
                    review_payload(config, shared)
                )
                review = parse_review(response)
                review["wall_seconds"] = wall
                review["cost"] = float(response.get("usage", {}).get("cost") or 0)
            except Exception as exc:
                review = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        reviews.append(
            {
                "turn": turn_index,
                "identity": bool(rewrite.get("identity")),
                "mechanical_pass": state_mechanical_pass,
                "requested_tool_matches": requested_tool_matches,
                "conservative_fidelity": fidelity,
                "review": review,
            }
        )
    semantic_pass = all(
        item["identity"] or bool(item["review"] and item["review"].get("passed"))
        for item in reviews
    )
    eligible = (
        len(source["shared"]) == 3
        and recomputed == source["source_sha256"]
        and fork["passed"]
        and mechanical_pass
        and rewritten_words < raw_words
        and semantic_pass
    )
    return {
        **job,
        "status": "audited",
        "source_file_sha256": file_sha256(
            run_dir / "sources" / f"{job['job_id']}.json"
        ),
        "source_sha256": source["source_sha256"],
        "source_hash_recomputed": recomputed == source["source_sha256"],
        "fork_audit": fork,
        "raw_words": raw_words,
        "rewritten_words": rewritten_words,
        "aggregate_shorter": rewritten_words < raw_words,
        "mechanical_pass": mechanical_pass,
        "semantic_pass": semantic_pass,
        "actual_human_review_performed": False,
        "reviews": reviews,
        "eligible": eligible,
    }


def audit(run_dir: Path) -> None:
    config, _tasks, jobs, _run = load_run(run_dir)
    configure_budget(config)
    sources_dir = run_dir / "sources"
    if any(not (sources_dir / f"{job['job_id']}.json").exists() for job in jobs["sources"]):
        raise RuntimeError("all planned source records must exist before audit")
    audits_dir = run_dir / "audits"
    audits_dir.mkdir(exist_ok=True)

    def execute(job: dict[str, Any]) -> None:
        path = audits_dir / f"{job['job_id']}.json"
        if path.exists():
            return
        source = strict_json_loads(
            (sources_dir / f"{job['job_id']}.json").read_bytes()
        )
        result = audit_source(run_dir, config, job, source)
        write_json_atomic(path, result)
        print(f"audit {job['job_id']} eligible={result['eligible']}", flush=True)

    pending = [
        job for job in jobs["sources"] if not (audits_dir / f"{job['job_id']}.json").exists()
    ]
    run_parallel(pending, config["workers"], execute)
    records = [
        strict_json_loads((audits_dir / f"{job['job_id']}.json").read_bytes())
        for job in jobs["sources"]
    ]
    eligible = [
        {
            "job_id": item["job_id"],
            "model_name": item["model_name"],
            "task_id": item["task_id"],
            "source_sha256": item.get("source_sha256"),
            "source_file_sha256": item.get("source_file_sha256"),
        }
        for item in records
        if item.get("eligible")
    ]
    write_json_atomic(
        run_dir / "source-audit.json",
        {
            "created_at": utc_now(),
            "requested_sources": len(records),
            "eligible_sources": len(eligible),
            "actual_human_review_performed": False,
            "status_counts": count_statuses(records),
            "raw_words": sum(int(item.get("raw_words") or 0) for item in records),
            "rewritten_words": sum(int(item.get("rewritten_words") or 0) for item in records),
            "model_review_cost_usd": sum(
                float(review["review"].get("cost") or 0)
                for item in records
                for review in item.get("reviews", [])
                if isinstance(review.get("review"), dict)
            ),
            "records": records,
        },
    )
    write_json_atomic(run_dir / "eligible-sources.json", eligible)
    write_json_atomic(
        run_dir / "audit-balance.json",
        {
            "checked_at": utc_now(),
            "remaining_usd": remaining_openrouter_credits(),
            "floor_usd": config["minimum_balance_usd"],
        },
    )


def continue_run(run_dir: Path) -> None:
    config, tasks, jobs, _run = load_run(run_dir)
    configure_budget(config)
    eligible_path = run_dir / "eligible-sources.json"
    if not eligible_path.exists():
        raise RuntimeError("source audit must finish before continuations")
    eligible = {
        item["job_id"]: item for item in strict_json_loads(eligible_path.read_bytes())
    }
    tasks_by_id = {task["id"]: task for task in tasks}
    sources_dir = run_dir / "sources"
    continuations_dir = run_dir / "continuations"
    continuations_dir.mkdir(exist_ok=True)

    def execute(job: dict[str, Any]) -> None:
        path = continuations_dir / f"{job['job_id']}.json"
        if path.exists():
            return
        binding = eligible[job["source_job_id"]]
        source_path = sources_dir / f"{job['source_job_id']}.json"
        if file_sha256(source_path) != binding["source_file_sha256"]:
            raise RuntimeError(f"eligible source file changed: {job['source_job_id']}")
        source = strict_json_loads(source_path.read_bytes())
        if source["source_sha256"] != binding["source_sha256"]:
            raise RuntimeError(f"eligible source hash changed: {job['source_job_id']}")
        print(f"continue start {job['job_id']}", flush=True)
        chat = DurableChat(
            run_dir / "journals" / "continue" / job["job_id"],
            max_returned_retries=config["max_returned_retries"],
        )
        expected = {
            source["model"]["model"]: source["model"]["expected_provider"]
        }
        try:
            result = continue_frozen_source(
                copy.deepcopy(tasks_by_id[job["task_id"]]),
                copy.deepcopy(source),
                job["continuation_seed"],
                job["replicate"],
                post_chat_fn=validate_route(chat, expected),
            )
        except Exception as exc:
            result = error_record(job, exc)
            result["source_sha256"] = source["source_sha256"]
        write_json_atomic(path, result)
        print(f"continue finish {job['job_id']} status={result['status']}", flush=True)

    selected_jobs = [
        job for job in jobs["continuations"] if job["source_job_id"] in eligible
    ]
    pending = [
        job
        for job in selected_jobs
        if not (continuations_dir / f"{job['job_id']}.json").exists()
    ]
    run_parallel(pending, config["workers"], execute)
    records = [
        strict_json_loads((continuations_dir / f"{job['job_id']}.json").read_bytes())
        for job in selected_jobs
    ]
    write_json_atomic(
        run_dir / "continuation-summary.json",
        {
            "eligible_sources": len(eligible),
            "requested_cases": len(selected_jobs),
            "status_counts": count_statuses(records),
            "remaining_openrouter_balance_usd": remaining_openrouter_credits(),
        },
    )


def count_statuses(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "acquire", "audit", "continue"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.command == "init":
        initialize(run_dir, args.config.resolve())
    elif args.command == "acquire":
        acquire(run_dir)
    elif args.command == "audit":
        audit(run_dir)
    else:
        continue_run(run_dir)


if __name__ == "__main__":
    main()
