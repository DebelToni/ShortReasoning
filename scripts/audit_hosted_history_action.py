#!/usr/bin/env python3
"""Audit whether hosted routes use an exact tool action stored only in prior reasoning."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import (  # noqa: E402
    load_dotenv,
    parse_arguments,
    post_chat,
    remaining_openrouter_credits,
    response_metrics,
)

TOOL_NAMES = ("state_probe_amber", "state_probe_cobalt", "state_probe_jade", "state_probe_silver")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", default="all")
    parser.add_argument("--reasoning-trials", type=int, default=5)
    parser.add_argument("--visible-trials", type=int, default=2)
    parser.add_argument("--negative-trials", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-balance", type=float, default=2.0)
    return parser.parse_args()


def selected_tool(model_name: str, condition: str, trial: int) -> str:
    digest = hashlib.sha256(f"{model_name}|{condition}|{trial}|action-audit-v1".encode()).digest()
    return TOOL_NAMES[digest[0] % len(TOOL_NAMES)]


def tool_schemas(model_name: str, condition: str, trial: int) -> list[dict[str, Any]]:
    order = list(TOOL_NAMES)
    random.Random(f"{model_name}|{condition}|{trial}").shuffle(order)
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "State-continuity probe. Use only if the prior assistant selected this exact probe.",
                "parameters": {
                    "type": "object",
                    "properties": {"case_id": {"type": "string"}},
                    "required": ["case_id"],
                    "additionalProperties": False,
                },
            },
        }
        for name in order
    ]


def payload_for(
    model_name: str,
    model: dict[str, Any],
    condition: str,
    trial: int,
) -> tuple[dict[str, Any], str, str]:
    expected = selected_tool(model_name, condition, trial)
    case_id = f"AUDIT-{trial:02d}"
    historical: dict[str, Any] = {"role": "assistant", "content": "Pending action recorded."}
    instruction = f"The exact next action is {expected} with case_id {case_id}."
    if condition == "reasoning":
        historical["reasoning"] = instruction
    elif condition == "visible":
        historical["content"] = instruction
        historical["reasoning"] = "Record the visible pending action."
    elif condition == "absent":
        historical["reasoning"] = "No pending action was selected."
    else:
        raise ValueError(condition)

    payload: dict[str, Any] = {
        "model": model["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are resuming a stateful tool workflow. Use exactly one tool call: the "
                    "exact probe selected in the previous assistant state. Do not infer from tool "
                    "order or names. If no probe was selected, return NO_PENDING_ACTION without a tool call."
                ),
            },
            {"role": "user", "content": "Record the next validation action."},
            historical,
            {"role": "user", "content": "Execute the exact pending validation action now."},
        ],
        "tools": tool_schemas(model_name, condition, trial),
        "tool_choice": "auto",
        "reasoning": {"effort": "low", "exclude": False},
        "include_reasoning": True,
        "temperature": 0,
        "max_tokens": 512,
        "provider": copy.deepcopy(model["provider"]),
    }
    if model.get("seed_supported", True):
        payload["seed"] = 74000 + trial
    return payload, expected, case_id


def run_trial(model_name: str, model: dict[str, Any], condition: str, trial: int) -> dict[str, Any]:
    payload, expected, case_id = payload_for(model_name, model, condition, trial)
    response, wall = post_chat(payload)
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    actual = calls[0]["function"].get("name") if len(calls) == 1 else None
    arguments = None
    if len(calls) == 1:
        try:
            arguments = parse_arguments(calls[0])
        except Exception:
            arguments = None
    correct = actual == expected and arguments == {"case_id": case_id}
    return {
        "model_name": model_name,
        "model": model["model"],
        "condition": condition,
        "trial": trial,
        "expected_tool": expected,
        "expected_arguments": {"case_id": case_id},
        "actual_tool": actual,
        "actual_arguments": arguments,
        "call_count": len(calls),
        "correct": correct,
        "request": payload,
        "response": response,
        "metrics": response_metrics(response, wall),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["OPENROUTER_MIN_BALANCE_USD"] = str(args.min_balance)
    os.environ["OPENROUTER_MAX_CALL_COST_USD"] = "0.10"
    config = json.loads((ROOT / "configs" / "models.json").read_text())
    targets = config["targets"]
    if args.models != "all":
        wanted = args.models.split(",")
        missing = set(wanted) - set(targets)
        if missing:
            raise SystemExit(f"unknown models: {sorted(missing)}")
        targets = {name: targets[name] for name in wanted}

    jobs = []
    for model_name, model in targets.items():
        for condition, count in (
            ("reasoning", args.reasoning_trials),
            ("visible", args.visible_trials),
            ("absent", args.negative_trials),
        ):
            for trial in range(1, count + 1):
                jobs.append((model_name, model, condition, trial))

    initial_balance = remaining_openrouter_credits()
    records = []
    errors = []

    def execute(job: tuple[Any, ...]) -> dict[str, Any]:
        model_name, model, condition, trial = job
        print(f"action-audit model={model_name} condition={condition} trial={trial}", flush=True)
        return run_trial(model_name, model, condition, trial)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            model_name, _model, condition, trial = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append({
                    "model_name": model_name,
                    "condition": condition,
                    "trial": trial,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    records.sort(key=lambda record: (record["model_name"], record["condition"], record["trial"]))

    summary = {}
    for model_name in targets:
        model_records = [record for record in records if record["model_name"] == model_name]
        by_condition = {
            condition: [record for record in model_records if record["condition"] == condition]
            for condition in ("reasoning", "visible", "absent")
        }
        hits = {condition: sum(record["correct"] for record in values) for condition, values in by_condition.items()}
        model_errors = [error for error in errors if error["model_name"] == model_name]
        summary[model_name] = {
            **{f"{condition}_hits": hits[condition] for condition in hits},
            **{f"{condition}_trials": len(by_condition[condition]) for condition in by_condition},
            "errors": len(model_errors),
            "functional_history_retention": (
                len(by_condition["reasoning"]) == args.reasoning_trials
                and hits["reasoning"] >= args.reasoning_trials - 1
                and len(by_condition["visible"]) == args.visible_trials
                and hits["visible"] == args.visible_trials
                and hits["absent"] <= 1
                and not model_errors
            ),
        }

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "recover an exact tool action stored only in historical assistant reasoning",
        "interpretation_limit": (
            "A positive result demonstrates functional use through the hosted route. A negative "
            "result can reflect serialization loss or inability to follow the state-resumption probe."
        ),
        "minimum_balance_usd": args.min_balance,
        "initial_balance_usd": initial_balance,
        "final_balance_usd": remaining_openrouter_credits(),
        "summary": summary,
        "errors": errors,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
