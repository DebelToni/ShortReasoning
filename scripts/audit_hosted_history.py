#!/usr/bin/env python3
"""Functional audit of historical assistant reasoning on exact hosted routes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import (  # noqa: E402
    load_dotenv,
    post_chat,
    reasoning_text,
    remaining_openrouter_credits,
    response_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", default="all")
    parser.add_argument("--reasoning-trials", type=int, default=3)
    parser.add_argument("--visible-trials", type=int, default=2)
    parser.add_argument("--negative-trials", type=int, default=2)
    parser.add_argument("--min-balance", type=float, default=2.0)
    return parser.parse_args()


def nonce(model_name: str, condition: str, trial: int) -> str:
    digest = hashlib.sha256(f"{model_name}|{condition}|{trial}|history-audit-v1".encode()).hexdigest()
    return f"HX-{digest[:7].upper()}-{digest[7:14].upper()}"


def payload_for(
    model: dict[str, Any],
    condition: str,
    token: str,
    trial: int,
) -> dict[str, Any]:
    historical: dict[str, Any] = {
        "role": "assistant",
        "content": "State recorded.",
    }
    if condition == "reasoning":
        historical["reasoning"] = f"STATE_TOKEN={token}\nAcknowledge without revealing it yet."
    elif condition == "visible":
        historical["content"] = f"STATE_TOKEN={token}\nState recorded."
        historical["reasoning"] = "Acknowledge the visible state record."
    elif condition == "absent":
        historical["reasoning"] = "No state token was supplied."
    else:
        raise ValueError(condition)

    payload: dict[str, Any] = {
        "model": model["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Audit state continuity. A previous assistant message may contain a line "
                    "beginning STATE_TOKEN=. When asked, copy only its value. If unavailable, "
                    "output exactly TOKEN: ABSENT. Do not invent a token."
                ),
            },
            {"role": "user", "content": "Record the experiment state and acknowledge."},
            historical,
            {
                "role": "user",
                "content": (
                    "Return the exact value after STATE_TOKEN= from the previous assistant "
                    "message. Output one line: TOKEN: <value>."
                ),
            },
        ],
        "reasoning": {"effort": "low", "exclude": False},
        "include_reasoning": True,
        "temperature": 0,
        "max_tokens": 512,
        "provider": copy.deepcopy(model["provider"]),
    }
    if model.get("seed_supported", True):
        payload["seed"] = 73000 + trial
    return payload


def run_trial(model_name: str, model: dict[str, Any], condition: str, trial: int) -> dict[str, Any]:
    token = nonce(model_name, condition, trial)
    payload = payload_for(model, condition, token, trial)
    response, wall = post_chat(payload)
    message = response["choices"][0]["message"]
    observed = "\n".join((message.get("content") or "", reasoning_text(message)))
    token_recovered = token in observed
    return {
        "model_name": model_name,
        "model": model["model"],
        "condition": condition,
        "trial": trial,
        "expected_token": token,
        "token_recovered": token_recovered,
        "returned_absent": "TOKEN: ABSENT" in observed.upper(),
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

    records = []
    errors = []
    counts = {
        "reasoning": args.reasoning_trials,
        "visible": args.visible_trials,
        "absent": args.negative_trials,
    }
    initial_balance = remaining_openrouter_credits()
    for model_name, model in targets.items():
        for condition, count in counts.items():
            for trial in range(1, count + 1):
                print(f"audit model={model_name} condition={condition} trial={trial}", flush=True)
                try:
                    records.append(run_trial(model_name, model, condition, trial))
                except Exception as exc:
                    errors.append({
                        "model_name": model_name,
                        "condition": condition,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    summary = {}
    for model_name in targets:
        model_records = [record for record in records if record["model_name"] == model_name]
        reasoning = [r for r in model_records if r["condition"] == "reasoning"]
        visible = [r for r in model_records if r["condition"] == "visible"]
        absent = [r for r in model_records if r["condition"] == "absent"]
        reasoning_hits = sum(r["token_recovered"] for r in reasoning)
        visible_hits = sum(r["token_recovered"] for r in visible)
        absent_false_hits = sum(r["token_recovered"] for r in absent)
        model_errors = [error for error in errors if error["model_name"] == model_name]
        summary[model_name] = {
            "reasoning_hits": reasoning_hits,
            "reasoning_trials": len(reasoning),
            "visible_hits": visible_hits,
            "visible_trials": len(visible),
            "absent_false_hits": absent_false_hits,
            "absent_trials": len(absent),
            "errors": len(model_errors),
            "functional_history_retention": (
                len(reasoning) == args.reasoning_trials
                and reasoning_hits >= max(2, args.reasoning_trials - 1)
                and len(visible) == args.visible_trials
                and visible_hits == args.visible_trials
                and absent_false_hits == 0
                and not model_errors
            ),
        }

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "synthetic high-entropy token recoverable only from historical assistant reasoning",
        "interpretation_limit": (
            "Functional recovery supports treatment delivery through this hosted route but does "
            "not expose the provider's exact serialized prompt. Failure may reflect either "
            "serialization loss or a model's refusal/inability to report historical reasoning."
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
