"""Frozen reasoning-history fork for the LiveCodeBench multi-turn adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from short_reasoning import (
    BudgetFloorReached,
    aggregate_metrics,
    assistant_history_message,
    call_target,
    parse_arguments,
    percent_change,
    post_chat,
    reasoning_text,
    rewrite_fidelity,
    rewrite_payload,
    strip_fence,
    target_payload,
)
from short_reasoning.frozen import frozen_source_hash, reasoning_only_fork_audit
from short_reasoning.livecodebench import official_check, static_code_summary

CODE_SYSTEM = """You are a careful Python coding agent operating through function tools.
Use exactly one tool call per assistant turn and use phases in this order:
record_algorithm_plan, record_candidate_solution, evaluate_recorded_candidate, submit_solution.
Do not skip, combine, repeat, or predict a tool result. Do not invent test outcomes.
All code must be Python 3 and must follow the requested stdin/stdout or starter-code interface exactly.
Reasoning style is your choice. Correctness and faithful state tracking are required.
The final submit_solution tool call ends the task."""

PHASES = (
    "record_algorithm_plan",
    "record_candidate_solution",
    "evaluate_recorded_candidate",
    "submit_solution",
)


def task_id(record: dict[str, Any]) -> str:
    return f"lcb-v6-{record['platform']}-{record['question_id']}"


def initial_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    starter = record.get("starter_code") or "(none; use stdin/stdout)"
    return [
        {"role": "system", "content": CODE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Case ID: {task_id(record)}\n"
                f"Platform: {record['platform']}\n"
                f"Difficulty: {record['difficulty']}\n"
                f"Title: {record['question_title']}\n\n"
                f"{record['question_content']}\n\n"
                f"Starter code:\n{starter}"
            ),
        },
    ]


def build_tools() -> list[dict[str, Any]]:
    def schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    case = {"case_id": {"type": "string"}}
    return [
        schema(
            PHASES[0],
            "Record the algorithm and proof sketch before coding.",
            {
                **case,
                "algorithm": {"type": "string"},
                "correctness_argument": {"type": "string"},
                "complexity": {"type": "string"},
            },
            ["case_id", "algorithm", "correctness_argument", "complexity"],
        ),
        schema(
            PHASES[1],
            "Record one complete Python candidate without running tests.",
            {**case, "python_code": {"type": "string"}},
            ["case_id", "python_code"],
        ),
        schema(
            PHASES[2],
            "Run the shared candidate on official public tests and return a deterministic syntax summary.",
            case,
            ["case_id"],
        ),
        schema(
            PHASES[3],
            "Submit the final complete Python solution; no test result is returned to the model.",
            {**case, "python_code": {"type": "string"}},
            ["case_id", "python_code"],
        ),
    ]


def validate_action(
    message: dict[str, Any],
    expected_name: str,
    expected_case_id: str,
    required_text_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    calls = message.get("tool_calls") or []
    result: dict[str, Any] = {
        "expected_tool": expected_name,
        "expected_case_id": expected_case_id,
        "call_count": len(calls),
        "correct": False,
    }
    if len(calls) != 1:
        return result
    call = calls[0]
    result["actual_tool"] = call["function"].get("name")
    try:
        arguments = parse_arguments(call)
    except Exception as exc:
        result["argument_error"] = str(exc)
        return result
    result["arguments"] = arguments
    expected_keys = {"case_id", *required_text_fields}
    result["correct"] = (
        result["actual_tool"] == expected_name
        and set(arguments) == expected_keys
        and arguments.get("case_id") == expected_case_id
        and all(isinstance(arguments.get(field), str) and arguments[field].strip() for field in required_text_fields)
    )
    return result


def rewrite_code_reasoning(
    rewriter: dict[str, Any],
    raw_reasoning: str,
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite without exposing argument values already preserved in the tool call."""
    function = tool_call["function"]
    arguments = parse_arguments(tool_call)
    payload = rewrite_payload(rewriter, raw_reasoning, tool_call)
    payload["messages"][1]["content"] = json.dumps(
        {
            "reasoning": raw_reasoning,
            "requested_tool": {
                "name": function["name"],
                "argument_keys": sorted(arguments),
                "argument_values": "preserved separately and intentionally withheld",
            },
        },
        ensure_ascii=False,
    )
    response, wall = post_chat(payload)
    rewritten = strip_fence(response["choices"][0]["message"].get("content") or "")
    if not rewritten:
        raise RuntimeError("code-history rewriter returned no content")
    return {
        "attempts": [{"request": copy.deepcopy(payload), "response": response, "wall_seconds": wall}],
        "request": copy.deepcopy(payload),
        "response": response,
        "wall_seconds": wall,
        "raw_reasoning": raw_reasoning,
        "rewritten": rewritten,
        "raw_words": len(raw_reasoning.split()),
        "rewritten_words": len(rewritten.split()),
        "requested_tool": copy.deepcopy(function),
        "rewriter_argument_values_withheld": True,
        "fidelity": rewrite_fidelity(raw_reasoning, rewritten),
    }


def append_result(
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    reasoning: str,
    result: dict[str, Any],
) -> None:
    assistant = assistant_history_message(assistant_message, reasoning)
    messages.append(assistant)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": assistant_message["tool_calls"][0]["id"],
            "name": assistant_message["tool_calls"][0]["function"]["name"],
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


def acquire_source(
    record: dict[str, Any],
    dataset_provenance: dict[str, Any],
    model_name: str,
    model: dict[str, Any],
    rewriter: dict[str, Any],
    source_seed: int,
) -> dict[str, Any]:
    case_id = task_id(record)
    tools = build_tools()
    clean = initial_messages(record)
    rewritten = copy.deepcopy(clean)
    shared = []
    required_fields = (
        ("algorithm", "correctness_argument", "complexity"),
        ("python_code",),
    )
    results = (
        {"case_id": case_id, "status": "plan_recorded"},
        {"case_id": case_id, "status": "candidate_recorded_without_execution"},
    )
    recorded_code = None
    for phase_index in range(2):
        target = call_target(model, clean, tools, source_seed + phase_index)
        message = target["response"]["choices"][0]["message"]
        action = validate_action(message, PHASES[phase_index], case_id, required_fields[phase_index])
        target.update({"turn": phase_index + 1, "phase": PHASES[phase_index], "action": action})
        if not action["correct"]:
            shared.append(target)
            return {
                "task_id": case_id,
                "model_name": model_name,
                "source_seed": source_seed,
                "status": "shared_action_failure",
                "shared": shared,
            }
        raw = reasoning_text(message)
        rewrite = rewrite_code_reasoning(rewriter, raw, message["tool_calls"][0])
        phase_result = copy.deepcopy(results[phase_index])
        if phase_index == 1:
            recorded_code = action["arguments"]["python_code"]
            phase_result["code_sha256"] = hashlib.sha256(recorded_code.encode()).hexdigest()
        append_result(clean, message, raw, phase_result)
        append_result(rewritten, message, rewrite["rewritten"], phase_result)
        target["rewrite"] = rewrite
        target["result"] = phase_result
        shared.append(target)
    fork_audit = reasoning_only_fork_audit(clean, rewritten)
    if not fork_audit["passed"]:
        raise RuntimeError(f"source fork audit failed: {fork_audit}")
    raw_words = sum(step["rewrite"]["raw_words"] for step in shared)
    rewritten_words = sum(step["rewrite"]["rewritten_words"] for step in shared)
    source_sha256 = frozen_source_hash(case_id, model_name, model, clean, rewritten)
    status = "selected" if rewritten_words <= raw_words * 0.9 else "rewrite_not_compact"
    return {
        "task_id": case_id,
        "question_id": record["question_id"],
        "title": record["question_title"],
        "platform": record["platform"],
        "difficulty": record["difficulty"],
        "dataset_provenance": copy.deepcopy(dataset_provenance),
        "model_name": model_name,
        "model": copy.deepcopy(model),
        "source_seed": source_seed,
        "status": status,
        "source_compaction": {
            "raw_words": raw_words,
            "rewritten_words": rewritten_words,
            "change_percent": percent_change(raw_words, rewritten_words),
            "required_maximum_ratio": 0.9,
        },
        "shared": shared,
        "recorded_code": recorded_code,
        "histories": {"clean": clean, "rewritten": rewritten},
        "fork_audit": fork_audit,
        "source_sha256": source_sha256,
    }


def continue_source(
    record: dict[str, Any],
    source: dict[str, Any],
    evaluator_repo: Any,
    continuation_seed: int,
    replicate: int,
) -> dict[str, Any]:
    if source.get("status") != "selected":
        raise ValueError("source is not selected")
    case_id = task_id(record)
    if case_id != source["task_id"]:
        raise ValueError("record/source mismatch")
    model = copy.deepcopy(source["model"])
    tools = build_tools()
    public_result = official_check(
        record, source["recorded_code"], evaluator_repo, public_only=True
    )
    candidate_observation = {
        "case_id": case_id,
        "recorded_code_sha256": hashlib.sha256(source["recorded_code"].encode()).hexdigest(),
        "public_tests": public_result,
        "static_summary": static_code_summary(record, source["recorded_code"]),
    }
    branches = {
        condition: {
            "messages": copy.deepcopy(source["histories"][condition]),
            "turns": [],
            "actions_correct": True,
        }
        for condition in ("clean", "rewritten")
    }
    def call_branch(branch: dict[str, Any], seed: int) -> dict[str, Any] | None:
        try:
            return call_target(model, branch["messages"], tools, seed)
        except BudgetFloorReached:
            raise
        except Exception as exc:
            branch["status"] = "endpoint_failure"
            branch["actions_correct"] = False
            branch["endpoint_error"] = f"{type(exc).__name__}: {exc}"
            branch["failed_request"] = target_payload(model, branch["messages"], tools, seed)
            return None

    fixed_phases = ((PHASES[2], candidate_observation),)
    for phase_offset, (phase_name, observation) in enumerate(fixed_phases):
        order = ["clean", "rewritten"]
        random.Random(continuation_seed + phase_offset).shuffle(order)
        for condition in order:
            branch = branches[condition]
            if branch.get("status") in {"action_failure", "endpoint_failure"}:
                continue
            target = call_branch(branch, continuation_seed + phase_offset)
            if target is None:
                continue
            message = target["response"]["choices"][0]["message"]
            action = validate_action(message, phase_name, case_id)
            target.update({"turn": phase_offset + 1, "phase": phase_name, "action": action})
            branch["turns"].append(target)
            branch["actions_correct"] = branch["actions_correct"] and action["correct"]
            if action["correct"]:
                append_result(branch["messages"], message, reasoning_text(message), observation)
            else:
                branch["status"] = "action_failure"

    order = ["clean", "rewritten"]
    random.Random(continuation_seed + 1).shuffle(order)
    for condition in order:
        branch = branches[condition]
        if branch.get("status") in {"action_failure", "endpoint_failure"}:
            continue
        target = call_branch(branch, continuation_seed + 1)
        if target is None:
            continue
        message = target["response"]["choices"][0]["message"]
        action = validate_action(message, PHASES[3], case_id, ("python_code",))
        target.update({"turn": 2, "phase": PHASES[3], "action": action})
        branch["turns"].append(target)
        branch["actions_correct"] = branch["actions_correct"] and action["correct"]
        if action["correct"]:
            code = action["arguments"]["python_code"]
            branch["submitted_code"] = code
            branch["evaluation"] = official_check(
                record, code, evaluator_repo, public_only=False
            )
            branch["status"] = "complete"
        else:
            branch["status"] = "action_failure"

    for branch in branches.values():
        branch.pop("messages", None)
        branch["aggregate"] = aggregate_metrics([turn["metrics"] for turn in branch["turns"]])
        branch["success"] = branch["actions_correct"] and branch.get("evaluation", {}).get("passed", False)
    clean = branches["clean"]["aggregate"]
    rewritten_metrics = branches["rewritten"]["aggregate"]
    comparable = (
        branches["clean"].get("status") == "complete"
        and branches["rewritten"].get("status") == "complete"
        and clean["turn_count"] == rewritten_metrics["turn_count"]
    )

    def paired(metric: str) -> float | None:
        return percent_change(clean.get(metric), rewritten_metrics.get(metric)) if comparable else None

    return {
        "task_id": case_id,
        "question_id": record["question_id"],
        "model_name": source["model_name"],
        "model": model,
        "source_seed": source["source_seed"],
        "source_sha256": source["source_sha256"],
        "continuation_seed": continuation_seed,
        "replicate": replicate,
        "status": "complete",
        "fixed_observations": {"candidate_evaluation": candidate_observation},
        "branches": branches,
        "pair": {
            "comparable_horizon": comparable,
            "reasoning_token_change_percent": paired("reasoning_tokens"),
            "completion_token_change_percent": paired("completion_tokens"),
            "prompt_token_change_percent": paired("prompt_tokens"),
        },
    }
