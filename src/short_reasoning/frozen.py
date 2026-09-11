"""Frozen-source acquisition and paired continuation for post-pilot experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

from . import (
    IDENTIFIER_RE,
    aggregate_metrics,
    append_tool_cycle,
    build_tools,
    call_target,
    collect_task_candidate_ids,
    evaluate_tool_response,
    final_is_correct,
    hard_thresholds_in_reasoning,
    initial_messages,
    percent_change,
    reasoning_text,
    rewrite_before_tool_result,
    task_hard_thresholds,
)


def call_target_with_transport(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    seed: int,
    post_chat_fn: Any,
) -> dict[str, Any]:
    if post_chat_fn is None:
        return call_target(model, messages, tools, seed)
    return call_target(model, messages, tools, seed, post_chat_fn=post_chat_fn)


def rewrite_with_transport(
    rewriter: dict[str, Any],
    raw: str,
    tool_call: dict[str, Any],
    required_ids: list[str],
    required_thresholds: list[str],
    post_chat_fn: Any,
) -> dict[str, Any]:
    if post_chat_fn is None:
        return rewrite_before_tool_result(
            rewriter, raw, tool_call, required_ids, required_thresholds
        )
    return rewrite_before_tool_result(
        rewriter,
        raw,
        tool_call,
        required_ids,
        required_thresholds,
        post_chat_fn=post_chat_fn,
    )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def reasoning_only_fork_audit(
    clean_messages: list[dict[str, Any]],
    rewritten_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(clean_messages) != len(rewritten_messages):
        return {
            "passed": False,
            "reason": "message_count",
            "clean_count": len(clean_messages),
            "rewritten_count": len(rewritten_messages),
            "reasoning_message_indices": [],
        }

    clean_stripped = copy.deepcopy(clean_messages)
    rewritten_stripped = copy.deepcopy(rewritten_messages)
    changed_indices = []
    for index, (clean, rewritten) in enumerate(zip(clean_stripped, rewritten_stripped)):
        clean_reasoning = clean.pop("reasoning", None)
        rewritten_reasoning = rewritten.pop("reasoning", None)
        if clean != rewritten:
            return {
                "passed": False,
                "reason": "non_reasoning_difference",
                "first_difference_index": index,
                "reasoning_message_indices": changed_indices,
            }
        if clean_reasoning != rewritten_reasoning:
            if clean.get("role") != "assistant":
                return {
                    "passed": False,
                    "reason": "non_assistant_reasoning_difference",
                    "first_difference_index": index,
                    "reasoning_message_indices": changed_indices,
                }
            changed_indices.append(index)

    passed = bool(changed_indices) and clean_stripped == rewritten_stripped
    return {
        "passed": passed,
        "reason": None if passed else "no_reasoning_difference",
        "reasoning_message_indices": changed_indices,
        "changed_reasoning_strings": len(changed_indices),
    }


def frozen_source_hash(
    task_id: str,
    model_name: str,
    model: dict[str, Any],
    clean_messages: list[dict[str, Any]],
    rewritten_messages: list[dict[str, Any]],
) -> str:
    return canonical_sha256({
        "task_id": task_id,
        "model_name": model_name,
        "model": model,
        "clean_messages": clean_messages,
        "rewritten_messages": rewritten_messages,
    })


def acquire_frozen_source(
    task: dict[str, Any],
    model_name: str,
    model: dict[str, Any],
    rewriter: dict[str, Any],
    source_seed: int,
    post_chat_fn: Any = None,
) -> dict[str, Any]:
    """Acquire one model-specific history fork before any continuation is sampled."""
    tools = build_tools(task)
    clean_messages = initial_messages(task)
    rewritten_messages = copy.deepcopy(clean_messages)
    shared_records = []
    candidate_ids = collect_task_candidate_ids(task)
    thresholds = task_hard_thresholds(task)

    for phase_index, phase in enumerate(task["phases"][: task["fork_after"]]):
        target = call_target_with_transport(
            model, clean_messages, tools, source_seed + phase_index, post_chat_fn
        )
        message = target["response"]["choices"][0]["message"]
        action = evaluate_tool_response(message, phase)
        target.update({
            "turn": phase_index + 1,
            "phase": phase["name"],
            "action": action,
        })
        if not action["correct"]:
            shared_records.append(target)
            return {
                "task_id": task["id"],
                "title": task["title"],
                "model_name": model_name,
                "model": copy.deepcopy(model),
                "source_seed": source_seed,
                "status": "shared_action_failure",
                "shared": shared_records,
            }

        raw = reasoning_text(message)
        required_ids = sorted(set(IDENTIFIER_RE.findall(raw.upper())) & candidate_ids)
        required_thresholds = hard_thresholds_in_reasoning(raw, thresholds)
        if raw.strip():
            rewrite = rewrite_with_transport(
                rewriter,
                raw,
                message["tool_calls"][0],
                required_ids,
                required_thresholds,
                post_chat_fn,
            )
        else:
            rewrite = {
                "identity": True,
                "attempts": [],
                "request": None,
                "response": None,
                "wall_seconds": 0,
                "raw_reasoning": raw,
                "rewritten": raw,
                "raw_words": 0,
                "rewritten_words": 0,
                "requested_tool": copy.deepcopy(message["tool_calls"][0]["function"]),
                "fidelity": {
                    "required_candidate_ids": [],
                    "missing_candidate_ids": [],
                    "all_candidate_ids_preserved": True,
                    "required_hard_thresholds": [],
                    "missing_hard_thresholds": [],
                    "all_hard_thresholds_preserved": True,
                },
            }
        # The pending result is appended only after the rewrite and fidelity gate complete.
        append_tool_cycle(clean_messages, message, raw, phase)
        append_tool_cycle(rewritten_messages, message, rewrite["rewritten"], phase)
        target["rewrite"] = rewrite
        shared_records.append(target)

    fork_audit = reasoning_only_fork_audit(clean_messages, rewritten_messages)
    if not fork_audit["passed"]:
        raise RuntimeError(f"frozen source failed fork audit: {fork_audit}")
    source_hash = frozen_source_hash(
        task["id"], model_name, model, clean_messages, rewritten_messages
    )
    return {
        "task_id": task["id"],
        "title": task["title"],
        "model_name": model_name,
        "model": copy.deepcopy(model),
        "source_seed": source_seed,
        "status": "selected",
        "fork_after": task["fork_after"],
        "shared": shared_records,
        "histories": {
            "clean": clean_messages,
            "rewritten": rewritten_messages,
        },
        "fork_audit": fork_audit,
        "source_sha256": source_hash,
    }


def continue_frozen_source(
    task: dict[str, Any],
    source: dict[str, Any],
    continuation_seed: int,
    replicate: int,
    post_chat_fn: Any = None,
) -> dict[str, Any]:
    """Continue both branches from an immutable model-specific source fork."""
    if source.get("status") != "selected":
        raise ValueError("only selected frozen sources can be continued")
    model = copy.deepcopy(source["model"])
    tools = build_tools(task)
    branches: dict[str, dict[str, Any]] = {
        condition: {
            "messages": copy.deepcopy(source["histories"][condition]),
            "turns": [],
            "actions_correct": True,
        }
        for condition in ("clean", "rewritten")
    }

    for phase_index in range(task["fork_after"], len(task["phases"])):
        phase = task["phases"][phase_index]
        order = ["clean", "rewritten"]
        random.Random(continuation_seed + phase_index).shuffle(order)
        for condition in order:
            branch = branches[condition]
            if branch.get("status") == "action_failure":
                continue
            target = call_target_with_transport(
                model,
                branch["messages"],
                tools,
                continuation_seed + phase_index,
                post_chat_fn,
            )
            message = target["response"]["choices"][0]["message"]
            action = evaluate_tool_response(message, phase)
            target.update({
                "turn": phase_index + 1,
                "phase": phase["name"],
                "action": action,
            })
            branch["turns"].append(target)
            branch["actions_correct"] = branch["actions_correct"] and action["correct"]
            if action["correct"]:
                append_tool_cycle(
                    branch["messages"], message, reasoning_text(message), phase
                )
            else:
                branch["status"] = "action_failure"

    final_turn = len(task["phases"]) + 1
    order = ["clean", "rewritten"]
    random.Random(continuation_seed + final_turn).shuffle(order)
    for condition in order:
        branch = branches[condition]
        if branch.get("status") == "action_failure":
            continue
        target = call_target_with_transport(
            model,
            branch["messages"],
            tools,
            continuation_seed + final_turn,
            post_chat_fn,
        )
        message = target["response"]["choices"][0]["message"]
        unexpected_calls = len(message.get("tool_calls") or [])
        correct = final_is_correct(message.get("content"), task["final_answer"])
        target.update({
            "turn": final_turn,
            "phase": "final",
            "final_correct": correct,
            "unexpected_tool_calls": unexpected_calls,
        })
        branch["turns"].append(target)
        branch["final_correct"] = correct and unexpected_calls == 0
        branch["final_content"] = message.get("content")
        branch["status"] = "complete"

    for branch in branches.values():
        branch.pop("messages", None)
        branch["aggregate"] = aggregate_metrics(
            [turn["metrics"] for turn in branch["turns"]]
        )
        branch.setdefault("final_correct", False)
        branch["success"] = branch["actions_correct"] and branch["final_correct"]

    clean = branches["clean"]["aggregate"]
    rewritten = branches["rewritten"]["aggregate"]
    comparable = (
        branches["clean"].get("status") == "complete"
        and branches["rewritten"].get("status") == "complete"
        and clean["turn_count"] == rewritten["turn_count"]
    )

    def paired_change(metric: str) -> float | None:
        if not comparable:
            return None
        return percent_change(clean.get(metric), rewritten.get(metric))

    return {
        "task_id": task["id"],
        "title": task["title"],
        "model_name": source["model_name"],
        "model": model,
        "source_seed": source["source_seed"],
        "source_sha256": source["source_sha256"],
        "continuation_seed": continuation_seed,
        "replicate": replicate,
        "status": "complete",
        "fork_after": task["fork_after"],
        "expected_final": task["final_answer"],
        "branches": branches,
        "pair": {
            "comparable_horizon": comparable,
            "reasoning_token_change_percent": paired_change("reasoning_tokens"),
            "completion_token_change_percent": paired_change("completion_tokens"),
            "reasoning_word_change_percent": paired_change("reasoning_words"),
            "prompt_token_change_percent": paired_change("prompt_tokens"),
        },
    }
