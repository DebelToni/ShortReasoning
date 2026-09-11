"""Exploratory frozen-history controls for length and surface syntax."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from short_reasoning import post_chat, rewrite_fidelity, strip_fence
from short_reasoning.frozen import reasoning_only_fork_audit

CONTROL_SYSTEMS = {
    "verbose_paraphrase": (
        "Rewrite the supplied assistant scratchpad in faithful alternate wording without compacting it. "
        "Output only the rewritten scratchpad. Preserve every fact, number, evidence-status distinction, "
        "uncertainty, correction, rejected branch, conclusion, and intended next action. Keep the word count "
        "inside the supplied target interval. Do not add knowledge, improve the solution, reveal unavailable "
        "action arguments, or predict the pending result."
    ),
    "compact_full_sentence": (
        "Rewrite the supplied compact assistant state using normal complete grammatical sentences. Output only "
        "the rewritten state. Preserve exactly the same information, uncertainty, conclusions, notation, and "
        "intended next action while removing telegraphic fragments. Keep the word count inside the supplied "
        "target interval and never add knowledge, consult an original verbose scratchpad, reveal unavailable "
        "action arguments, or predict the pending result."
    ),
}
FORBIDDEN_TAGS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")


def word_interval(control: str, raw_words: int, compact_words: int) -> tuple[int, int]:
    if control == "verbose_paraphrase":
        return max(1, round(raw_words * 0.9)), max(1, round(raw_words * 1.1))
    if control == "compact_full_sentence":
        return max(1, round(compact_words * 0.8)), max(1, round(compact_words * 1.25))
    raise ValueError(f"unknown history control: {control}")


def control_payload(
    rewriter: dict[str, Any],
    control: str,
    raw_reasoning: str,
    compact_reasoning: str,
    requested_tool: dict[str, Any],
) -> dict[str, Any]:
    lower, upper = word_interval(
        control, len(raw_reasoning.split()), len(compact_reasoning.split())
    )
    source_text = raw_reasoning if control == "verbose_paraphrase" else compact_reasoning
    arguments = requested_tool.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    action_context = {"name": requested_tool["name"]}
    if control == "verbose_paraphrase":
        action_context.update({
            "argument_keys": sorted(arguments),
            "argument_values": "preserved separately and intentionally withheld",
        })
    user = {
        "source_scratchpad": source_text,
        "requested_action": action_context,
        "target_word_interval_inclusive": [lower, upper],
    }
    return {
        "model": rewriter["model"],
        "messages": [
            {"role": "system", "content": CONTROL_SYSTEMS[control]},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "reasoning": {"effort": rewriter["reasoning_effort"], "exclude": True},
        "include_reasoning": False,
        "seed": 73191 if control == "verbose_paraphrase" else 73192,
        "max_tokens": rewriter["max_tokens"],
        "provider": copy.deepcopy(rewriter["provider"]),
    }


def automatic_control_audit(
    control: str,
    raw_reasoning: str,
    compact_reasoning: str,
    variant: str,
) -> dict[str, Any]:
    raw_words = len(raw_reasoning.split())
    compact_words = len(compact_reasoning.split())
    lower, upper = word_interval(control, raw_words, compact_words)
    variant_words = len(variant.split())
    reference = raw_reasoning if control == "verbose_paraphrase" else compact_reasoning
    fidelity = rewrite_fidelity(reference, variant)
    reference_ids = set(fidelity["raw_identifiers"])
    variant_ids = set(rewrite_fidelity(variant, variant)["raw_identifiers"])
    reference_numbers = set(fidelity["raw_numeric_literals"])
    variant_numbers = set(rewrite_fidelity(variant, variant)["raw_numeric_literals"])
    added_ids = sorted(variant_ids - reference_ids)
    added_numbers = sorted(variant_numbers - reference_numbers)
    lower_variant = variant.lower()
    forbidden = next((tag for tag in FORBIDDEN_TAGS if tag.lower() in lower_variant), None)
    passed = (
        lower <= variant_words <= upper
        and fidelity["all_identifiers_preserved"]
        and fidelity["all_numeric_literals_preserved"]
        and not added_ids
        and not added_numbers
        and forbidden is None
    )
    return {
        "passed": passed,
        "reference": "raw" if control == "verbose_paraphrase" else "compact",
        "raw_words": raw_words,
        "compact_words": compact_words,
        "variant_words": variant_words,
        "target_word_interval_inclusive": [lower, upper],
        "forbidden_template_tag": forbidden,
        "added_identifiers": added_ids,
        "added_numeric_literals": added_numbers,
        "fidelity": fidelity,
    }


def generate_control_turn(
    rewriter: dict[str, Any],
    control: str,
    raw_reasoning: str,
    compact_reasoning: str,
    requested_tool: dict[str, Any],
    on_attempt: Any | None = None,
) -> dict[str, Any]:
    payload = control_payload(
        rewriter, control, raw_reasoning, compact_reasoning, requested_tool
    )
    attempts = []
    for attempt_index in range(3):
        response, wall = post_chat(payload)
        variant = strip_fence(response["choices"][0]["message"].get("content") or "")
        audit = automatic_control_audit(
            control, raw_reasoning, compact_reasoning, variant
        )
        attempts.append(
            {
                "request": copy.deepcopy(payload),
                "response": response,
                "wall_seconds": wall,
                "variant": variant,
                "automatic_audit": audit,
            }
        )
        if on_attempt is not None:
            on_attempt(copy.deepcopy(attempts))
        if variant and audit["passed"]:
            return {
                "control": control,
                "attempts": attempts,
                "variant": variant,
                "audit_inputs": {
                    "raw_reasoning": raw_reasoning,
                    "compact_reasoning": compact_reasoning,
                },
                "automatic_audit": audit,
                "requested_action_name": requested_tool["name"],
                "argument_values_withheld": True,
            }
        payload = copy.deepcopy(payload)
        payload["messages"][1]["content"] = json.dumps(
            {
                **json.loads(payload["messages"][1]["content"]),
                "prior_draft": variant,
                "repair_instruction": (
                    "Try again within the exact word interval and preserve every identifier and number from "
                    "the supplied source without appending an unexplained list."
                ),
            },
            ensure_ascii=False,
        )
        payload["seed"] += attempt_index + 1
    raise RuntimeError(f"{control} failed its automatic gate three times")


def control_variant_hash(
    parent_source_sha256: str,
    control: str,
    baseline_condition: str,
    baseline_history: list[dict[str, Any]],
    variant_history: list[dict[str, Any]],
) -> str:
    payload = {
        "parent_source_sha256": parent_source_sha256,
        "control": control,
        "baseline_condition": baseline_condition,
        "baseline_history": baseline_history,
        "variant_history": variant_history,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_control_variant(variant: dict[str, Any]) -> None:
    expected = control_variant_hash(
        variant["parent_source_sha256"],
        variant["control"],
        variant["baseline_condition"],
        variant["histories"]["baseline"],
        variant["histories"]["variant"],
    )
    if variant.get("variant_sha256") != expected:
        raise RuntimeError("control variant hash mismatch")
    audit = reasoning_only_fork_audit(
        variant["histories"]["baseline"], variant["histories"]["variant"]
    )
    if not audit["passed"] or audit != variant.get("fork_audit"):
        raise RuntimeError("control variant reasoning-only audit mismatch")
    generated = variant.get("generated_turns") or []
    assistant_reasoning = [
        message["reasoning"]
        for message in variant["histories"]["variant"]
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    if len(generated) != len(assistant_reasoning):
        raise RuntimeError("control variant contains an incomplete turn set")
    for turn, history_reasoning in zip(generated, assistant_reasoning):
        inputs = turn.get("audit_inputs", {})
        recomputed = automatic_control_audit(
            variant["control"],
            inputs.get("raw_reasoning", ""),
            inputs.get("compact_reasoning", ""),
            turn.get("variant", ""),
        )
        if (
            turn.get("variant") != history_reasoning
            or recomputed != turn.get("automatic_audit")
            or not recomputed["passed"]
        ):
            raise RuntimeError("control variant automatic audit does not recompute")


def build_control_variant(
    source: dict[str, Any],
    control: str,
    generated_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_condition = "clean" if control == "verbose_paraphrase" else "rewritten"
    baseline = copy.deepcopy(source["histories"][baseline_condition])
    variant = copy.deepcopy(baseline)
    assistant_indices = [
        index
        for index, message in enumerate(variant)
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    if len(assistant_indices) != len(generated_turns):
        raise ValueError("control turns do not match historical assistant reasoning turns")
    for index, generated in zip(assistant_indices, generated_turns):
        variant[index]["reasoning"] = generated["variant"]
    fork_audit = reasoning_only_fork_audit(baseline, variant)
    if not fork_audit["passed"]:
        raise RuntimeError(f"control variant failed reasoning-only audit: {fork_audit}")
    variant_hash = control_variant_hash(
        source["source_sha256"], control, baseline_condition, baseline, variant
    )
    return {
        "task_id": source["task_id"],
        "model_name": source["model_name"],
        "model": copy.deepcopy(source["model"]),
        "parent_source_sha256": source["source_sha256"],
        "control": control,
        "baseline_condition": baseline_condition,
        "status": "pending_semantic_audit",
        "generated_turns": generated_turns,
        "histories": {"baseline": baseline, "variant": variant},
        "fork_audit": fork_audit,
        "variant_sha256": variant_hash,
    }
