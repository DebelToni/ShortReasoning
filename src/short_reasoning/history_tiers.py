"""Pre-outcome construction and verification for a five-arm history dose study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Sequence

from short_reasoning import rewrite_fidelity, strict_json_loads
from short_reasoning.frozen import reasoning_only_fork_audit
from short_reasoning.history_controls import FORBIDDEN_TAGS

GENERATED_TIERS = (
    "ultra_telegraphic",
    "full_sentence_compact",
    "relaxed_full_sentence_compact",
)
FOUR_FRESH_TIERS = (
    "ultra_telegraphic",
    "telegraphic_compact",
    "full_sentence_compact",
    "relaxed_full_sentence_compact",
)
ARMS = (
    "clean",
    "ultra_telegraphic",
    "telegraphic_compact",
    "full_sentence_compact",
    "relaxed_full_sentence_compact",
)

TIER_INSTRUCTIONS = {
    "ultra_telegraphic": (
        "Write the shortest unambiguous state record possible. Use fragments, symbols, semicolons, and shared "
        "modifiers aggressively. Remove rhetoric, repeated rules, repeated arithmetic, headings, and prose that "
        "does not change the next decision. Omit no decision-relevant state."
    ),
    "telegraphic_compact": (
        "Write a compact telegraphic state record using brief fragments, labels, symbols, and semicolon-shared "
        "structure. Remove rhetoric, repeated arithmetic, and repetition while preserving every decision-relevant "
        "fact and exact action. Remain more explicit and readable than the ultra-telegraphic tier, but do not turn "
        "the state into ordinary full-sentence prose."
    ),
    "full_sentence_compact": (
        "Express the compact state in ordinary complete grammatical sentences. Make dependencies explicit "
        "without adding explanation, repetition, or facts. Avoid telegraphic fragments and note headings."
    ),
    "relaxed_full_sentence_compact": (
        "Express the state as relaxed, natural, complete prose with modest connective context. Remain materially "
        "shorter than the raw scratchpad. Never repeat facts merely to reach the interval."
    ),
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
RESULT_STATUS_RE = re.compile(
    r"(?i)\b(?:result|output|brief)\b.{0,24}\b(?:pending|withheld|forthcoming)\b"
    r"|\b(?:pending|withheld|forthcoming)\b.{0,24}\b(?:result|output|brief)\b"
    r"|\bno\s+(?:simulation\s+)?output\s+yet\b"
)

SYSTEM_PROMPT = """Rewrite one historical assistant reasoning block into the requested style tier.
Output only the rewritten state through the required JSON schema.
Preserve every decision-relevant fact, number, unit, comparator, uncertainty, correction, rejected branch,
evidence-status distinction, conclusion, dependency, and intended next action supported by the raw scratchpad and
accepted compact anchor. The raw scratchpad is authoritative if the anchor adds process-control metadata; do not
restore irrelevant repetition.
Include the requested action's exact function name, argument keys, and argument values in the state. The action
has not executed: do not add process metadata saying its result is pending, withheld, absent, or forthcoming.
Preserve an unknown/pending measurement only when the reasoning texts themselves state it. State every inequality
direction and tie-break direction explicitly; `limit X to N` or a bare metric name is not a substitute for ≤/≥,
lower/higher, at-most/at-least, minimum/maximum, or the source's exact direction. Never import a fact absent from
both reasoning texts, improve the solution, use a later turn, observation, or final answer. Preserve opaque
identifiers exactly; do not invent shorthand such as P2 for Phase 2 or DAY-21 for day 21.
Stay inside the exact inclusive word interval. Do not emit template-control tags, code fences, labels, commentary,
or an unexplained token list."""

SELF_SYSTEM_PROMPT = """Rewrite one of your own historical reasoning blocks into the requested style tier.
Output only the rewritten state through the required JSON schema.
The raw scratchpad is the sole semantic source. Preserve every decision-relevant fact, number, unit, comparator,
uncertainty, correction, rejected branch, evidence-status distinction, conclusion, dependency, and intended next
action. Remove only rhetoric, repeated arithmetic, transitions, and redundant restatement.
Include the requested action's exact function name, argument keys, and argument values. The action has not executed:
do not claim its result is pending, withheld, absent, or forthcoming. State every inequality and tie-break direction
explicitly; `limit X to N` or a bare metric is not a substitute for the original ≤/≥, lower/higher,
at-most/at-least, minimum/maximum direction. Never improve the solution, import future information, or invent an
identifier. If an unrelated style example is supplied, imitate only its compression pattern and never copy its
facts, identifiers, numbers, action, or conclusions. Stay inside the exact inclusive word interval. Output no code
fence, commentary, or unexplained token list."""


def generated_tiers_for(mode: str = "shared_telegraphic") -> tuple[str, ...]:
    if mode == "shared_telegraphic":
        return GENERATED_TIERS
    if mode == "four_fresh":
        return FOUR_FRESH_TIERS
    raise ValueError(f"unknown tier mode: {mode}")


def word_interval(tier: str, raw_words: int, compact_words: int) -> tuple[int, int]:
    """Derive one turn's dose without consulting any later historical turn."""
    if raw_words <= 0 or compact_words <= 0:
        raise ValueError("tier dose requires nonempty reasoning blocks")
    gap = raw_words - compact_words
    if tier == "ultra_telegraphic":
        lower = max(1, math.ceil(0.60 * compact_words))
        upper = min(max(1, compact_words - 1), max(lower, math.floor(0.90 * compact_words)))
    elif tier == "telegraphic_compact":
        _ultra_lower, ultra_upper = word_interval(
            "ultra_telegraphic", raw_words, compact_words
        )
        lower = min(compact_words, ultra_upper + 1)
        upper = compact_words
    elif gap <= 0 and tier in {"full_sentence_compact", "relaxed_full_sentence_compact"}:
        lower = upper = compact_words
    elif tier == "full_sentence_compact":
        lower = compact_words + math.ceil(0.05 * gap)
        upper = max(lower, compact_words + max(3, math.floor(0.30 * gap)))
    elif tier == "relaxed_full_sentence_compact":
        lower = compact_words + math.ceil(0.35 * gap)
        upper = max(lower, compact_words + max(3, math.floor(0.65 * gap)))
    else:
        raise ValueError(f"unknown history tier: {tier}")
    if not 0 < lower <= upper:
        raise ValueError(f"empty word interval for {tier}: {lower}..{upper}")
    return lower, upper


def turn_word_intervals(
    tier: str, raw_words: Sequence[int], compact_words: Sequence[int]
) -> list[tuple[int, int]]:
    if len(raw_words) != 3 or len(compact_words) != 3:
        raise ValueError("tier study requires exactly three historical turns")
    return [
        word_interval(tier, raw, compact)
        for raw, compact in zip(raw_words, compact_words)
    ]


def tier_payload(
    rewriter: dict[str, Any],
    tier: str,
    turn: dict[str, Any],
    turn_index: int,
    interval: tuple[int, int],
    prior_draft: str | None = None,
    repair_instruction: str | None = None,
) -> dict[str, Any]:
    if tier not in TIER_INSTRUCTIONS:
        raise ValueError(f"unknown generated tier: {tier}")
    reference_mode = rewriter.get("reference_mode", "compact_anchor")
    if reference_mode not in {"compact_anchor", "raw_only"}:
        raise ValueError(f"unknown rewriter reference mode: {reference_mode}")
    user: dict[str, Any] = {
        "tier": tier,
        "tier_instruction": TIER_INSTRUCTIONS[tier],
        "historical_turn": turn_index,
        "raw_reasoning": turn["raw_reasoning"],
    }
    if reference_mode == "compact_anchor":
        user["accepted_compact_anchor"] = turn["compact_reasoning"]
    user.update(
        requested_action=copy.deepcopy(turn["requested_tool"]),
        word_interval_inclusive=list(interval),
    )
    if rewriter.get("style_example") is not None:
        user["unrelated_style_example"] = copy.deepcopy(rewriter["style_example"])
    if prior_draft is not None:
        user["prior_draft"] = prior_draft
        user["repair_instruction"] = repair_instruction
    schema = {
        "type": "object",
        "properties": {"state": {"type": "string"}},
        "required": ["state"],
        "additionalProperties": False,
    }
    payload = {
        "model": rewriter["model"],
        "messages": [
            {
                "role": "system",
                "content": SELF_SYSTEM_PROMPT if reference_mode == "raw_only" else SYSTEM_PROMPT,
            },
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "reasoning": {"effort": rewriter["reasoning_effort"], "exclude": True},
        "include_reasoning": False,
        "seed": {
            "ultra_telegraphic": 84101,
            "full_sentence_compact": 84102,
            "relaxed_full_sentence_compact": 84103,
            "telegraphic_compact": 84104,
        }[tier]
        + turn_index,
        "provider": copy.deepcopy(rewriter["provider"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "history_tier_rewrite",
                "strict": True,
                "schema": schema,
            },
        },
    }
    if rewriter.get("max_tokens") is not None:
        payload["max_tokens"] = rewriter["max_tokens"]
    if "temperature" in rewriter:
        payload["temperature"] = rewriter["temperature"]
    return payload


def parse_tier_response(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("rewriter response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("rewriter response did not finish with stop")
    message = choice.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        raise ValueError("rewriter response lacks one assistant string message content")
    parsed = strict_json_loads(message["content"])
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"state"}
        or not isinstance(parsed["state"], str)
    ):
        raise ValueError("rewriter response has the wrong object shape")
    state = parsed["state"].strip()
    if not state:
        raise ValueError("rewriter returned an empty state")
    return state


def _disambiguate_numeric_commas(text: str) -> str:
    def bracket_list(match: re.Match[str]) -> str:
        value = match.group(0)
        if "$" in value:
            return value
        if re.search(r"(?<=\d),\s+(?=\d)", value):
            return re.sub(r"(?<=\d),\s+(?=\d)", " ", value)
        numeric_commas = re.findall(r"(?<=\d),(?=\d)", value)
        if len(numeric_commas) == 1 and re.fullmatch(r"\[\s*\d{1,3},\d{3}\s*\]", value):
            return value
        return re.sub(r"(?<=\d),(?=\d)", " ", value)

    text = re.sub(r"\[[^\]\n]*\]", bracket_list, text)

    def comma_run(match: re.Match[str]) -> str:
        value = match.group(0)
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", value):
            return value
        return value.replace(",", " ")

    return re.sub(r"\d(?:[\d,]*\d)?", comma_run, text)


def _lexical_sets(text: str) -> tuple[set[str], set[str]]:
    disambiguated = _disambiguate_numeric_commas(text)
    audit = rewrite_fidelity(disambiguated, disambiguated)
    return set(audit["raw_identifiers"]), set(audit["raw_numeric_literals"])


def _schema_field_numeric_literals(text: str) -> set[str]:
    fields = re.findall(
        r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*\d+[A-Za-z0-9_]*\b", text
    )
    return {number for field in fields for number in re.findall(r"\d+", field)}


def _list_length_numeric_literals(text: str) -> set[str]:
    result = set()
    for contents in re.findall(r"\[([^\]\n]+)\]", text):
        items = [item for item in contents.split(",") if item.strip()]
        if len(items) >= 2:
            result.add(str(len(items)))
    return result


def _number_word_literals(text: str) -> set[str]:
    return {
        value
        for word, value in NUMBER_WORDS.items()
        if re.search(rf"(?i)\b{word}\b", text)
    }


def _requested_action_audit(turn: dict[str, Any], state: str) -> dict[str, Any]:
    requested = turn.get("requested_tool", {})
    function_name = requested.get("name")
    try:
        arguments = strict_json_loads(requested.get("arguments", ""))
    except (ValueError, TypeError, json.JSONDecodeError):
        arguments = None
    keys: list[str] = []
    string_values: list[str] = []
    numeric_values: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                keys.append(key)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            string_values.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_values.append(str(value))

    if isinstance(arguments, dict):
        collect(arguments)
    def contains_symbol(value: Any) -> bool:
        return isinstance(value, str) and bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
                state,
            )
        )

    _state_ids, state_numbers = _lexical_sets(state)
    result = {
        "function_name": function_name,
        "arguments_parseable_object": isinstance(arguments, dict),
        "missing_function_name": (
            function_name
            if not contains_symbol(function_name)
            else None
        ),
        "missing_argument_keys": sorted(
            {key for key in keys if not contains_symbol(key)}
        ),
        "missing_string_argument_values": sorted(
            {value for value in string_values if value not in state}
        ),
        "missing_numeric_argument_values": sorted(
            {value for value in numeric_values if value not in state_numbers}
        ),
    }
    result["passed"] = (
        result["arguments_parseable_object"]
        and result["missing_function_name"] is None
        and not result["missing_argument_keys"]
        and not result["missing_string_argument_values"]
        and not result["missing_numeric_argument_values"]
    )
    return result


def automatic_turn_audit(
    turn: dict[str, Any],
    state: str,
    interval: tuple[int, int],
    *,
    require_exact_action: bool = False,
    require_compact_difference: bool = True,
) -> dict[str, Any]:
    anchor_ids, anchor_numbers = _lexical_sets(turn["compact_reasoning"])
    allowed_ids, allowed_numbers = _lexical_sets(
        turn["raw_reasoning"] + "\n" + turn["compact_reasoning"]
    )
    state_ids, state_numbers = _lexical_sets(state)
    state_schema_numbers = _schema_field_numeric_literals(state)
    state_list_lengths = _list_length_numeric_literals(state)
    state_number_words = _number_word_literals(state)
    forbidden = next((tag for tag in FORBIDDEN_TAGS if tag.lower() in state.lower()), None)
    result_status_match = RESULT_STATUS_RE.search(state)
    words = len(state.split())
    result = {
        "raw_words": len(turn["raw_reasoning"].split()),
        "compact_words": len(turn["compact_reasoning"].split()),
        "variant_words": words,
        "target_word_interval_inclusive": list(interval),
        "missing_anchor_identifiers": sorted(anchor_ids - state_ids),
        "missing_anchor_numeric_literals": sorted(
            anchor_numbers
            - state_numbers
            - state_schema_numbers
            - state_list_lengths
            - state_number_words
        ),
        "state_schema_field_numeric_literals": sorted(state_schema_numbers),
        "state_list_length_numeric_literals": sorted(state_list_lengths),
        "state_number_word_literals": sorted(state_number_words),
        "added_identifiers": sorted(state_ids - allowed_ids),
        "added_numeric_literals": sorted(state_numbers - allowed_numbers),
        "forbidden_template_tag": forbidden,
        "unsupported_result_status": (
            result_status_match.group(0) if result_status_match else None
        ),
        "differs_from_raw": state.strip() != turn["raw_reasoning"].strip(),
        "differs_from_compact": state.strip() != turn["compact_reasoning"].strip(),
    }
    if require_exact_action:
        result["requested_action_audit"] = _requested_action_audit(turn, state)
    result["passed"] = (
        interval[0] <= words <= interval[1]
        and not result["missing_anchor_identifiers"]
        and not result["missing_anchor_numeric_literals"]
        and not result["added_identifiers"]
        and not result["added_numeric_literals"]
        and forbidden is None
        and result["unsupported_result_status"] is None
        and result["differs_from_raw"]
        and (result["differs_from_compact"] or not require_compact_difference)
        and (
            not require_exact_action
            or result["requested_action_audit"]["passed"]
        )
    )
    return result


def automatic_tier_audit(
    tier: str,
    turns: Sequence[dict[str, Any]],
    states: Sequence[str],
    *,
    require_exact_action: bool = False,
) -> dict[str, Any]:
    raw_words = [len(turn["raw_reasoning"].split()) for turn in turns]
    compact_words = [len(turn["compact_reasoning"].split()) for turn in turns]
    intervals = turn_word_intervals(tier, raw_words, compact_words)
    if not (len(states) == len(turns) == 3):
        raise ValueError("tier states do not match the three source turns")
    audits = [
        automatic_turn_audit(
            turn,
            state,
            interval,
            require_exact_action=require_exact_action,
        )
        for turn, state, interval in zip(turns, states, intervals)
    ]
    source_interval = (
        sum(lower for lower, _ in intervals),
        sum(upper for _, upper in intervals),
    )
    variant_total = sum(audit["variant_words"] for audit in audits)
    return {
        "passed": all(audit["passed"] for audit in audits)
        and source_interval[0] <= variant_total <= source_interval[1],
        "tier": tier,
        "clean_total_words": sum(raw_words),
        "compact_total_words": sum(compact_words),
        "variant_total_words": variant_total,
        "target_total_word_interval_inclusive": list(source_interval),
        "turn_intervals": [list(interval) for interval in intervals],
        "turns": audits,
    }


def source_turns(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "raw_reasoning": step["rewrite"]["raw_reasoning"],
            "compact_reasoning": step["rewrite"]["rewritten"],
            "requested_tool": copy.deepcopy(step["rewrite"]["requested_tool"]),
        }
        for step in source["shared"]
    ]


def tier_variant_hash(
    parent_source_sha256: str,
    tier: str,
    clean_history: list[dict[str, Any]],
    variant_history: list[dict[str, Any]],
) -> str:
    value = {
        "parent_source_sha256": parent_source_sha256,
        "tier": tier,
        "clean_history": clean_history,
        "variant_history": variant_history,
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_tier_variant(
    source: dict[str, Any],
    tier: str,
    generated_turns: Sequence[dict[str, Any]],
    *,
    require_exact_action: bool = False,
) -> dict[str, Any]:
    turns = source_turns(source)
    states = [turn["state"] for turn in generated_turns]
    audit = automatic_tier_audit(
        tier,
        turns,
        states,
        require_exact_action=require_exact_action,
    )
    if not audit["passed"]:
        raise ValueError("cannot build a tier variant that failed its automatic audit")
    clean = copy.deepcopy(source["histories"]["clean"])
    variant = copy.deepcopy(clean)
    indices = [
        index
        for index, message in enumerate(variant)
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    if len(indices) != len(states):
        raise ValueError("generated tier does not match historical reasoning turns")
    for index, state in zip(indices, states):
        variant[index]["reasoning"] = state
    fork_audit = reasoning_only_fork_audit(clean, variant)
    if not fork_audit["passed"]:
        raise ValueError("tier variant changes fields outside historical reasoning")
    digest = tier_variant_hash(source["source_sha256"], tier, clean, variant)
    result = {
        "task_id": source["task_id"],
        "model_name": source["model_name"],
        "model": copy.deepcopy(source["model"]),
        "parent_source_sha256": source["source_sha256"],
        "tier": tier,
        "status": "pending_semantic_audit",
        "generated_turns": copy.deepcopy(list(generated_turns)),
        "automatic_audit": audit,
        "histories": {"clean": clean, "variant": variant},
        "fork_audit": fork_audit,
        "variant_sha256": digest,
    }
    if require_exact_action:
        result["require_exact_action"] = True
    return result


def verify_tier_variant(
    variant: dict[str, Any], source: dict[str, Any] | None = None
) -> None:
    turns = [turn["source"] for turn in variant["generated_turns"]]
    if source is not None and (
        variant["task_id"] != source["task_id"]
        or variant["model_name"] != source["model_name"]
        or variant["model"] != source["model"]
        or variant["parent_source_sha256"] != source["source_sha256"]
        or variant["histories"]["clean"] != source["histories"]["clean"]
        or turns != source_turns(source)
    ):
        raise RuntimeError("tier variant does not bind to its immutable parent source")
    states = [turn["state"] for turn in variant["generated_turns"]]
    require_exact_action = variant.get("require_exact_action", False)
    if not isinstance(require_exact_action, bool):
        raise RuntimeError("tier variant exact-action mode is invalid")
    audit = automatic_tier_audit(
        variant["tier"],
        turns,
        states,
        require_exact_action=require_exact_action,
    )
    if audit != variant.get("automatic_audit") or not audit["passed"]:
        raise RuntimeError("tier variant automatic audit does not recompute")
    for generated, turn_audit in zip(variant["generated_turns"], audit["turns"]):
        if generated.get("automatic_audit") != turn_audit:
            raise RuntimeError("stored turn audit does not recompute")
    fork = reasoning_only_fork_audit(
        variant["histories"]["clean"], variant["histories"]["variant"]
    )
    if fork != variant.get("fork_audit") or not fork["passed"]:
        raise RuntimeError("tier variant fork audit does not recompute")
    reasoning = [
        message["reasoning"]
        for message in variant["histories"]["variant"]
        if message.get("role") == "assistant" and message.get("reasoning")
    ]
    if reasoning != states:
        raise RuntimeError("tier states do not match the variant history")
    expected = tier_variant_hash(
        variant["parent_source_sha256"],
        variant["tier"],
        variant["histories"]["clean"],
        variant["histories"]["variant"],
    )
    if expected != variant.get("variant_sha256"):
        raise RuntimeError("tier variant hash mismatch")
