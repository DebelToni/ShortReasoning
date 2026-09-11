#!/usr/bin/env python3
import argparse
import copy
import decimal
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from short_reasoning import (  # noqa: E402
    IDENTIFIER_RE,
    hard_thresholds_in_reasoning,
    load_tasks,
    normalized_numeric_literals,
    reasoning_text,
    rewrite_fidelity,
)


def collect_candidate_ids(value):
    found = set()
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            found.add(value["id"].upper())
        for child in value.values():
            found.update(collect_candidate_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_candidate_ids(child))
    return found


def canonical_number(value):
    return format(decimal.Decimal(str(value)).normalize(), "f")


def first_fork_only_reasoning_differs(case):
    clean = copy.deepcopy(case["branches"]["clean"]["turns"][0]["request"])
    rewritten = copy.deepcopy(case["branches"]["rewritten"]["turns"][0]["request"])
    clean_reasoning = []
    rewritten_reasoning = []
    if len(clean["messages"]) != len(rewritten["messages"]):
        return False, 0
    for clean_message, rewritten_message in zip(clean["messages"], rewritten["messages"]):
        if clean_message.get("role") == "assistant":
            clean_reasoning.append(clean_message.pop("reasoning", None))
            rewritten_reasoning.append(rewritten_message.pop("reasoning", None))
    return clean == rewritten and clean_reasoning != rewritten_reasoning, len(clean_reasoning)


def target_systems_have_no_concision_instruction(case):
    banned = ("think concisely", "be concise", "concise", "be terse")
    requests = [record["request"] for record in case["shared"]]
    requests.extend(turn["request"] for branch in case["branches"].values() for turn in branch["turns"])
    for request in requests:
        for message in request["messages"]:
            if message.get("role") != "system":
                continue
            text = (message.get("content") or "").lower()
            if any(phrase in text for phrase in banned):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("runs", nargs="*", type=Path)
    args = parser.parse_args()

    tasks = {task["id"]: task for task in load_tasks(ROOT / "manifests" / "tasks" / "tasks.json")}
    records = []
    case_audits = []
    paths = []
    for run in args.runs:
        paths.extend(sorted((run.resolve() / "cases").glob("*.json")))
    if args.source_index:
        for item in json.loads(args.source_index.read_text()):
            path = Path(item["path"])
            paths.append(path if path.is_absolute() else ROOT / path)
    if not paths:
        raise SystemExit("provide at least one run or --source-index")
    for path in paths:
        case = json.loads(path.read_text())
        task = tasks[case["task_id"]]
        candidate_ids = collect_candidate_ids(task)
        hard_thresholds = {
            canonical_number(value)
            for value in task["phases"][0]["result"]["hard_constraints"].values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        only_reasoning, reasoning_count = first_fork_only_reasoning_differs(case)
        case_audits.append({
            "model": case["model_name"],
            "task": case["task_id"],
            "seed": case["base_seed"],
            "source": str(path.resolve()),
            "fork_request_only_assistant_reasoning_differs": only_reasoning,
            "fork_reasoning_strings_changed": reasoning_count,
            "target_system_has_no_concision_instruction": target_systems_have_no_concision_instruction(case),
            "clean_actions_correct": case["branches"]["clean"]["actions_correct"],
            "rewritten_actions_correct": case["branches"]["rewritten"]["actions_correct"],
        })
        for turn, shared in enumerate(case["shared"], start=1):
            message = shared["response"]["choices"][0]["message"]
            raw = reasoning_text(message)
            rewritten = shared["rewrite"]["rewritten"]
            generic = rewrite_fidelity(raw, rewritten)
            raw_ids = set(IDENTIFIER_RE.findall(raw.upper()))
            rewritten_ids = set(IDENTIFIER_RE.findall(rewritten.upper()))
            raw_candidates = sorted(raw_ids & candidate_ids)
            missing_candidates = [item for item in raw_candidates if item not in rewritten_ids]
            rewritten_numbers = set(normalized_numeric_literals(rewritten))
            raw_thresholds = hard_thresholds_in_reasoning(raw, hard_thresholds)
            missing_thresholds = [item for item in raw_thresholds if item not in rewritten_numbers]
            tool_call = message["tool_calls"][0]["function"]
            rewrite_user = json.loads(shared["rewrite"]["attempts"][0]["request"]["messages"][1]["content"])
            records.append({
                "model": case["model_name"],
                "task": case["task_id"],
                "seed": case["base_seed"],
                "shared_turn": turn,
                "raw_reasoning": raw,
                "rewritten_reasoning": rewritten,
                "raw_words": len(raw.split()),
                "rewritten_words": len(rewritten.split()),
                "requested_tool": copy.deepcopy(tool_call),
                "rewrite_input_matches_raw_and_tool": rewrite_user == {
                    "reasoning": raw,
                    "requested_tool": tool_call,
                },
                "raw_candidate_ids": raw_candidates,
                "missing_candidate_ids": missing_candidates,
                "all_candidate_ids_preserved": not missing_candidates,
                "raw_hard_thresholds": raw_thresholds,
                "missing_hard_thresholds": missing_thresholds,
                "all_hard_thresholds_preserved": not missing_thresholds,
                "conservative_all_identifiers_and_numbers": generic,
            })

    summary = {
        "cases": len(case_audits),
        "rewrites": len(records),
        "fork_request_invariants_passed": sum(item["fork_request_only_assistant_reasoning_differs"] for item in case_audits),
        "target_prompt_invariants_passed": sum(item["target_system_has_no_concision_instruction"] for item in case_audits),
        "rewrite_inputs_match_raw_and_tool": sum(item["rewrite_input_matches_raw_and_tool"] for item in records),
        "candidate_id_audits_passed": sum(item["all_candidate_ids_preserved"] for item in records),
        "hard_threshold_audits_passed": sum(item["all_hard_thresholds_preserved"] for item in records),
        "candidate_id_failures": [
            {key: item[key] for key in ("model", "task", "seed", "shared_turn", "missing_candidate_ids")}
            for item in records if not item["all_candidate_ids_preserved"]
        ],
        "hard_threshold_failures": [
            {key: item[key] for key in ("model", "task", "seed", "shared_turn", "missing_hard_thresholds")}
            for item in records if not item["all_hard_thresholds_preserved"]
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "cases": case_audits, "rewrites": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
