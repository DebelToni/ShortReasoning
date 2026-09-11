#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

TERMINAL = {"LimitsExceeded", "RepeatedFormatError", "Submitted", "TimeExceeded"}


def trajectory_counts(path):
    trajectory = json.loads(path.read_text())
    assistants = [m for m in trajectory["messages"] if m.get("role") == "assistant"]
    usages = [
        ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
        for m in assistants
    ]
    reasoning_tokens = [
        int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        for usage in usages
    ]
    return {
        "target_responses": len(assistants),
        "responses_with_usage": sum(bool(usage) for usage in usages),
        "reasoning_tokens": sum(reasoning_tokens),
        "completion_tokens": sum(int(usage.get("completion_tokens") or 0) for usage in usages),
        "prompt_tokens": sum(int(usage.get("prompt_tokens") or 0) for usage in usages),
        "tool_calls": sum(len(m.get("tool_calls") or []) for m in assistants),
        "tool_results": sum(m.get("role") == "tool" for m in trajectory["messages"]),
        "exit_status": trajectory.get("info", {}).get("exit_status"),
    }


def summarize(name, paths):
    tasks = {}
    for instance, path in sorted(paths.items()):
        tasks[instance] = trajectory_counts(path)
    tool_calls = [row["tool_calls"] for row in tasks.values()]
    return {
        "arm": name,
        "tasks": len(tasks),
        "target_responses": sum(row["target_responses"] for row in tasks.values()),
        "responses_with_usage": sum(row["responses_with_usage"] for row in tasks.values()),
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in tasks.values()),
        "completion_tokens": sum(row["completion_tokens"] for row in tasks.values()),
        "prompt_tokens": sum(row["prompt_tokens"] for row in tasks.values()),
        "tool_calls": sum(tool_calls),
        "tool_results": sum(row["tool_results"] for row in tasks.values()),
        "median_tool_calls_per_task": median(tool_calls),
        "min_tool_calls": min(tool_calls),
        "max_tool_calls": max(tool_calls),
        "exit_statuses": dict(sorted(Counter(row["exit_status"] for row in tasks.values()).items())),
        "task_detail": tasks,
    }


def selected_continuous_paths(root):
    progress = json.loads((root / "progress.json").read_text())
    paths = {}
    for instance, attempts in progress["attempts"].items():
        selected = next(
            row
            for row in attempts
            if row.get("exit_status") in TERMINAL and row.get("trajectory")
        )
        paths[instance] = root / selected["trajectory"]
    assert len(paths) == 50
    return paths


def deepseek_normal_paths(original, recovery, recovery_status):
    paths = {}
    for row in json.loads((original / "item-statuses.json").read_text()):
        if row["trajectory_status"] == "Submitted":
            matches = list((original / "inference" / row["instance_id"]).glob("*.traj.json"))
            assert len(matches) == 1
            paths[row["instance_id"]] = matches[0]
    assert len(paths) == 44
    for row in json.loads(recovery_status.read_text()):
        instance = row["instance_id"]
        attempt = row["completed_attempt"]
        matches = list(
            (recovery / "attempts" / f"attempt-{attempt:03d}" / "inference" / instance).glob("*.traj.json")
        )
        assert len(matches) == 1
        paths[instance] = matches[0]
    assert len(paths) == 50
    return paths


def comparison(treatment, normal):
    shared = sorted(set(normal["task_detail"]) & set(treatment["task_detail"]))
    differences = [
        treatment["task_detail"][key]["tool_calls"] - normal["task_detail"][key]["tool_calls"]
        for key in shared
    ]
    changes = [
        100 * (
            treatment["task_detail"][key]["tool_calls"]
            / normal["task_detail"][key]["tool_calls"]
            - 1
        )
        for key in shared
    ]
    return {
        "tasks": len(shared),
        "aggregate_tool_call_change_percent": 100 * (treatment["tool_calls"] / normal["tool_calls"] - 1),
        "aggregate_target_response_change_percent": 100 * (treatment["target_responses"] / normal["target_responses"] - 1),
        "aggregate_reasoning_token_change_percent": 100 * (treatment["reasoning_tokens"] / normal["reasoning_tokens"] - 1),
        "reasoning_tokens_per_response": {
            "normal": normal["reasoning_tokens"] / normal["target_responses"],
            "treatment": treatment["reasoning_tokens"] / treatment["target_responses"],
            "change_percent": 100 * (
                (treatment["reasoning_tokens"] / treatment["target_responses"])
                / (normal["reasoning_tokens"] / normal["target_responses"])
                - 1
            ),
        },
        "median_paired_tool_call_difference": median(differences),
        "median_paired_tool_call_change_percent": median(changes),
        "tasks_with_fewer_equal_more_calls": [
            sum(value < 0 for value in differences),
            sum(value == 0 for value in differences),
            sum(value > 0 for value in differences),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-normal-original", type=Path, required=True)
    parser.add_argument("--deepseek-normal-recovery", type=Path, required=True)
    parser.add_argument("--deepseek-recovery-status", type=Path, required=True)
    parser.add_argument("--deepseek-matrix-root", type=Path, required=True)
    parser.add_argument("--minimax-matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = {}
    arms["deepseek-v4-flash__normal"] = summarize(
        "deepseek-v4-flash__normal",
        deepseek_normal_paths(
            args.deepseek_normal_original,
            args.deepseek_normal_recovery,
            args.deepseek_recovery_status,
        ),
    )
    for treatment in ("luna-compact", "self-compact"):
        name = f"deepseek-v4-flash__{treatment}"
        arms[name] = summarize(name, selected_continuous_paths(args.deepseek_matrix_root / name))
    for treatment in ("normal", "luna-compact", "self-compact"):
        name = f"minimax-m3__{treatment}"
        arms[name] = summarize(name, selected_continuous_paths(args.minimax_matrix_root / name))

    comparisons = {}
    for model in ("deepseek-v4-flash", "minimax-m3"):
        normal = arms[f"{model}__normal"]
        for treatment in ("luna-compact", "self-compact"):
            name = f"{model}__{treatment}"
            comparisons[f"{name}_vs_normal"] = comparison(arms[name], normal)

    output = {"metric_contract": "Assistant-issued tool calls in the one selected terminal trajectory per task.", "arms": arms, "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
