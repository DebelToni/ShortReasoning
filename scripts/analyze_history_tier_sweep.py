#!/usr/bin/env python3
"""Verify and summarize a completed five-arm history-tier sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_history_tiers as runner  # noqa: E402

DOSE_ORDER = (
    "clean",
    "relaxed_full_sentence_compact",
    "full_sentence_compact",
    "telegraphic_compact",
    "ultra_telegraphic",
)
ADJACENT = tuple(zip(DOSE_ORDER, DOSE_ORDER[1:]))
DEVELOPMENT_TASKS = {
    "hospital-decision-v2",
    "manufacturing-recall-v2",
    "wildfire-evacuation-v2",
}


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def clustered_median_ci(records: Sequence[dict[str, Any]], seed: int = 962001) -> list[float] | None:
    by_task: dict[str, list[float]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], []).append(float(record["value"]))
    tasks = sorted(by_task)
    if len(tasks) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(10_000):
        selected = [rng.choice(tasks) for _ in tasks]
        values = [value for task in selected for value in by_task[task]]
        samples.append(statistics.median(values))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for position in ordered[cursor:end]:
            ranks[position] = rank
        cursor = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return None if denominator == 0 else sum(a * b for a, b in zip(dx, dy)) / denominator


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def history_words(messages: Sequence[dict[str, Any]]) -> int:
    return sum(
        len(message.get("reasoning", "").split())
        for message in messages
        if message.get("role") == "assistant"
        and isinstance(message.get("reasoning"), str)
    )


def percent_change(reference: float, treatment: float) -> float | None:
    return None if reference == 0 else (treatment - reference) / reference * 100


def verify_and_load(
    run_dir: Path, partial_replicate_limit: int | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run = json.loads((run_dir / "run.json").read_text())
    complete = run.get("status") == "five_arm_outcomes_complete_pending_analysis"
    if partial_replicate_limit is None:
        if not complete:
            raise RuntimeError("tier sweep is not complete and frozen for analysis")
    elif (
        partial_replicate_limit not in {1, 2}
        or run.get("status")
        not in {
            "five_arm_outcomes_in_progress",
            "five_arm_outcomes_complete_pending_analysis",
        }
        or int(run.get("completed_replicate_limit", 0)) < partial_replicate_limit
    ):
        raise RuntimeError("partial analysis does not match a captured replicate prefix")
    runner._verify_frozen_file_manifest(run_dir / "blocks", run["block_manifest_sha256"])
    if complete:
        runner._verify_frozen_file_manifest(
            run_dir / "continuations", run["continuation_manifest_sha256"]
        )
        runner._verify_frozen_file_manifest(
            run_dir / "outcome-journals", run["outcome_journal_manifest_sha256"]
        )
    tasks = {
        task["id"]: task for task in json.loads((run_dir / "tasks.snapshot.json").read_text())
    }
    blocks = [json.loads(path.read_text()) for path in sorted((run_dir / "blocks").glob("*.json"))]
    block_by_key = {(block["model_name"], block["task_id"]): block for block in blocks}
    randomization = json.loads((run_dir / "randomization.json").read_text())
    random_by_key = {
        (entry["model_name"], entry["task_id"], entry["replicate"]): entry
        for entry in randomization["entries"]
    }
    all_continuation_paths = sorted((run_dir / "continuations").glob("*.json"))
    all_journal_paths = sorted((run_dir / "outcome-journals").glob("*.json"))
    replicate_limit = 3 if partial_replicate_limit is None else partial_replicate_limit
    available_limit = 3 if complete else int(run.get("completed_replicate_limit", 0))
    expected_available_names = {
        f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}.json"
        for block in blocks
        for replicate in range(1, available_limit + 1)
    }
    if (
        {path.name for path in all_continuation_paths} != expected_available_names
        or {path.name for path in all_journal_paths} != expected_available_names
    ):
        raise RuntimeError("available analysis captures are not an exact replicate prefix")
    expected_names = {
        f"{block['model_name']}__{block['task_id']}__rep-{replicate:02d}.json"
        for block in blocks
        for replicate in range(1, replicate_limit + 1)
    }
    continuation_paths = [
        path for path in all_continuation_paths if path.name in expected_names
    ]
    journal_paths = [path for path in all_journal_paths if path.name in expected_names]
    if (
        {path.name for path in continuation_paths} != expected_names
        or {path.name for path in journal_paths} != expected_names
    ):
        raise RuntimeError("analysis capture paths do not match the exact replicate subset")
    cases = []
    for path in continuation_paths:
        case = json.loads(path.read_text())
        key = (case["model_name"], case["task_id"])
        block = block_by_key[key]
        random_entry = random_by_key[(*key, case["replicate"])]
        journal_path = run_dir / "outcome-journals" / path.name
        journal = json.loads(journal_path.read_text())
        runner._validate_outcome_journal(
            journal, block, tasks[case["task_id"]], random_entry, case["replicate"]
        )
        expected = runner._capture_from_journal(journal, block, tasks[case["task_id"]])
        if {key: value for key, value in case.items() if key != "created_at"} != expected:
            raise RuntimeError(f"capture does not reconstruct: {path}")
        cases.append(case)
    return run, blocks, cases


def summarize(
    run_dir: Path, partial_replicate_limit: int | None = None
) -> dict[str, Any]:
    run, blocks, cases = verify_and_load(run_dir, partial_replicate_limit)
    block_by_key = {(block["model_name"], block["task_id"]): block for block in blocks}
    input_words = {
        (*key, arm): history_words(block["histories"][arm])
        for key, block in block_by_key.items()
        for arm in DOSE_ORDER
    }
    arm_rows = {}
    for arm in DOSE_ORDER:
        branches = [(case, case["branches"][arm]) for case in cases]
        complete = [(case, branch) for case, branch in branches if branch.get("status") == "complete"]
        reasoning = [branch["aggregate"]["reasoning_tokens"] for _, branch in complete]
        arm_rows[arm] = {
            "branches": len(branches),
            "complete_horizons": len(complete),
            "strict_successes": sum(branch["success"] for _, branch in branches),
            "action_failures": sum(branch.get("status") == "action_failure" for _, branch in branches),
            "reasoning_tokens_total_complete": sum(reasoning),
            "reasoning_tokens_median_complete": statistics.median(reasoning) if reasoning else None,
            "input_history_words_by_source": {
                task_id: input_words[("deepseek-v4-flash", task_id, arm)]
                for task_id in sorted({case["task_id"] for case in cases})
            },
        }
    contrasts = {}
    for reference, treatment in (("clean", arm) for arm in DOSE_ORDER[1:]):
        key = f"{reference}__to__{treatment}"
        records = []
        for case in cases:
            left = case["branches"][reference]
            right = case["branches"][treatment]
            if (
                left.get("status") == right.get("status") == "complete"
                and left["aggregate"]["turn_count"] == right["aggregate"]["turn_count"]
            ):
                value = percent_change(
                    left["aggregate"]["reasoning_tokens"],
                    right["aggregate"]["reasoning_tokens"],
                )
                if value is not None:
                    records.append({"task_id": case["task_id"], "value": value})
        values = [record["value"] for record in records]
        contrasts[key] = {
            "comparable": len(values),
            "shorter": sum(value < 0 for value in values),
            "median_percent_change": statistics.median(values) if values else None,
            "source_clustered_95_ci": clustered_median_ci(records),
            "values": records,
        }
    adjacent = {}
    for reference, treatment in ADJACENT:
        key = f"{reference}__to__{treatment}"
        records = []
        for case in cases:
            left = case["branches"][reference]
            right = case["branches"][treatment]
            if (
                left.get("status") == right.get("status") == "complete"
                and left["aggregate"]["turn_count"] == right["aggregate"]["turn_count"]
            ):
                value = percent_change(
                    left["aggregate"]["reasoning_tokens"],
                    right["aggregate"]["reasoning_tokens"],
                )
                if value is not None:
                    records.append({"task_id": case["task_id"], "value": value})
        values = [record["value"] for record in records]
        adjacent[key] = {
            "comparable": len(values),
            "shorter": sum(value < 0 for value in values),
            "median_percent_change": statistics.median(values) if values else None,
            "source_clustered_95_ci": clustered_median_ci(records, seed=963001 + len(adjacent)),
            "values": records,
        }
    correlations = []
    monotonic = 0
    for case in cases:
        complete = all(case["branches"][arm].get("status") == "complete" for arm in DOSE_ORDER)
        if not complete:
            continue
        words = [input_words[(case["model_name"], case["task_id"], arm)] for arm in DOSE_ORDER]
        generated = [case["branches"][arm]["aggregate"]["reasoning_tokens"] for arm in DOSE_ORDER]
        correlation = spearman(words, generated)
        correlations.append(
            {
                "task_id": case["task_id"],
                "replicate": case["replicate"],
                "input_words": dict(zip(DOSE_ORDER, words)),
                "reasoning_tokens": dict(zip(DOSE_ORDER, generated)),
                "spearman_input_words_vs_reasoning_tokens": correlation,
            }
        )
        if all(generated[index] >= generated[index + 1] for index in range(4)):
            monotonic += 1
    rho_values = [
        item["spearman_input_words_vs_reasoning_tokens"]
        for item in correlations
        if item["spearman_input_words_vs_reasoning_tokens"] is not None
    ]
    pooled_words = []
    pooled_reasoning = []
    for item in correlations:
        pooled_words.extend(item["input_words"][arm] for arm in DOSE_ORDER)
        pooled_reasoning.extend(item["reasoning_tokens"][arm] for arm in DOSE_ORDER)
    is_self_development = (
        run["snapshot"].get("protocol", {}).get("name")
        == "deepseek_self_development"
        and run["snapshot"].get("tier_mode") == "four_fresh"
        and run["snapshot"].get("selected_models") == ["deepseek-v4-flash"]
        and run["snapshot"].get("rewriter", {}).get("profile_name")
        in {"deepseek_self", "deepseek_self_examples"}
    )
    first_replicate_gate = None
    if (
        partial_replicate_limit == 1
        and is_self_development
        and set(run["snapshot"]["selected_tasks"]) == DEVELOPMENT_TASKS
    ):
        first_replicate_gate = runner.first_replicate_development_gate(cases)
    full_development_gate = None
    if (
        partial_replicate_limit is None
        and is_self_development
        and set(run["snapshot"]["selected_tasks"]) == DEVELOPMENT_TASKS
        and len(cases) == 9
    ):
        full_development_gate = runner.full_development_expansion_gate(
            cases, blocks
        )
    capture_names = {
        f"{case['model_name']}__{case['task_id']}__rep-{case['replicate']:02d}.json"
        for case in cases
    }
    analysis_input_sha256 = runner.canonical_hash(
        {
            "block_manifest_sha256": run["block_manifest_sha256"],
            "randomization_sha256": runner.file_sha256(
                run_dir / "randomization.json"
            ),
            "continuations": {
                name: runner.file_sha256(run_dir / "continuations" / name)
                for name in sorted(capture_names)
            },
            "outcome_journals": {
                name: runner.file_sha256(run_dir / "outcome-journals" / name)
                for name in sorted(capture_names)
            },
        }
    )
    protocol_name = run["snapshot"].get("protocol", {}).get("name")
    purpose = (
        "deepseek_self_rewrite_five_tier_analysis"
        if protocol_name in {
            "deepseek_self_development",
            "deepseek_self_expansion",
        }
        else "deepseek_five_tier_history_dose_analysis"
    )
    claim_scope = {
        "deepseek_self_development": (
            "exploratory development used for prompt selection and expansion gating"
        ),
        "deepseek_self_expansion": (
            "untouched seven-task confirmatory self-rewrite expansion"
        ),
    }.get(
        protocol_name,
        "retrospective reused-source tier-pattern evidence; not an unbiased primary-effect, pure-dose, compressor, or mechanism estimate",
    )
    return {
        "purpose": purpose,
        "analysis_input_sha256": analysis_input_sha256,
        "claim_scope": claim_scope,
        "run_status": (
            "partial_capture_verified"
            if partial_replicate_limit is not None
            else run["status"]
        ),
        "models": run["snapshot"]["selected_models"],
        "tasks": run["snapshot"]["selected_tasks"],
        "replicates": run["snapshot"]["replicates"],
        "captured_replicates": sorted({case["replicate"] for case in cases}),
        "partial_replicate_limit": partial_replicate_limit,
        "cases": len(cases),
        "dose_order_long_to_short": list(DOSE_ORDER),
        "arms": arm_rows,
        "clean_contrasts": contrasts,
        "adjacent_contrasts": adjacent,
        "dose_association": {
            "complete_five_arm_cases": len(correlations),
            "strictly_nonincreasing_reasoning_cases": monotonic,
            "median_spearman_input_words_vs_reasoning_tokens": (
                statistics.median(rho_values) if rho_values else None
            ),
            "pooled_spearman_input_words_vs_reasoning_tokens": spearman(
                pooled_words, pooled_reasoning
            ),
            "records": correlations,
        },
        "first_replicate_development_gate": first_replicate_gate,
        "full_development_expansion_gate": full_development_gate,
    }


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1f}%"


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek five-tier history sweep",
        "",
        f"Cases: {summary['cases']} ({len(summary['tasks'])} sources × "
        f"{len(summary['captured_replicates'])} captured replicates).",
        "",
        "## Arm outcomes",
        "",
        "| Arm | Complete | Strict success | Reasoning-token median | Complete pooled tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in DOSE_ORDER:
        row = summary["arms"][arm]
        lines.append(
            f"| {arm} | {row['complete_horizons']}/{row['branches']} | "
            f"{row['strict_successes']}/{row['branches']} | "
            f"{row['reasoning_tokens_median_complete']} | "
            f"{row['reasoning_tokens_total_complete']} |"
        )
    lines += ["", "## Versus clean", ""]
    for key, row in summary["clean_contrasts"].items():
        lines.append(
            f"- `{key}`: {row['shorter']}/{row['comparable']} shorter; median "
            f"{format_percent(row['median_percent_change'])}; clustered 95% CI "
            f"{row['source_clustered_95_ci']}."
        )
    dose = summary["dose_association"]
    gate = summary.get("first_replicate_development_gate")
    if gate is not None:
        lines += [
            "",
            "## Prespecified replicate-1 development gate",
            "",
            f"- Decision: {'PASS' if gate['passed'] else 'STOP/USE FALLBACK'}.",
            f"- Strict successes: {gate['strict_successes']}/{gate['branches']}.",
            f"- Nonclean-versus-clean shorter: {gate['nonclean_shorter']}/"
            f"{gate['nonclean_comparisons']}.",
        ]
    lines += [
        "",
        "## Dose ordering",
        "",
        f"- Nonincreasing generated reasoning: {dose['strictly_nonincreasing_reasoning_cases']}/"
        f"{dose['complete_five_arm_cases']} complete five-arm cases.",
        f"- Median within-case Spearman(input history words, generated reasoning tokens): "
        f"{dose['median_spearman_input_words_vs_reasoning_tokens']}.",
        "",
        f"Scope: {summary['claim_scope']}. Quality counts and intervals must accompany every token comparison.",
    ]
    return "\n".join(lines) + "\n"


def write_identical_or_new(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to replace differing analysis: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--partial-replicate-limit", type=int, choices=(1, 2))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = summarize(run_dir, args.partial_replicate_limit)
    suffix = (
        "" if args.partial_replicate_limit is None
        else f"-rep-{args.partial_replicate_limit:02d}"
    )
    analysis_name = f"analysis{suffix}.json"
    report_name = f"report{suffix}.md"
    manifest_name = f"ANALYSIS{suffix.upper()}_SHA256SUMS"
    analysis_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode()
    report_bytes = report(summary).encode()
    write_identical_or_new(run_dir / analysis_name, analysis_bytes)
    write_identical_or_new(run_dir / report_name, report_bytes)
    manifest = (
        f"{hashlib.sha256(analysis_bytes).hexdigest()}  {analysis_name}\n"
        f"{hashlib.sha256(report_bytes).hexdigest()}  {report_name}\n"
    ).encode()
    write_identical_or_new(run_dir / manifest_name, manifest)
    print(run_dir / report_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
