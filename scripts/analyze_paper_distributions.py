#!/usr/bin/env python3
"""Build distribution statistics requested for the final paper tables."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/20260730-paper-distribution-statistics-v1"
SOURCES = {
    "controlled": ROOT / "results/20260724-v2-12-task-two-route-screen/analysis.json",
    "controls": ROOT / "results/20260725-frozen-history-controls/analysis.json",
    "code": ROOT / "results/20260724-livecodebench-v6-12-item-screen/analysis.json",
    "moderate": ROOT / "results/20260730-deepseek-moderate-dose-analysis-dev-v2/analysis.json",
}
BOOTSTRAPS = 10_000
SEED = 20260730


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    cluster: Callable[[dict[str, Any]], str],
) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[cluster(row)].append(row)
    keys = sorted(groups)
    rng = random.Random(SEED + len(rows) + len(keys))
    medians: list[float] = []
    means: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample: list[float] = []
        for key in (rng.choice(keys) for _ in keys):
            sample.extend(value(row) for row in groups[key])
        medians.append(float(statistics.median(sample)))
        means.append(float(statistics.mean(sample)))
    return {
        "median_95_interval": [
            round(percentile(medians, 0.025), 1),
            round(percentile(medians, 0.975), 1),
        ],
        "mean_95_interval": [
            round(percentile(means, 0.025), 1),
            round(percentile(means, 0.975), 1),
        ],
    }


def summarize(
    name: str,
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    cluster: Callable[[dict[str, Any]], str],
    status: str,
) -> dict[str, Any]:
    values = [value(row) for row in rows]
    if len(values) < 2:
        raise RuntimeError(f"{name} has too few values")
    return {
        "name": name,
        "status": status,
        "pairs": len(values),
        "clusters": len({cluster(row) for row in rows}),
        "mean_percent": round(float(statistics.mean(values)), 1),
        "sample_variance_percent_squared": round(float(statistics.variance(values)), 1),
        "sample_standard_deviation_percent": round(float(statistics.stdev(values)), 1),
        "median_percent": round(float(statistics.median(values)), 1),
        "q1_percent": round(percentile(values, 0.25), 1),
        "q3_percent": round(percentile(values, 0.75), 1),
        "min_percent": round(min(values), 1),
        "max_percent": round(max(values), 1),
        "clustered_bootstrap": bootstrap(rows, value, cluster),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    data = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    rows: list[dict[str, Any]] = []

    controlled = [row for row in data["controlled"]["cases"] if row["comparable_horizon"]]
    controlled_value = lambda row: float(row["reasoning_token_change_percent"])
    source_cluster = lambda row: f"{row['model_name']}::{row['source_sha256']}"
    rows.append(
        summarize(
            "Controlled / combined",
            controlled,
            controlled_value,
            source_cluster,
            "measured_exploratory",
        )
    )
    for model, label in (
        ("deepseek-v4-flash", "Controlled / DeepSeek"),
        ("laguna-s-2.1", "Controlled / Laguna"),
    ):
        rows.append(
            summarize(
                label,
                [row for row in controlled if row["model_name"] == model],
                controlled_value,
                source_cluster,
                "measured_exploratory",
            )
        )

    code = [row for row in data["code"]["cases"] if row["comparable_horizon"]]
    code_value = lambda row: float(row["reasoning_change_percent"])
    rows.append(
        summarize(
            "Contest code / combined",
            code,
            code_value,
            source_cluster,
            "measured_exploratory_acquisition_selected",
        )
    )

    controls = data["controls"]["cases"]
    for variant, label in (
        ("verbose_paraphrase", "Length-matched paraphrase"),
        ("compact_full_sentence", "Full-sentence vs telegraphic compact"),
    ):
        selected = [
            row
            for row in controls
            if row["control"] == variant and row["comparable_horizon"]
        ]
        rows.append(
            summarize(
                label,
                selected,
                controlled_value,
                lambda row: f"{row['model_name']}::{row['parent_source_sha256']}",
                "measured_exploratory_control",
            )
        )

    moderate = data["moderate"]["rows"]
    for arm, dose in (
        ("moderate_25", "Moderate dose / 25%"),
        ("moderate_33", "Moderate dose / 33%"),
        ("moderate_42", "Moderate dose / 42%"),
        ("moderate_50", "Moderate dose / 50%"),
    ):
        selected = [
            row
            for row in moderate
            if row["arm"] == arm
            and row["comparable_five_cycle_horizon"]
            and row["reasoning_change_percent"] is not None
        ]
        rows.append(
            summarize(
                dose,
                selected,
                lambda row: float(row["reasoning_change_percent"]),
                lambda row: row["task_id"],
                "development_non_evidence",
            )
        )

    analysis = {
        "schema_version": 1,
        "status": "complete",
        "metric": "paired percent change in cumulative future provider-reported reasoning tokens",
        "rows": rows,
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in SOURCES.values()
        },
        "interpretation": "Means and variances expose heavy right tails hidden by favorable medians. They are descriptive complements, not replacements for paired medians and task/source-clustered intervals.",
        "limitations": [
            "Percentage changes can be unstable when the clean denominator is small.",
            "Replicates from one frozen source are correlated; cluster counts are reported explicitly.",
            "The moderate-dose intervals have only three task clusters.",
            "Rows use different task and horizon contracts and must not be pooled across sections.",
        ],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "files": {
            "analysis.json": hashlib.sha256(
                (OUTPUT / "analysis.json").read_bytes()
            ).hexdigest()
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "complete", "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
