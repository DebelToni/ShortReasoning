#!/usr/bin/env python3
"""Derive continuation and end-to-end latency/cost views for the paper."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/20260724-v2-12-task-two-route-screen"
OUTPUT = ROOT / "results/20260730-paper-latency-cost-analysis-v1"
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


def percent(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline * 100


def clustered_interval(rows: list[dict[str, Any]], field: str) -> list[float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[row["source_key"]].append(float(row[field]))
    keys = sorted(clusters)
    rng = random.Random(SEED + sum(ord(char) for char in field))
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample: list[float] = []
        for key in (rng.choice(keys) for _ in keys):
            sample.extend(clusters[key])
        estimates.append(float(statistics.median(sample)))
    return [round(percentile(estimates, 0.025), 1), round(percentile(estimates, 0.975), 1)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def median(field: str) -> float:
        return round(float(statistics.median(row[field] for row in rows)), 3)

    return {
        "cases": len(rows),
        "sources": len({row["source_key"] for row in rows}),
        "continuation_only": {
            "faster_cases": sum(row["continuation_wall_delta_seconds"] < 0 for row in rows),
            "median_wall_change_percent": median("continuation_wall_change_percent"),
            "source_clustered_median_95_interval": clustered_interval(
                rows, "continuation_wall_change_percent"
            ),
            "median_wall_delta_seconds": median("continuation_wall_delta_seconds"),
            "pooled_wall_change_percent": round(
                percent(
                    sum(row["rewritten_wall_seconds"] for row in rows),
                    sum(row["clean_wall_seconds"] for row in rows),
                ),
                1,
            ),
            "median_cost_change_percent": median("continuation_cost_change_percent"),
            "pooled_cost_change_percent": round(
                percent(
                    sum(row["rewritten_cost_usd"] for row in rows),
                    sum(row["clean_cost_usd"] for row in rows),
                ),
                1,
            ),
        },
        "one_use_end_to_end": {
            "faster_cases": sum(row["one_use_wall_delta_seconds"] < 0 for row in rows),
            "median_wall_change_percent": median("one_use_wall_change_percent"),
            "median_wall_delta_seconds": median("one_use_wall_delta_seconds"),
            "cheaper_cases": sum(row["one_use_cost_delta_usd"] < 0 for row in rows),
            "median_cost_change_percent": median("one_use_cost_change_percent"),
            "median_cost_delta_usd": round(
                float(statistics.median(row["one_use_cost_delta_usd"] for row in rows)),
                6,
            ),
        },
        "three_replicate_amortized": {
            "faster_cases": sum(row["amortized_wall_delta_seconds"] < 0 for row in rows),
            "median_wall_change_percent": median("amortized_wall_change_percent"),
            "median_wall_delta_seconds": median("amortized_wall_delta_seconds"),
            "cheaper_cases": sum(row["amortized_cost_delta_usd"] < 0 for row in rows),
            "median_cost_change_percent": median("amortized_cost_change_percent"),
            "median_cost_delta_usd": round(
                float(statistics.median(row["amortized_cost_delta_usd"] for row in rows)),
                6,
            ),
        },
        "rewrite_preprocessing": {
            "median_wall_seconds_per_source": median("rewrite_wall_seconds"),
            "median_cost_usd_per_source": round(
                float(statistics.median(row["rewrite_cost_usd"] for row in rows)), 6
            ),
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    source_files = sorted((SOURCE / "sources").glob("*.json"))
    for path in source_files:
        value = json.loads(path.read_text())
        key = (value["model_name"], value["task_id"])
        rewrite_wall = sum(float(turn["rewrite"]["wall_seconds"]) for turn in value["shared"])
        rewrite_cost = sum(
            float(attempt["response"]["usage"]["cost"])
            for turn in value["shared"]
            for attempt in turn["rewrite"]["attempts"]
        )
        sources[key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rewrite_wall_seconds": rewrite_wall,
            "rewrite_cost_usd": rewrite_cost,
        }
    if len(sources) != 24:
        raise RuntimeError(f"expected 24 frozen sources, found {len(sources)}")

    rows: list[dict[str, Any]] = []
    continuation_files = sorted((SOURCE / "continuations").glob("*.json"))
    for path in continuation_files:
        value = json.loads(path.read_text())
        if not value["pair"]["comparable_horizon"]:
            continue
        source = sources[(value["model_name"], value["task_id"])]
        clean = value["branches"]["clean"]["aggregate"]
        rewritten = value["branches"]["rewritten"]["aggregate"]
        rewrite_wall = source["rewrite_wall_seconds"]
        rewrite_cost = source["rewrite_cost_usd"]
        clean_wall = float(clean["wall_seconds"])
        rewritten_wall = float(rewritten["wall_seconds"])
        clean_cost = float(clean["cost"])
        rewritten_cost = float(rewritten["cost"])
        row = {
            "case": path.stem,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_name": value["model_name"],
            "task_id": value["task_id"],
            "source_key": f"{value['model_name']}--{value['task_id']}",
            "replicate": value["replicate"],
            "reasoning_change_percent": value["pair"]["reasoning_token_change_percent"],
            "clean_wall_seconds": clean_wall,
            "rewritten_wall_seconds": rewritten_wall,
            "continuation_wall_delta_seconds": rewritten_wall - clean_wall,
            "continuation_wall_change_percent": percent(rewritten_wall, clean_wall),
            "rewrite_wall_seconds": rewrite_wall,
            "one_use_wall_delta_seconds": rewritten_wall + rewrite_wall - clean_wall,
            "one_use_wall_change_percent": percent(rewritten_wall + rewrite_wall, clean_wall),
            "amortized_wall_delta_seconds": rewritten_wall + rewrite_wall / 3 - clean_wall,
            "amortized_wall_change_percent": percent(
                rewritten_wall + rewrite_wall / 3, clean_wall
            ),
            "clean_cost_usd": clean_cost,
            "rewritten_cost_usd": rewritten_cost,
            "continuation_cost_change_percent": percent(rewritten_cost, clean_cost),
            "rewrite_cost_usd": rewrite_cost,
            "one_use_cost_delta_usd": rewritten_cost + rewrite_cost - clean_cost,
            "one_use_cost_change_percent": percent(
                rewritten_cost + rewrite_cost, clean_cost
            ),
            "amortized_cost_delta_usd": rewritten_cost + rewrite_cost / 3 - clean_cost,
            "amortized_cost_change_percent": percent(
                rewritten_cost + rewrite_cost / 3, clean_cost
            ),
        }
        rows.append(row)
    if len(rows) != 69:
        raise RuntimeError(f"expected 69 comparable cases, found {len(rows)}")

    analysis = {
        "schema_version": 1,
        "status": "complete",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_analysis_sha256": hashlib.sha256(
            (SOURCE / "analysis.json").read_bytes()
        ).hexdigest(),
        "overall": summarize(rows),
        "by_model": {
            model: summarize([row for row in rows if row["model_name"] == model])
            for model in sorted({row["model_name"] for row in rows})
        },
        "rows": rows,
        "interpretation": "Compact histories reduced hosted continuation wall time in most comparable cases, but one-use preprocessing made the median end-to-end trajectory slower and much more expensive. Reusing one frozen rewrite across three endpoint replicates amortized latency enough for a small median speedup, but never amortized monetary cost.",
        "limitations": [
            "Wall time includes network and provider queue variation, not only inference.",
            "The source rewriter ran offline before continuation and could overlap other work; one-use addition is a serial end-to-end counterfactual.",
            "Three-replicate amortization describes this experimental design, not a typical single-use deployment.",
            "Latency was not the prespecified primary outcome and provider capacity may vary over time.",
        ],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    files = {
        "analysis.json": hashlib.sha256((OUTPUT / "analysis.json").read_bytes()).hexdigest()
    }
    manifest = {"schema_version": 1, "files": files}
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "complete", "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
