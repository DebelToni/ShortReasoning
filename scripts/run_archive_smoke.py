#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"configs", "environments", "figures", "manifests", "results", "scripts", "src", "tests"}


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def check_manifest() -> int:
    manifest = load("archive_manifest.json")
    entries = manifest["entries"]
    listed = {row["path"] for row in entries}
    if len(listed) != len(entries) or manifest["files"] != len(entries):
        raise AssertionError("manifest count or uniqueness failure")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "archive_manifest.json" and "__pycache__" not in path.parts and ".git" not in path.parts
    }
    if actual != listed:
        raise AssertionError(f"manifest coverage mismatch: missing={sorted(listed-actual)[:3]}, extra={sorted(actual-listed)[:3]}")
    total = 0
    for row in entries:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError(f"unsafe manifest path: {relative}")
        size, observed = digest(ROOT / relative)
        total += size
        if size != row["bytes"] or observed != row["sha256"]:
            raise AssertionError(f"manifest mismatch: {relative}")
    if total != manifest["bytes"]:
        raise AssertionError("manifest byte total mismatch")
    return len(entries)


def check_structure() -> None:
    roots = {path.name for path in ROOT.iterdir() if path.is_dir() and path.name != ".git"}
    if roots != REQUIRED:
        raise AssertionError(f"unexpected root directories: {sorted(roots ^ REQUIRED)}")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_symlink():
            raise AssertionError(f"symlink: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in {".tex", ".bib", ".sty", ".bst", ".latex"}:
            raise AssertionError(f"LaTeX file: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() == ".md" and path.name != "README.md":
            raise AssertionError(f"non-README Markdown: {path.relative_to(ROOT)}")


def check_metrics() -> None:
    camera = load("results/20260910-camera-ready-controlled-audit-v2/analysis.json")
    headline = camera["headline_verification"]
    assert (headline["all_assigned_pairs"], headline["comparable_pairs"], headline["compact_shorter_comparable_pairs"]) == (174, 172, 151)
    assert (headline["clean_exact_success"], headline["compact_exact_success"]) == (165, 166)
    assert camera["cohort_accounting"]["attempted_route_task_sources"] == 72
    assert camera["cohort_accounting"]["accepted_sources"] == 58
    assert camera["causal_timing_audit"]["first_request_reasoning_only_invariant"] == 174

    controlled = load("results/20260804-six-model-common-slice-infrastructure-corrected-v2/analysis.json")
    overall = controlled["corrected_primary_overall"]
    assert (overall["cases"], overall["comparable"], overall["shorter"]) == (72, 72, 62)
    assert abs(overall["median_change_percent"] - -39.24057252136497) < 1e-12
    rows = {row["model_name"]: row for row in controlled["table1_rows"]}
    expected = {
        "deepseek-v4-flash": (36, 30, -29.486037848713906),
        "laguna-s-2.1": (36, 32, -46.17020609083218),
        "minimax-m3": (30, 25, -27.30061023580192),
        "glm-4.7-flash": (13, 13, -50.52816901408451),
        "glm-5.1": (30, 28, -20.156966833088358),
        "deepseek-v4-pro": (27, 23, -32.59668508287293),
    }
    for model, values in expected.items():
        assert (rows[model]["comparable"], rows[model]["shorter"]) == values[:2]
        assert abs(rows[model]["median_change_percent"] - values[2]) < 1e-12

    controls = load("results/20260725-frozen-history-controls/analysis.json")["controls"]
    verbose = controls["verbose_paraphrase"]["overall"]
    full = controls["compact_full_sentence"]["overall"]
    assert (verbose["comparable_horizon_cases"], verbose["reasoning_reduced_cases"]) == (65, 31)
    assert abs(verbose["median_reasoning_token_change_percent"] - 0.6751687921980495) < 1e-12
    assert (full["comparable_horizon_cases"], full["reasoning_reduced_cases"]) == (60, 33)
    assert abs(full["median_reasoning_token_change_percent"] - -4.689605363860773) < 1e-12

    code = load("results/20260724-livecodebench-v6-12-item-screen/analysis.json")["overall"]
    assert (code["cases"], code["unique_sources"], code["comparable_cases"], code["reasoning_reduced_cases"]) == (36, 12, 33, 29)
    assert abs(code["median_reasoning_change_percent"] - -57.14285714285714) < 1e-12

    style = load("results/20260726-deepseek-five-tier-luna-v6-manual-style-adjudication/analysis.json")
    sol = load("results/20260726-deepseek-five-tier-sol-anchor-dev-manual-style/analysis.json")
    assert style["dose_association"]["complete_five_arm_cases"] == 30
    assert style["dose_association"]["strictly_nonincreasing_reasoning_cases"] == 6
    assert sol["dose_association"]["complete_five_arm_cases"] == 9

    swe = load("results/20260803-swebench-verified-mini-continuous-compaction-interim-v1/analysis.json")
    assert swe["tasks_per_arm"] == 50 and swe["status"] == "interim_completed_arms_only"
    assert swe["arms"]["deepseek-v4-flash__normal"]["official_resolved"] == 32
    assert swe["arms"]["deepseek-v4-flash__luna-compact"]["official_resolved"] == 32
    assert swe["arms"]["deepseek-v4-flash__self-compact"]["official_resolved"] == 29
    assert swe["arms"]["minimax-m3__luna-compact"]["official_resolved"] == 31
    assert swe["arms"]["minimax-m3__self-compact"]["official_resolved"] == 30

    latency = load("results/20260730-paper-latency-cost-analysis-v1/analysis.json")["overall"]
    assert latency["cases"] == 69 and latency["continuation_only"]["faster_cases"] == 51
    assert abs(latency["continuation_only"]["median_wall_change_percent"] - -25.931) < 1e-9

    deployment = load("results/20260910-camera-ready-deployment-audit-v2/analysis.json")
    assert deployment["deployment"]["accuracy_change_range_percentage_points"] == [-6.0, 6.0]
    assert len(deployment["deployment"]["models"]) == 4
    assert deployment["hosted_matched_amortization"]["sources"] == 24
    assert deployment["b300_latency"]["original_10"]["rows"] == 10
    assert deployment["b300_latency"]["overload_primary_23"]["rows"] == 23
    assert deployment["b300_latency"]["overload_retrospective_25"]["rows"] == 25
    inputs = load("results/20260910-camera-ready-deployment-audit-v2/inputs.json")
    assert inputs["ledger"]["records"] == 77228
    assert deployment["response_ledger_sha256"] == inputs["ledger"]["sha256"]
    expected_spend = {
        "kimi-k2.6": (48.836254107, 45.1071252514, 33.2587870974),
        "nemotron-3-ultra": (23.8236573, 38.3348833, 49.9521061),
    }
    for model, totals in expected_spend.items():
        arms = deployment["deployment"]["models"][model]["arms"]
        observed = tuple(arms[arm]["all_acquisition_spend"]["total_usd"] for arm in ("normal", "luna-compact", "self-compact"))
        assert all(abs(left - right) < 1e-9 for left, right in zip(observed, totals))


def check_figure_data() -> int:
    path = ROOT / "figures/source-data/controlled-effect-distributions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 174:
        raise AssertionError("controlled figure row count changed")
    expected = {
        row["model_name"]: row["median_change_percent"]
        for row in load("results/20260804-six-model-common-slice-infrastructure-corrected-v2/analysis.json")["table1_rows"]
    }
    for model, target in expected.items():
        values = sorted(
            float(row["reasoning_change_percent"])
            for row in rows
            if row["model_name"] == model and row["comparable_horizon"] == "True"
        )
        middle = len(values) // 2
        observed = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        if abs(observed - target) > 1e-12:
            raise AssertionError(f"figure data median mismatch: {model}")
    active = load("figures/source-data/active-figure-values.json")
    for relative, expected_hash in active["input_sha256"].items():
        if digest(ROOT / relative)[1] != expected_hash:
            raise AssertionError(f"active figure input changed: {relative}")
    reductions = [row["reduction_percent"] for row in active["turn_reductions"]]
    if [round(value, 1) for value in reductions] != [8.8, 34.4, 56.1]:
        raise AssertionError("controlled-summary turn values changed")
    length = active["length_quality"]
    if [round(row["reasoning_change_percent"], 1) for row in length] != [-39.2, 0.7]:
        raise AssertionError("length-quality values changed")
    expected_pdfs = {
        "paired-fork-protocol.pdf", "controlled-summary.pdf",
        "controlled-effect-raincloud.pdf", "length-quality-control.pdf",
    }
    observed_pdfs = {path.name for path in (ROOT / "figures").glob("*.pdf")}
    if observed_pdfs != expected_pdfs:
        raise AssertionError(f"active figure set mismatch: {sorted(observed_pdfs ^ expected_pdfs)}")
    return 2


def check_json_and_python() -> tuple[int, int]:
    json_count = 0
    python_count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            json_count += 1
        elif path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
            python_count += 1
    return json_count, python_count


def is_json_subset(candidate, reference) -> bool:
    if isinstance(reference, dict):
        return isinstance(candidate, dict) and all(
            key in candidate and is_json_subset(candidate[key], value)
            for key, value in reference.items()
        )
    if isinstance(reference, list):
        return (
            isinstance(candidate, list)
            and len(candidate) == len(reference)
            and all(is_json_subset(left, right) for left, right in zip(candidate, reference))
        )
    return candidate == reference


def run_checked(command: list[str]) -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    subprocess.run(
        command, cwd=ROOT, env=environment, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def check_response_ledger() -> int:
    result = "results/20260910-camera-ready-deployment-audit-v2"
    inputs = load(f"{result}/inputs.json")
    ledger_path = ROOT / result / "response-ledger.jsonl.gz"
    expected = inputs["ledger"]
    size, observed_hash = digest(ledger_path)
    if observed_hash != expected["sha256"] or size <= 0:
        raise AssertionError("response ledger compressed-byte binding differs")
    records = 0
    kinds: dict[str, int] = {}
    with gzip.open(ledger_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AssertionError("response ledger row is not an object")
            records += 1
            kind = row.get("kind")
            kinds[kind] = kinds.get(kind, 0) + 1
            response_hash = row.get("response_id_sha256")
            if response_hash is not None and (
                not isinstance(response_hash, str) or len(response_hash) != 64
                or any(character not in "0123456789abcdef" for character in response_hash)
            ):
                raise AssertionError("response ledger contains a malformed hashed ID")
    if records != expected["records"] or records != 77228:
        raise AssertionError(f"response ledger row count differs: {records}")
    if kinds != {"model_response": 77021, "error_body": 207}:
        raise AssertionError(f"response ledger kind counts differ: {kinds}")
    return records


def replay_analyses(work_dir: Path) -> dict[str, str]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    controlled_out = work_dir / "controlled"
    run_checked([
        sys.executable, "scripts/analyze_camera_ready_controlled.py", "--root", ".",
        "--input", "results/20260804-six-model-common-slice-infrastructure-corrected-v2",
        "--output", str(controlled_out),
    ])
    controlled_replay = json.loads((controlled_out / "analysis.json").read_text())
    if controlled_replay != load("results/20260910-camera-ready-controlled-audit-v2/analysis.json"):
        raise AssertionError("camera-ready controlled analysis replay differs")

    deployment_result = "results/20260910-camera-ready-deployment-audit-v2"
    central = ROOT / "scripts/analyze_camera_ready_deployment.py"
    detached = ROOT / deployment_result / "analyze.py"
    if central.read_bytes() != detached.read_bytes():
        raise AssertionError("detached deployment analyzer differs from canonical analyzer")
    deployment_out = work_dir / "deployment.json"
    run_checked([
        sys.executable, f"{deployment_result}/analyze.py", "analyze",
        "--input", f"{deployment_result}/inputs.json",
        "--ledger", f"{deployment_result}/response-ledger.jsonl.gz",
        "--output", str(deployment_out),
    ])
    deployment_replay = json.loads(deployment_out.read_text())
    if deployment_replay != load(f"{deployment_result}/analysis.json"):
        raise AssertionError("camera-ready deployment analysis replay differs")

    b300_out = work_dir / "b300-selections.json"
    run_checked([
        sys.executable, f"{deployment_result}/b300_selection_transform.py",
        "--input", f"{deployment_result}/inputs.json", "--output", str(b300_out),
    ])
    selections = json.loads(b300_out.read_text())
    if len(selections.get("primary_23_contemporaneous", [])) != 23:
        raise AssertionError("B300 contemporaneous selection differs")
    if len(selections.get("retrospective_25_retry_augmented", [])) != 25:
        raise AssertionError("B300 retry-augmented selection differs")

    controls = work_dir / "history-controls"
    shutil.copytree(ROOT / "results/20260725-frozen-history-controls", controls)
    run_checked([sys.executable, "scripts/analyze_history_controls.py", str(controls), "--replace"])
    controls_replay = json.loads((controls / "analysis.json").read_text())
    controls_frozen = load("results/20260725-frozen-history-controls/analysis.json")
    if not is_json_subset(controls_replay, controls_frozen):
        raise AssertionError("frozen history-control values differ from enriched replay")

    livecode = work_dir / "livecodebench"
    shutil.copytree(ROOT / "results/20260724-livecodebench-v6-12-item-screen", livecode)
    run_checked([sys.executable, "scripts/analyze_livecodebench_frozen.py", str(livecode), "--replace"])
    livecode_replay = json.loads((livecode / "analysis.json").read_text())
    if livecode_replay != load("results/20260724-livecodebench-v6-12-item-screen/analysis.json"):
        raise AssertionError("LiveCodeBench analysis replay differs")

    return {
        "camera_ready_controlled": "exact",
        "camera_ready_deployment": "exact detached replay from 77,228-row ledger",
        "b300_overload": "23-row contemporaneous and 25-row retry-augmented selections reconstructed",
        "history_controls": "frozen analysis is an exact subset of enriched replay",
        "livecodebench": "exact",
    }


def check_cli_help() -> int:
    scripts = (
        "run_frozen_experiment.py", "run_four_model_replication.py",
        "run_history_controls.py", "run_history_tiers.py",
        "run_livecodebench_frozen.py", "analyze_camera_ready_controlled.py",
        "analyze_camera_ready_deployment.py",
        "run_reasoning_off_diagnostic.py", "run_reasoning_low_diagnostic.py",
    )
    for script in scripts:
        run_checked([sys.executable, f"scripts/{script}", "--help"])
    return len(scripts)


def run_tests() -> int:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    total = 0
    for pattern in (
        "test_archive.py", "test_camera_ready_controlled.py",
        "test_camera_ready_deployment_stdlib.py",
    ):
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"],
            cwd=ROOT, env=environment, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        match = list(__import__("re").finditer(r"Ran (\d+) tests?", result.stdout))
        if not match:
            raise AssertionError(f"unittest count missing for {pattern}")
        total += int(match[-1].group(1))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path.home() / "tmp/short-reasoning-archive-smoke-work",
    )
    args = parser.parse_args()
    files = check_manifest()
    check_structure()
    ledger_records = check_response_ledger()
    replays = replay_analyses(args.work_dir.expanduser().resolve())
    check_metrics()
    effort_summary = ROOT / "results/20260915-deepseek-compressor-effort-audit-v1/summary.json"
    before_effort = effort_summary.read_bytes()
    run_checked([sys.executable, "scripts/analyze_compressor_effort_diagnostic.py"])
    assert effort_summary.read_bytes() == before_effort, "compressor-effort replay differs"
    replays["compressor_effort"] = "exact"
    figure_tables = check_figure_data()
    json_files, python_files = check_json_and_python()
    cli_help = check_cli_help()
    tests = run_tests()
    print(json.dumps({"status": "passed", "manifest_files": files, "json_files": json_files, "python_files": python_files, "figure_tables": figure_tables, "analysis_replays": replays, "response_ledger_records": ledger_records, "cli_help": cli_help, "stdlib_tests": tests}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
