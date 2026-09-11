"""Pinned LiveCodeBench loading and deterministic code-evaluation helpers."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
import pickle
import sys
import zlib
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    ids = [item["question_id"] for item in manifest["items"]]
    if len(ids) != 12 or len(ids) != len(set(ids)):
        raise ValueError("LiveCodeBench screen must contain 12 unique question IDs")
    return manifest


def fetch_dataset(manifest: dict[str, Any]) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            manifest["dataset_repo"],
            manifest["dataset_file"],
            repo_type="dataset",
            revision=manifest["dataset_revision"],
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["dataset_sha256"]:
        raise RuntimeError(f"dataset SHA-256 mismatch: {digest}")
    return path


def select_records(dataset_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {}
    for line in dataset_path.read_text().splitlines():
        record = json.loads(line)
        by_id[record["question_id"]] = record
    selected = []
    for item in manifest["items"]:
        if item["question_id"] not in by_id:
            raise KeyError(item["question_id"])
        record = by_id[item["question_id"]]
        if record["platform"] != item["platform"] or record["difficulty"] != item["difficulty"]:
            raise ValueError(f"manifest metadata mismatch for {item['question_id']}")
        selected.append(record)
    return selected


def decode_test_cases(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        # The pinned official dataset stores larger private suites this way.
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(value))))
    if not isinstance(decoded, list):
        raise ValueError("test cases must decode to a list")
    return decoded


def evaluation_sample(record: dict[str, Any], *, public_only: bool) -> dict[str, str]:
    public = decode_test_cases(record["public_test_cases"])
    tests = public if public_only else public + decode_test_cases(record["private_test_cases"])
    metadata = json.loads(record["metadata"])
    return {
        "input_output": json.dumps(
            {
                "inputs": [test["input"] for test in tests],
                "outputs": [test["output"] for test in tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def static_code_summary(record: dict[str, Any], code: str) -> dict[str, Any]:
    metadata = json.loads(record["metadata"])
    expected_function = metadata.get("func_name")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_valid": False,
            "syntax_error": {"line": exc.lineno, "offset": exc.offset, "message": exc.msg},
            "line_count": len(code.splitlines()),
            "expected_function": expected_function,
            "expected_function_present": False,
            "imports": [],
            "defined_functions": [],
            "defined_classes": [],
        }
    imports = []
    functions = []
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {
        "syntax_valid": True,
        "syntax_error": None,
        "line_count": len(code.splitlines()),
        "expected_function": expected_function,
        "expected_function_present": expected_function is None or expected_function in functions,
        "imports": sorted(set(imports)),
        "defined_functions": sorted(set(functions)),
        "defined_classes": sorted(set(classes)),
    }


def official_check(
    record: dict[str, Any],
    code: str,
    evaluator_repo: Path,
    *,
    public_only: bool,
    timeout: int = 6,
) -> dict[str, Any]:
    if not evaluator_repo.exists():
        raise FileNotFoundError(evaluator_repo)
    repo_text = str(evaluator_repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    module = importlib.import_module("lcb_runner.evaluation.compute_code_generation_metrics")
    results, metadata = module.check_correctness(
        evaluation_sample(record, public_only=public_only),
        code,
        timeout=timeout,
        debug=False,
    )
    normalized = [bool(value) if type(value).__module__ == "numpy" else value for value in results]
    return {
        "passed": bool(normalized) and all(value is True for value in normalized),
        "passed_tests": sum(value is True for value in normalized),
        "total_tests": len(normalized),
        "results": normalized,
        "metadata": metadata,
    }
