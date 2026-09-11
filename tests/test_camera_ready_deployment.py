import base64
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_camera_ready_deployment.py"
RESULT = ROOT / "results" / "20260910-camera-ready-deployment-audit-v2"
spec = importlib.util.spec_from_file_location("camera_ready_deployment", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def write_journal(root, relative, response, wall=2.0):
    path = root / "journals" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(response, sort_keys=True).encode()
    row = {
        "status": "complete",
        "wall_seconds": wall,
        "raw_response_base64": base64.b64encode(raw).decode(),
        "raw_response_sha256": module.sha256_bytes(raw),
        "request_sha256": "a" * 64,
    }
    path.write_text(json.dumps(row))
    return path


def response(response_id, cost, *, choices=True):
    row = {
        "provider": "fixture-provider",
        "model": "fixture-model",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }
    if response_id is not None:
        row["id"] = response_id
    if choices:
        row["choices"] = [{"finish_reason": "length"}]
    else:
        row["error"] = {"message": "fixture"}
    return row


def test_ledger_includes_orphan_success_excludes_error_and_deduplicates(tmp_path):
    trajectory_path = tmp_path / "attempts" / "attempt-001" / "inference" / "task-1" / "task-1.traj.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(json.dumps({"messages": [{"role": "assistant", "extra": {"response": {"id": "in-trajectory"}}}]}))
    selected = {"task-1": trajectory_path}

    write_journal(tmp_path, "attempt-001/model-a/call-001/attempt-01.json", response("in-trajectory", 0.25))
    write_journal(tmp_path, "attempt-001/model-a/call-002/attempt-01.json", response("journal-only", 0.75))
    write_journal(tmp_path, "attempt-001/model-a/call-003/attempt-01.json", response(None, 0.50))
    write_journal(tmp_path, "attempt-001/model-a/call-004/attempt-01.json", response(None, 0.0, choices=False), wall=7.0)
    write_journal(tmp_path, "attempt-001/model-a/call-005/attempt-01.json", response("in-trajectory", 0.25))

    records, reconciliation = module.collect_response_ledger("fixture", "normal", [tmp_path], selected)
    successes = [row for row in records if row["kind"] == "model_response"]
    errors = [row for row in records if row["kind"] == "error_body"]

    assert len(successes) == 3
    assert sum(row["usage"]["reported_cost_usd"] for row in successes) == 1.50
    assert sum(not row["in_any_trajectory"] for row in successes) == 2
    assert sum(row["response_id_sha256"] is None for row in successes) == 1
    assert successes[0]["finish_reasons"] == ["length"]
    assert len(errors) == 1 and errors[0]["wall_seconds"] == 7.0
    assert reconciliation["successful_journal_responses_not_in_trajectory"] == 2
    assert reconciliation["trajectory_only_response_ids"] == 0


def test_conflicting_duplicate_response_id_fails(tmp_path):
    trajectory_path = tmp_path / "inference" / "task-1" / "task-1.traj.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(json.dumps({"messages": []}))
    write_journal(tmp_path, "attempt-001/model-a/call-001/attempt-01.json", response("duplicate", 0.25))
    write_journal(tmp_path, "attempt-001/model-a/call-002/attempt-01.json", response("duplicate", 0.75))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        module.collect_response_ledger("fixture", "normal", [tmp_path], {"task-1": trajectory_path})


def b300_arm(status="complete", seconds=100.0):
    return {
        "status": status,
        "complete_task_seconds": seconds,
        "summed_request_seconds": seconds - 2,
        "summed_generation_phase_seconds": seconds - 3,
        "tool_execution_seconds": 1,
        "future_reasoning_tokens": 10,
        "total_output_tokens": 20,
        "turns": 2,
        "patch_sha256": "b" * 64 if status == "complete" else None,
        "evaluation": {"resolved": True} if status == "complete" else None,
        "error": None if status == "complete" else "protocol failure",
    }


def test_b300_selection_reconstructs_23_and_noncontemporaneous_25():
    original = []
    for block in range(1, 26):
        arms = {arm: b300_arm(seconds=100 - index * 10) for index, arm in enumerate(("normal", "self", "luna"))}
        if block == 2:
            arms["normal"] = b300_arm("failure", 58.0)
        if block == 17:
            arms["self"] = b300_arm("failure", 39.0)
        original.append({"block": block, "regime": "overloaded", "source_sha256": f"{block:064x}", "arms": arms})
    retries = [
        {"block": 2, "arm": "normal", "parent_source_sha256": f"{2:064x}", "retry_source_sha256": "c" * 64, "record": b300_arm(seconds=124.0)},
        {"block": 17, "arm": "self", "parent_source_sha256": f"{17:064x}", "retry_source_sha256": "d" * 64, "record": b300_arm(seconds=91.0)},
    ]

    selections = module.b300_selections(original, retries)

    assert len(selections["primary_23_contemporaneous"]) == 23
    assert len(selections["retrospective_25_retry_augmented"]) == 25
    augmented = {row["block"]: row for row in selections["retrospective_25_retry_augmented"]}
    assert augmented[2]["arms"]["normal"]["complete_task_seconds"] == 124.0
    assert augmented[17]["arms"]["self"]["complete_task_seconds"] == 91.0
    assert augmented[2]["arms"]["normal"]["retrospective_replacement"] is True


def test_bundled_v2_detached_reanalysis():
    inputs = module.load(RESULT / "inputs.json")
    ledger_path = RESULT / "response-ledger.jsonl.gz"
    reproduced = module.analyze(inputs, module.read_ledger(ledger_path), ledger_path)
    assert reproduced == module.load(RESULT / "analysis.json")
    deployment = reproduced["deployment"]
    assert deployment["accuracy_change_range_percentage_points"] == [-6.0, 6.0]
    expected_spend = {
        "kimi-k2.6": [48.836254107, 45.1071252514, 33.2587870974],
        "nemotron-3-ultra": [23.8236573, 38.3348833, 49.9521061],
    }
    for model, expected in expected_spend.items():
        arms = deployment["models"][model]["arms"]
        actual = [arms[arm]["all_acquisition_spend"]["total_usd"] for arm in module.ARMS]
        assert actual == pytest.approx(expected)
    kimi = deployment["models"]["kimi-k2.6"]["arms"]
    assert [kimi[arm]["journal_trajectory_reconciliation"]["target_success_not_in_any_trajectory"] for arm in module.ARMS] == [122, 420, 254]
    nemotron = deployment["models"]["nemotron-3-ultra"]["arms"]
    assert [nemotron[arm]["journal_trajectory_reconciliation"]["target_success_not_in_any_trajectory"] for arm in module.ARMS] == [98, 205, 72]
    b300 = reproduced["b300_latency"]
    assert b300["overload_primary_23"]["rows"] == 23
    assert b300["overload_retrospective_25"]["rows"] == 25
    failures = {(row["block"], next(iter(row["failed_arms"]))): next(iter(row["failed_arms"].values()))["complete_task_seconds"] for row in b300["original_failed_arm_records"]}
    assert failures == {(2, "normal"): 58.331674856, (17, "self"): 39.593234483}
    assert all(row["record"]["evaluation"]["resolved"] for row in b300["retry_bindings"])
