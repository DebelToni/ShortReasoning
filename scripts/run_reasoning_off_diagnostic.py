#!/usr/bin/env python3
"""One-off exploratory replays of slow self-compactions; no automatic retries."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from short_reasoning import load_dotenv, openrouter_key

API = "https://openrouter.ai/api/v1/chat/completions"
RECORDED = ROOT / "results/20260915-deepseek-0731-reasoning-off-20-v1"
JUDGE_PROMPT = """Compare two candidate rewrites against the ORIGINAL historical coding-agent reasoning.
Judge each independently for preservation of decision-relevant observations, identifiers,
numbers, boolean conditions, code structure, conclusions, corrections, unresolved uncertainty,
rejected alternatives, and intended next actions. Do not solve or improve the underlying task.
Distinguish harmless stylistic shortening from material omissions, changes, or unsupported additions.
The pending tool result is unavailable. Do not infer facts from outside the supplied original.
A compact state need not be executable code, but its meaning and control flow must remain clear.
For every Problem, quote specific evidence from the original and candidate; do not invent quotations.
Return only a JSON object with these keys:
A and B: objects with verdict (Pass, Problem, or Unsure), flags (array chosen from OM, CF, CU, UA, NA),
evidence (array of objects with original_quote, candidate_quote, explanation), and summary (string).
OM=omission; CF=changed fact/condition/code meaning; CU=changed uncertainty/evidence status;
UA=unsupported addition; NA=changed next action.
preferred: A, B, Tie, or Unsure, based on fidelity, not merely length.
comparison: a short explanation. Be concise; keep the entire answer under 700 words."""


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def select_cases(output, count):
    # Portable release: use the exact selected captures supplied with the artifact.
    selected = json.loads((RECORDED / "selection.json").read_text())
    assert count == len(selected["cases"]) == 20
    cases = []
    for rank in range(1, count + 1):
        historical = json.loads((RECORDED / f"case-{rank:02d}" / "historical.json").read_text())
        request = historical["request"]
        response = historical["response"]
        case_dir = output / f"case-{rank:02d}"
        case_dir.mkdir()
        dump(case_dir / "historical.json", historical)
        cases.append({"rank": rank, "directory": str(case_dir), "request": request,
                      "raw": historical["raw_reasoning"],
                      "old_text": response["choices"][0]["message"].get("content") or "",
                      "historical_seconds": historical["ledger"]["wall_seconds"]})
    dump(output / "selection.json", {"selection": selected["selection"],
         "eligible_responses": selected["eligible_responses"],
         "cases": [{k: c[k] for k in ("rank", "historical_seconds", "directory")} for c in cases]})
    return cases


def call(payload, directory, name, key, monitor_reasoning=False, stop_on_reasoning=True):
    dump(directory / f"{name}-request.json", payload)
    result = {"status": "incomplete", "started_utc": now(), "content": "", "usage": None,
              "reasoning_observed": False}
    dump(directory / f"{name}-started.json", {"utc": result["started_utc"]})
    request = urllib.request.Request(API, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            if not monitor_reasoning:
                data = json.load(response)
                dump(directory / f"{name}-response.json", data)
                result.update(usage=data.get("usage"), provider=data.get("provider"), id=data.get("id"))
                if data.get("error"):
                    result.update(status="provider_error", error=data["error"])
                else:
                    choice = data["choices"][0]
                    result.update(status="complete", content=choice["message"].get("content") or "",
                                  finish_reason=choice.get("finish_reason"))
            else:
                with (directory / f"{name}-events.jsonl").open("w") as journal:
                    for line in response:
                        line = line.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        text = line[5:].strip()
                        if text == "[DONE]":
                            result["status"] = "complete"
                            break
                        event = json.loads(text)
                        journal.write(json.dumps(event) + "\n"); journal.flush()
                        for field in ("id", "provider", "model", "usage"):
                            if event.get(field): result[field] = event[field]
                        if event.get("error"):
                            result.update(status="provider_error", error=event["error"]); break
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            result["content"] += delta.get("content") or ""
                            if choice.get("finish_reason"): result["finish_reason"] = choice["finish_reason"]
                            if any(delta.get(k) for k in ("reasoning", "reasoning_content", "reasoning_details")):
                                result["reasoning_observed"] = True
                                result["reasoning_evidence"] = delta
                        details = (result.get("usage") or {}).get("completion_tokens_details") or {}
                        if (details.get("reasoning_tokens") or 0) > 0 or any(tag in result["content"].lower() for tag in ("<think>", "</think>")):
                            result["reasoning_observed"] = True
                        if result["reasoning_observed"] and stop_on_reasoning:
                            result["status"] = "stopped_reasoning_observed"
                            break
    except urllib.error.HTTPError as exc:
        result.update(status="http_error", http_status=exc.code, error=exc.read().decode(errors="replace"))
    except Exception as exc:
        result.update(status="client_error", error=f"{type(exc).__name__}: {exc}")
    result["wall_seconds"] = time.monotonic() - start
    dump(directory / f"{name}-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-after-transport-error", action="store_true")
    parser.add_argument("--retry-incomplete-after-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    retry_ranks = []
    if args.retry_incomplete_after_run:
        while True:
            progress = json.loads((output / "progress.json").read_text())
            if progress["status"] not in ("running", "selecting"):
                break
            time.sleep(3)
        assert progress["status"] == "complete", "Never bypass a reasoning or budget stop."
        for row in progress["cases"]:
            result = json.loads((output / f"case-{row['rank']:02d}" / "reasoning-off-result.json").read_text())
            if result["status"] != "complete" or result.get("finish_reason") == "length":
                assert not result["reasoning_observed"]
                retry_ranks.append(row["rank"])
        if not retry_ranks:
            return
        before = output / "progress-before-uncapped-retry.json"
        assert not before.exists(), "Refusing an automatic repeated retry sweep."
        dump(before, progress)
        for rank in retry_ranks:
            directory = output / f"case-{rank:02d}"
            archive = directory / "initial-attempt"
            archive.mkdir()
            for path in list(directory.iterdir()):
                if path.is_file() and path.name != "historical.json":
                    path.rename(archive / path.name)
        progress["cases"] = [r for r in progress["cases"] if r["rank"] not in retry_ranks]
        progress["finished"] = len(progress["cases"])
        progress["successful_judgments"] = sum("judgments" in r for r in progress["cases"])
        progress["uncapped_retry_ranks"] = retry_ranks
        start = time.monotonic() - progress["elapsed_seconds"]
    elif args.resume_after_transport_error:
        progress = json.loads((output / "progress.json").read_text())
        assert progress["status"] == "stopped_rewrite_error"
        last = json.loads((output / f"case-{progress['finished']:02d}" / "reasoning-off-result.json").read_text())
        assert not last["reasoning_observed"] and last["status"] == "client_error"
        start = time.monotonic() - progress["elapsed_seconds"]
    else:
        output.mkdir(parents=True, exist_ok=False)  # Never accidentally rerun paid calls.
        start = time.monotonic()
        progress = {"status": "selecting", "started_utc": now(), "total": 20,
                    "finished": 0, "successful_judgments": 0, "reported_cost_usd": 0.0, "cases": []}
    load_dotenv(ROOT / ".env")
    key = openrouter_key()

    def update():
        progress["elapsed_seconds"] = time.monotonic() - start
        n = progress["finished"]
        progress["eta_remaining_seconds"] = progress["elapsed_seconds"] / n * (20 - n) if n else None
        temporary = output / "progress.tmp"
        dump(temporary, progress)
        temporary.replace(output / "progress.json")

    update()
    if args.resume_after_transport_error or args.retry_incomplete_after_run:
        cases = []
        ranks = retry_ranks if args.retry_incomplete_after_run else range(progress["finished"] + 1, 21)
        for rank in ranks:
            directory = output / f"case-{rank:02d}"
            old = json.loads((directory / "historical.json").read_text())
            cases.append({"rank": rank, "directory": str(directory), "request": old["request"],
                          "raw": old["raw_reasoning"],
                          "old_text": old["response"]["choices"][0]["message"].get("content") or "",
                          "historical_seconds": old["ledger"]["wall_seconds"]})
        progress["resumed_utc"] = now()
        progress["resume_note"] = ("User-authorized retry of interrupted/capped cases without a client timeout or DeepSeek output cap; initial attempts retained."
                                   if args.retry_incomplete_after_run else "Continue unattempted cases; retain timed-out case without retry.")
    else:
        cases = select_cases(output, 20)
    progress["status"] = "running"
    # Conservative per-call dollar ceilings exceed current published prices at
    # these context lengths; stop well inside the user's $10 revision budget.
    for case in cases:
        if progress["reported_cost_usd"] + 0.50 > 8.0:
            progress["status"] = "stopped_budget"; update(); return
        directory = Path(case["directory"])
        progress.update(current_case=case["rank"], phase="reasoning_off")
        update()
        payload = copy.deepcopy(case["request"])
        payload.update(model="deepseek/deepseek-v4-flash-0731",
                       provider={"only": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True},
                       reasoning={"effort": "none", "exclude": False},
                       stream=True, stream_options={"include_usage": True})
        payload.pop("include_reasoning", None)
        payload.pop("max_tokens", None)
        payload.pop("max_completion_tokens", None)
        rewrite = call(payload, directory, "reasoning-off", key, monitor_reasoning=True)
        progress["reported_cost_usd"] += (rewrite.get("usage") or {}).get("cost", 0) or 0
        row = {"rank": case["rank"], "historical_seconds": case["historical_seconds"],
               "rewrite_status": rewrite["status"], "rewrite_seconds": rewrite["wall_seconds"],
               "raw_words": len(case["raw"].split()), "historical_words": len(case["old_text"].split()),
               "new_words": len(rewrite["content"].split())}
        if rewrite["reasoning_observed"]:
            progress["cases"].append(row)
            progress.update(status="stopped_reasoning_observed", phase="stopped")
            update(); return
        if rewrite["status"] != "complete" or not rewrite["content"]:
            row["judge_status"] = "not_run"
            progress["cases"].append(row)
            progress["finished"] += 1
            # Retain failed attempts; neither retry nor change providers.
            update()
            continue
        # Labels are randomly assigned, recorded locally, and omitted from the judge prompt.
        import random
        labels = ["historical", "reasoning_off"]
        random.Random(20260915 + case["rank"]).shuffle(labels)
        mapping = dict(zip(("A", "B"), labels))
        dump(directory / "judge-labels.json", mapping)
        texts = {"historical": case["old_text"], "reasoning_off": rewrite["content"]}
        user = json.dumps({"original": case["raw"], "A": texts[mapping["A"]], "B": texts[mapping["B"]]})
        # UTF-8 byte count upper-bounds ordinary text token counts conservatively.
        judge_upper = (len((JUDGE_PROMPT + user).encode()) + 2048) * 0.000002 + 4096 * 0.00001
        if judge_upper > 0.50:
            progress["cases"].append(row)
            progress.update(status="stopped_judge_budget", phase="stopped")
            update(); return
        progress["phase"] = "sol_judgment"; update()
        judge_payload = {"model": "openai/gpt-5.6-sol",
            "provider": {"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
            "messages": [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
            "reasoning": {"effort": "medium", "exclude": True}, "max_tokens": 4096,
            "response_format": {"type": "json_object"}, "usage": {"include": True}}
        judged = call(judge_payload, directory, "judge", key)
        progress["reported_cost_usd"] += (judged.get("usage") or {}).get("cost", 0) or 0
        row.update(judge_status=judged["status"], judge_seconds=judged["wall_seconds"],
                   label_mapping=mapping, new_is_shorter=row["new_words"] < row["raw_words"],
                   historical_is_shorter=row["historical_words"] < row["raw_words"])
        if judged["status"] == "complete":
            try:
                verdict = json.loads(judged["content"])
                for label in ("A", "B"):
                    assert verdict[label]["verdict"] in ("Pass", "Problem", "Unsure")
                row["judgments"] = {mapping[label]: verdict[label] for label in ("A", "B")}
                row["preferred"] = mapping.get(verdict.get("preferred"), verdict.get("preferred"))
                dump(directory / "judgment.json", verdict)
                progress["successful_judgments"] += 1
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                row.update(judge_status="invalid_judgment", parse_error=str(exc))
        progress["cases"].append(row)
        progress["finished"] += 1
        update()
        print(json.dumps({k: progress[k] for k in ("finished", "successful_judgments", "elapsed_seconds", "eta_remaining_seconds", "reported_cost_usd")}), flush=True)
        if judged["status"] != "complete":
            progress.update(status="stopped_judge_error", phase="stopped"); update(); return
    progress.update(status="complete", phase="complete")
    update()


if __name__ == "__main__":
    main()
