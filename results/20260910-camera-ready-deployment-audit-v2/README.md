# Camera-ready deployment response-ledger audit v2

This is the current deployment audit. An earlier aggregate incorrectly restricted target accounting to response IDs present in saved trajectories and is intentionally omitted. This audit enumerates completed target and compressor journals symmetrically and derives spend, token, and request-duration totals from the bundled response ledger. No model, network, GPU, or evaluator runs were performed.

## Response-ledger contract

`response-ledger.jsonl.gz` contains 77,228 privacy-minimized records across all 12 deployment arms. It stores SHA-256-hashed response IDs, journal and no-ID response source hashes, minimal usage, reported cost, client duration, provider/model, finish reasons, attempt/call/task linkage when recoverable, and trajectory/selection flags. It contains no response text, prompts, patches, requests, credentials, or filesystem paths.

The primary accounting unit is a journal with `status=complete` whose returned JSON contains one or more `choices`. Every finish reason, including `length`, is retained. Records are deduplicated by response ID; a successful response without an ID is retained once by raw-response SHA-256. Conflicting duplicate IDs fail collection. Returned JSON error bodies without choices are excluded from primary response totals and reported separately. All current successful responses have IDs; the no-ID path is covered by a regression fixture.

Bidirectional reconciliation found no trajectory response lacking a successful journal response. Journal-only successful target responses are real acquisitions and are included:

| Model | Normal | Luna | Self |
|---|---:|---:|---:|
| DeepSeek V4 Flash | 0 | 0 | 0 |
| MiniMax M3 | 0 | 0 | 0 |
| Kimi K2.6 | 122 | 420 | 254 |
| Nemotron 3 Ultra | 98 | 205 | 72 |

Task linkage is unavailable for these journal-only target responses because no trajectory binds their model-call group to a benchmark task. Their attempt/call identifiers, response/source hashes, usage, and timing remain in the ledger.

## Corrected all-acquisition spending

Spend includes every successful target and compressor response in the journal ledger, including successful calls omitted from trajectories. Kimi uses captured provider-reported usage. Nemotron target usage is reconstructed at the bundled common DeepInfra tariff; its compressor cost remains captured. The expected independent-audit totals reproduce to sub-microdollar precision.

| Model | Arm | Total USD | Delta vs Normal |
|---|---|---:|---:|
| DeepSeek V4 Flash | Normal | 2.696627 | ref. |
|  | Luna | 1.602617 | -40.6% |
|  | Self | 1.521109 | -43.6% |
| MiniMax M3 | Normal | 23.682061 | ref. |
|  | Luna | 13.929600 | -41.2% |
|  | Self | 11.576406 | -51.1% |
| Kimi K2.6 | Normal | 48.836254 | ref. |
|  | Luna | 45.107125 | -7.6% |
|  | Self | 33.258787 | -31.9% |
| Nemotron 3 Ultra | Normal | 23.823657 | ref. |
|  | Luna | 38.334883 | +60.9% |
|  | Self | 49.952106 | +109.7% |

Six of eight treatment arms reduce all-acquisition spend. V1's Kimi Luna and Nemotron percentages are superseded.

## Successful response duration

These values sum client-observed durations for returned model responses with choices. Target duration includes the newly recovered journal-only responses. Inclusive duration adds successful compressor responses. Queueing, prefill, decoding, and network transfer are included; tools, orchestration, failed transport attempts, and full task wall time are not.

| Model | Arm | Target seconds | Target delta | Inclusive seconds | Inclusive delta |
|---|---|---:|---:|---:|---:|
| DeepSeek V4 Flash | Normal | 13,539 | ref. | 13,539 | ref. |
|  | Luna | 8,061 | -40.5% | 12,560 | -7.2% |
|  | Self | 7,918 | -41.5% | 24,136 | +78.3% |
| MiniMax M3 | Normal | 151,850 | ref. | 151,850 | ref. |
|  | Luna | 41,930 | -72.4% | 45,913 | -69.8% |
|  | Self | 35,683 | -76.5% | 41,321 | -72.8% |
| Kimi K2.6 | Normal | 111,867 | ref. | 111,867 | ref. |
|  | Luna | 87,638 | -21.7% | 102,648 | -8.2% |
|  | Self | 47,170 | -57.8% | 84,348 | -24.6% |
| Nemotron 3 Ultra | Normal | 28,692 | ref. | 28,692 | ref. |
|  | Luna | 28,346 | -1.2% | 41,608 | +45.0% |
|  | Self | 36,797 | +28.3% | 63,063 | +119.8% |

Returned error bodies have zero captured cost. Their excluded durations are bound in `analysis.json`: Kimi target Normal 4/41.738 s and Luna 24/249.152 s, Kimi Luna compressor 6/3.998 s; Nemotron target Normal 34/95.244 s, Luna 17/20.099 s, Self 64/104.773 s, Nemotron Luna compressor 1/0.814 s and Self compressor 54/28.072 s; DeepSeek Luna compressor 1/0.471 s; MiniMax Luna compressor 2/1.356 s.

## Outcomes and selected trajectories

All 600 official binary task outcomes and task IDs remain bundled. The per-model paired task-bootstrap intervals and discordances are unchanged; accuracy changes span **-6 to +6 percentage points**. These intervals condition on the fixed 50-task cohort and one captured run per arm, estimate neither provider/seed variability nor noninferiority, and must not be pooled across models.

Selected-trajectory input/output/reasoning/response/tool-call totals and deltas remain separate from all-acquisition accounting. Kimi's result-level response counters are 1,866 Normal and 2,327 Luna (+24.7%); serialized selected trajectories contain 1,842 and 2,304 assistant responses (+25.1%); selected tool calls are 1,990 and 2,384 (+19.8%). `inputs.json` additionally binds each selected prediction to task ID, model, patch SHA-256/size, and official outcome without copying patch code.

## B300 selections and provenance

The portable input now contains the ten original per-block three-arm records, all 25 original overload blocks including failed-arm work, and the two individual retry records. Every record binds its source JSON hash, patch hash, and available official evaluation evidence.

| Selection | Rows | Luna paired median | Self paired median |
|---|---:|---:|---:|
| Original overload, contemporaneous complete triples | 23 | -26.1% | -23.4% |
| Retrospective retry-augmented sensitivity | 25 | -26.4% | -23.4% |

The primary selection excludes block 2, where Normal failed after 58.332 s, and block 17, where Self failed after 39.593 s. The 25-row sensitivity replaces only those arms with later 124.513 s and 91.482 s reruns; it is not 25 contemporaneous three-arm blocks. Parent/retry hashes and replacement flags are retained.

All 30 original ten-block branches and 73 successful original overload branches bind to resolved official evaluations. Self block 17's retry patch binds to an official report. Normal block 2's retry evaluator report file is unavailable; its computed patch hash and evaluator summary log establish one submitted/completed/resolved instance with zero errors. The archived `patch.sha256` text has one trailing extra character, recorded as a declared-hash mismatch rather than silently corrected.

`b300_selection_transform.py` independently reconstructs the 23-row primary and 25-row retrospective selections from the original/retry records. B300 timing is marginal full postfork task time after compact history is available on one pinned task; it includes tools and orchestration but excludes rewriting and setup.

## Detached reproduction

The result directory includes its exact analyzer and pricing file:

```bash
python3 analyze.py analyze \
  --input inputs.json \
  --ledger response-ledger.jsonl.gz \
  --output ~/tmp/deployment-v2-analysis.json
cmp ~/tmp/deployment-v2-analysis.json analysis.json

python3 b300_selection_transform.py \
  --input inputs.json --output ~/tmp/b300-selections.json
```

`collect --help` documents the explicit local-source import interface. `manifest.json` binds the analyzer, ledger, pricing, compact inputs, outputs, tests, local source files, archives, official reports, and selected-prediction files through relative artifact paths or logical source labels. V2 remains an existing-data reconstruction: deployment full task wall time, task linkage for journal-only responses, and uncaptured cost for transport errors remain unavailable.
