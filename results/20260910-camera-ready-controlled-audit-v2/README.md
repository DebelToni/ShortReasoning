# Camera-ready controlled robustness audit v2

Offline recomputation from immutable continuation captures. The paper result is retained but classified as **retrospective retry-augmented**; original first-captured outcomes are reported separately. No model calls or network access are used.

## Retry-augmented paper estimand

- Cohort accounting is **72 attempted route-task sources → 58 accepted sources → 174 assigned continuation pairs** (three replicates per accepted source). These are not 174 independent tasks.
- Compact history is shorter in **151/172 equal-horizon pairs** (87.8%). Exact success over every assigned pair is **165/174 → 166/174** (+0.6 pp).
- At the first continuation, where both arms have the same just-revealed result and no post-treatment generated mediator, the paired median is **-20.0%** and the pooled token change is **-16.4%** over 174 pairs. Subsequent reasoning has paired median **-37.8%** and pooled change **-40.0%**; it is a trajectory effect, not a direct effect.
- Full-trajectory reasoning over all assigned pairs changes from **135778 to 97790 tokens** (-28.0%). The equal-horizon paired median is **-32.0%**, with source-cluster bootstrap 95% interval **[-38.8, -24.3]%**.
- In raw token units, the equal-horizon paired median is **-178 tokens** (bootstrap interval **[-224, -126]**). There are 20 increases and 1 equality; the maximum increase is **+2807 tokens**, so the right tail remains practically important.
- Without equal-horizon filtering, turns are **521 → 520**, accumulated input tokens **1429111 → 1277229**, and completion tokens **220049 → 163876**. There are **9 → 8 exact failures**, including one action-failure termination in each arm.

The source-cluster 95% interval treats 58 route-task sources as exchangeable. Resampling the 12 shared task definitions instead gives **[-37.4, -26.0]%**; this preserves cross-route task dependence and observed gate missingness.

## Original-first-capture sensitivity

The three clean Laguna length-cap outcomes and the compact provider-502 outcome remain exactly as first captured. Exact success is **162/174 → 165/174** (+1.7 pp). Full-trajectory reasoning is **152475 → 94357 tokens** (-38.1%), and first-turn paired median change is **-20.72%**.

| Route | Comparable/all | Shorter | Median change | Exact success clean→compact |
|---|---:|---:|---:|---:|
| deepseek-v4-flash | 36/36 | 30 | -29.5% | 36→35 |
| laguna-s-2.1 | 33/36 | 30 | -47.1% | 33→34 |
| minimax-m3 | 30/30 | 25 | -27.3% | 30→30 |
| glm-4.7-flash | 13/15 | 13 | -50.5% | 11→14 |
| glm-5.1 | 30/30 | 28 | -20.2% | 25→25 |
| deepseek-v4-pro | 27/27 | 23 | -32.6% | 27→27 |

Across the original **169** comparable pairs, **149** shorten; raw median delta is **-177 tokens**, with 19 increases, equality count 1, and maximum increase **+1849 tokens**. Without horizon filtering, turns are **515 → 518** (+0.6%), input tokens **1399681 → 1259755** (-10.0%), completion tokens **234817 → 159887** (-31.9%), and exact failures **12 → 9**. These failure-inclusive workload totals retain cap outcomes; shorter failed traces are not efficiency successes.

## Retry classification

Three clean branches—Laguna grid rep 2, manufacturing rep 3, and rescue rep 2—ended at the stated 8,192 completion cap. The available records do not establish that a cap outcome is an infrastructure fault. Laguna rescue rep 2 compact instead captured a provider HTTP 502. All four were retrospectively rerun under a rule permitting up to 20 attempts per retried turn; recovery-run request-attempt totals were 3, 3, 5, and 3; these totals exclude the original first-capture branches. `retry_augmentation_records.json` exports both branch versions, trigger classification, per-turn recovery attempt counts, and metric changes.

## Leave-one-route-out sensitivity of retry-augmented result

| Omitted route | Comparable pairs | Shorter | Median change | Median token delta | Exact success clean→compact |
|---|---:|---:|---:|---:|---:|
| deepseek-v4-flash | 136 | 121 | -32.7% | -188 | 129→131 (+1.4 pp) |
| laguna-s-2.1 | 136 | 119 | -29.2% | -156 | 129→131 (+1.4 pp) |
| minimax-m3 | 142 | 126 | -33.7% | -207 | 135→136 (+0.7 pp) |
| glm-4.7-flash | 159 | 138 | -29.3% | -160 | 154→152 (-1.3 pp) |
| glm-5.1 | 142 | 123 | -36.7% | -202 | 140→141 (+0.7 pp) |
| deepseek-v4-pro | 145 | 128 | -31.8% | -180 | 138→139 (+0.7 pp) |

The equal-horizon token effect remains negative after omitting any route. The aggregate quality sign changes when GLM 4.7 Flash is omitted, showing that the +0.6 pp total is not route-robust.

## Selection, timing, and uncertainty

The additional-route gate excluded 5 sources with no action-qualified source and 9 for model-judged fidelity, leaving 58 accepted sources. GPT-5.6 Sol generated and separately judged rewrites; there was no independent blinded human validation. The controlled estimand is therefore conditional on passing a one-sided, compressor-family gate. `per_source.json` preserves each gate decision and compact rejection explanation available in the audits.

Route retention in displayed order is 12/12, 12/12, 10/12, 5/12, 10/12, and 9/12. The rewrite was completed before the pending tool result was executed or exposed. The first continuation received the same result in 174/174 pairs; the preceding result was also identical in 173/173 pairs reaching turn 2 in both arms and 172/172 reaching turn 3, but later text still follows treatment-affected generation and can contain mediated effects. Bootstrap intervals resample the 58 route-task source clusters and retain their continuation replicates; they do not estimate new routes, rejected sources, provider reruns, or seed variability.

## Provenance

`results/20260804-six-model-common-slice-infrastructure-corrected-v2/analysis.json` defines the four retry augmentations and the accepted paper reference rows; `results/20260803-table1-infrastructure-reruns-v1/protocol.json` defines their retry policy. Pair-level values are recalculated from `results/20260724-v2-12-task-two-route-screen/continuations/` for DeepSeek V4 Flash and Laguna S 2.1; `results/20260730-four-model-paired-replication-v1/continuations/` for MiniMax M3; `results/20260730-glm-four-model-replication-v2/continuations/` for GLM 4.7 Flash; and `results/20260730-latest-model-paired-replication-v1/continuations/` for GLM 5.1 and DeepSeek V4 Pro. Source decisions come from the corresponding source audits. The exact additional-route model-review rubric is copied from `scripts/run_four_model_replication.py` and exported in `per_source.json`; the initial source audit used additional candidate-ID and threshold gates documented in its immutable audit files.

## Reproduction

```bash
python3 scripts/analyze_camera_ready_controlled.py \
  --root . \
  --input results/20260804-six-model-common-slice-infrastructure-corrected-v2 \
  --output results/20260910-camera-ready-controlled-audit-v2
python3 -m unittest tests/test_camera_ready_controlled.py
```

Outputs are deterministic: `analysis.json` contains aggregate estimands and caveats; explicitly named per-pair files preserve both versions; `retry_augmentation_records.json` binds original and replacement branches; `per_source.json` preserves source gates; and `manifest.json` hashes every consumed capture/audit plus generated output. Paths in JSON are repository-relative. The script uses only the Python standard library.
