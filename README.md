# ShortReasoning

Code and frozen experimental records for **[Why Pay for Long Reasoning When Rewrite Do Trick?](paper.pdf)** by **Anton Hristov**, accepted at REALM Workshop, EMNLP 2026.

In this study, shorter rewrites of earlier assistant reasoning were often followed by fewer provider-reported future reasoning tokens in selected hosted paired forks. The distributions include reversals and source-selection effects, and the evidence does not establish semantic faithfulness, quality noninferiority, or a model- or provider-general speedup.

## Quick start

The tested interpreter is CPython 3.14.3; retained code requires Python 3.11 or newer. From the archive root:

```bash
python3 -m venv ~/tmp/short-reasoning-smoke
~/tmp/short-reasoning-smoke/bin/python -m pip install --no-index -r environments/smoke/requirements.txt
~/tmp/short-reasoning-smoke/bin/python scripts/run_archive_smoke.py \
  --work-dir ~/tmp/short-reasoning-smoke-work
```

The dependency-free, network-free smoke verifies every exported hash, recomputes the controlled, deployment, history-control, and LiveCodeBench analyses from bundled inputs, checks active-figure values, compiles every Python file, and runs the bundled standard-library verification tests. The complete included pytest suite is specified separately under `environments/tests/`.

## Current coverage

- The controlled audit contains 72 attempted route-task sources, 58 accepted source clusters, and 174 assigned continuation pairs. Its paper estimand is explicitly retrospective and retry-augmented: 151/172 equal-horizon pairs shorten, the median change is -32.0%, and exact success is 165/174 for clean history versus 166/174 for compact history. The separate original-first-capture sensitivity has 149/169 comparable pairs shortening; these quality counts are descriptive and route-sensitive.
- The deployment audit contains all 600 binary outcomes for four models, three arms, and 50 tasks per arm, plus a privacy-minimized per-response ledger that validates acquisition coverage rather than aggregate arithmetic alone. Accuracy differences range from -6 to +6 percentage points and do not constitute noninferiority evidence. Corrected all-response accounting gives Kimi K2.6 spend changes of -7.6% and -31.9%; common-tariff Nemotron estimates are +60.9% and +109.7%.
- The deployment ledger contains all 77,021 captured completed model responses with choices, including responses orphaned from saved trajectories. The 207 known returned error bodies and their durations are reported separately; full deployment task wall time was not captured.
- The same deployment inputs contain the original 10 B300 blocks, the 23 contemporaneous complete overload triples, the retrospective retry-augmented 25-block sensitivity, and the explicit selection transform. These measurements concern one pinned task and exclude rewrite/setup time.
- The archive also retains the two history controls, retrospective style-dose diagnostics, the acquisition-selected LiveCodeBench screen, completed SWE-bench Verified Mini records, hosted amortization inputs, exact prompts/rubrics, task manifests, accepted traces, exclusions, and correction captures.

## Reproduction map

- `scripts/README.md` gives exact offline replay and fresh-acquisition entry points.
- `results/README.md` separates current camera-ready evidence from exploratory and retained audit scopes.
- `figures/README.md` documents the four active PDF snapshots and their archive-specific generator.
- `environments/README.md` separates smoke, plotting, and acquisition dependencies.
- `manifests/README.md` defines source-byte, anonymization, exclusion, and exported-byte records.

The current paper PDF is included as [`paper.pdf`](paper.pdf). LaTeX sources, peer-review material, agent documents, provider journals redundant with frozen captures, unselected development runs, raw SWE-bench workspaces, credentials, and third-party datasets are omitted.

## Integrity and anonymization

`manifests/source_files.json` records original hashes before anonymization; `archive_manifest.json` covers the published payload, excluding itself and local Git metadata. Experimental records have known personal names, usernames, hosts, private roots/remotes, and internal project revisions replaced. Public author information and the paper PDF are added after this sanitization. Credential families and exact local `.env`/`e.env` values are checked without recording those values. PDF metadata and extracted text are inspected. The audit is broad and deterministic over its enumerated checks, not a universal proof that no unknown identifying string exists.

## License and third-party terms

A first-party public license has not yet been assigned. Public-license selection is pending; absent a grant, first-party files remain all rights reserved. This statement does not replace third-party terms.

No third-party model weights, full datasets, or source repositories are redistributed. LiveCodeBench software is MIT licensed (Copyright 2024 LiveCodeBench); problem statements and platform content may have additional terms. SWE-bench software is MIT licensed; benchmark instances and target repositories remain subject to upstream terms. mini-swe-agent is referenced but not copied. Hosted outputs transfer no model or provider license.

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Consult upstream sources for authoritative copyright notices and dataset-specific terms before fresh acquisition or redistribution.
