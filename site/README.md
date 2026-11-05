# ShortReasoning project site

Static HTML, CSS and JavaScript for https://debeltoni.github.io/ShortReasoning/.
The Pages workflow publishes only this directory, not the research working tree.

## Preview

From the repository root:

```sh
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory site
```

## Scientific boundaries

- `data/results.json` contains public aggregate values from the revised arXiv paper, https://arxiv.org/abs/2611.03852: controlled results from Table 1, retained-run deployment results from Table 3, and absolute/inclusive request timing from Table 10.
- Controlled token comparisons are conditional on accepted histories and equal horizons. Acceptance differs by route. Task success does not establish quality noninferiority.
- All 50 assigned tasks remain in each deployment arm. Spend includes rewriting. Nemotron target costs are common-tariff estimates with captured compressor charges.
- Target-only and rewrite-inclusive request durations are distinct. Neither is end-to-end agent latency. The history-word totals are not billed input-token totals.
- The five compactor diagrams are illustrations of design choices. Self/external rewriting and per-block rewriting were tested; other settings shown are proposed experiments.
- The training diagram is a proposal, not a measured result: frozen same-model teacher on compact history, trainable student on original history, matching a common continuation prefix and vocabulary. Forward KL updates the student only. The animated distributions are explicitly illustrative, not model outputs or training measurements.
- Source figure exports preserve the paper's original controlled-effects and length/quality figures.

No raw reasoning, task patches, response journals, credentials, or private paths belong in this site.

## Deployment

`.github/workflows/pages.yml` checks syntax and public aggregate invariants,
then deploys `site/` using GitHub Pages Actions. Resource paths are relative so
the site works under the `/ShortReasoning/` project prefix. No server, package
installation, model call, or account credential is required by the site itself.
