# Benchmark boundaries

The following registry is transcribed from the supplied manuscript, not independently certified against downloaded official partitions.

| Dataset | Paper contract | Current executable status |
|---|---|---|
| CASIA-B | Train001–074; test075–124; gallery NM01–04; probe NM05–06/BG01–02/CL01–02; 11 views; exclude identical views | Generic per-gallery-view evaluator implemented; official parity NOT established; validation/refit must be resolved |
| OU-MVLP | 5153 train / 5154 test; gallery01/probe00; 14 views; exclude identical views | Explicit official subject lists required; generic evaluator usable once manifests supplied; missing-sequence convention requires official parity test |
| CCGR | Official split; easy NM1; hard covariate-pair evaluation | Generic evaluator intentionally NOT used as official hard/easy; research runner refuses CCGR |

For each probe and each eligible gallery view, the generic evaluator takes maximum cosine score over gallery sequences and assigns the winning identity. Rank-1 is computed per condition/probe-view/gallery-view cell and averaged equally across nonempty cells. It also records micro comparison accuracy and macro classification metrics. These are distinct aggregates. AUC is left null because hard labels cannot reproduce the reported ROC-AUC.

For official benchmark claims, pin the upstream evaluator, retain config/code checksum and demonstrate agreement on shared embeddings. For CCGR, export embeddings in the exact pinned evaluator format and retain the official hard/easy output. No placeholder official metric is emitted.

Modern baseline reference: [OpenGait](https://github.com/ShiqiYu/OpenGait) (official project inspected2026-09-18). This archive does not vendor or claim to have trained OpenGait/GaitBase. `external_baselines/provenance.template.json` lists the files required; `scripts/check_external_baseline.py` verifies their existence and hashes. `scripts/evaluate_embeddings.py` evaluates row-aligned exported embeddings under this package's generic contract.
