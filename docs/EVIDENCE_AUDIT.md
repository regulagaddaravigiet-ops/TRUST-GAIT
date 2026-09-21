# Evidence audit against the supplied paper

## Findings

The earlier archive contains source fragments, aggregate tables and reconstructed seed values. Its checkpoint registry has no corresponding trained model files. The new release adds executable pipelines and genuine synthetic test artifacts, but cannot recover historical experiments from aggregates.

| Paper claim | Supplied evidence | Release handling | Remaining requirement |
|---|---|---|---|
| Table 5: 97.1% accuracy / 96.1% F1 / 0.98 AUC | Aggregate CSV only | Preserve and plot as reported | Named dataset, exact split, evaluation unit, raw scores/predictions and original checkpoint |
| Table 6: score drop 0.56 / flips 68% | Aggregate CSV only | Preserve; add a new matched-coordinate audit | Original sample list, target score, baseline, attribution budget and raw interventions |
| Table 8: ablation and 176–194 ms | Aggregate CSV, including accuracy/F1 not tabulated in manuscript Table 8 | Preserve with exact source label | Every ablation checkpoint, preprocessing switches, raw timing and hardware |
| Table 9: stress gains | Aggregate CSV only | Preserve; supply separately defined executable diagnostics | Original perturbation parameters and matched per-sample outcomes |
| Table 10: five-run summary | Means/SD plus deliberately reconstructed seed tuples | Quarantine reconstructed tuples | Original five independent seed runs; no inference from reconstructed p-values |
| CASIA-B / OU-MVLP / CCGR | Protocol descriptions only | Protocol registry and generic evaluator | Authorized data, actual frozen sample lists and official evaluator parity |
| Matched modern baseline | Literature context only | Provenance schema and embedding-evaluation interface | Pinned upstream architecture/config, trained checkpoint, raw predictions under same contract |

## Issues requiring author decisions

1. **The headline development result is still unidentified.** Renaming it "controlled-development" does not resolve the reviewer's request. The abstract and conclusion still foreground it. Identify the actual experiment or remove it as headline evidence.
2. **The rule loss has no gradient to neural weights in the supplied design.** GIAT values, predicate slope/midpoint and rule memberships are fixed. The code records the term honestly as a diagnostic constant. A claimed gain from optimizing that term cannot be established. A learned descriptor/predicate mechanism would be a methodological change and needs new experiments.
3. **Equation 11 is ambiguous.** It appears to complement non-member predicates, while the fixed rule list and earlier implementation multiply only member predicates. This release follows the explicit rule list and records that choice.
4. **TS normalization differs across text/equation descriptions.** The new extractor uses variance after normalizing centroid coordinates by body height. This dimensionless convention is explicit; it does not claim to reconstruct the unknown historical extractor.
5. **No silhouette-to-pose substitute is physically identifiable for all six GIATs.** When pose is absent, reliabilities become zero. We do not manufacture anatomical landmarks or claim a working physical-trait branch from unavailable pose.
6. **Reconstructed paired p-values are not research evidence.** Do not use them to establish significance. Sorting synthetic tuples can impose artificial correlation.
7. **SVM/RF closed-set classification cannot identify unseen test labels directly.** Their new gallery adaptation is documented and must not be presented as an exact reproduction of Table 5.

## Release boundary

The package can be shared as reconstructed implementation plus software validation. It is not yet a complete experimental reproduction of the paper and does not close the reviewer requests above. `scripts/release_gate.py` intentionally returns a nonzero status until the research evidence checklist is completed truthfully. Software test success does not change this research gate.
