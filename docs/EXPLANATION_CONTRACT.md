# Executed explanation contract

The audit compares three rankings of the **same eight reliability-weighted rule coordinates**. The neural representation and gallery embeddings are frozen. Each coordinate is neutralized to zero at the input to final fusion.

- Rule removal: observe all eight individual removals and select the largest score reduction.
- Integrated Gradients: straight-line path from all-zero rule vector to the observed vector;32 trapezoidal integration intervals.
- Exact Shapley: all256 subsets of the eight coordinates; no sampling approximation.

The target is the originally predicted gallery identity. Its score is obtained by max cosine over that identity's gallery sequences followed by a class softmax with temperature0.1. This is a fixed diagnostic temperature, **not** validation-calibrated probability. A flip changes the winning identity.

All methods are evaluated with a one-coordinate removal budget. Attribution-computation cost differs: rule removal uses8 alternatives, IG33 gradient points, Shapley256 coalitions. Therefore this is not an equal-compute comparison. Rule removal also selects on the very score-drop quantity being reported, so apparent superiority is selection-biased and descriptive.

Outputs:
- `rule_interventions.csv`: all individual score changes, activations, flips and scope.
- `attributions.csv`: IG/Shapley/removal attribution for each coordinate.
- `matched_top1_audit.csv`: selected coordinate and observed effect for each method.
- `audit_contract.json`: checkpoint/manifest hashes and exact definitions.

`positive_effect` is a binary positive-score-drop diagnostic. It is not the manuscript's intervention-precision score. The paper does not supply the original correctness reference needed for that metric. No Table6 number is reconstructed from these fixture outputs.
