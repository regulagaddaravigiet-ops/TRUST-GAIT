# Reviewer navigation

This archive provides a reconstructed executable implementation and synthetic software validation. It does **not** contain the historical research evidence required to verify the manuscript's reported97.1% accuracy.

1. `EVIDENCE_AUDIT.md`: result-by-result source status and unresolved reviewer requirements.
2. `METHOD_MAPPING.md`: implementation details and deviations/ambiguities, including non-trainable rule consistency.
3. `../validation/FINAL_STATUS.json`: actual software execution summary and separate research gate.
4. `../validation/executed/evidence_ledger.csv`: run-level hashes connecting inputs, checkpoint, predictions and metrics.
5. `../validation/executed/runs/`: five seeds for all five author-model implementations.
6. `../validation/executed/audits/`: per-rule intervention and attribution logs.
7. `../reported/`: unchanged aggregate CSVs from the earlier supplied package.
8. `../reported/reconstructed/`: reconstructed historical seed tuples, isolated from experimental analysis.
9. `../supplementary/SUPPLEMENT.html`: readable supplementary guide.
10. `PROTOCOLS.md`: official evaluator and modern baseline work still required.

The supplied manuscript is retained verbatim in `Manuscript_as_supplied.docx` for traceability, not endorsed as empirically validated. In particular, the unidentified development tuple remains in its abstract/conclusion, and its reconstructed significance values cannot replace original seed logs.

A pass in software tests means the tested code paths execute correctly under the fixture contract. It does not imply official benchmark agreement, original experimental recovery, physiological validity, or reviewer acceptance.
