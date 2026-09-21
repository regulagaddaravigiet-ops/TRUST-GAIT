# Command reference

Run from the extracted package root after `pip install -e . --no-deps` and installing requirements.

```bash
# All fixture training, five-seed rule audits, robustness, knockouts, timing and statistics:
python scripts/run_all.py --out validation/fresh

# Verify included artifact provenance and metrics:
python scripts/verify_executed.py

# Inspect original evidence gaps (expected exit code2 with supplied inputs):
python scripts/release_gate.py

# Replot the supplied aggregates and summarize the included fixture runs:
python scripts/make_supplement.py

# Model-only timing, one sequence per forward pass:
python scripts/benchmark_latency.py --checkpoint validation/executed/runs/seed_11/tritrust/best.pt --out new_timing.json
```

`run_all.py` does not invoke the research gate because passing software checks cannot satisfy missing historical evidence. The report generator reads the included `validation/executed` tree; newly executed trees remain separate until explicitly reviewed.

| Script | Required input | Output |
|---|---|---|
| import_casia.py | Authorized raw sequence folders; explicit subject splits | NPZ sequences and hash manifest |
| diagnostics.py | Validated manifest | Class/view counts and leakage report |
| run_research.py | Real research manifest and completed contract | Actual new training checkpoints/metrics |
| evaluate_embeddings.py | Manifest; NPZ embedding+sample_ids | Shared generic gallery/probe evaluation |
| audit_rules.py | TriTrust checkpoint; matching manifest | Single-rule changes, IG and exact Shapley |
| robustness.py | Deep checkpoint; matching manifest | Defined perturbation predictions |
| ablate_inference.py | TriTrust checkpoint; matching manifest | Frozen-checkpoint branch knockouts |
| compare_seeds.py | Matched per-seed run directories | Paired difference and confidence interval |
| check_external_baseline.py | Completed external provenance record | Hash/config consistency check |
| verify_release.py | Included SHA256SUMS | Archive integrity check |

GPU execution requires an appropriate PyTorch installation. Included runs used CPU, two threads. Timings on the fixture are not comparable to the manuscript's unspecified original hardware. Intermediate source modifications invalidate recorded source hashes; retrain and regenerate validation outputs after modifying code.
