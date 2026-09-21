# Data contract

One NPZ file per sequence, loaded with `allow_pickle=False`:

- `silhouettes`: numeric T×H×W, at least three frames; nonzero means foreground. Empty/nonfinite frames are rejected rather than silently discarded.
- `pose` (optional): float T×17×3 COCO keypoints, columns x/y/confidence, same original coordinate system and temporal order as silhouettes. Confidence below0.50 is invalid. Coordinates must not be normalized to a different crop. Record estimator name/version separately.

`samples.csv` fields:

`sample_id,subject_id,split,role,view,condition,sequence,path,sha256,source_kind`

- IDs are strings; unique sample IDs required.
- `split`: train / val / test; identities must be disjoint across splits.
- `role`: train for training; gallery or probe for validation/test.
- `path`: relative to the CSV directory; absolute paths also resolve.
- `sha256`: SHA-256 of sequence NPZ bytes.
- `source_kind`: `research` or `synthetic_fixture`; never relabel fixture inputs as research.
- Every probe identity must have gallery coverage in every eligible gallery view.

Train-only robust GIAT fitting is recomputed deterministically from the locked training data. The stored scaler supports audit. Full source-byte hashes detect changes; this is a content-integrity check, not biometric near-duplicate detection. All identical sequence hashes are rejected, even within a split, to force review.

For CASIA-style raw folders, run:

```bash
python scripts/import_casia.py --root /authorized/CASIA-B --splits /path/subject_splits.json --out research/data
```

The split file maps train/val/test to subject lists. A validation subset from001–074 reduces the training set; it must be declared as a development contract. To claim the standard74-training-subject benchmark, establish validation/model-selection on a development fold and add a documented full74-subject refit. The current runner does not perform this automatically.

The importer supplies silhouettes only. Without separately measured pose the symbolic branch is neutral, and that run must not be called a full physical-trait TriTrust evaluation. Use generic NPZ conversion with explicit manifests for OU-MVLP/other sources; do not guess official subject lists from numerical ID ordering.
