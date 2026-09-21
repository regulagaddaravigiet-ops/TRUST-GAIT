# Implementation and paper mapping

| Component | Implementation | Status / choice |
|---|---|---|
| GEI CNN | Conv 3×3 channels 32/64/128/256, BN/ReLU; pool first 3; GAP; linear 256 | Matches supplied architecture |
| Frame encoder | Conv32/pool/Conv64/GAP, projection128 | Retained from old source; paper only specifies output dimension |
| Temporal branch | 2-layer bidirectional LSTM, hidden256 each direction, dropout0.3, attention, projection256 | Matches supplied architecture |
| Fusion | GEI+temporal → linear512×256+ReLU; append eight weighted rules → linear264×256 | Retained old implementation; exact historical weights absent |
| Predicates | sigmoid(10(a−0.5)); q·p+(1−q)0.5 | Matches fixed specification |
| Rules | Product of members; reliability=min member q | Matches explicit eight-rule list |
| Rule consistency | Mean ReLU(0.5−rule), weight0.1 | Constant wrt model parameters; threshold0.5 is declared implementation choice |
| Triplet | Batch-hard on L2-normalized embeddings, margin0.30, weight1 | Weight and mining strategy are new explicit choices |
| Optimizer | Adam 1e−4; weight decay1e−5; batch32 research / batch8 fixture | Weight decay implements parameter regularization |
| Selection | Maximum validation mean cross-view Rank-1, earliest tie, patience15 | Explicit new selection rule; historical selection unknown |
| Augmentation | None in all matched new models | Historical exact policy absent; not silently invented |
| Input normalization | Binary 0/1 silhouettes; crop/center/resize/pad | Equation 5 z-score is not applied; ambiguity disclosed |
| Frame sampling | Deterministic linearly spaced 30 frames; ordered | No random order; new explicit convention |
| GIAT scaling | Training 1st/99th percentiles and median; missing q=0 | No test fitting |
| Pose | COCO17 x,y,confidence in original silhouette coordinates | Upstream estimator and its revision must be supplied for research |
| Short gaps | Internal ≤3 frames linearly interpolated; endpoint min confidence | Longer gaps not bridged |
| SR / SWC | Peaks in smoothed ankle separation, min distance3 and prominence0.01; CV regularity | Proxy definition; cycle spacing NOT learned from historical training data |
| BS | Hip-centered left leg versus x-mirrored right leg | Explicit aligned coordinate convention |
| TS | exp(−sum variance(normalized torso centroid)−variance torso angle) | Dimensionless normalization choice |
| MS | exp(−mean norm second difference / body height) over contiguous valid triples | Does not bridge long gaps |
| LRC | exp(−SD mean-leg-length / torso-length) | Pose chain defined in source |
| Explanation | Frozen neural+gallery; intervene on weighted rule vector | Different from input-space IG/SHAP; not Table 6 reproduction |

Important: the fixture exercises code paths. Two epochs on eight synthetic identities do not validate recognition, causal explanations, physiological descriptor accuracy or real-world robustness. No original benchmark checkpoint is included. Existing source was reused where structurally compatible; source hashes bind the executed code to its outputs.
