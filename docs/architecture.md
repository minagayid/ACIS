# Architecture

```text
recording manifest ─┐
                    ├─> validated joined examples ─> leakage-aware split ─> transparent baseline
annotation JSONL ───┘                                                        │
                                                                               ├─> score distribution + abstention
                                                                               └─> nearest evidence examples
                                                                                          │
                                                                                       JSON API
```

## Layers

1. **Raw provenance:** `Recording` stores where a source came from without copying media into Git.
2. **Observable annotation:** `AnnotationSegment` stores intervals, labels, visible behavior, trigger notes, and quality flags.
3. **Feature adapter:** deterministic acoustic summaries provide a stable interface for the fixture and a future audio backend.
4. **Dataset join and split:** recording metadata and annotations are joined only after independent validation; partitions are grouped by animal.
5. **Species routing:** the multispecies mode trains an independent standardized nearest-centroid model per species and filters evidence retrieval to the query species.
6. **Baseline inference:** a standardized nearest-centroid model produces an uncalibrated score distribution and can abstain.
7. **Evidence retrieval:** nearest labeled examples are returned with provenance fields.
8. **Local service:** the HTTP layer exposes contracts for annotation tools and later research interfaces.

## Deliberate omissions

No database, vector store, foundation model, model-serving cluster, unrestricted LLM explanation, playback actuator, or continuous training loop is included. Those components would increase operational surface area before the dataset and evaluation design are trustworthy.

## Extension points

- Replace the measured-feature adapter with canonical WAV preprocessing.
- Add multi-label learning without changing the annotation schema.
- Add calibrated classifiers and explicit OOD thresholds.
- Add model/dataset version IDs to an experiment registry.
- Add per-species calibration and cross-population/device holdouts before comparing species.
- Add controlled playback only behind a separate welfare-reviewed protocol.
