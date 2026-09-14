# ACIS Context Lab synthetic contract smoke report

Dataset status: `synthetic_fixture`

> **Synthetic fixture warning:** these features are hand-authored templates for software tests. The score is not evidence of performance on dogs.


## Interpretation boundary

This is a predictive evaluation of operational context labels. It is not evidence of animal intent, semantics, emotion, or translation.

## Dataset

16 recordings and 16 annotations. The split is grouped by animal; this fixture is a contract/demo corpus, not evidence of real-world performance.

| Partition | Animals | Segments |
| --- | --- | --- |
| Train | dog_001, dog_002 | seg_0001, seg_0002, seg_0003, seg_0004, seg_0005, seg_0006, seg_0007, seg_0008 |
| Validation | dog_003 | seg_0009, seg_0010, seg_0011, seg_0012 |
| Test | dog_004 | seg_0013, seg_0014, seg_0015, seg_0016 |

## Model

```json
{
  "model_name": "standardized-nearest-centroid",
  "model_version": "0.1.0",
  "feature_names": [
    "duration_ms",
    "peak_frequency_hz",
    "spectral_centroid_hz",
    "energy_db",
    "repetition_count",
    "voiced_fraction"
  ],
  "abstention_threshold": 0.55
}
```

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 1.000 |
| Macro-F1 | 1.000 |
| Balanced accuracy | 1.000 |
| Brier score | 0.036 |
| Abstention rate | 0.000 |
| Selective accuracy | 1.000 |
| Majority-class accuracy | 0.250 |

## Test performance by animal

| Animal | Count | Accuracy |
| --- | ---: | ---: |
| `dog_004` | 4 | 1.000 |

## Required follow-up

- Freeze a real evaluation manifest before model tuning.
- Repeat with held-out sessions, devices, sites, and seasons where available.
- Add background/no-call negatives and an identity shortcut diagnostic.
- Report annotator agreement and allow `unknown`/disagreement.
- Do not make a functional or semantic claim without controlled behavioral validation.
