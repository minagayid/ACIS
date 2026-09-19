# Widening ACIS across animals

ACIS now supports a species-routed baseline and shared recording metadata for
multispecies research. The route is deliberately conservative: the label
vocabulary and API can be shared, but feature scaling, centroids, evidence, and
uncertainty remain species-specific.

## Supported expansion pattern

1. Add a consented recording manifest with a stable species identifier.
2. Add population, region, habitat, domestic/wild status, and social-group
   metadata when those fields are known and safe to expose.
3. Collect observable context labels and response evidence using the same
   operational definitions where they remain valid.
4. Train one baseline per species with subject-disjoint splits.
5. Report performance per species and per holdout condition; never pool scores
   into a single cross-species claim.

`configs/species_profiles.json` records the intended sensor and annotation
surface for the first expansion targets. It is a planning registry, not proof
that data exists or that the listed modalities are sufficient.

## Running a routed model

Use `configs/multispecies_baseline.json` with a multispecies manifest:

```bash
python -m acis.cli validate \
  --recordings path/to/recordings.jsonl \
  --annotations path/to/annotations.jsonl \
  --config configs/multispecies_baseline.json

python -m acis.cli train \
  --recordings path/to/recordings.jsonl \
  --annotations path/to/annotations.jsonl \
  --config configs/multispecies_baseline.json \
  --output evaluations/multispecies-model.json
```

Every configured species needs enough eligible, documented-consent examples
for at least two context labels. If a species does not meet that bar, training
fails instead of silently borrowing another species' acoustic scale.

## Scientific boundary

The router does not create a universal animal language model. A prediction is a
species-specific association between measured signal features and an observed
context. Dialects, individual signatures, sequence structure, and controlled
playback remain separate research stages requiring independent annotations,
welfare review, and replication.
