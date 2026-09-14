# Problem definition

## In scope

ACIS Context Lab studies whether transparent acoustic features contain reproducible signal for predicting operationally observed context labels in a single species and setting.

The first slice uses domestic dogs in consented household recordings. The unit of analysis is a time-bounded vocalization segment linked to a recording manifest and a human annotation.

## Falsifiable question

> Given a short segment from a dog not present in training, does the baseline predict the documented context better than a training-set majority baseline, while remaining calibrated and able to abstain?

## Claims the system may support

- A segment is acoustically similar to previously annotated examples.
- A model predicts an operational context label under a specified evaluation split.
- The prediction has a measured confidence, known provenance, and an abstention policy.
- Performance changes across animals, sessions, devices, or sites.

## Claims the system cannot support by itself

- The dog “said” a human-language sentence.
- A signal has a universal meaning across dogs, breeds, regions, or species.
- The model detects a private emotion, pain, medical condition, or intent.
- Clustering proves a vocabulary; sequence prediction proves grammar; or classification proves communication.
- A synthetic signal is part of the animal’s natural communication system.

## Evidence ladder

1. **Observation:** an annotator records what was visible/audible and under what conditions.
2. **Prediction:** a model estimates an operational label from held-out data.
3. **Association hypothesis:** repeated predictive association survives shortcut and negative-control checks.
4. **Functional hypothesis:** a preregistered intervention changes a predefined receiver response.
5. **Replicated finding:** independent animals, sessions, and researchers reproduce the effect.

This repository implements levels 1–2 and prepares evidence for level 3. Levels 4–5 require a separate welfare-reviewed research program.
