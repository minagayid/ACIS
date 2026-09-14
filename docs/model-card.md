# Model card: standardized nearest centroid

## Model summary

The baseline standardizes six supplied acoustic summary features using training data only, learns one centroid per operational context, and converts distances into a soft score distribution. It abstains when the top score is below a configured threshold. The scores are not calibrated probabilities; Brier score is reported as a diagnostic and a future calibration layer is required before interpreting them probabilistically.

This is a diagnostic baseline. Its purpose is to measure whether a narrow task contains recoverable signal and to provide inspectable evidence, not to compete with modern audio models. The API labels its confidence as `uncalibrated_model_score`; it must not be read as a probability of correctness.

## Features

- segment duration;
- peak frequency;
- spectral centroid;
- energy;
- repetition count;
- voiced fraction.

## Evaluation protocol

The CLI groups partitions by `animal_id`. A serious study must additionally audit session, day, site, device, season, social group, and population holdouts. Random segment splits are invalid when nearby segments share recording conditions.

## Intended use

Research exploration of context-associated vocalization patterns in the data domain described by the data card.

## Out of scope

Translation, intent/emotion detection, medical advice, unrestricted signal generation, cross-species claims, and unsupervised semantic alignment.

## Failure modes

The model may learn individual identity, device artifacts, room acoustics, human speech, trigger leakage, or annotation artifacts. The API therefore returns evidence counts, nearest examples, training-membership metadata, an abstention flag, and a claim-scope disclaimer. These controls do not eliminate confounding; they make it easier to detect and investigate.
