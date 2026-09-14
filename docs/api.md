# Local API contract

The server binds to loopback by default and has no authentication. Do not expose it to an untrusted network. Responses redact raw file URIs, collector IDs, animal IDs, device IDs, and session/site identifiers.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, inference mode, and claim policy. |
| `POST` | `/v1/recordings` | Register a manifest entry. |
| `POST` | `/v1/annotations` | Add one annotation segment. |
| `POST` | `/v1/recordings/{recording_id}/segments` | Add one segment with the path supplying `recording_id`. |
| `POST` | `/v1/predict` | Return an uncalibrated baseline score, abstention, and retrievable evidence. |
| `GET` | `/v1/examples/{segment_id}` | Return a redacted example for local review. |
| `GET` | `/v1/evaluations/latest` | Return an evaluation artifact loaded at startup, if provided. |

## Prediction request

Register a recording and annotation first, then query by segment ID:

```json
{"segment_id": "seg_0001"}
```

The response contains `confidence_kind: "uncalibrated_model_score"`. `predicted_label` is `null` when the model abstains; inspect `abstention_reason` and `ood_flags`. A result is an association hypothesis, not an animal-language translation.

The service reports `inference_mode: "in_sample_demo"` when it fits on the loaded fixture corpus. That mode is for API contract testing only and must not be treated as held-out evaluation. Artifact mode is selected with `--model`.
