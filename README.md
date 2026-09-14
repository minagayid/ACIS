# ACIS Context Lab

ACIS Context Lab is a small, reproducible research scaffold for studying animal vocalizations in context.

It is the first, deliberately narrow slice of the Animal Communication Intelligence System (ACIS) vision from the shared design conversation. The project starts with domestic dogs, observable context labels, transparent acoustic features, subject-disjoint evaluation, evidence retrieval, threshold-based uncertainty handling, and a human review loop.

It does **not** translate animal language, infer private emotions, or generate animal signals. A model prediction is a hypothesis about an observed association. Establishing communicative function requires controlled experiments and independent replication.

## What is included

- Separate recording and annotation contracts with provenance.
- JSONL examples that are safe to commit because they contain metadata only, not recordings.
- WAV validation and interval checks.
- Deterministic acoustic feature extraction from supplied measurements.
- Animal/session/site-aware splits to reduce leakage.
- A transparent nearest-centroid baseline with threshold-based abstention.
- Nearest-example retrieval so predictions are inspectable.
- Evaluation metrics for macro-F1, balanced accuracy, Brier score, and majority-class comparison.
- A dependency-free JSON HTTP service and CLI.
- Data/model cards, annotation guidance, a protocol outline, and an adversarial decision log.

## Research question for the first vertical slice

> On short domestic-dog vocalization segments, can a transparent acoustic baseline predict a small set of operationally defined contexts on unseen animals, while exposing uncertainty and evidence?

The initial labels are intentionally behavioral and operational:

| Label | Operational definition |
| --- | --- |
| `play_context` | Play activity is visibly or explicitly documented. |
| `separation_context` | Owner absence, departure, or reunion is documented. |
| `alert_context` | The animal responds to a sound, person, object, or environmental event. |
| `food_context` | Food preparation, feeding, or food anticipation is documented. |
| `resting_context` | No target stimulus is present and the animal is resting or neutral. |
| `unknown` | Evidence is insufficient, contradictory, or not visible. |

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m acis.cli validate \
  --recordings data/examples/recordings.jsonl \
  --annotations data/examples/annotations.jsonl
python -m acis.cli evaluate \
  --recordings data/examples/recordings.jsonl \
  --annotations data/examples/annotations.jsonl \
  --allow-synthetic \
  --output evaluations/example-report.md
```

Start the local service with the example corpus loaded:

```bash
python -m acis.cli serve \
  --recordings data/examples/recordings.jsonl \
  --annotations data/examples/annotations.jsonl \
  --allow-synthetic
```

Then inspect it:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/examples/seg_0001
```

The service is intentionally local and in-memory. It is an API boundary for experiments, not a production deployment.

The fixture service consumes the six acoustic summaries supplied in the annotation JSONL. It does not read WAV bytes or infer features directly from audio yet. Start it with `--allow-synthetic` only for a contract demo; use `--model` with an artifact trained from a separate training manifest for a real holdout workflow.

## Repository layout

```text
configs/                    Example label and baseline configuration
data/examples/              Metadata-only example corpus
docs/                       Scientific, annotation, governance, and design notes
evaluations/                Generated report location and fixtures
schemas/                    JSON Schema contracts
src/acis/                   Python package
tests/                      Standard-library unit and contract tests
```

## Scientific guardrails

The repository enforces the following vocabulary in its documentation and API responses:

- `associated_context`, not “meaning”;
- `hypothesis`, not “translation”;
- `predictive_confidence`, an uncalibrated baseline score until calibration is demonstrated, not a probability of animal intent;
- `abstained`, when evidence is weak or out of distribution;
- `evidence`, with retrievable examples and metadata provenance.

Do not use random clip-level splits. Clips from one session can share identity, room acoustics, equipment, background events, and observer behavior. The example evaluator splits by animal and keeps session/site metadata attached for later audits.

Before claiming that a signal has a communicative function, use a preregistered, welfare-reviewed, randomized playback or interaction study with matched controls, blinded scoring where possible, predefined outcomes, and replication. See [`docs/research-protocol.md`](docs/research-protocol.md).

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

The test suite uses only the Python standard library. Optional development tooling is declared in `pyproject.toml`.

## Scope roadmap

1. Freeze an annotation ontology and collect a consented, subject-disjoint dog corpus.
2. Replace measured-feature fixtures with a canonical audio preprocessing pipeline.
3. Add independent annotators, agreement statistics, and negative/background windows.
4. Add device/site holdouts and shortcut-learning diagnostics.
5. Add calibrated models and out-of-distribution monitoring after the data contract is stable.
6. Design a controlled behavioral validation study before any functional claim.
7. Treat dialects, sequence structure, synthetic signals, cross-species comparison, and generation as later research questions—not MVP capabilities.

## License

MIT. See [`LICENSE`](LICENSE).
