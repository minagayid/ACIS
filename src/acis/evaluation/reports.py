"""Human-readable evaluation report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_report(
    *,
    metrics: dict[str, Any],
    split: dict[str, list[str]],
    model: dict[str, Any],
    dataset_status: str,
    dataset_note: str,
) -> str:
    per_animal = metrics.get("per_animal", {})
    per_animal_lines = "\n".join(
        f"| `{animal_id}` | {values['count']:.0f} | {values['accuracy']:.3f} |"
        for animal_id, values in sorted(per_animal.items())
    ) or "| — | 0 | — |"
    title = "ACIS Context Lab synthetic contract smoke report" if dataset_status == "synthetic_fixture" else "ACIS Context Lab baseline report"
    warning = (
        "> **Synthetic fixture warning:** these features are hand-authored templates for software tests. "
        "The score is not evidence of performance on dogs.\n"
        if dataset_status == "synthetic_fixture" else ""
    )
    return f"""# {title}

Dataset status: `{dataset_status}`

{warning}

## Interpretation boundary

This is a predictive evaluation of operational context labels. It is not evidence of animal intent, semantics, emotion, or translation.

## Dataset

{dataset_note}

| Partition | Animals | Segments |
| --- | --- | --- |
| Train | {', '.join(split.get('train_animals', [])) or '—'} | {', '.join(split.get('train_segments', [])) or '—'} |
| Validation | {', '.join(split.get('validation_animals', [])) or '—'} | {', '.join(split.get('validation_segments', [])) or '—'} |
| Test | {', '.join(split.get('test_animals', [])) or '—'} | {', '.join(split.get('test_segments', [])) or '—'} |

## Model

```json
{json.dumps({key: model.get(key) for key in ('model_name', 'model_version', 'feature_names', 'abstention_threshold')}, indent=2)}
```

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {metrics.get('accuracy', 0):.3f} |
| Macro-F1 | {metrics.get('macro_f1', 0):.3f} |
| Balanced accuracy | {metrics.get('balanced_accuracy', 0):.3f} |
| Brier score | {metrics.get('brier_score', 0):.3f} |
| Abstention rate | {metrics.get('abstention_rate', 0):.3f} |
| Selective accuracy | {metrics.get('selective_accuracy', 0):.3f} |
| Majority-class accuracy | {metrics.get('majority_accuracy', 0):.3f} |

## Test performance by animal

| Animal | Count | Accuracy |
| --- | ---: | ---: |
{per_animal_lines}

## Required follow-up

- Freeze a real evaluation manifest before model tuning.
- Repeat with held-out sessions, devices, sites, and seasons where available.
- Add background/no-call negatives and an identity shortcut diagnostic.
- Report annotator agreement and allow `unknown`/disagreement.
- Do not make a functional or semantic claim without controlled behavioral validation.
"""


def write_report(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
