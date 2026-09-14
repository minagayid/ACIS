# Annotation guide

The annotation contract separates observable evidence from interpretation. Annotators must not label a dog as “happy,” “sad,” “angry,” “lonely,” or “trying to say X.” Those are hypotheses, not ground truth for this slice.

## Context labels

- `play_context`: play activity is visibly or explicitly documented.
- `separation_context`: owner absence, departure, or reunion is documented.
- `alert_context`: the animal responds to a sound, person, object, or environmental event.
- `food_context`: food preparation, feeding, or food anticipation is documented.
- `resting_context`: no target stimulus is present and the animal is resting or neutral.
- `unknown`: evidence is insufficient, contradictory, or not visible.

Labels may be multi-valued when the evidence genuinely supports more than one context. The baseline currently trains only on single-label examples; multi-label records remain valid data and are flagged for a future head.

## Record observations, not explanations

Prefer:

- “standing, head oriented toward doorbell”;
- “food bowl visible; animal approaches feeding area”;
- “owner leaves frame at 00:12; vocalization begins at 00:15”;
- “receiver response: orientation; latency: 350 ms.”

Avoid:

- “the dog is anxious”;
- “the bark means danger”;
- “the dog wants attention”;
- “the animal understands the command.”

## Quality and disagreement

- Preserve `unknown` and quality flags instead of forcing a guess.
- Use at least two independent annotators for evaluation samples.
- Keep annotator IDs, annotation version, timestamps, and adjudication decisions.
- Mark overlapping vocalizations and background speech/noise when visible.
- Do not infer caller identity when the source is not observable; record uncertainty.
- Do not include raw recordings in Git. Keep sensitive media access-controlled.
