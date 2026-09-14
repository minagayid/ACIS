# Decision log

## 2026-09-14 — Narrow the first release

**Decision:** build a species-specific context-analysis scaffold for domestic dogs.

**Reason:** the shared vision combines detection, identity, repertoire discovery, functional inference, semantics, generation, and cross-species alignment. Those are separate scientific questions. A narrow vertical slice has a falsifiable evaluation boundary.

## 2026-09-14 — Use observable labels

**Decision:** use `play_context`, `separation_context`, `alert_context`, `food_context`, `resting_context`, and `unknown`.

**Reason:** emotion and intent labels invite circular annotation and human projection. Observable context can be audited and independently reviewed.

## 2026-09-14 — Split by animal

**Decision:** the default evaluator holds out animals rather than random clips.

**Reason:** random clips can leak identity, session, room, device, observer, or trigger artifacts. Production studies should add site/device/session holdouts.

## 2026-09-14 — Keep the baseline transparent

**Decision:** standardized nearest centroid over six measured features.

**Reason:** the first bottleneck is data quality and leakage, not model scale. The baseline is easy to inspect, serialize, test, and replace.

## 2026-09-14 — No generated animal signals

**Decision:** generation and playback remain out of scope.

**Reason:** generated stimuli require welfare review, experimental controls, and species-specific safeguards. An unvalidated generator could cause harm and create false scientific confidence.
