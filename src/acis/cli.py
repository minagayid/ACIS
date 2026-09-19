"""Command-line entry points for local validation, evaluation, and serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from acis.api.service import ACISStore, create_server
from acis.audio.features import acoustic_vector
from acis.config import ProjectConfig, load_config
from acis.dataset.loaders import join_training_examples
from acis.dataset.split import grouped_split
from acis.evaluation.metrics import evaluate_predictions
from acis.evaluation.reports import render_report, write_report
from acis.ingest.manifest import validate_dataset
from acis.models.baseline import ContextBaseline
from acis.models.inference import predict_with_evidence
from acis.models.loader import create_model, load_model_artifact


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recordings", required=True, help="recording manifest JSONL")
    parser.add_argument("--annotations", required=True, help="annotation JSONL")
    parser.add_argument("--config", default="configs/dog_baseline.json", help="runtime JSON configuration")


def _is_synthetic(path: str) -> bool:
    normalized = Path(path).as_posix().rstrip("/")
    return normalized.startswith("data/examples/") or normalized.endswith("/data/examples") or "/data/examples/" in normalized


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _missing_features(example) -> bool:
    try:
        acoustic_vector(example.annotation)
    except ValueError:
        return True
    return False


def _artifact_metadata(config: ProjectConfig, recordings, annotations, training_examples, config_path: str, dataset_status: str) -> dict[str, Any]:
    return {
        "dataset_status": dataset_status,
        "dataset_fingerprint": _fingerprint({"recordings": [item.to_dict() for item in recordings], "annotations": [item.to_dict() for item in annotations]}),
        "config_fingerprint": _fingerprint(asdict(config)),
        "config_path": str(config_path),
        "training_segment_ids": [item.annotation.segment_id for item in training_examples],
        "training_animal_ids": sorted({item.animal_id for item in training_examples}),
        "training_species": sorted({item.species for item in training_examples}),
        "species_scope": list(config.species_scope),
    }


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    if config.model not in {ContextBaseline.name, "species-routed-standardized-nearest-centroid"}:
        raise ValueError(f"unsupported configured model: {config.model}")
    recordings, annotations = validate_dataset(args.recordings, args.annotations)
    mismatched_species = sorted({item.species for item in recordings if item.species not in config.species_scope})
    if mismatched_species:
        raise ValueError(f"recordings contain species outside configured species scope {list(config.species_scope)}: {mismatched_species}")
    mismatched_labels = sorted({label for item in annotations for label in item.context_labels if label not in config.labels})
    if mismatched_labels:
        raise ValueError(f"annotations contain labels outside config: {mismatched_labels}")
    return config, recordings, annotations, join_training_examples(recordings, annotations)


def _metrics(model: ContextBaseline, examples, *, baseline_labels: list[str] | None = None):
    truth: list[str] = []
    predicted: list[str] = []
    probabilities: list[dict[str, float]] = []
    abstained: list[bool] = []
    animal_ids: list[str] = []
    excluded = {"unknown": 0, "multi_label": 0, "missing_acoustic_features": 0}
    for example in examples:
        if not example.is_single_label:
            excluded["multi_label"] += 1
            continue
        if example.target_label == "unknown":
            excluded["unknown"] += 1
            continue
        if _missing_features(example):
            excluded["missing_acoustic_features"] += 1
            continue
        result = predict_with_evidence(model, example.annotation, species=example.species, neighbors=None)
        truth.append(example.label)
        predicted.append(result.predicted_label if result.predicted_label is not None else "abstained")
        probabilities.append(result.probabilities)
        abstained.append(result.abstained)
        animal_ids.append(example.animal_id)
    metrics = evaluate_predictions(
        truth,
        predicted,
        probabilities,
        abstained=abstained,
        animal_ids=animal_ids,
        baseline_labels=baseline_labels,
    )
    metrics["excluded"] = excluded
    return metrics


def command_validate(args: argparse.Namespace) -> int:
    config, recordings, annotations, examples = _load(args)
    excluded = {
        "unknown": sum(item.target_label == "unknown" for item in examples if item.is_single_label),
        "multi_label": sum(not item.is_single_label for item in examples),
        "missing_acoustic_features": sum(
            1 for item in examples
            if item.is_single_label and item.target_label != "unknown" and _missing_features(item)
        ),
    }
    print(json.dumps({"valid": True, "config_species": config.species, "species_scope": list(config.species_scope), "recordings": len(recordings), "annotations": len(annotations), "excluded_from_single_label_baseline": excluded}, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config, recordings, annotations, examples = _load(args)
    if _is_synthetic(args.recordings) and not args.allow_synthetic:
        raise SystemExit("refusing to train from the synthetic fixture without --allow-synthetic")
    model = create_model(config.model, abstention_threshold=config.abstention_threshold).fit(examples, allow_excluded=args.allow_exclusions)
    model.artifact_metadata = _artifact_metadata(config, recordings, annotations, model.training_examples, args.config, "synthetic_fixture" if _is_synthetic(args.recordings) else "empirical")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(destination), "labels": model.labels, "dataset_status": model.artifact_metadata["dataset_status"]}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config, recordings, annotations, examples = _load(args)
    if _is_synthetic(args.recordings) and not args.allow_synthetic:
        raise SystemExit("refusing to evaluate the synthetic fixture without --allow-synthetic")
    split = grouped_split(examples)
    model = create_model(config.model, abstention_threshold=config.abstention_threshold).fit(split.train, allow_excluded=args.allow_exclusions)
    metrics = {
        "validation": _metrics(model, split.validation, baseline_labels=[item.label for item in split.train]),
        "test": _metrics(model, split.test, baseline_labels=[item.label for item in split.train]),
    }
    split_payload = {
        "train_animals": list(split.train_animals),
        "validation_animals": list(split.validation_animals),
        "test_animals": list(split.test_animals),
        "train_segments": [item.annotation.segment_id for item in split.train],
        "validation_segments": [item.annotation.segment_id for item in split.validation],
        "test_segments": [item.annotation.segment_id for item in split.test],
    }
    report_metrics = metrics["test"]
    report = render_report(
        metrics=report_metrics,
        split=split_payload,
        model=model.to_dict(),
        dataset_status="synthetic_fixture" if _is_synthetic(args.recordings) else "empirical",
        dataset_note=(
            f"{len(recordings)} recordings and {len(annotations)} annotations. "
            "The split is grouped by animal; this fixture is a contract/demo corpus, not evidence of real-world performance."
        ),
    )
    write_report(args.output, report)
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"dataset_status": "synthetic_fixture" if _is_synthetic(args.recordings) else "empirical", "metrics": metrics, "split": split_payload, "model": model.to_dict()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": args.output, "metrics_output": args.metrics_output, "dataset_status": "synthetic_fixture" if _is_synthetic(args.recordings) else "empirical", "validation": metrics["validation"], "test": metrics["test"]}, indent=2))
    return 0


def command_serve(args: argparse.Namespace) -> int:
    config, recordings, annotations, examples = _load(args)
    store = ACISStore()
    for recording in recordings:
        store.add_recording(recording)
    for annotation in annotations:
        store.add_annotation(annotation)
    if args.model:
        artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
        expected_config = _fingerprint(asdict(config))
        artifact_config = artifact.get("metadata", {}).get("config_fingerprint")
        if artifact_config and artifact_config != expected_config:
            raise ValueError("model artifact configuration does not match --config")
        store.model = load_model_artifact(artifact)
        store.inference_mode = "artifact"
    else:
        if not args.allow_synthetic and _is_synthetic(args.recordings):
            raise SystemExit("refusing to serve the synthetic fixture without --allow-synthetic")
        store.model = create_model(config.model, abstention_threshold=config.abstention_threshold).fit(examples, allow_excluded=args.allow_exclusions)
        store.inference_mode = "in_sample_demo"
    store.training_segment_ids = set(item.annotation.segment_id for item in store.model.training_examples)
    store.training_segment_ids.update(store.model.artifact_metadata.get("training_segment_ids", []))
    if args.evaluation:
        store.evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    server = create_server(store, host=args.host, port=args.port, allow_unsafe_host=args.allow_unsafe_host)
    if args.allow_unsafe_host and args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: unauthenticated ACIS API is exposed beyond loopback; do not use on an untrusted network.", flush=True)
    print(f"ACIS Context Lab listening on http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACIS Context Lab local research tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate recording and annotation JSONL")
    _paths(validate)
    validate.set_defaults(handler=command_validate)

    train = subparsers.add_parser("train", help="fit and serialize the transparent baseline")
    _paths(train)
    train.add_argument("--output", default="evaluations/baseline-model.json")
    train.add_argument("--allow-synthetic", action="store_true")
    train.add_argument("--allow-exclusions", action="store_true", help="explicitly skip multi-label and unknown annotations")
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate", help="run grouped validation/test evaluation")
    _paths(evaluate)
    evaluate.add_argument("--output", default="evaluations/example-report.md")
    evaluate.add_argument("--metrics-output", default="evaluations/example-metrics.json")
    evaluate.add_argument("--allow-synthetic", action="store_true")
    evaluate.add_argument("--allow-exclusions", action="store_true", help="explicitly skip multi-label and unknown annotations")
    evaluate.set_defaults(handler=command_evaluate)

    serve = subparsers.add_parser("serve", help="serve the local JSON API")
    _paths(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--model", help="serialized model artifact; without it, service runs in in-sample demo mode")
    serve.add_argument("--evaluation", help="machine-readable evaluation JSON to expose at /v1/evaluations/latest")
    serve.add_argument("--allow-synthetic", action="store_true")
    serve.add_argument("--allow-exclusions", action="store_true", help="explicitly skip multi-label and unknown annotations")
    serve.add_argument("--allow-unsafe-host", action="store_true", help="allow binding beyond loopback; unauthenticated")
    serve.set_defaults(handler=command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
