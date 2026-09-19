"""Dependency-free local JSON API for the research scaffold."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from acis.dataset.loaders import TrainingExample, join_training_examples
from acis.models.baseline import ContextBaseline
from acis.models.inference import predict_with_evidence
from acis.schemas.annotation import AnnotationSegment
from acis.schemas.common import ValidationError
from acis.schemas.recording import Recording


class ACISStore:
    """In-memory store suitable for local experiments and contract tests."""

    def __init__(self) -> None:
        self.recordings: dict[str, Recording] = {}
        self.annotations: dict[str, AnnotationSegment] = {}
        self.model: ContextBaseline | None = None
        self.evaluation: dict[str, Any] | None = None
        self.inference_mode = "unconfigured"
        self.training_segment_ids: set[str] = set()
        self.unsafe_exposure = False

    @property
    def examples(self) -> list[TrainingExample]:
        return join_training_examples(list(self.recordings.values()), list(self.annotations.values()))

    def add_recording(self, recording: Recording) -> None:
        if recording.recording_id in self.recordings:
            raise ValidationError(f"recording already exists: {recording.recording_id}")
        self.recordings[recording.recording_id] = recording

    def add_annotation(self, annotation: AnnotationSegment) -> None:
        if annotation.recording_id not in self.recordings:
            raise ValidationError(f"unknown recording: {annotation.recording_id}")
        if annotation.segment_id in self.annotations:
            raise ValidationError(f"segment already exists: {annotation.segment_id}")
        recording = self.recordings[annotation.recording_id]
        if recording.duration_ms is not None and annotation.end_ms > recording.duration_ms:
            raise ValidationError(f"segment exceeds recording duration: {annotation.segment_id}")
        self.annotations[annotation.segment_id] = annotation


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _public_recording(recording: Recording) -> dict[str, Any]:
    return {
        "recording_id": recording.recording_id,
        "species": recording.species,
        "subspecies": recording.subspecies,
        "population": recording.population,
        "region": recording.region,
        "habitat": recording.habitat,
        "wild_or_domestic": recording.wild_or_domestic,
        "social_group_id": recording.social_group_id,
        "age_class": recording.age_class,
        "sex": recording.sex,
        "location_class": recording.location_class,
        "duration_ms": recording.duration_ms,
        "sample_rate_hz": recording.sample_rate_hz,
        "channels": recording.channels,
        "consent_status": recording.consent_status,
    }


def _public_annotation(annotation: AnnotationSegment) -> dict[str, Any]:
    return {
        "segment_id": annotation.segment_id,
        "recording_id": annotation.recording_id,
        "start_ms": annotation.start_ms,
        "end_ms": annotation.end_ms,
        "vocalization_type": annotation.vocalization_type,
        "context_labels": annotation.context_labels,
        "observed_behavior": annotation.observed_behavior,
        "annotation_confidence": annotation.annotation_confidence,
        "acoustic_features": annotation.acoustic_features,
        "response_observed": annotation.response_observed,
        "response_latency_ms": annotation.response_latency_ms,
        "quality_flags": annotation.quality_flags,
    }


def create_server(
    store: ACISStore,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    allow_unsafe_host: bool = False,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_unsafe_host:
        raise ValueError("refusing non-loopback host without allow_unsafe_host=True; the API has no authentication")
    store.unsafe_exposure = host not in {"127.0.0.1", "localhost", "::1"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ACISContextLab/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            # Keep research runs quiet and avoid logging potentially sensitive paths.
            return

        def _send(self, status: int, value: Any) -> None:
            payload = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValidationError("Content-Length must be an integer") from exc
            if length <= 0 or length > 1_000_000:
                raise ValidationError("request body must be between 1 byte and 1MB")
            try:
                value = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ValidationError("request body must be valid JSON") from exc
            if not isinstance(value, dict):
                raise ValidationError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP API
            path = unquote(urlparse(self.path).path.rstrip("/")) or "/"
            if path == "/health":
                self._send(HTTPStatus.OK, {
                    "status": "ok",
                    "service": "acis-context-lab",
                    "model_loaded": store.model is not None,
                    "inference_mode": store.inference_mode,
                    "recording_count": len(store.recordings),
                    "annotation_count": len(store.annotations),
                    "unsafe_exposure": store.unsafe_exposure,
                    "claims_policy": "context-associated hypothesis; not a translation or intent claim",
                })
                return
            if path.startswith("/v1/examples/"):
                segment_id = path.rsplit("/", 1)[-1]
                annotation = store.annotations.get(segment_id)
                if annotation is None:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "example not found"})
                    return
                recording = store.recordings[annotation.recording_id]
                self._send(HTTPStatus.OK, {"recording": _public_recording(recording), "annotation": _public_annotation(annotation)})
                return
            if path == "/v1/evaluations/latest":
                if store.evaluation is None:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "no evaluation loaded"})
                else:
                    self._send(HTTPStatus.OK, store.evaluation)
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "route not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP API
            path = unquote(urlparse(self.path).path.rstrip("/")) or "/"
            try:
                body = self._body()
                if path == "/v1/recordings":
                    recording = Recording.from_dict(body)
                    store.add_recording(recording)
                    self._send(HTTPStatus.CREATED, _public_recording(recording))
                    return
                if path == "/v1/annotations" or (path.startswith("/v1/recordings/") and path.endswith("/segments")):
                    if path != "/v1/annotations":
                        recording_id = path.split("/", 4)[3]
                        if "recording_id" in body and body["recording_id"] != recording_id:
                            raise ValidationError("path recording_id and body recording_id disagree")
                        body = {**body, "recording_id": recording_id}
                    annotation = AnnotationSegment.from_dict(body)
                    store.add_annotation(annotation)
                    self._send(HTTPStatus.CREATED, _public_annotation(annotation))
                    return
                if path == "/v1/predict":
                    if store.model is None:
                        self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no model loaded"})
                        return
                    segment_id = body.get("segment_id")
                    if segment_id is not None:
                        if not isinstance(segment_id, str) or segment_id not in store.annotations:
                            self._send(HTTPStatus.NOT_FOUND, {"error": "segment not found"})
                            return
                        segment = store.annotations[segment_id]
                    else:
                        raw_segment = body.get("segment")
                        if not isinstance(raw_segment, dict):
                            raise ValidationError("provide segment_id or an inline segment object")
                        segment = AnnotationSegment.from_dict(raw_segment)
                    recording = store.recordings.get(segment.recording_id)
                    if recording is None:
                        raise ValidationError("prediction recording must be registered first")
                    prediction = predict_with_evidence(
                        store.model,
                        segment,
                        query_recording_animal_id=recording.animal_id if recording else None,
                        query_recording_device_id=recording.device_id if recording else None,
                        query_recording_id=recording.recording_id,
                        query_session_id=recording.session_id,
                        species=recording.species,
                        neighbors=store.examples,
                        training_segment_ids=store.training_segment_ids,
                    )
                    self._send(HTTPStatus.OK, prediction.to_dict())
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "route not found"})
            except (ValidationError, ValueError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), Handler)
