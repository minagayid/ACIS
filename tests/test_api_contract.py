import http.client
import json
import threading
import unittest

from acis.api.service import ACISStore, create_server
from acis.dataset.loaders import join_training_examples
from acis.ingest.manifest import validate_dataset
from acis.models.baseline import ContextBaseline


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recordings, annotations = validate_dataset(
            "data/examples/recordings.jsonl", "data/examples/annotations.jsonl"
        )
        cls.store = ACISStore()
        for recording in recordings:
            cls.store.add_recording(recording)
        for annotation in annotations:
            cls.store.add_annotation(annotation)
        cls.store.model = ContextBaseline().fit(join_training_examples(recordings, annotations))
        cls.store.inference_mode = "in_sample_demo"
        cls.store.training_segment_ids = set(item.annotation.segment_id for item in cls.store.model.training_examples)
        cls.server = create_server(cls.store, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        response = connection.getresponse()
        data = json.loads(response.read())
        connection.close()
        return response.status, data

    def test_health_exposes_claim_scope(self):
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["model_loaded"])
        self.assertIn("not a translation", body["claims_policy"])

    def test_prediction_is_structured_and_evidence_aware(self):
        status, body = self.request("POST", "/v1/predict", {"segment_id": "seg_0013"})
        self.assertEqual(status, 200)
        self.assertIn("probabilities", body)
        self.assertIn("neighbors", body)
        self.assertEqual(body["confidence_kind"], "uncalibrated_model_score")
        self.assertIn("claim_scope", body)
        self.assertTrue(body["evidence"]["query_in_training_set"])
        self.assertNotIn("seg_0013", [item["segment_id"] for item in body["neighbors"]])

    def test_missing_example_is_not_found(self):
        status, body = self.request("GET", "/v1/examples/not-there")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_non_loopback_binding_requires_explicit_unsafe_flag(self):
        from acis.api.service import create_server
        with self.assertRaises(ValueError):
            create_server(ACISStore(), host="0.0.0.0", port=0)
