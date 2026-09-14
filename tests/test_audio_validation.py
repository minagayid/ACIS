import tempfile
import unittest
import wave
from pathlib import Path

from acis.audio.segment import SegmentWindow, intersection_over_union
from acis.audio.validate import validate_wav


class AudioValidationTests(unittest.TestCase):
    def test_valid_wav_metadata_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                handle.writeframes(b"\0\0" * 800)
            spec = validate_wav(path)
            self.assertEqual(spec.sample_rate_hz, 8000)
            self.assertEqual(spec.duration_ms, 100)

    def test_interval_iou(self):
        self.assertAlmostEqual(
            intersection_over_union(SegmentWindow(0, 100), SegmentWindow(50, 150)),
            1 / 3,
        )

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            SegmentWindow(100, 100)
