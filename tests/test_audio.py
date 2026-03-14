"""Tests for audio capture module."""

import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.capture import AudioCapture, list_input_devices
from config.defaults import CHUNK_SIZE, SAMPLE_RATE


class TestAudioCapture(unittest.TestCase):
    """Unit tests for AudioCapture using a mocked sounddevice stream."""

    def _make_capture(self) -> AudioCapture:
        return AudioCapture(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)

    @patch("audio.capture.sd.InputStream")
    def test_start_opens_stream(self, mock_stream_cls: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_stream_cls.return_value = mock_stream

        capture = self._make_capture()
        capture.start()

        mock_stream_cls.assert_called_once()
        mock_stream.start.assert_called_once()

    @patch("audio.capture.sd.InputStream")
    def test_stop_closes_stream(self, mock_stream_cls: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_stream_cls.return_value = mock_stream

        capture = self._make_capture()
        capture.start()
        capture.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("audio.capture.sd.InputStream")
    def test_double_start_raises(self, mock_stream_cls: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_stream.active = True  # simulate already running
        mock_stream_cls.return_value = mock_stream

        capture = self._make_capture()
        capture._stream = mock_stream  # inject directly

        with self.assertRaises(RuntimeError):
            capture.start()

    def test_audio_callback_queues_chunk(self) -> None:
        capture = self._make_capture()

        # Simulate a 2-channel input (capture flattens to mono)
        fake_indata = np.ones((CHUNK_SIZE, 1), dtype="float32") * 0.5
        capture._audio_callback(fake_indata, CHUNK_SIZE, None, MagicMock())

        chunk = capture._queue.get_nowait()
        self.assertEqual(chunk.shape, (CHUNK_SIZE,))
        np.testing.assert_allclose(chunk, 0.5)

    def test_drain_clears_queue(self) -> None:
        capture = self._make_capture()
        for _ in range(5):
            capture._queue.put(np.zeros(CHUNK_SIZE, dtype="float32"))

        capture.drain()
        self.assertTrue(capture._queue.empty())

    def test_stream_yields_chunks_and_stops_on_none(self) -> None:
        capture = self._make_capture()
        expected = [np.ones(CHUNK_SIZE, dtype="float32") * i for i in range(3)]
        for chunk in expected:
            capture._queue.put(chunk)
        capture._queue.put(None)  # sentinel

        collected = list(capture.stream())
        self.assertEqual(len(collected), 3)
        for i, chunk in enumerate(collected):
            np.testing.assert_allclose(chunk, i)

    @patch("audio.capture.sd.query_devices")
    def test_list_input_devices_filters_inputs(self, mock_query: MagicMock) -> None:
        mock_query.return_value = [
            {"name": "Built-in Mic", "max_input_channels": 1, "default_samplerate": 44100},
            {"name": "Built-in Output", "max_input_channels": 0, "default_samplerate": 44100},
        ]
        devices = list_input_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "Built-in Mic")


class TestAudioCaptureContextManager(unittest.TestCase):
    """Test AudioCapture used as a context manager."""

    @patch("audio.capture.sd.InputStream")
    def test_context_manager_starts_and_stops(self, mock_stream_cls: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_stream_cls.return_value = mock_stream

        with AudioCapture() as capture:
            mock_stream.start.assert_called_once()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
