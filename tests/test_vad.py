"""Tests for Voice Activity Detection module."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio.vad import VoiceActivityDetector
from config.defaults import SAMPLE_RATE


class TestVoiceActivityDetector(unittest.TestCase):
    """Unit tests for VoiceActivityDetector using a mocked Silero model."""

    def test_invalid_sample_rate_raises(self) -> None:
        with self.assertRaises(ValueError):
            VoiceActivityDetector(sample_rate=44100)

    def test_invalid_sensitivity_raises(self) -> None:
        with self.assertRaises(ValueError):
            VoiceActivityDetector(sensitivity="ultra")

    def test_is_speech_before_load_raises(self) -> None:
        vad = VoiceActivityDetector()
        with self.assertRaises(RuntimeError):
            vad.is_speech(np.zeros(1600, dtype="float32"))

    @patch("audio.vad.load_silero_vad")
    @patch("audio.vad.torch")
    def test_load_sets_model(self, mock_torch: MagicMock, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        vad = VoiceActivityDetector()
        vad.load()

        mock_load.assert_called_once()
        mock_model.eval.assert_called_once()
        self.assertIsNotNone(vad._model)

    @patch("audio.vad.load_silero_vad")
    @patch("audio.vad.torch")
    def test_is_speech_above_threshold(self, mock_torch: MagicMock, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.8  # above medium threshold (0.5)
        mock_load.return_value = mock_model

        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        vad = VoiceActivityDetector(sensitivity="medium")
        vad.load()

        chunk = np.zeros(1600, dtype="float32")
        result = vad.is_speech(chunk)
        self.assertTrue(result)

    @patch("audio.vad.load_silero_vad")
    @patch("audio.vad.torch")
    def test_is_speech_below_threshold(self, mock_torch: MagicMock, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.2  # below medium threshold (0.5)
        mock_load.return_value = mock_model

        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        vad = VoiceActivityDetector(sensitivity="medium")
        vad.load()

        chunk = np.zeros(1600, dtype="float32")
        result = vad.is_speech(chunk)
        self.assertFalse(result)

    @patch("audio.vad.load_silero_vad")
    @patch("audio.vad.torch")
    def test_load_idempotent(self, mock_torch: MagicMock, mock_load: MagicMock) -> None:
        """Calling load() twice should not reload the model."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        vad = VoiceActivityDetector()
        vad.load()
        vad.load()

        mock_load.assert_called_once()

    @patch("audio.vad.load_silero_vad")
    @patch("audio.vad.torch")
    def test_reset_state_calls_model(self, mock_torch: MagicMock, mock_load: MagicMock) -> None:
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        vad = VoiceActivityDetector()
        vad.load()
        vad.reset_state()

        mock_model.reset_states.assert_called_once()


if __name__ == "__main__":
    unittest.main()
