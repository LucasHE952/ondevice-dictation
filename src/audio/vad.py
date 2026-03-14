"""Voice Activity Detection (VAD) for ondevice-dictation.

Uses Silero VAD — a neural VAD model that runs on CPU with ~100ms latency.

Why Silero VAD over webrtcvad:
- Significantly higher accuracy across diverse speakers, accents, and environments
- Handles music, background noise, and non-speech audio more robustly
- Simple Python API; no manual frame-size constraints
- Actively maintained (webrtcvad is effectively unmaintained)
- Runs in ~1-2ms per 100ms chunk on Apple Silicon CPU
"""

import logging
from typing import Optional

import numpy as np

from config.defaults import CHUNK_DURATION_MS, SAMPLE_RATE, VAD_SENSITIVITY_LEVELS

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Wraps Silero VAD to classify audio chunks as speech or silence.

    Silero VAD is loaded lazily on first use to avoid slowing app startup.

    Args:
        sensitivity: One of "low", "medium", "high". Maps to a probability
            threshold above which audio is classified as speech.
        sample_rate: Must be 8000 or 16000 (Silero VAD constraint).
    """

    def __init__(
        self,
        sensitivity: str = "medium",
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Silero VAD requires 8000 or 16000 Hz, got {sample_rate}")
        if sensitivity not in VAD_SENSITIVITY_LEVELS:
            raise ValueError(
                f"sensitivity must be one of {list(VAD_SENSITIVITY_LEVELS)}, got {sensitivity!r}"
            )

        self.sample_rate = sample_rate
        self.threshold = VAD_SENSITIVITY_LEVELS[sensitivity]
        self._model: Optional[object] = None
        self._get_speech_timestamps: Optional[object] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the Silero VAD model into memory.

        Called once at startup. Subsequent calls are no-ops.
        """
        if self._model is not None:
            return

        logger.info("Loading Silero VAD model…")
        # Import here so the app can start even if torch is slow to import
        import torch
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._model.eval()
        logger.info("Silero VAD loaded (threshold=%.2f)", self.threshold)

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Return True if the audio chunk contains speech.

        Args:
            chunk: 1-D float32 numpy array. Length must correspond to
                30ms, 60ms, or 100ms at the configured sample_rate.

        Returns:
            True if speech probability exceeds the configured threshold.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        if self._model is None:
            raise RuntimeError("VoiceActivityDetector.load() must be called before is_speech()")

        import torch

        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            prob: float = self._model(tensor, self.sample_rate).item()

        return prob >= self.threshold

    def reset_state(self) -> None:
        """Reset Silero VAD's internal hidden state between sessions.

        Silero VAD is stateful (LSTM-based). Call this at the start of each
        new push-to-talk session to prevent bleed-over from prior audio.
        """
        if self._model is not None:
            self._model.reset_states()
            logger.debug("VAD state reset")
