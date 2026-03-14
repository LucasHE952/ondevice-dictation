"""Token streaming and audio buffer management for ondevice-dictation.

This module handles the buffering strategy between audio capture and the
transcription model: accumulate speech chunks while the hotkey is held,
then flush the buffer to the model when silence is detected or the key
is released.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from config.defaults import CHUNK_SIZE, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Maximum audio buffer duration in seconds. If the user speaks longer than
# this without a pause, we segment and transcribe incrementally to avoid
# unbounded memory growth.
MAX_BUFFER_SECONDS: float = 30.0
MAX_BUFFER_SAMPLES: int = int(MAX_BUFFER_SECONDS * SAMPLE_RATE)


class AudioBuffer:
    """Accumulates raw audio chunks from the microphone into a contiguous buffer.

    Speech chunks (as determined by VAD) are appended. Silence chunks are
    discarded unless ``keep_silence`` is set (useful for model context).

    Args:
        keep_silence_ms: Milliseconds of trailing silence to retain after each
            speech segment. Helps the model understand sentence boundaries.
    """

    def __init__(self, keep_silence_ms: int = 300) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._silence_budget: int = int(keep_silence_ms / 1000 * SAMPLE_RATE)
        self._silence_carried: int = 0

    def append_speech(self, chunk: np.ndarray) -> None:
        """Append a speech chunk to the buffer.

        Args:
            chunk: 1-D float32 numpy array of audio samples.
        """
        self._chunks.append(chunk)
        self._total_samples += len(chunk)
        self._silence_carried = 0  # reset silence budget on new speech

        if self._total_samples > MAX_BUFFER_SAMPLES:
            self._trim_oldest()

    def append_silence(self, chunk: np.ndarray) -> bool:
        """Optionally append a silence chunk (up to the silence budget).

        Args:
            chunk: 1-D float32 numpy array of audio samples.

        Returns:
            True if the chunk was appended (within budget), False if discarded.
        """
        if self._silence_carried < self._silence_budget:
            self._chunks.append(chunk)
            self._total_samples += len(chunk)
            self._silence_carried += len(chunk)
            return True
        return False

    def flush(self) -> Optional[np.ndarray]:
        """Concatenate and return all buffered audio, then clear the buffer.

        Returns:
            1-D float32 numpy array, or None if the buffer is empty.
        """
        if not self._chunks:
            return None
        audio = np.concatenate(list(self._chunks))
        self.clear()
        return audio

    def clear(self) -> None:
        """Discard all buffered audio."""
        self._chunks.clear()
        self._total_samples = 0
        self._silence_carried = 0

    @property
    def duration_seconds(self) -> float:
        """Current buffer duration in seconds."""
        return self._total_samples / SAMPLE_RATE

    @property
    def is_empty(self) -> bool:
        """True if no audio has been buffered."""
        return self._total_samples == 0

    def _trim_oldest(self) -> None:
        """Drop the oldest chunks to stay within MAX_BUFFER_SAMPLES."""
        while self._total_samples > MAX_BUFFER_SAMPLES and self._chunks:
            dropped = self._chunks.popleft()
            self._total_samples -= len(dropped)
            logger.debug("Buffer trim: dropped %.1fs of oldest audio", len(dropped) / SAMPLE_RATE)
