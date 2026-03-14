"""Voxtral model loading and inference via MLX.

Model: mistralai/Voxtral-Mini-4B-Realtime-2602
Weights: mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit (~2–2.5GB)
Library: mlx-audio (pip install mlx-audio[stt])

Voxtral-Mini-4B-Realtime-2602 is architecturally distinct from Voxtral-Mini-3B-2507:
- Purpose-built for streaming/real-time transcription (causal audio encoder)
- Sub-200ms latency; configurable delay from 80ms to 2400ms in 80ms steps
- Streaming decoder: yields tokens as audio is processed (not batch-only)
- Official MLX support via mlx-audio (Mistral-endorsed community integration)

Why mlx-audio over mlx-lm:
- mlx-lm is designed for text LLMs; it has no audio encoder/processor support
- mlx-audio provides the full STT pipeline: mel spectrogram → encoder → LM decoder
- Realtime streaming API matches our push-to-talk architecture exactly

Why MLX over PyTorch:
- Unified memory: tensors shared between CPU and GPU without PCIe transfers
- Metal-backed kernels compiled once and cached; fast after first run
- Lower latency for streaming workloads on Apple Silicon
"""

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import numpy as np

from config.defaults import MODEL_LOCAL_DIR, MODEL_REPO_ID, SAMPLE_RATE

logger = logging.getLogger(__name__)


class VoxtralModel:
    """Wraps the Voxtral Realtime model for streaming transcription via MLX.

    The model is loaded lazily. Call ``load()`` once at startup (takes ~5s
    on first use while MLX compiles kernels; subsequent loads use the cache).

    Args:
        model_path: Path to local MLX model directory. If the directory does
            not exist, ``load()`` will raise a clear error directing the user
            to run the setup script.
    """

    def __init__(self, model_path: Path = MODEL_LOCAL_DIR) -> None:
        self.model_path = Path(model_path)
        self._model: Optional[object] = None
        self._processor: Optional[object] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights into MLX (GPU/Neural Engine).

        Raises:
            FileNotFoundError: If model weights are not present locally.
                The error message instructs the user to run setup.sh.
            ImportError: If mlx or mlx_lm are not installed.
        """
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Voxtral weights not found at {self.model_path}.\n"
                "Run the setup script to download them:\n\n"
                "    bash setup.sh\n"
            )

        logger.info("Loading Voxtral model from %s…", self.model_path)
        t0 = time.perf_counter()

        try:
            # mlx_audio provides the full STT pipeline for Voxtral:
            # mel spectrogram extraction → causal audio encoder → LM decoder.
            # load_models() returns the model graph and its associated processor.
            from mlx_audio.stt.models.voxtral import load_models

            self._model, self._processor = load_models(str(self.model_path))
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is not installed. Run: pip install 'mlx-audio[stt]'"
            ) from exc

        elapsed = time.perf_counter() - t0
        logger.info("Voxtral loaded in %.1fs", elapsed)

    def is_loaded(self) -> bool:
        """Return True if model weights are in memory."""
        return self._model is not None

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        max_tokens: int = 448,
    ) -> str:
        """Transcribe a complete audio buffer to text (non-streaming).

        Suitable for Phase 1 testing. Phase 2 uses ``transcribe_stream()``.

        Args:
            audio: 1-D float32 numpy array at ``SAMPLE_RATE`` Hz.
            language: BCP-47 language code (e.g. "en", "fr").
            max_tokens: Maximum output tokens to generate.

        Returns:
            Transcribed text string (stripped of leading/trailing whitespace).

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._assert_loaded()

        logger.debug("Transcribing %d samples (%.1fs audio)", len(audio), len(audio) / SAMPLE_RATE)
        t0 = time.perf_counter()

        # mlx_audio transcribe() handles: mel spectrogram → encoder → greedy decode.
        # temperature=0.0 gives deterministic output (recommended for dictation).
        result = self._model.transcribe(
            audio,
            processor=self._processor,
            language=language,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text: str = result["text"].strip()
        elapsed = time.perf_counter() - t0
        logger.debug("Transcription complete in %.2fs: %r", elapsed, text[:80])

        return text

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str = "en",
        max_tokens: int = 448,
    ) -> Generator[str, None, None]:
        """Transcribe audio and yield text tokens as they are generated.

        Used in Phase 2 for low-latency text injection — each token is
        injected into the target application as soon as it arrives.

        Args:
            audio: 1-D float32 numpy array at ``SAMPLE_RATE`` Hz.
            language: BCP-47 language code.
            max_tokens: Maximum output tokens.

        Yields:
            str: Individual decoded token strings (may be subwords).

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._assert_loaded()

        # mlx_audio's streaming transcribe yields partial text chunks as the
        # causal encoder processes audio — ideal for low-latency token injection.
        for chunk in self._model.transcribe_stream(
            audio,
            processor=self._processor,
            language=language,
            temperature=0.0,
            max_tokens=max_tokens,
        ):
            token_str: str = chunk.get("text", "")
            if token_str:
                yield token_str

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call VoxtralModel.load() first."
            )
