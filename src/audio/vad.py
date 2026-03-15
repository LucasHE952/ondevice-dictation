"""Voice Activity Detection (VAD) for VoxVault.

Reimplements the Silero VAD v5 LSTM architecture using MLX, eliminating the
~500MB PyTorch dependency. Weights are extracted once from the Silero PyTorch
model (see scripts/extract_vad_weights.py) and stored as a ~1.2MB .npz file.

Architecture (16kHz path):
  1. Prepend 64-sample context from the previous call
  2. STFT via Conv1d with pre-computed basis → magnitude spectrum [1, 129, 4]
  3. Encoder: 4× Conv1d+ReLU blocks (strides 1,2,2,1) → [1, 128, 1]
  4. Decoder: LSTM cell (128→128) → ReLU → Conv1d(128,1,1) → Sigmoid
  5. Output: speech probability scalar in [0, 1]

Why MLX over PyTorch for VAD:
- MLX is already required for the main Voxtral transcription model
- Removing PyTorch cuts the .app bundle from ~887MB to ~350MB
- The VAD model is tiny (~1.2MB weights) — MLX CPU inference is <1ms per window
- No accuracy loss: identical weights, identical forward pass
"""

import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from config.defaults import CHUNK_DURATION_MS, SAMPLE_RATE, VAD_SENSITIVITY_LEVELS

logger = logging.getLogger(__name__)

# Path to the pre-extracted Silero VAD v5 weights (bundled with the app)
_WEIGHTS_PATH = Path(__file__).parent / "silero_vad_v5.npz"

# ── STFT constants (match Silero VAD v5 16kHz) ──────────────────────────────
_FILTER_LENGTH = 256
_HOP_LENGTH = 128
_STFT_CUTOFF = _FILTER_LENGTH // 2 + 1  # 129 frequency bins
_CONTEXT_SIZE = 64  # samples prepended from the previous window

# Encoder Conv1d strides per block
_ENCODER_STRIDES = [1, 2, 2, 1]


class VoiceActivityDetector:
    """Classifies audio chunks as speech or silence using MLX inference.

    Port of Silero VAD v5 (LSTM-based) from PyTorch to MLX. The public API
    is identical to the original torch-based implementation.

    Args:
        sensitivity: One of "low", "medium", "high". Maps to a probability
            threshold above which audio is classified as speech.
        sample_rate: Must be 16000 (only 16kHz is supported in this MLX port).
    """

    def __init__(
        self,
        sensitivity: str = "medium",
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError(f"MLX VAD only supports 16000 Hz, got {sample_rate}")
        if sensitivity not in VAD_SENSITIVITY_LEVELS:
            raise ValueError(
                f"sensitivity must be one of {list(VAD_SENSITIVITY_LEVELS)}, got {sensitivity!r}"
            )

        self.sample_rate = sample_rate
        self.threshold = VAD_SENSITIVITY_LEVELS[sensitivity]

        # Weights (loaded lazily)
        self._weights: Optional[dict[str, mx.array]] = None

        # Stateful context and LSTM hidden state
        self._context: Optional[mx.array] = None
        self._h: Optional[mx.array] = None  # LSTM hidden state [1, 128]
        self._c: Optional[mx.array] = None  # LSTM cell state   [1, 128]

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the VAD model weights into memory.

        Called once at startup. Subsequent calls are no-ops.
        """
        if self._weights is not None:
            return

        if not _WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"VAD weights not found at {_WEIGHTS_PATH}. "
                "Run: python scripts/extract_vad_weights.py"
            )

        logger.info("Loading MLX VAD weights from %s…", _WEIGHTS_PATH.name)
        npz = np.load(_WEIGHTS_PATH)
        self._weights = {key: mx.array(npz[key]) for key in npz.files}
        mx.eval(*self._weights.values())
        logger.info("MLX VAD loaded (threshold=%.2f)", self.threshold)

    # Silero VAD requires exactly this many samples per call at 16kHz.
    _WINDOW_SAMPLES: dict[int, int] = {16000: 512}

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Return True if the audio chunk contains speech.

        Splits the chunk into 512-sample windows and returns True if any
        window exceeds the configured threshold.

        Args:
            chunk: 1-D float32 numpy array of any length >= 512.

        Returns:
            True if speech probability exceeds the threshold in any window.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        if self._weights is None:
            raise RuntimeError("VoiceActivityDetector.load() must be called before is_speech()")

        window = self._WINDOW_SAMPLES[self.sample_rate]

        for start in range(0, len(chunk) - window + 1, window):
            audio_np = chunk[start : start + window]
            prob = self._forward(mx.array(audio_np, dtype=mx.float32))
            if prob >= self.threshold:
                return True

        return False

    def reset_state(self) -> None:
        """Reset LSTM hidden state and context between push-to-talk sessions."""
        self._context = None
        self._h = None
        self._c = None
        logger.debug("VAD state reset")

    # ── Private: forward pass ────────────────────────────────────────────────

    def _forward(self, x: mx.array) -> float:
        """Run a single 512-sample window through the model.

        Args:
            x: 1-D float32 MLX array of exactly 512 samples.

        Returns:
            Speech probability as a Python float in [0, 1].
        """
        w = self._weights
        assert w is not None

        # Batch dimension: [512] → [1, 512]
        x = x.reshape(1, -1)

        # Prepend context from previous call (or zeros on first call)
        if self._context is None:
            self._context = mx.zeros((1, _CONTEXT_SIZE))
        x = mx.concatenate([self._context, x], axis=1)  # [1, 576]

        # Save context for next call (last 64 samples)
        self._context = x[:, -_CONTEXT_SIZE:]

        # ── STFT ─────────────────────────────────────────────────────────────
        # Reflect-pad right by 64: [1, 576] → [1, 640]
        x_padded = _reflect_pad_right(x, _CONTEXT_SIZE)

        # Conv1d for STFT: input [1, 640, 1], weight [258, 256, 1]
        x_stft = x_padded.reshape(1, -1, 1)  # [1, 640, 1]
        stft_out = mx.conv1d(x_stft, w["stft.basis"], stride=_HOP_LENGTH)  # [1, 4, 258]

        # Split into real and imaginary, compute magnitude
        real = stft_out[:, :, :_STFT_CUTOFF]   # [1, 4, 129]
        imag = stft_out[:, :, _STFT_CUTOFF:]   # [1, 4, 129]
        mag = mx.sqrt(real * real + imag * imag)  # [1, 4, 129]

        # ── Encoder ──────────────────────────────────────────────────────────
        # mag is [1, time=4, freq=129] — already in NLC format for MLX conv1d
        enc = mag
        for i, stride in enumerate(_ENCODER_STRIDES):
            enc = mx.conv1d(
                enc,
                w[f"encoder.{i}.weight"],
                stride=stride,
                padding=1,
            )
            enc = enc + w[f"encoder.{i}.bias"]
            enc = mx.maximum(enc, 0.0)  # ReLU

        # enc: [1, 1, 128]

        # ── Decoder ──────────────────────────────────────────────────────────
        # Squeeze time dimension: [1, 1, 128] → [1, 128]
        lstm_in = enc.squeeze(1)

        # LSTM cell
        self._h, self._c = _lstm_cell(
            lstm_in,
            self._h,
            self._c,
            w["decoder.rnn.weight_ih"],
            w["decoder.rnn.weight_hh"],
            w["decoder.rnn.bias_ih"],
            w["decoder.rnn.bias_hh"],
        )

        # Output head: ReLU → Conv1d(128, 1, 1) → Sigmoid
        # h: [1, 128] → [1, 1, 128] for conv1d
        out = mx.maximum(self._h, 0.0)  # ReLU
        out = out.reshape(1, 1, -1)  # [1, 1, 128]
        out = mx.conv1d(out, w["decoder_out.weight"])  # [1, 1, 1]
        out = out + w["decoder_out.bias"]
        out = mx.sigmoid(out)

        # Scalar output
        prob = out.item()
        mx.eval(self._context, self._h, self._c)
        return float(prob)


def _reflect_pad_right(x: mx.array, pad_size: int) -> mx.array:
    """Reflect-pad a 2-D array [batch, length] on the right side.

    Mirrors the last `pad_size` samples (excluding the boundary) to the right,
    matching PyTorch's F.pad(x, [0, pad_size], 'reflect').
    """
    # Reflect: take x[:, -2:-2-pad_size:-1] (reversed slice before the last element)
    reflected = x[:, -2 : -2 - pad_size : -1]
    return mx.concatenate([x, reflected], axis=1)


def _lstm_cell(
    x: mx.array,
    h: Optional[mx.array],
    c: Optional[mx.array],
    weight_ih: mx.array,
    weight_hh: mx.array,
    bias_ih: mx.array,
    bias_hh: mx.array,
) -> tuple[mx.array, mx.array]:
    """Single LSTM cell step (matches torch.lstm_cell).

    Args:
        x: Input tensor [batch, input_size].
        h: Previous hidden state [batch, hidden_size] or None.
        c: Previous cell state [batch, hidden_size] or None.
        weight_ih: Input-hidden weights [4*hidden_size, input_size].
        weight_hh: Hidden-hidden weights [4*hidden_size, hidden_size].
        bias_ih: Input-hidden bias [4*hidden_size].
        bias_hh: Hidden-hidden bias [4*hidden_size].

    Returns:
        (h_new, c_new) each of shape [batch, hidden_size].
    """
    hidden_size = weight_hh.shape[1]

    if h is None:
        h = mx.zeros((x.shape[0], hidden_size))
    if c is None:
        c = mx.zeros((x.shape[0], hidden_size))

    # gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
    gates = x @ weight_ih.T + h @ weight_hh.T + bias_ih + bias_hh  # [batch, 4*hidden]

    # Split into i, f, g, o (PyTorch LSTM gate order)
    i, f, g, o = mx.split(gates, 4, axis=1)
    i = mx.sigmoid(i)
    f = mx.sigmoid(f)
    g = mx.tanh(g)
    o = mx.sigmoid(o)

    c_new = f * c + i * g
    h_new = o * mx.tanh(c_new)

    return h_new, c_new
