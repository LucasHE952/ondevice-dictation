#!/usr/bin/env python3
"""Extract Silero VAD v5 weights and save as MLX-compatible .npz.

This script loads the Silero VAD PyTorch JIT model, extracts all weights
for the 16kHz path, and saves them as a numpy .npz file that the MLX-based
VoiceActivityDetector can load without any PyTorch dependency.

Usage:
    python scripts/extract_vad_weights.py

Output:
    src/audio/silero_vad_v5.npz (~1.5MB)
"""

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    # Lazy import — torch is only needed for this one-time extraction
    import torch
    from silero_vad import load_silero_vad

    print("Loading Silero VAD model…")
    model = load_silero_vad()
    m = model._model  # 16kHz sub-model

    weights: dict[str, np.ndarray] = {}

    # ── STFT basis (used as a Conv1d kernel) ─────────────────────────────────
    # PyTorch shape: [258, 1, 256]  (out_channels, in_channels, kernel_size)
    # MLX conv1d expects weight: [out_channels, kernel_size, in_channels]
    stft_basis = m.stft.forward_basis_buffer.detach().numpy()
    weights["stft.basis"] = stft_basis.transpose(0, 2, 1)  # [258, 256, 1]

    # ── Encoder: 4 Conv1d blocks ─────────────────────────────────────────────
    # PyTorch Conv1d weight: [out_ch, in_ch, kernel_size]
    # MLX conv1d weight: [out_ch, kernel_size, in_ch]
    encoder_configs = [
        (0, 1),  # stride=1
        (1, 2),  # stride=2
        (2, 2),  # stride=2
        (3, 1),  # stride=1
    ]
    for idx, stride in encoder_configs:
        prefix = f"encoder.{idx}"
        for name, param in m.named_parameters():
            if f"encoder.{idx}.reparam_conv.weight" in name:
                w = param.detach().numpy()
                weights[f"{prefix}.weight"] = w.transpose(0, 2, 1)
            elif f"encoder.{idx}.reparam_conv.bias" in name:
                weights[f"{prefix}.bias"] = param.detach().numpy()

    # ── Decoder LSTM cell ────────────────────────────────────────────────────
    for name, param in m.named_parameters():
        if "decoder.rnn." in name:
            key = name.replace("_model.decoder.rnn.", "lstm.")
            weights[key] = param.detach().numpy()

    # ── Decoder output Conv1d(128, 1, 1) ─────────────────────────────────────
    # PyTorch shape: weight [1, 128, 1], bias [1]
    # MLX conv1d weight: [1, 1, 128]
    for name, param in m.named_parameters():
        if "decoder.decoder.2.weight" in name:
            w = param.detach().numpy()
            weights["decoder_out.weight"] = w.transpose(0, 2, 1)
        elif "decoder.decoder.2.bias" in name:
            weights["decoder_out.bias"] = param.detach().numpy()

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "src" / "audio" / "silero_vad_v5.npz"
    np.savez(out_path, **weights)
    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {len(weights)} arrays to {out_path} ({size_kb:.0f} KB)")

    # Print summary
    for key, arr in sorted(weights.items()):
        print(f"  {key}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
