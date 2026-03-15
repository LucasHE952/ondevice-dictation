# ARCHITECTURE.md — VoxVault

Technical reference for contributors. Read this before modifying core pipeline components.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User holds hotkey                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AudioCapture (sounddevice / PortAudio)                             │
│  • 16kHz mono float32 stream                                        │
│  • Non-blocking callback → thread-safe queue                        │
│  • Chunk size: 100ms (1600 samples)                                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  raw audio chunks
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VoiceActivityDetector (Silero VAD)                                 │
│  • Neural LSTM model, runs on CPU                                   │
│  • Per-chunk binary classification: speech / silence                │
│  • Threshold configurable: low=0.3 / medium=0.5 / high=0.7         │
└─────────────┬───────────────────────────────────┬───────────────────┘
              │ speech chunk                       │ silence chunk
              ▼                                   ▼
┌─────────────────────────┐            ┌──────────────────────────────┐
│  AudioBuffer            │            │  Discarded (or kept ≤300ms   │
│  • Accumulates speech   │            │  for sentence boundary ctx)  │
│  • Max 30s rolling      │            └──────────────────────────────┘
│  • Trimmed on overflow  │
└───────────┬─────────────┘
            │ flush on: key release | long silence
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VoxtralModel (MLX / Apple Silicon GPU + Neural Engine)             │
│  • Model: mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit         │
│  • Causal audio encoder: streams mel spectrogram chunks             │
│  • LM decoder: autoregressive token generation, greedy (temp=0)    │
│  • Streaming: tokens yielded as generated (Phase 2)                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  token stream
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TextInjector (Quartz CGEvent / pyobjc)                             │
│  • Simulates keyboard events at the OS level                        │
│  • Works in every app: terminals, browsers, native Cocoa apps       │
│  • Requires Accessibility permission                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Library Choices and Rationale

### Inference: MLX + mlx-audio

**Model:** `mistralai/Voxtral-Mini-4B-Realtime-2602` (weights: `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`)

Voxtral-Mini-4B-Realtime-2602 is architecturally different from the non-realtime Voxtral-Mini-3B-2507:
- **Causal audio encoder** — processes audio in a streaming fashion, not all-at-once
- Sub-200ms latency; configurable transcription delay from 80ms–2400ms in 80ms steps
- Purpose-built for live dictation; the 3B-2507 variant is a batch audio-understanding model

**Why MLX over PyTorch:**
- MLX is Apple's own ML framework, built specifically for unified memory on M-series chips
- Tensors live in a single memory space shared by CPU and GPU — no PCIe transfer overhead
- Metal-backed GPU kernels are compiled and cached by MLX at first run
- Neural Engine offload is automatic for supported ops
- PyTorch's MPS backend exists but lags behind native MLX performance for inference workloads

**Why mlx-audio over mlx-lm:**
- mlx-lm is designed for text-only LLMs — it has no audio encoder or mel spectrogram support
- mlx-audio provides the full STT pipeline: mel spectrogram → causal encoder → LM decoder
- It is the official MLX community integration listed on Mistral's Voxtral model card
- `transcribe_stream()` yields partial text chunks as audio is processed — maps directly to our push-to-talk streaming architecture

### Audio Capture: sounddevice

**Why sounddevice over PyAudio:**
- PyAudio's last release was 2017; sounddevice is actively maintained
- sounddevice integrates directly with numpy — no manual `struct.unpack` or buffer casting
- PortAudio backend is identical (both wrap PortAudio), so latency is the same
- Non-blocking callback model maps cleanly to a producer/consumer queue pattern
- No build-time compilation issues on macOS ARM

### Voice Activity Detection: Silero VAD

**Why Silero VAD over webrtcvad:**
- webrtcvad is a simple energy + zero-crossing detector — high false-positive rate on noise
- Silero is a neural LSTM model trained on diverse speech data: multiple languages, noise types, accents
- ~100ms latency on CPU (same as webrtcvad but far more accurate)
- webrtcvad requires very specific frame sizes (10/20/30ms) and is awkward to integrate
- Silero's Python API is straightforward; its `reset_states()` method correctly handles session boundaries

**Tradeoff:** Silero pulls in `torch` as a lightweight CPU-only dependency. We deliberately avoid using PyTorch for the primary inference path (Voxtral), but it's acceptable here — VAD runs on CPU and uses negligible memory.

### Text Injection: Quartz CGEvent (pyobjc)

**Why CGEvent over pynput's key injection or pyautogui:**
- CGEvent operates at the macOS window server level — it's the same mechanism as real hardware keystrokes
- Works system-wide including: sandboxed apps, Electron apps, native Cocoa apps, terminals, browsers
- pynput can inject keystrokes but relies on CGEvent internally; using pyobjc directly gives more control
- pyautogui uses a different mechanism and has known issues with some apps on Apple Silicon

**Limitation:** Requires Accessibility permission. Without it, CGEvent injection is silently ignored by the OS. We detect this at startup and guide the user to grant it.

**Unicode handling:** Most characters can be injected via `CGEventKeyboardSetUnicodeString`. For characters outside the BMP or in special positions, we use `kCGEventKeyboardEventKeycode` with the appropriate modifier flags.

### Global Hotkey: pynput

**Why pynput:**
- Uses Quartz `CGEventTap` to intercept keyboard events at the OS level
- Works even when the app is not the frontmost window (system-wide)
- Well-maintained; supports key press/release callbacks needed for push-to-talk
- Alternative (`keyboard` library) requires root on macOS; pynput does not

### Menu Bar UI: rumps

**Why rumps:**
- Built on AppKit via pyobjc — fully native macOS menu bar behaviour
- Minimal API surface; excellent for status icon + dropdown menu pattern
- Handles the macOS event loop correctly (required for AppKit apps)
- Alternatives (pystray, systray) have known rendering issues on macOS 13+

---

## Streaming and Buffering Strategy

### During a hotkey session (Phase 2)

```
hotkey down
    │
    ├── AudioCapture.start()
    ├── VoiceActivityDetector.reset_state()
    │
    └── per 100ms chunk:
            VAD.is_speech(chunk)?
               yes → AudioBuffer.append_speech(chunk)
               no  → AudioBuffer.append_silence(chunk)  # up to 300ms silence kept

            if silence > SILENCE_TIMEOUT (e.g. 1.5s):
                audio = AudioBuffer.flush()
                for token in VoxtralModel.transcribe_stream(audio):
                    TextInjector.type(token)

hotkey up
    │
    ├── AudioCapture.stop()
    ├── audio = AudioBuffer.flush()  # flush any remaining buffered speech
    ├── for token in VoxtralModel.transcribe_stream(audio):
    │       TextInjector.type(token)
    └── done
```

### Why we keep 300ms of silence

The Voxtral model uses sentence-boundary context to improve accuracy. Including a small amount of trailing silence prevents it from cutting off words at the end of clauses. 300ms is the empirically chosen value — enough for the model, not enough to add noticeable latency.

### Why we segment on long silence (not just key release)

For multi-sentence dictation, the user may pause naturally between sentences. Segmenting on silence allows us to begin transcribing earlier sentence while the user continues speaking, reducing the perceived latency of text injection.

---

## How VAD Interacts with the Hotkey

```
State machine:

         ┌──────────────────┐
    ───▶ │  IDLE            │
         │  (hotkey up)     │
         └────────┬─────────┘
                  │ hotkey down
                  ▼
         ┌──────────────────┐
         │  LISTENING       │◀──────────────────────┐
         │  (VAD active)    │                        │ speech detected
         └────────┬─────────┘                        │
                  │ speech detected                  │
                  ▼                                  │
         ┌──────────────────┐                        │
         │  RECORDING       │────────────────────────┘
         │  (buffering)     │
         └────────┬─────────┘
                  │ silence > threshold OR hotkey up
                  ▼
         ┌──────────────────┐
         │  TRANSCRIBING    │
         │  (model running) │
         └────────┬─────────┘
                  │ done
                  ▼
         LISTENING (if hotkey still held) or IDLE (if hotkey released)
```

---

## macOS-Specific Implementation Details

### Accessibility Permission Detection

```python
from Quartz import AXIsProcessTrusted
has_permission: bool = AXIsProcessTrusted()
```

This is checked at startup. If `False`, we display a step-by-step guide and refuse to start the injection component (failing safely rather than silently).

### Unicode Keystroke Injection

Standard ASCII characters map to known keycodes. For arbitrary Unicode (e.g. accented characters, emoji in custom vocabulary), we use:

```python
from Quartz import CGEventCreateKeyboardEvent, CGEventKeyboardSetUnicodeString, CGEventPost
event = CGEventCreateKeyboardEvent(None, 0, True)
CGEventKeyboardSetUnicodeString(event, len(char), char)
CGEventPost(kCGHIDEventTap, event)
```

This bypasses keycode mapping entirely and injects the Unicode codepoint directly.

---

## Known Limitations

1. **First-run latency**: MLX compiles and caches Metal kernels on first use. This adds ~30s to the first `python src/main.py --phase1` run. Subsequent runs are fast.

2. **Accessibility prompt**: macOS will never automatically prompt for Accessibility access. Some users miss this step. The setup script and app startup both include explicit instructions.

3. **VAD torch dependency**: Silero VAD requires PyTorch for CPU inference. We keep PyTorch out of the primary inference path but cannot eliminate it entirely in this version.

4. **Single-speaker optimisation**: The VAD and model are tuned for single-speaker dictation. Multi-speaker scenarios (e.g. transcribing a meeting) are not a target use case.

5. **No background noise suppression**: Raw microphone audio is passed to VAD. In very noisy environments, VAD false-positive rate increases. A future improvement would add a lightweight denoiser before VAD.

---

## Future Improvement Areas

- Replace Silero's torch dependency with an MLX-native VAD to eliminate PyTorch entirely
- Add a lightweight denoiser (e.g. RNNoise) before VAD for noisy environments
- Implement incremental/streaming audio feeding to Voxtral for even lower latency
- Add a floating overlay UI (Phase 5) showing tentative transcription before commitment
- Explore context biasing via custom vocabulary injection into the model prompt
