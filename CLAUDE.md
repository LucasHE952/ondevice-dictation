# CLAUDE.md — On-Device Dictation App

This document is the authoritative specification for this project. Read it in full before
writing any code. Refer back to it whenever making architectural or implementation decisions.

---

## Project Overview

This is an open-source, on-device dictation app for macOS. It allows users to dictate text
into any focused application by holding a hotkey. All transcription happens locally on the
user's machine using Mistral's Voxtral Realtime model — no audio or text ever leaves the device.

**Core value proposition:**
- More accurate than macOS built-in dictation
- More private than cloud-based tools like Wispr Flow (fully offline after setup)
- Free and open-source, auditable by anyone
- Optimised for knowledge workers: coders, researchers, writers

**This is a portfolio/showcase open-source project.** Code quality, documentation, and project
structure matter as much as functionality.

---

## Target Users

- macOS users on Apple Silicon (M1 or later)
- Knowledge workers: software developers, researchers, writers
- Users who are already familiar with AI tools and value privacy
- Users comfortable with a one-time terminal-based setup

---

## Hardware Requirements

- Apple Silicon Mac (M1, M2, M3, or M4) — Intel Macs are NOT supported
- Minimum 8GB unified memory (16GB recommended)
- Approximately 3GB free disk space for model weights (Voxtral 4-bit is 2.9GB)
- macOS 13 (Ventura) or later

---

## How It Works — The Core Pipeline

Understanding this pipeline is essential before writing any code:

```
[Microphone] 
     ↓
[Audio capture — continuous stream of raw audio chunks]
     ↓
[Voice Activity Detection (VAD) — is the user speaking right now?]
     ↓  (only when speech detected AND hotkey is held)
[Voxtral Realtime — local inference, streaming transcription]
     ↓
[Text output — streamed tokens]
     ↓
[Text injection — simulated keystrokes into the currently focused app]
```

The hotkey is push-to-talk: the user holds it to record, releases to stop. VAD runs
continuously while the hotkey is held to handle pauses within a dictation session.

---

## Activation Model

**Push-to-talk only.** The user holds a configurable hotkey (default: Right Option key) to
activate recording. Releasing the hotkey ends the session and commits any buffered text.

- While the hotkey is held: audio is captured, VAD filters silence, Voxtral transcribes speech
- On hotkey release: finalise any in-progress transcription, inject remaining text, stop recording
- There is no toggle mode in this version

---

## The Voxtral Model

**Model:** `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`
- 4B parameter model, 4-bit quantised for Apple Silicon (~2.9GB on disk)
- Apache 2.0 license
- Purpose-built for transcription (not general audio understanding)
- Inference via `mlx-audio` (`mlx_audio.stt.load`) — MLX-native, no PyTorch
- Supports 13 languages

**Model weights are NOT bundled with the app.** Downloaded once via `curl` directly
from HuggingFace and stored at `models/voxtral-realtime/` (gitignored). After
download, the app runs fully offline forever.

**Download method:** Use `curl -L -C -` with an HF auth token. Do NOT use
`huggingface_hub.snapshot_download` — the xet transfer protocol it uses for large
safetensors files stalls unpredictably on some networks. Plain HTTPS via curl is
reliable and supports resuming with `-C -`.

---

## Tech Stack

Claude Code should select the best available libraries for each component. Decisions must be
justified in code comments and in ARCHITECTURE.md. The following constraints apply:

**Language:** Python 3.12 (required — 3.11 and below are not supported)

**Inference backend:** `mlx-audio` (`pip install mlx-audio[stt]`) — provides
`mlx_audio.stt.load()` which auto-detects the model type and returns a model with
`.generate(audio, stream=False)` returning an `STTOutput` with `.text`. MLX-native,
runs on the Neural Engine / GPU. Do not use PyTorch as the inference backend.

**Audio capture:** `sounddevice` (PortAudio bindings) — clean numpy integration,
actively maintained, non-blocking callback stream. Chunks are 100ms / 1600 samples.

**Voice Activity Detection:** `silero-vad` — neural VAD (LSTM), more accurate than
webrtcvad across accents and noise. Requires exactly 512 samples per call at 16kHz.
Split larger chunks into 512-sample windows internally before calling the model.
Uses PyTorch for VAD inference only (not for the main transcription backend).

**Text injection:** Quartz CGEvent via `pyobjc-framework-Quartz` — works system-wide
at the window server layer. `AXIsProcessTrusted()` for permission check lives in
`pyobjc-framework-ApplicationServices`, NOT in Quartz.

**Global hotkey:** `pynput` — uses CGEventTap internally, works system-wide without
root access, supports key press/release callbacks needed for push-to-talk.

**UI (menu bar):** `rumps` — standard Python macOS menu bar library (Phase 3).

Do NOT use Electron, React, or any web-based UI framework. This is a native macOS utility app.

---

## Build Phases

Build in strict phase order. Do not proceed to the next phase until the current one is
complete and manually tested.

### Phase 1 — Proof of Concept (no UI, no injection) ✅ COMPLETE
- Load the Voxtral model via mlx-audio
- Capture microphone audio
- Transcribe a short recording
- Print the transcription to the terminal
- Run with: `python src/main.py --phase1 --duration 5`
- Goal: confirm the model loads and produces accurate output

### Phase 2 — Core Dictation Loop ✅ COMPLETE
- Push-to-talk hotkey (system-wide via pynput/CGEventTap)
- VAD filters silence within a held-key session (Silero, 512-sample windows)
- Audio buffered on hotkey hold, transcribed on release (batch, not streaming)
- Transcribed text injected via CGEvent into the focused application
- Run with: `python src/main.py`
- Goal: basic dictation works end-to-end

### Phase 3 — Menu Bar App
- Add a menu bar icon showing app status (idle / recording / processing)
- Add a basic settings menu: change hotkey, select language
- Add a "Check for model" option that shows whether weights are downloaded
- Goal: the app feels like a real macOS utility

### Phase 4 — Setup & Onboarding
- Write a setup script (`setup.sh`) that: checks hardware compatibility, installs Python
  dependencies, downloads Voxtral weights from Hugging Face, checks macOS permissions
- Add guided onboarding for Accessibility and Microphone permissions
- Goal: a new user can go from zero to working dictation by following the README

### Phase 5 — Context Biasing & Polish
- Add a settings screen for custom vocabulary (user-defined words/phrases the model should
  prioritise — names, technical terms, domain jargon)
- Expose VAD sensitivity as a user setting
- Add a floating overlay UI (small, non-intrusive window showing recording status and
  tentative transcription text before it is committed to the target app)
- Goal: power-user features that differentiate from built-in dictation

---

## Repository Structure

```
ondevice-dictation/
├── CLAUDE.md                  # This file
├── README.md                  # User-facing documentation
├── ARCHITECTURE.md            # Technical deep-dive for contributors
├── LICENSE                    # Apache 2.0
├── setup.sh                   # One-command setup script
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── main.py                # Entry point
│   ├── audio/
│   │   ├── capture.py         # Microphone input
│   │   └── vad.py             # Voice activity detection
│   ├── transcription/
│   │   ├── model.py           # Voxtral model loading and inference
│   │   └── streaming.py       # Token streaming and buffering logic
│   ├── injection/
│   │   └── text_injector.py   # System-wide text injection
│   ├── hotkey/
│   │   └── listener.py        # Global hotkey detection
│   ├── ui/
│   │   ├── menu_bar.py        # Menu bar icon and menu
│   │   └── overlay.py         # Floating transcription overlay (Phase 5)
│   └── config/
│       ├── settings.py        # User settings management
│       └── defaults.py        # Default configuration values
│
├── models/                    # Downloaded model weights live here (gitignored)
│   └── .gitkeep
│
└── tests/
    ├── test_audio.py
    ├── test_vad.py
    └── test_injection.py
```

---

## macOS Permissions

The app requires two macOS permissions. Both must be handled gracefully:

**Microphone access**
- macOS will prompt automatically on first use
- If denied, show a clear error message explaining how to re-enable in System Settings

**Accessibility access** (required for text injection)
- macOS will NEVER prompt for this automatically — the user must grant it manually
- Location: System Settings → Privacy & Security → Accessibility
- The app must detect whether this permission has been granted at startup
- If not granted, the app must display a clear, step-by-step guide to granting it
- Do NOT silently fail if this permission is missing

---

## Configuration

User settings must be persisted between sessions. Store in `~/.config/ondevice-dictation/settings.json`.

Default settings:
```json
{
  "hotkey": "right_option",
  "language": "en",
  "vad_sensitivity": "medium",
  "custom_vocabulary": [],
  "model_path": "~/.cache/ondevice-dictation/models/voxtral-realtime"
}
```

---

## Code Quality Standards

- **Type hints** on all function signatures
- **Docstrings** on all classes and public methods (Google style)
- **No silent failures** — all errors must be logged and surfaced to the user appropriately
- **No hardcoded paths** — all paths derived from config or user home directory
- **Logging** — use Python's standard `logging` module, not `print` statements
- Keep functions small and single-purpose
- Write tests for audio capture, VAD, and text injection logic

---

## What NOT to Do

These are guardrails. Do not deviate from them without flagging it explicitly.

- **Do NOT add any cloud or network features** beyond the one-time Hugging Face model download
- **Do NOT send audio, text, or any user data anywhere** — this is a privacy-first app
- **Do NOT support Windows or Linux** in this version — macOS only
- **Do NOT use the App Store distribution path** — direct install only
- **Do NOT use PyTorch as the inference backend** — use MLX
- **Do NOT build a web UI** — this is a native macOS utility
- **Do NOT skip phases** — build in order, test each phase before moving on
- **Do NOT store transcription history** unless the user explicitly opts in (Phase 5+)
- **Do NOT use `print()` for logging** — use the `logging` module

---

## Known Gotchas

These are non-obvious issues discovered during development. Check here before debugging.

- **Silero VAD chunk size:** The model requires exactly 512 samples at 16kHz (32ms).
  Passing any other size raises a TorchScript ValueError. Always split chunks into
  512-sample windows before calling `vad.is_speech()`.

- **AXIsProcessTrusted import:** This function is in `ApplicationServices`, not `Quartz`.
  `from Quartz import AXIsProcessTrusted` silently returns an ImportError.
  Correct: `from ApplicationServices import AXIsProcessTrusted`.

- **HuggingFace xet download stalls:** `snapshot_download` uses HF's xet protocol for
  large safetensors files and stalls at ~384MB on many networks. Use `curl -L -C -` with
  a bearer token instead. Resume interrupted downloads with `-C -`.

- **mlx-audio is STT + TTS:** The package description says "text-to-speech" but it also
  has a full `stt/` subpackage including `stt/models/voxtral_realtime/`. Import via
  `from mlx_audio.stt import load`.

- **Python version:** 3.12 is required. The system Python on macOS is 3.9 and is
  incompatible. Install via `brew install python@3.12` and create the venv explicitly:
  `/opt/homebrew/bin/python3.12 -m venv .venv`.

- **Homebrew PATH:** After installing Homebrew, add to PATH before using it in new shells:
  `eval "$(/opt/homebrew/bin/brew shellenv zsh)"` (add to `~/.zprofile` permanently).

---

## README Requirements

The README.md must include (in this order):
1. A one-paragraph description of what the app does and why it exists
2. Hardware and OS requirements (prominently, near the top)
3. A demo GIF or screenshot (placeholder is fine initially)
4. One-command install instructions
5. How to grant the required macOS permissions
6. How to use the app (hotkey, settings)
7. How the app works (brief technical summary, link to ARCHITECTURE.md)
8. How to contribute
9. License

---

## ARCHITECTURE.md Requirements

This file is for developers and contributors. It must explain:
- The full audio-to-text pipeline with a diagram
- Why each library was chosen
- The streaming and buffering strategy for text injection
- How VAD interacts with the hotkey
- macOS-specific implementation details (permissions, text injection approach)
- Known limitations and future improvement areas

---

## Naming

The repository and app are currently named **ondevice-dictation** as a placeholder. A proper
name will be decided later. Do not hard-code a branding name anywhere — use the config value
`app_name` (default: `"ondevice-dictation"`) wherever the app name appears in UI or logs.