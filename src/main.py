"""Entry point for ondevice-dictation.

Phase 1 goal: confirm the model loads and produces accurate output.
Run with: python src/main.py --phase1

Full push-to-talk dictation (Phase 2+) will be added in subsequent phases.
"""

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure src/ is on the path when running as `python src/main.py`
sys.path.insert(0, str(Path(__file__).parent))

from config.defaults import APP_NAME, APP_VERSION, LOG_FILE, SAMPLE_RATE
from config.settings import Settings


def _configure_logging(verbose: bool = False) -> None:
    """Set up logging to both stderr and the log file."""
    level = logging.DEBUG if verbose else logging.INFO
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def phase1_smoke_test(settings: Settings, duration_seconds: float = 5.0) -> None:
    """Phase 1: Record audio from the mic, transcribe with Voxtral, print result.

    This is the minimal end-to-end test. No hotkey, no injection — just the
    audio → model → text pipeline.

    Args:
        settings: User settings instance.
        duration_seconds: How many seconds of audio to record.
    """
    logger = logging.getLogger(__name__)

    # ── Lazy imports (heavy; keep startup fast) ───────────────────────────────
    from audio.capture import AudioCapture
    from transcription.model import VoxtralModel

    local_model_path = Path(__file__).parent.parent / "models" / "voxtral-realtime"
    model = VoxtralModel(
        model_path=local_model_path,
        language=settings["language"],
    )

    print(f"\n{APP_NAME} v{APP_VERSION} — Phase 1 smoke test")
    print("=" * 50)

    # Load model (downloads ~2.5GB on first run via HuggingFace, then cached)
    print("Loading Voxtral Realtime (downloads ~2.5GB on first run, then cached) …")
    try:
        model.load()
    except ImportError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Model ready. Recording {duration_seconds:.0f}s of audio …\n")
    print("Speak now!")
    print("-" * 50)

    # Record audio
    chunks: list[np.ndarray] = []
    capture = AudioCapture(sample_rate=SAMPLE_RATE)

    try:
        capture.start()
        deadline = time.perf_counter() + duration_seconds
        for chunk in capture.stream():
            chunks.append(chunk)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            # Progress indicator
            elapsed = duration_seconds - remaining
            bar = "#" * int(elapsed * 10 / duration_seconds)
            print(f"\r[{bar:<10}] {elapsed:.1f}s / {duration_seconds:.0f}s", end="", flush=True)
    finally:
        capture.stop()

    print("\n" + "-" * 50)
    print("Recording complete. Transcribing …\n")

    if not chunks:
        print("No audio captured. Check your microphone.", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(chunks)
    logger.info("Captured %.1fs of audio (%d samples)", len(audio) / SAMPLE_RATE, len(audio))

    # Transcribe
    text = model.transcribe(audio, language=settings["language"])

    print("Transcription:")
    print("=" * 50)
    print(text or "(empty — no speech detected)")
    print("=" * 50)


def run_dictation_loop(settings: Settings) -> None:
    """Phase 2: Full push-to-talk dictation loop.

    Hold the configured hotkey to record. On release, the audio is
    transcribed and injected into the focused application.

    Args:
        settings: User settings instance.
    """
    logger = logging.getLogger(__name__)

    from audio.capture import AudioCapture
    from audio.vad import VoiceActivityDetector
    from hotkey.listener import HotkeyListener
    from injection.text_injector import TextInjector, check_accessibility_permission
    from transcription.model import VoxtralModel
    from transcription.streaming import AudioBuffer

    # ── Accessibility permission check ────────────────────────────────────────
    if not check_accessibility_permission():
        print(
            "\nERROR: Accessibility permission is required for text injection.\n"
            "\nTo grant access:\n"
            "  1. Open System Settings -> Privacy & Security -> Accessibility\n"
            "  2. Enable 'Terminal' (or whichever app you launch this from)\n"
            "  3. Re-run this app\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    local_model_path = Path(__file__).parent.parent / "models" / "voxtral-realtime"
    model = VoxtralModel(model_path=local_model_path, language=settings["language"])
    vad = VoiceActivityDetector(sensitivity=settings["vad_sensitivity"])
    injector = TextInjector()

    print(f"\n{APP_NAME} v{APP_VERSION} — dictation mode")
    print("=" * 50)
    print("Loading Voxtral Realtime...")
    model.load()
    print("Loading Silero VAD...")
    vad.load()
    print("Ready.\n")

    # ── Shared state ──────────────────────────────────────────────────────────
    buffer = AudioBuffer()
    _recording = threading.Event()
    _transcribe_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()

    # ── Hotkey callbacks (run in pynput's listener thread) ────────────────────
    def on_press() -> None:
        buffer.clear()
        vad.reset_state()
        capture.drain()
        _recording.set()
        print("\n[Recording...]", end="", flush=True)

    def on_release() -> None:
        _recording.clear()
        audio = buffer.flush()
        print(" done", flush=True)
        if audio is not None and len(audio) > 0:
            _transcribe_queue.put(audio)

    # ── Audio collection thread ───────────────────────────────────────────────
    capture = AudioCapture(sample_rate=SAMPLE_RATE)

    def _collect_audio() -> None:
        for chunk in capture.stream():
            if _recording.is_set():
                if vad.is_speech(chunk):
                    buffer.append_speech(chunk)
                else:
                    buffer.append_silence(chunk)

    capture.start()
    collect_thread = threading.Thread(target=_collect_audio, daemon=True, name="audio-collector")
    collect_thread.start()

    # ── Hotkey listener ───────────────────────────────────────────────────────
    hotkey = HotkeyListener(
        hotkey=settings["hotkey"],
        on_press=on_press,
        on_release=on_release,
    )
    hotkey.start()

    print(f"Hold [{settings['hotkey']}] to dictate. Ctrl+C to quit.\n")

    # ── Transcription loop (main thread) ──────────────────────────────────────
    try:
        while True:
            try:
                audio = _transcribe_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio is None:
                break

            logger.info("Transcribing %.1fs of audio...", len(audio) / SAMPLE_RATE)
            try:
                text = model.transcribe(audio, language=settings["language"])
                if text:
                    logger.info("Transcription: %r", text)
                    injector.type(text)
                else:
                    logger.debug("Empty transcription — no speech detected")
            except PermissionError as exc:
                logger.error("%s", exc)
            except Exception as exc:
                logger.error("Transcription failed: %s", exc, exc_info=True)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        hotkey.stop()
        capture.stop()
        _transcribe_queue.put(None)


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="On-device dictation powered by Voxtral Realtime on Apple Silicon.",
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1 smoke test: record audio and print transcription.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Recording duration for --phase1 (default: 5s).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {APP_VERSION}",
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    settings = Settings.load()

    if args.phase1:
        phase1_smoke_test(settings, duration_seconds=args.duration)
    else:
        run_dictation_loop(settings)


if __name__ == "__main__":
    main()
