"""Global push-to-talk hotkey listener (Phase 2).

Uses pynput to monitor keyboard events system-wide via Quartz CGEventTap.
Works even when the app is not the focused window.

Why pynput over alternatives:
- Uses CGEventTap internally — same OS mechanism as CGEvent injection
- Works system-wide without root access (unlike the `keyboard` library)
- Supports key press/release callbacks needed for push-to-talk semantics
- Actively maintained
"""

import logging
import threading
from collections.abc import Callable
from typing import Optional

from pynput import keyboard

from config.defaults import DEFAULT_HOTKEY

logger = logging.getLogger(__name__)

# Map config hotkey names to pynput Key objects
HOTKEY_MAP: dict[str, keyboard.Key] = {
    "right_option": keyboard.Key.alt_r,
    "left_option": keyboard.Key.alt,
    "right_ctrl": keyboard.Key.ctrl_r,
    "left_ctrl": keyboard.Key.ctrl,
    "right_shift": keyboard.Key.shift_r,
    "left_shift": keyboard.Key.shift,
    "right_cmd": keyboard.Key.cmd_r,
    "left_cmd": keyboard.Key.cmd,
    "fn": keyboard.Key.f13,  # best approximation; Fn key is OS-level
}


class HotkeyListener:
    """Monitors a configurable push-to-talk key and fires callbacks on press/release.

    Args:
        hotkey: Config name of the key (e.g. "right_option"). See HOTKEY_MAP.
        on_press: Called when the hotkey is pressed (recording starts).
        on_release: Called when the hotkey is released (recording stops).
    """

    def __init__(
        self,
        hotkey: str = DEFAULT_HOTKEY,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
    ) -> None:
        if hotkey not in HOTKEY_MAP:
            raise ValueError(
                f"Unknown hotkey {hotkey!r}. Valid options: {list(HOTKEY_MAP)}"
            )

        self._key = HOTKEY_MAP[hotkey]
        self._on_press = on_press or (lambda: None)
        self._on_release = on_release or (lambda: None)
        self._listener: Optional[keyboard.Listener] = None
        self._held = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start listening for hotkey events in a background thread."""
        self._listener = keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
        )
        self._listener.start()
        logger.info("Hotkey listener started (key=%s)", self._key)

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Hotkey listener stopped")

    def _handle_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == self._key:
            with self._lock:
                if not self._held:
                    self._held = True
                    logger.debug("Hotkey pressed — recording start")
                    self._on_press()

    def _handle_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == self._key:
            with self._lock:
                if self._held:
                    self._held = False
                    logger.debug("Hotkey released — recording stop")
                    self._on_release()

    def __enter__(self) -> "HotkeyListener":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
