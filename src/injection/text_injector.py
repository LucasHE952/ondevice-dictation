"""System-wide text injection via Quartz CGEvent (Phase 2).

Uses macOS Quartz CGEvent to simulate keystrokes at the OS level.
This approach works in every macOS app — terminals, browsers, Electron apps,
and native Cocoa apps — because it operates at the window server layer.

Requires: Accessibility permission (System Settings → Privacy & Security → Accessibility).
Without it, injected events are silently dropped by the OS.

Why CGEvent over alternatives:
- pynput key injection ultimately calls CGEvent; this is the direct path
- pyautogui uses a different mechanism with known issues on Apple Silicon
- CGEvent Unicode injection bypasses keycode mapping for arbitrary characters
"""

import logging
import time
from typing import Optional

from config.defaults import KEYSTROKE_DELAY

logger = logging.getLogger(__name__)


def check_accessibility_permission() -> bool:
    """Return True if the process has Accessibility (assistive technology) access.

    This is checked at startup. If False, text injection will silently fail.
    The app must guide the user to grant the permission before proceeding.

    Returns:
        True if Accessibility permission is granted, False otherwise.
    """
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except ImportError:
        logger.error("pyobjc-framework-ApplicationServices not installed — cannot check accessibility")
        return False


class TextInjector:
    """Injects text into the currently focused application via CGEvent.

    Args:
        keystroke_delay: Seconds to wait between individual keystrokes.
            0 works in most apps; increase to 0.005 if characters are dropped.
    """

    def __init__(self, keystroke_delay: float = KEYSTROKE_DELAY) -> None:
        self.keystroke_delay = keystroke_delay
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Return True if injection is possible (Accessibility permission granted).

        Result is cached after first check.
        """
        if self._available is None:
            self._available = check_accessibility_permission()
            if not self._available:
                logger.warning(
                    "Accessibility permission not granted — text injection disabled.\n"
                    "Grant access in: System Settings → Privacy & Security → Accessibility"
                )
        return self._available

    def type(self, text: str) -> None:
        """Inject text into the focused application character by character.

        Uses CGEventKeyboardSetUnicodeString so arbitrary Unicode works,
        including accented characters and non-ASCII scripts.

        Args:
            text: String to inject. May contain any Unicode characters.

        Raises:
            PermissionError: If Accessibility permission is not granted.
            ImportError: If pyobjc-framework-Quartz is not installed.
        """
        if not text:
            return

        if not self.is_available():
            raise PermissionError(
                "Accessibility permission required for text injection.\n"
                "Grant access in: System Settings → Privacy & Security → Accessibility"
            )

        try:
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventKeyboardSetUnicodeString,
                CGEventPost,
                kCGHIDEventTap,
            )
        except ImportError as exc:
            raise ImportError(
                "pyobjc-framework-Quartz is required for text injection.\n"
                "Run: pip install pyobjc-framework-Quartz"
            ) from exc

        for char in text:
            # Key down
            event_down = CGEventCreateKeyboardEvent(None, 0, True)
            CGEventKeyboardSetUnicodeString(event_down, len(char), char)
            CGEventPost(kCGHIDEventTap, event_down)

            # Key up
            event_up = CGEventCreateKeyboardEvent(None, 0, False)
            CGEventKeyboardSetUnicodeString(event_up, len(char), char)
            CGEventPost(kCGHIDEventTap, event_up)

            if self.keystroke_delay > 0:
                time.sleep(self.keystroke_delay)

        logger.debug("Injected %d characters", len(text))
