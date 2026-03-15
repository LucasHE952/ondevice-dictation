"""Global push-to-talk hotkey listener (Phase 2).

Uses a Quartz CGEventTap directly (no pynput) to monitor modifier key
events system-wide.  The tap's run-loop source is added to the **main
thread's** CFRunLoop so that all callbacks execute on the main queue.
This avoids the SIGTRAP crash on macOS 15+ caused by pynput calling
TSMGetInputSourceProperty from a background thread.

The tap is created before rumps' app.run() takes over the main thread.
Once the run loop starts, events flow automatically.
"""

import logging
import threading
from collections.abc import Callable
from typing import Optional

import Quartz
from Quartz import (
    CGEventGetIntegerValueField,
    CGEventMaskBit,
    CGEventTapCreate,
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetMain,
    kCFRunLoopCommonModes,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGHeadInsertEventTap,
    kCGSessionEventTap,
)

from config.defaults import DEFAULT_HOTKEY

logger = logging.getLogger(__name__)

# macOS virtual key codes for modifier keys
_KEYCODE_TO_NAME: dict[int, str] = {
    61: "right_option",
    58: "left_option",
    62: "right_ctrl",
    59: "left_ctrl",
    60: "right_shift",
    56: "left_shift",
    54: "right_cmd",
    55: "left_cmd",
    63: "fn",
}

# Map config hotkey names to virtual key codes (kept as module-level
# constant so settings_window can list available keys via HOTKEY_MAP).
HOTKEY_MAP: dict[str, int] = {name: code for code, name in _KEYCODE_TO_NAME.items()}

# CGEvent flag bits for each modifier key
_KEYCODE_TO_FLAG: dict[int, int] = {
    54: Quartz.kCGEventFlagMaskCommand,   # right_cmd
    55: Quartz.kCGEventFlagMaskCommand,   # left_cmd
    56: Quartz.kCGEventFlagMaskShift,     # left_shift
    58: Quartz.kCGEventFlagMaskAlternate,  # left_option
    59: Quartz.kCGEventFlagMaskControl,   # left_ctrl
    60: Quartz.kCGEventFlagMaskShift,     # right_shift
    61: Quartz.kCGEventFlagMaskAlternate,  # right_option
    62: Quartz.kCGEventFlagMaskControl,   # right_ctrl
    63: Quartz.kCGEventFlagMaskSecondaryFn,  # fn
}

_KCEVENT_KEYCODE = 9  # kCGKeyboardEventKeycode


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
        on_escape: Optional[Callable[[], None]] = None,
    ) -> None:
        if hotkey not in HOTKEY_MAP:
            raise ValueError(
                f"Unknown hotkey {hotkey!r}. Valid options: {list(HOTKEY_MAP)}"
            )

        self._keycode: int = HOTKEY_MAP[hotkey]
        self._on_press = on_press or (lambda: None)
        self._on_release = on_release or (lambda: None)
        self._on_escape = on_escape or (lambda: None)
        self._held = False
        self._lock = threading.Lock()
        self._tap = None
        self._source = None

    def start(self) -> None:
        """Create a CGEventTap and add it to the main CFRunLoop.

        Must be called before the run loop starts (i.e. before rumps app.run()).
        Events will begin flowing once the run loop is active.
        """
        mask = CGEventMaskBit(kCGEventFlagsChanged) | CGEventMaskBit(kCGEventKeyDown)

        self._tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._tap_callback,
            None,
        )
        if self._tap is None:
            logger.error(
                "CGEventTapCreate failed — Accessibility permission is required"
            )
            return

        self._source = CFMachPortCreateRunLoopSource(None, self._tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), self._source, kCFRunLoopCommonModes)
        logger.info("Hotkey listener started (keycode=%d)", self._keycode)

    def set_hotkey(self, hotkey: str) -> None:
        """Change the hotkey without recreating the event tap."""
        if hotkey not in HOTKEY_MAP:
            raise ValueError(
                f"Unknown hotkey {hotkey!r}. Valid options: {list(HOTKEY_MAP)}"
            )
        with self._lock:
            self._keycode = HOTKEY_MAP[hotkey]
            self._held = False
        logger.info("Hotkey changed to %s (keycode=%d)", hotkey, self._keycode)

    def stop(self) -> None:
        """Disable the event tap."""
        if self._tap is not None:
            Quartz.CGEventTapEnable(self._tap, False)
            self._tap = None
            self._source = None
            logger.info("Hotkey listener stopped")

    _ESCAPE_KEYCODE = 53

    def _tap_callback(self, proxy, event_type, event, refcon):
        """CGEventTap callback — runs on the main thread."""
        keycode = CGEventGetIntegerValueField(event, _KCEVENT_KEYCODE)

        if event_type == kCGEventKeyDown and keycode == self._ESCAPE_KEYCODE:
            self._on_escape()
            return event

        if event_type == kCGEventFlagsChanged:
            with self._lock:
                if keycode != self._keycode:
                    return event

                flag_bit = _KEYCODE_TO_FLAG.get(keycode, 0)
                flags = Quartz.CGEventGetFlags(event)
                is_pressed = bool(flags & flag_bit)

                if is_pressed and not self._held:
                    self._held = True
                    logger.debug("Hotkey pressed — recording start")
                    self._on_press()
                elif not is_pressed and self._held:
                    self._held = False
                    logger.debug("Hotkey released — recording stop")
                    self._on_release()

        return event

    def __enter__(self) -> "HotkeyListener":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
