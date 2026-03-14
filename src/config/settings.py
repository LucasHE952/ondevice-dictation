"""User settings management for ondevice-dictation.

Settings are persisted as JSON in ~/.config/ondevice-dictation/settings.json.
On first run the file is created with DEFAULT_SETTINGS. On subsequent runs
the file is loaded and any missing keys are backfilled from defaults (forward
compatibility for new settings added in future versions).
"""

import json
import logging
from pathlib import Path
from typing import Any

from config.defaults import CONFIG_DIR, CONFIG_FILE, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


class Settings:
    """Manages loading, saving, and accessing user configuration.

    Usage:
        settings = Settings.load()
        settings["hotkey"]           # read a value
        settings["language"] = "fr"  # update a value
        settings.save()              # persist to disk
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data: dict[str, Any] = data

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from disk, creating the file if it does not exist.

        Returns:
            A Settings instance populated with persisted or default values.
        """
        if not CONFIG_FILE.exists():
            logger.info("No settings file found — creating defaults at %s", CONFIG_FILE)
            instance = cls(DEFAULT_SETTINGS.copy())
            instance.save()
            return instance

        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read settings (%s) — falling back to defaults", exc)
            return cls(DEFAULT_SETTINGS.copy())

        # Backfill any keys added in newer versions of the app.
        merged = {**DEFAULT_SETTINGS, **data}
        if merged != data:
            logger.info("Settings backfilled with new defaults; saving updated file")
            instance = cls(merged)
            instance.save()
            return instance

        return cls(merged)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist current settings to disk.

        Raises:
            OSError: If the config directory or file cannot be written.
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            logger.debug("Settings saved to %s", CONFIG_FILE)
        except OSError as exc:
            logger.error("Failed to save settings: %s", exc)
            raise

    # ── Dict-like access ─────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for key, or default if key is not present."""
        return self._data.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the settings dict."""
        return self._data.copy()
