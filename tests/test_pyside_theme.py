"""Regression tests for ``pyside_app.theme``.

Covers the pure-model surface — palette lookup, QSettings
persistence, notifier signal emission — without touching any real
pyqtgraph widgets.  Skipped when PySide6 (or its transitive native
deps) cannot be imported, which is the case in the sandbox but not
on Marco's Mac.

These tests run with the ``offscreen`` Qt platform so they do not
require a real display server.
"""
from __future__ import annotations

import os
import sys

import pytest

# ``offscreen`` lets Qt create a hidden platform backend so
# QApplication() doesn't require a real X/Wayland/macOS session.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ``importorskip`` alone is not enough on headless Linux sandboxes:
# PySide6 imports cleanly but pulls in libEGL/libGL at widget-class
# import time.  Guard both steps so the tests skip cleanly rather
# than failing at collection time on machines without a display
# driver installed (our sandbox; CI without ``xvfb``).
pytest.importorskip("PySide6")
try:
    from PySide6.QtCore import QSettings  # noqa: E402
    from PySide6.QtWidgets import QApplication  # noqa: E402
except ImportError as exc:   # pragma: no cover — platform-dependent
    pytest.skip(
        f"PySide6 widget libs unavailable in this environment: {exc}",
        allow_module_level=True,
    )


# ── QApplication fixture ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    """One ``QApplication`` shared across all theme tests.

    Qt forbids more than one ``QApplication`` per process, so we
    reuse whatever instance is live (e.g. created by another test
    module) and fall back to creating one ourselves.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    # A stable (organisation, application) pair makes QSettings write
    # into a predictable key namespace — individual tests clear this
    # namespace in their setup so the suite is isolation-safe.
    app.setOrganizationName("CardiacFP-Test")
    app.setApplicationName("theme-unit")
    yield app
    # Don't call app.quit() — other test modules in the same process
    # may still need it.


@pytest.fixture(autouse=True)
def _clear_theme_settings(qapp):
    """Reset the theme key before each test."""
    s = QSettings()
    s.remove("theme/mode")
    s.sync()
    yield
    s.remove("theme/mode")
    s.sync()


# ── Palette presets ──────────────────────────────────────────────────

def test_dark_palette_matches_historical_constants(qapp):
    """Dark preset must keep the exact hex codes that were hard-coded
    in every call site before the extraction — changing them would be
    a visible, user-facing regression."""
    from pyside_app import theme

    assert theme.DARK.background == "#0e1117"
    assert theme.DARK.foreground == "#cccccc"
    assert theme.DARK.name == "dark"


def test_light_palette_is_distinct_and_well_formed(qapp):
    """Sanity: the light preset exists, is a different colour, and
    has a higher grid alpha (white bg needs a stronger grid to be
    readable)."""
    from pyside_app import theme

    assert theme.LIGHT.name == "light"
    assert theme.LIGHT.background != theme.DARK.background
    assert theme.LIGHT.foreground != theme.DARK.foreground
    assert theme.LIGHT.grid_alpha > theme.DARK.grid_alpha


def test_palette_lookup_by_name(qapp):
    from pyside_app import theme

    assert theme.palette("dark") is theme.DARK
    assert theme.palette("light") is theme.LIGHT
    with pytest.raises(ValueError):
        theme.palette("blueprint")  # type: ignore[arg-type]


# ── Persistence round-trip ──────────────────────────────────────────

def test_default_mode_is_dark_when_unset(qapp):
    """Fresh QSettings has no theme key → caller sees ``'dark'``.

    This is the preference most users will inherit at first launch,
    and it matches the pre-refactor hard-coded default.
    """
    from pyside_app import theme

    assert theme.current_mode() == "dark"


def test_set_mode_persists_across_lookups(qapp):
    from pyside_app import theme

    theme.set_mode("light")
    assert theme.current_mode() == "light"
    assert theme.current_palette() is theme.LIGHT

    theme.set_mode("dark")
    assert theme.current_mode() == "dark"
    assert theme.current_palette() is theme.DARK


def test_set_mode_rejects_unknown_values(qapp):
    from pyside_app import theme

    with pytest.raises(ValueError):
        theme.set_mode("neon")  # type: ignore[arg-type]


def test_current_mode_falls_back_on_corrupted_settings(qapp):
    """Tampered settings (e.g. a user hand-edits the INI file to
    ``theme/mode=neon``) must not crash the app — ``current_mode``
    must silently fall back to the default preset."""
    from pyside_app import theme

    QSettings().setValue("theme/mode", "neon")
    assert theme.current_mode() == "dark"


# ── Notifier signal ─────────────────────────────────────────────────

def test_set_mode_emits_theme_changed(qapp):
    """``theme.set_mode`` must fire ``notifier().theme_changed`` with
    the new mode so subscribed plots can live-repaint."""
    from pyside_app import theme

    received: list[str] = []
    theme.connect(received.append)

    theme.set_mode("light")
    theme.set_mode("dark")

    # One emission per ``set_mode`` call; payload is the new mode.
    assert received == ["light", "dark"]


def test_label_color_tracks_current_palette(qapp):
    """``theme.label_color()`` resolves to the current palette's
    foreground — it's the colour call sites pass to pyqtgraph's
    ``setLabel(color=...)``, so it must follow the toggle."""
    from pyside_app import theme

    theme.set_mode("dark")
    assert theme.label_color() == theme.DARK.foreground

    theme.set_mode("light")
    assert theme.label_color() == theme.LIGHT.foreground
