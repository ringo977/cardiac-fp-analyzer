"""Plot theme presets + application helpers.

The PySide UI has three pyqtgraph ``PlotWidget``s — the Signal-tab
viewer (`signal_viewer.py`), the Battiti-tab template overlay
(`main.py`), and the dose-response dialog (`study_panel.py`) — and
they must read as a single coherent surface.  Before this module each
call site hard-coded its own ``setBackground('#0e1117')`` + axis pens,
and one call site slipped through with a white background, making the
dose-response dialog the only light-themed pane in an otherwise dark
app.

This module centralises the colour palette in two presets (``DARK``
and ``LIGHT``), exposes a pair of idempotent "apply" helpers, and
persists the user's choice via ``QSettings`` so the selection
survives restarts.

Design notes
------------
* The presets are deliberately minimal — ``background`` / ``foreground``
  / ``grid_alpha`` is everything the three plots need.  Per-series
  colours (signal trace, beat markers, drug curves) stay at their call
  sites because they are semantic, not chromatic: swapping their
  absolute RGB would change meaning, not just appearance.

* ``apply_to_plot`` is intentionally tolerant of being called again on
  the same widget — the menu toggle re-applies to every existing
  widget on theme change, and re-applying must be a no-op if the
  theme hasn't actually changed.

* QSettings uses the keys ``theme/mode`` = ``'dark' | 'light'``.  The
  organisation / application name are set by ``main.main()`` so the
  preference lands in the same backing store as other future prefs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from PySide6.QtCore import QObject, QSettings, Signal


ThemeMode = Literal["dark", "light"]


# ── Presets ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThemePalette:
    """Minimal palette consumed by the plot-application helpers.

    ``background`` is passed to ``pg.PlotWidget.setBackground``;
    ``foreground`` is used for both axis pens and axis tick text so
    the plot reads uniformly.  ``grid_alpha`` is a 0–1 scalar forwarded
    to ``PlotWidget.showGrid(alpha=...)``.
    """

    name: ThemeMode
    background: str
    foreground: str
    grid_alpha: float


DARK = ThemePalette(
    name="dark",
    # ``#0e1117`` matches the GitHub dark canvas — the Signal viewer
    # and Battiti template plot both used this literal before the
    # refactor; keeping the value identical avoids a visible shift for
    # existing users.
    background="#0e1117",
    foreground="#cccccc",
    grid_alpha=0.2,
)

LIGHT = ThemePalette(
    name="light",
    # Pure white is harsh under office lighting and eats detail in
    # grid lines; ``#f7f7f9`` is the "paper" tone used by matplotlib's
    # ``seaborn-whitegrid`` default and reads well on LCDs.
    background="#f7f7f9",
    # Dark grey rather than black: pyqtgraph's axis tick text is
    # anti-aliased and pure black can look blurry at smaller font
    # sizes.  ``#333333`` preserves contrast (WCAG AAA vs the ``f7``
    # background) without the fuzz.
    foreground="#333333",
    grid_alpha=0.35,
)


_PRESETS: dict[ThemeMode, ThemePalette] = {"dark": DARK, "light": LIGHT}


# ── QSettings-backed current theme ──────────────────────────────────────

_SETTINGS_GROUP = "theme"
_SETTINGS_KEY = "mode"
_DEFAULT: ThemeMode = "dark"


def _settings() -> QSettings:
    # QSettings(no-args) picks up organisation/application name set by
    # ``QApplication`` — see ``main.main()`` where we register those.
    # If they aren't set we still get a working in-memory fallback on
    # the INI backend, so this doesn't crash in tests or headless
    # tooling.
    return QSettings()


def current_mode() -> ThemeMode:
    """Return the persisted theme mode, defaulting to ``'dark'``.

    The value is validated against ``_PRESETS`` so a tampered settings
    file can never produce a crash — any unknown string silently falls
    back to the default.
    """
    value = _settings().value(f"{_SETTINGS_GROUP}/{_SETTINGS_KEY}", _DEFAULT)
    # ``QSettings.value`` returns ``str`` on most platforms but can hand
    # back ``QVariant`` edge cases on older Qt; coerce defensively.
    mode = str(value) if value is not None else _DEFAULT
    return mode if mode in _PRESETS else _DEFAULT


def set_mode(mode: ThemeMode) -> None:
    """Persist ``mode`` and notify observers.

    Raises ``ValueError`` if ``mode`` is not one of the registered
    presets — catching this mistake at the call site is far better than
    silently writing garbage into the settings store.
    """
    if mode not in _PRESETS:
        raise ValueError(
            f"Unknown theme mode {mode!r}; expected one of {list(_PRESETS)}"
        )
    _settings().setValue(f"{_SETTINGS_GROUP}/{_SETTINGS_KEY}", mode)
    _notifier.theme_changed.emit(mode)


def current_palette() -> ThemePalette:
    """Return the resolved palette for the currently-persisted mode."""
    return _PRESETS[current_mode()]


def palette(mode: ThemeMode) -> ThemePalette:
    """Return a palette by explicit name.

    Useful for menu-building code that wants to label entries with both
    modes without actually switching to them.
    """
    if mode not in _PRESETS:
        raise ValueError(
            f"Unknown theme mode {mode!r}; expected one of {list(_PRESETS)}"
        )
    return _PRESETS[mode]


# ── Change-notifier singleton ──────────────────────────────────────────
#
# Widgets that want to live-update on theme change connect to this
# signal.  Keeping the notifier a singleton rather than a MainWindow
# attribute lets the dose-response dialog (which is spawned from the
# Studi dock, not the main window proper) subscribe without threading
# a reference through every constructor.

class _ThemeNotifier(QObject):
    """Emits ``theme_changed(mode)`` when ``set_mode`` is called."""

    theme_changed = Signal(str)  # payload: ``'dark'`` or ``'light'``


_notifier = _ThemeNotifier()


def notifier() -> _ThemeNotifier:
    """Return the singleton notifier for subscribing to theme changes."""
    return _notifier


# ── Application helpers ────────────────────────────────────────────────

def apply_to_plot(plot_widget, theme: ThemePalette | None = None) -> None:
    """Apply background + axis colours + grid to a pyqtgraph plot.

    Parameters
    ----------
    plot_widget
        Either a ``pg.PlotWidget`` or a ``pg.PlotItem``-returning
        object; the helper duck-types on the two methods it needs
        (``setBackground`` and ``getAxis``).
    theme
        Optional override.  When ``None`` the currently-persisted theme
        is used — this is the common case during app startup and
        theme-toggle events.

    Safe to call multiple times on the same widget: each call simply
    resets the style to the requested palette.  The grid is re-shown
    on every call because pyqtgraph's ``showGrid(alpha=...)`` does not
    update alpha for an already-visible grid, and we want the ``LIGHT``
    preset to be able to bump the alpha up for legibility.
    """
    t = theme if theme is not None else current_palette()

    # ``setBackground`` exists on PlotWidget but not PlotItem — the
    # Signal-viewer subclasses PlotWidget, the dose-response dialog
    # and Battiti overlay use plain PlotWidget instances, so both
    # hit this branch.
    if hasattr(plot_widget, "setBackground"):
        plot_widget.setBackground(t.background)

    for ax_name in ("left", "bottom"):
        ax = plot_widget.getAxis(ax_name)
        ax.setPen(t.foreground)
        ax.setTextPen(t.foreground)

    # Grid always on with alpha from the preset.  The Battiti plot and
    # dose-response dialog already enable the grid; the Signal viewer
    # does too.  Re-calling ``showGrid`` with different alpha is the
    # supported way to update the grid opacity in pyqtgraph 0.13+.
    plot_widget.showGrid(x=True, y=True, alpha=t.grid_alpha)


def apply_to_legend(legend, theme: ThemePalette | None = None) -> None:
    """Set a pyqtgraph ``LegendItem``'s label text colour.

    Applied to legends added via ``PlotWidget.addLegend()``.  Like
    ``apply_to_plot`` this is idempotent — calling twice with the same
    palette is a no-op from the user's perspective.

    ``setLabelTextColor`` in pyqtgraph 0.13+ both stores the default
    for future items and repaints existing ones, so we do not need to
    iterate over ``legend.items`` manually.
    """
    t = theme if theme is not None else current_palette()
    legend.setLabelTextColor(t.foreground)


# ── Label helpers ──────────────────────────────────────────────────────

def label_color(theme: ThemePalette | None = None) -> str:
    """Return the foreground colour to pass to ``setLabel(color=...)``.

    ``pg.PlotItem.setLabel`` takes an explicit ``color`` kwarg that
    controls the axis title (distinct from axis tick text, which
    ``apply_to_plot`` sets via ``setTextPen``).  Call sites that
    customise their axis titles use this helper so the title follows
    the active theme.
    """
    t = theme if theme is not None else current_palette()
    return t.foreground


# ── Subscription convenience ───────────────────────────────────────────

def connect(callback: Callable[[str], None]) -> None:
    """Subscribe ``callback`` to theme changes.

    Equivalent to ``notifier().theme_changed.connect(callback)``.  The
    callback receives the new mode string; it is responsible for
    re-applying the palette to whatever widgets it owns.
    """
    _notifier.theme_changed.connect(callback)


# ── Bulk re-apply helper ───────────────────────────────────────────────

def reapply_to(widgets: Iterable, legends: Iterable = ()) -> None:
    """Re-apply the current palette to a batch of widgets and legends.

    Convenience used by the toggle action in ``main.py``: it owns
    references to every long-lived plot in the app and needs to refresh
    them all on a single click.  Dialogs that outlive the toggle
    (like a currently-open dose-response) subscribe to ``notifier()``
    themselves; this helper handles only the widgets the main window
    already knows about.
    """
    theme = current_palette()
    for w in widgets:
        apply_to_plot(w, theme)
    for lg in legends:
        apply_to_legend(lg, theme)
