"""PySide6 settings dialog for AnalysisConfig (task #74).

Iterates over ``settings_dialog_helpers.SPEC`` to build one tab per
logical page (Segnale / Detection / Repolarizzazione / QC / Inclusione)
with the right widget kind for each field.  The dialog itself is a
thin view — all config ↔ dict mapping lives in the pure helpers module
so it can be unit-tested without a Qt display.

Usage from MainWindow:

    dlg = SettingsDialog(self._config, parent=self)
    dlg.applied.connect(self._on_settings_applied)   # re-runs analysis
    dlg.exec()

Semantics of the three footer buttons:

* **Annulla** — close without touching the config.
* **Applica** — copy the widget values back into ``cfg`` (the same
  instance passed in) and emit ``applied``.  Dialog stays open.
* **OK**     — same as Applica, then close.
* **Ripristina default** — fill every widget with the default
  ``AnalysisConfig()`` values.  Does NOT touch the real config until
  the user clicks Applica / OK.

A preset ``QComboBox`` at the top of the dialog re-populates every
widget from one of the built-in presets
(``AnalysisConfig.preset(name)``).  The preset choice is NOT persisted —
the picker is a shortcut to "fill widgets with these values"; whatever
the user does afterwards (including reverting individual fields) is what
gets applied.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cardiac_fp_analyzer.config import AnalysisConfig
from pyside_app.settings_dialog_helpers import (
    SPEC,
    FieldSpec,
    apply_values_to_config,
    fields_for_page,
    page_order,
    values_from_config,
)

# ═══════════════════════════════════════════════════════════════════════
#   Widget factory — one QWidget per FieldSpec kind
# ═══════════════════════════════════════════════════════════════════════


def _make_widget(fs: FieldSpec, parent: QWidget | None = None) -> QWidget:
    """Create the appropriate input widget for a ``FieldSpec``.

    Keeps widget configuration in one place so SPEC authors only need
    to think about semantics (min/max/step), not Qt plumbing.  Tooltip,
    suffix and range are wired identically for every int/float field
    which keeps the dialog visually consistent.
    """
    if fs.kind == "float":
        w = QDoubleSpinBox(parent)
        w.setRange(fs.minimum, fs.maximum)
        w.setSingleStep(fs.step)
        w.setDecimals(fs.decimals)
        if fs.suffix:
            w.setSuffix(fs.suffix)
        if fs.tooltip:
            w.setToolTip(fs.tooltip)
        return w
    if fs.kind == "int":
        w = QSpinBox(parent)
        w.setRange(int(fs.minimum), int(fs.maximum))
        w.setSingleStep(int(fs.step))
        if fs.suffix:
            w.setSuffix(fs.suffix)
        if fs.tooltip:
            w.setToolTip(fs.tooltip)
        return w
    if fs.kind == "bool":
        w = QCheckBox(parent)
        if fs.tooltip:
            w.setToolTip(fs.tooltip)
        return w
    if fs.kind == "choice":
        w = QComboBox(parent)
        for c in fs.choices:
            w.addItem(c, userData=c)
        if fs.tooltip:
            w.setToolTip(fs.tooltip)
        return w
    raise ValueError(f"Unsupported FieldSpec.kind={fs.kind!r}")


def _read_widget(fs: FieldSpec, w: QWidget) -> Any:
    """Read the current value out of a widget as the Python type
    expected by ``apply_values_to_config``.
    """
    if fs.kind == "float":
        return float(w.value())  # type: ignore[attr-defined]
    if fs.kind == "int":
        return int(w.value())    # type: ignore[attr-defined]
    if fs.kind == "bool":
        return bool(w.isChecked())  # type: ignore[attr-defined]
    if fs.kind == "choice":
        return str(w.currentData())  # type: ignore[attr-defined]
    raise ValueError(f"Unsupported FieldSpec.kind={fs.kind!r}")


def _set_widget(fs: FieldSpec, w: QWidget, value: Any) -> None:
    """Write ``value`` into the widget without firing change signals.

    The dialog relies on widget values being the source of truth ONLY
    when the user clicks Applica/OK — intermediate programmatic sets
    (preset loading, defaults restore) must not ripple through to the
    config until the user confirms.
    """
    block = w.blockSignals(True)
    try:
        if fs.kind == "float":
            w.setValue(float(value))         # type: ignore[attr-defined]
        elif fs.kind == "int":
            w.setValue(int(value))           # type: ignore[attr-defined]
        elif fs.kind == "bool":
            w.setChecked(bool(value))        # type: ignore[attr-defined]
        elif fs.kind == "choice":
            # Match by userData string — falls back to index 0 if the
            # stored value is not a valid choice (e.g. legacy config).
            idx = 0
            for i in range(w.count()):       # type: ignore[attr-defined]
                if w.itemData(i) == value:   # type: ignore[attr-defined]
                    idx = i
                    break
            w.setCurrentIndex(idx)           # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported FieldSpec.kind={fs.kind!r}")
    finally:
        w.blockSignals(block)


# ═══════════════════════════════════════════════════════════════════════
#   SettingsDialog
# ═══════════════════════════════════════════════════════════════════════


class SettingsDialog(QDialog):
    """Tabbed settings dialog for ``AnalysisConfig``.

    The dialog holds a reference to the config passed in and mutates
    it in place on Applica/OK via ``apply_values_to_config``.  It emits
    ``applied`` AFTER every successful mutation so MainWindow can
    re-run the analysis on the currently-open CSV.

    Not modal — but MainWindow opens it with ``exec()``, which makes it
    behave modally at the app level without blocking background work
    inside the pipeline.
    """

    applied = Signal()  # fires after every Applica / OK, post-mutation

    def __init__(
        self, config: AnalysisConfig, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Impostazioni analisi"))
        self._config = config
        # Widget registry keyed by the same ``{section}.{attr}`` string
        # that ``values_from_config`` uses — lets us copy widget values
        # into a dict in one comprehension.
        self._widgets: dict[str, QWidget] = {}

        root = QVBoxLayout(self)

        # ── Preset row ───────────────────────────────────────────
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(self.tr("Preset:")))
        self._combo_preset = QComboBox(self)
        for name in ("default", "conservative", "sensitive",
                     "peak_method", "no_filters"):
            self._combo_preset.addItem(name)
        self._combo_preset.setToolTip(self.tr(
            "Carica un profilo di parametri nei campi sottostanti. "
            "Le modifiche non vengono applicate finché non si clicca "
            "Applica o OK."
        ))
        preset_row.addWidget(self._combo_preset)
        btn_load_preset = QPushButton(self.tr("Carica preset"), self)
        btn_load_preset.clicked.connect(self._on_load_preset)
        preset_row.addWidget(btn_load_preset)
        preset_row.addStretch(1)
        root.addLayout(preset_row)

        # ── Tabs ─────────────────────────────────────────────────
        self._tabs = QTabWidget(self)
        for page in page_order():
            self._tabs.addTab(self._build_page(page), page)
        root.addWidget(self._tabs)

        # ── Buttons ──────────────────────────────────────────────
        bb = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Apply
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.RestoreDefaults,
            parent=self,
        )
        bb.button(QDialogButtonBox.Ok).setText(self.tr("OK"))
        bb.button(QDialogButtonBox.Apply).setText(self.tr("Applica"))
        bb.button(QDialogButtonBox.Cancel).setText(self.tr("Annulla"))
        bb.button(QDialogButtonBox.RestoreDefaults).setText(
            self.tr("Ripristina default")
        )
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._on_restore_defaults
        )
        root.addWidget(bb)

        # Populate widgets with the current config.
        self._load_values(values_from_config(self._config))

    # ─── Page construction ────────────────────────────────────────────

    def _build_page(self, page: str) -> QWidget:
        """Build one ``QWidget`` holding a ``QFormLayout`` of fields."""
        container = QWidget(self)
        form = QFormLayout(container)
        for fs in fields_for_page(page):
            w = _make_widget(fs, container)
            key = f"{fs.section}.{fs.attr}"
            self._widgets[key] = w
            form.addRow(self.tr(fs.label), w)
        return container

    # ─── Value plumbing ──────────────────────────────────────────────

    def _current_values(self) -> dict[str, Any]:
        """Read every registered widget into a dict.

        Iterates ``SPEC`` (the source of truth) rather than
        ``self._widgets`` so a missing widget errors loud instead of
        silently dropping a key.
        """
        out: dict[str, Any] = {}
        for fs in SPEC:
            key = f"{fs.section}.{fs.attr}"
            w = self._widgets[key]
            out[key] = _read_widget(fs, w)
        return out

    def _load_values(self, values: dict[str, Any]) -> None:
        """Push ``values`` into every matching widget (silently).

        Unknown / missing keys are skipped — the registry only contains
        SPEC-declared keys, so extras from a forward-compat future are
        a no-op here too.
        """
        spec_by_key = {f"{fs.section}.{fs.attr}": fs for fs in SPEC}
        for key, w in self._widgets.items():
            fs = spec_by_key.get(key)
            if fs is None or key not in values:
                continue
            _set_widget(fs, w, values[key])

    # ─── Slots ───────────────────────────────────────────────────────

    def _on_ok(self) -> None:
        self._on_apply()
        self.accept()

    def _on_apply(self) -> None:
        """Commit widget values to the real config and notify MainWindow."""
        apply_values_to_config(self._config, self._current_values())
        self.applied.emit()

    def _on_restore_defaults(self) -> None:
        """Fill every widget with ``AnalysisConfig()`` defaults.

        Does NOT touch ``self._config`` — the user still has to click
        Applica / OK to commit.  This mirrors the "preview, then
        commit" ergonomics of every desktop app's Preferences panel.
        """
        self._load_values(values_from_config(AnalysisConfig()))

    def _on_load_preset(self) -> None:
        """Populate widgets from the selected built-in preset."""
        name = self._combo_preset.currentText()
        try:
            preset_cfg = AnalysisConfig.preset(name)
        except ValueError:
            return
        self._load_values(values_from_config(preset_cfg))
