"""Pure helpers for the PySide6 Settings dialog (task #74).

The actual ``SettingsDialog`` widget lives in ``settings_dialog.py`` and
depends on PySide6 — which means its import fails in any environment
without ``libEGL`` (our headless sandbox, CI containers, etc.).

Keeping the config ↔ dict round-trip here — in a module that does NOT
import Qt — lets us unit-test the mapping logic under ``pytest`` in
every environment.  The dialog just wires widgets to ``SPEC`` and
delegates to these functions on ``Applica`` / ``OK``.

Two contracts are pinned by the tests:

1. ``values_from_config(cfg)`` is a faithful snapshot — applying it back
   via ``apply_values_to_config`` must give a config equal (on the
   exposed fields) to the original.
2. ``apply_values_to_config(cfg, values)`` does NOT replace the config
   object; it mutates in-place and returns the same instance, so any
   caller holding a reference (MainWindow._config) sees the update
   without re-plumbing.

The ``SPEC`` module constant enumerates every exposed field as a
``FieldSpec`` with its section, type, range and human-readable label.
``SettingsDialog`` iterates over ``SPEC`` to build its widgets, so
adding a field is a one-line change here — no duplicated list in the
dialog code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cardiac_fp_analyzer.config import AnalysisConfig

# ═══════════════════════════════════════════════════════════════════════
#   Field specification
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FieldSpec:
    """Description of one user-editable config field.

    ``section`` is either ``"_top"`` (for top-level ``AnalysisConfig``
    attributes like ``amplifier_gain`` / ``use_overrides``) or the name
    of a sub-config attribute (``"filtering"``, ``"beat_detection"``,
    ``"repolarization"``, ``"quality"``, ``"inclusion"``).

    ``kind`` drives widget selection in ``SettingsDialog``:
        ``"float"``  → ``QDoubleSpinBox``
        ``"int"``    → ``QSpinBox``
        ``"bool"``   → ``QCheckBox``
        ``"choice"`` → ``QComboBox`` (with ``choices``)

    ``page`` is the dialog-tab label the field appears on — we don't
    tie it to ``section`` because the UI groups differ from the dataclass
    layout (e.g. the Segnale tab pulls from both ``_top`` and
    ``filtering``).
    """

    attr: str
    section: str          # "_top" | "filtering" | "beat_detection" | ...
    kind: str             # "float" | "int" | "bool" | "choice"
    page: str             # dialog-tab label
    label: str            # field label in the UI
    tooltip: str = ""
    # Numeric constraints — ignored for bool/choice
    minimum: float = 0.0
    maximum: float = 1e9
    step: float = 1.0
    decimals: int = 3
    suffix: str = ""
    # Choice widget
    choices: tuple[str, ...] = ()
    # Optional coercion (e.g. scientific-notation amplifier gain).
    # Applied to the value coming FROM the widget before setattr.
    coerce: Callable[[Any], Any] | None = None


# ═══════════════════════════════════════════════════════════════════════
#   SPEC — every user-editable field in one place
# ═══════════════════════════════════════════════════════════════════════
#
# Kept deliberately compact: only fields that the Streamlit sidebar
# exposed, plus ``use_overrides`` (from #64).  The long tail of
# ``AnalysisConfig`` (Normalization / Arrhythmia / ChannelSelection
# weights, scoring bonuses, adaptive-search fine-tuning…) stays
# power-user territory, editable only via a saved JSON preset.  Adding
# a field here is a single-line change; removing one is safe as long as
# tests don't pin it.

PAGE_SIGNAL = "Segnale"
PAGE_DETECTION = "Detection"
PAGE_REPOL = "Repolarizzazione"
PAGE_QC = "QC"
PAGE_INCLUSION = "Inclusione"


SPEC: tuple[FieldSpec, ...] = (
    # ── Segnale ─────────────────────────────────────────────────────
    FieldSpec(
        attr="amplifier_gain", section="_top", kind="float",
        page=PAGE_SIGNAL, label="Guadagno amplificatore",
        tooltip=(
            "Fattore di guadagno hardware (raw / gain = volt reale). "
            "Default µECG-Pharma Digilent: 10000 (×10⁴).  Imposta a 1 "
            "se il CSV è già in volt fisici."
        ),
        minimum=1.0, maximum=1e7, step=100.0, decimals=1,
    ),
    FieldSpec(
        attr="notch_freq_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Notch (Hz)",
        tooltip="Frequenza di rete per il filtro notch (50 Hz EU, 60 Hz US).",
        minimum=0.0, maximum=500.0, step=1.0, decimals=1, suffix=" Hz",
    ),
    FieldSpec(
        attr="bandpass_low_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Banda passante — basso",
        tooltip="Cutoff inferiore del filtro passa-banda.",
        minimum=0.0, maximum=1000.0, step=0.1, decimals=2, suffix=" Hz",
    ),
    FieldSpec(
        attr="bandpass_high_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Banda passante — alto",
        tooltip="Cutoff superiore del filtro passa-banda.",
        minimum=1.0, maximum=5000.0, step=10.0, decimals=1, suffix=" Hz",
    ),
    FieldSpec(
        attr="bandpass_order", section="filtering", kind="int",
        page=PAGE_SIGNAL, label="Banda passante — ordine",
        tooltip="Ordine del filtro Butterworth (1–8).",
        minimum=1, maximum=8, step=1,
    ),
    FieldSpec(
        attr="highpass_cutoff_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Passa-alto (drift) — cutoff",
        tooltip="Rimozione drift di baseline: cutoff del passa-alto.",
        minimum=0.0, maximum=100.0, step=0.1, decimals=2, suffix=" Hz",
    ),
    FieldSpec(
        attr="highpass_order", section="filtering", kind="int",
        page=PAGE_SIGNAL, label="Passa-alto — ordine",
        tooltip="Ordine del passa-alto drift (1–8).",
        minimum=1, maximum=8, step=1,
    ),
    FieldSpec(
        attr="lowpass_cutoff_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Passa-basso (anti-alias) — cutoff",
        tooltip="Smoothing anti-alias: cutoff del passa-basso.",
        minimum=1.0, maximum=5000.0, step=10.0, decimals=1, suffix=" Hz",
    ),
    FieldSpec(
        attr="lowpass_order", section="filtering", kind="int",
        page=PAGE_SIGNAL, label="Passa-basso — ordine",
        tooltip="Ordine del passa-basso (1–8).",
        minimum=1, maximum=8, step=1,
    ),

    # ── Detection ───────────────────────────────────────────────────
    FieldSpec(
        attr="method", section="beat_detection", kind="choice",
        page=PAGE_DETECTION, label="Metodo",
        tooltip=(
            "Algoritmo di picking: 'auto' sceglie il migliore via "
            "scoring fisiologico; 'derivative' è lo standard CiPA "
            "(dV/dt); 'prominence' è più sensibile ma può confondere "
            "onda T con depolarizzazione in costrutti 3D."
        ),
        choices=("auto", "prominence", "derivative", "peak"),
    ),
    FieldSpec(
        attr="min_distance_ms", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Distanza minima fra battiti",
        tooltip="Intervallo inter-battito minimo consentito (ms).",
        minimum=50.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="threshold_factor", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Soglia adattativa (× noise)",
        tooltip=(
            "Moltiplicatore della deviazione standard del rumore per "
            "determinare la soglia di rilevamento. Più alto → meno "
            "sensibile."
        ),
        minimum=0.5, maximum=20.0, step=0.5, decimals=2,
    ),
    FieldSpec(
        attr="enable_morphology_validation", section="beat_detection",
        kind="bool", page=PAGE_DETECTION,
        label="Validazione morfologica (template)",
        tooltip=(
            "Costruisce un template dai candidati più forti e rigetta "
            "i battiti con bassa correlazione. Disattivare per segnali "
            "molto irregolari o bradicardici."
        ),
    ),
    FieldSpec(
        attr="morphology_min_corr", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Correlazione minima template",
        tooltip=(
            "Soglia di correlazione template per accettare un battito "
            "(default 0.7; CardioMDA usa 0.95 su MEA planari)."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="enable_amplitude_cluster_filter", section="beat_detection",
        kind="bool", page=PAGE_DETECTION,
        label="Filtro cluster ampiezze",
        tooltip=(
            "Rigetta il cluster di battiti a bassa ampiezza quando il "
            "gap vs il cluster dominante supera la soglia (utile su "
            "segnali con T-wave residua mascherata da battito)."
        ),
    ),

    # ── Repolarizzazione ────────────────────────────────────────────
    FieldSpec(
        attr="fpd_method", section="repolarization", kind="choice",
        page=PAGE_REPOL, label="Metodo FPD",
        tooltip=(
            "tangent = tangente al punto di max-downslope (gold "
            "standard); peak = picco di repolarizzazione "
            "(sottostima ~25%); max_slope = punto di max-downslope; "
            "50pct = 50% ampiezza discesa; baseline_return = "
            "zero-crossing; consensus = voto tra metodi."
        ),
        choices=("tangent", "peak", "max_slope", "50pct",
                 "baseline_return", "consensus"),
    ),
    FieldSpec(
        attr="correction", section="repolarization", kind="choice",
        page=PAGE_REPOL, label="Correzione FPD",
        tooltip=(
            "fridericia = FPDcF = FPD / RR^(1/3); bazett = "
            "FPDcB = FPD / √RR; none = FPD grezzo."
        ),
        choices=("fridericia", "bazett", "none"),
    ),
    FieldSpec(
        attr="search_start_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Inizio finestra ricerca T-wave",
        tooltip="Inizia a cercare la repolarizzazione N ms dopo lo spike.",
        minimum=0.0, maximum=1000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="search_end_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Fine finestra ricerca T-wave",
        tooltip="Fine ricerca T-wave (sarà esteso adattivamente via %RR).",
        minimum=100.0, maximum=5000.0, step=50.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="search_end_pct_rr", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Estensione adattiva (%RR)",
        tooltip=(
            "Su segnali bradicardici estende la fine-finestra fino a "
            "questa frazione di RR. 0 disabilita."
        ),
        minimum=0.0, maximum=0.95, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="min_fpd_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="FPD minimo (pavimento fisso)",
        tooltip=(
            "Floor fisiologico: FPD sotto questa soglia è quasi "
            "certamente un afterpotential. Default 120 ms."
        ),
        minimum=0.0, maximum=1000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="min_fpd_pct_rr", section="repolarization", kind="float",
        page=PAGE_REPOL, label="FPD minimo adattivo (%RR)",
        tooltip=(
            "Floor adattivo: FPD deve essere almeno questa frazione "
            "dell'intervallo RR. 0 disabilita."
        ),
        minimum=0.0, maximum=0.95, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="max_adaptive_min_fpd_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Cap sul floor adattivo",
        tooltip=(
            "Cap sulla parte adattiva del floor (ms). Evita che su "
            "bradicardie estreme il floor salti oltre la T-wave reale. "
            "0 disabilita il cap."
        ),
        minimum=0.0, maximum=2000.0, step=50.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="min_valid_fpd_ratio", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Frazione minima FPD validi",
        tooltip=(
            "Se la frazione di battiti con FPD valido scende sotto "
            "questa soglia, il report flagga 'fpd_reliable=False'."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),

    # ── QC ─────────────────────────────────────────────────────────
    FieldSpec(
        attr="use_morphology", section="quality", kind="bool",
        page=PAGE_QC, label="Abilita QC morfologico",
        tooltip=(
            "Filtro QC post-segmentazione basato su correlazione con "
            "template. Disabilitare per accettare tutti i battiti."
        ),
    ),
    FieldSpec(
        attr="morphology_threshold", section="quality", kind="float",
        page=PAGE_QC, label="Soglia correlazione morfologica",
        tooltip="Correlazione minima per accettare un battito in QC.",
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="morphology_marginal", section="quality", kind="float",
        page=PAGE_QC, label="Soglia morfologica marginale",
        tooltip=(
            "Sotto questa soglia il battito viene rigettato con "
            "certezza (non recuperabile via strict-rhythm)."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="max_rejection_rate", section="quality", kind="float",
        page=PAGE_QC, label="Massima frazione battiti rigettati",
        tooltip=(
            "Sopra questa frazione di rigetti il QC grade viene "
            "abbassato a D."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="min_beats_for_analysis", section="quality", kind="int",
        page=PAGE_QC, label="Battiti minimi per analisi",
        tooltip="Meno di N battiti accettati → grade F.",
        minimum=1, maximum=50, step=1,
    ),

    # ── Inclusione ─────────────────────────────────────────────────
    FieldSpec(
        attr="enabled_cv", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="CV(BP) — attiva criterio",
        tooltip="Escludi baseline con CV del beat-period troppo alta.",
    ),
    FieldSpec(
        attr="max_cv_bp", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="CV(BP) — massimo (%)",
        tooltip="Limite superiore CV% del beat-period.",
        minimum=0.0, maximum=100.0, step=1.0, decimals=1, suffix=" %",
    ),
    FieldSpec(
        attr="enabled_fpdc_range", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="FPDcF — attiva range plausibile",
        tooltip="Escludi baseline con FPDcF fuori dal range di sicurezza.",
    ),
    FieldSpec(
        attr="fpdc_range_min", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="FPDcF — minimo",
        tooltip="Limite inferiore del range di sicurezza FPDcF.",
        minimum=0.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="fpdc_range_max", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="FPDcF — massimo",
        tooltip="Limite superiore del range di sicurezza FPDcF.",
        minimum=100.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="enabled_confidence", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="Confidence — attiva",
        tooltip="Escludi baseline con bassa confidence FPD.",
    ),
    FieldSpec(
        attr="min_fpd_confidence", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="Confidence minima FPD",
        tooltip="Soglia di confidence per includere una baseline.",
        minimum=0.0, maximum=1.0, step=0.01, decimals=2,
    ),
    FieldSpec(
        attr="enabled_fpdc_physiol", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="Range fisiologico — attiva",
        tooltip="Escludi baseline con FPDcF fuori dal range fisiologico.",
    ),
    FieldSpec(
        attr="fpdc_physiol_min", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="Range fisiologico — minimo",
        tooltip="Limite inferiore fisiologico FPDcF (letteratura: ~350 ms).",
        minimum=0.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="fpdc_physiol_max", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="Range fisiologico — massimo",
        tooltip="Limite superiore fisiologico FPDcF (letteratura: ~800 ms).",
        minimum=100.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),

    # ── Use overrides (cross-cutting, mostrato nel footer del dialog) ──
    FieldSpec(
        attr="use_overrides", section="_top", kind="bool",
        page=PAGE_DETECTION,  # semantically detection-related
        label="Applica correzioni manuali (sidecar .overrides.json)",
        tooltip=(
            "Carica e applica ad ogni analisi il sidecar di correzioni "
            "manuali salvato accanto al CSV. Spegni per una rilettura "
            "pulita dall'automatico."
        ),
    ),
)


# ═══════════════════════════════════════════════════════════════════════
#   Config ↔ values round-trip
# ═══════════════════════════════════════════════════════════════════════


def _get_section(cfg: AnalysisConfig, section: str):
    """Return the sub-config for a ``FieldSpec.section``.

    ``"_top"`` is the ``AnalysisConfig`` itself (for top-level fields
    like ``amplifier_gain``).  Unknown sections raise — the spec has
    a typo and we want to fail loud at test time rather than silently
    in the dialog.
    """
    if section == "_top":
        return cfg
    try:
        return getattr(cfg, section)
    except AttributeError as exc:
        raise KeyError(
            f"FieldSpec.section={section!r} does not match any "
            f"AnalysisConfig attribute"
        ) from exc


def values_from_config(cfg: AnalysisConfig) -> dict[str, Any]:
    """Snapshot the current value of every ``SPEC`` field.

    Key format: ``"{section}.{attr}"`` (e.g. ``"repolarization.fpd_method"``,
    ``"_top.amplifier_gain"``).  The dotted key lets a future UI store a
    subset without ambiguity, and matches how tests assert on individual
    fields.
    """
    out: dict[str, Any] = {}
    for fs in SPEC:
        section = _get_section(cfg, fs.section)
        out[f"{fs.section}.{fs.attr}"] = getattr(section, fs.attr)
    return out


def apply_values_to_config(
    cfg: AnalysisConfig, values: dict[str, Any],
) -> AnalysisConfig:
    """Mutate ``cfg`` in place with ``values`` and return the same object.

    Unknown keys are silently ignored (forward-compatible: a UI from a
    newer version can store extra fields without breaking old code).
    Type coercion is minimal — the dialog already returns the right
    Python type for each kind — but we still cast numeric widget output
    through the current attribute's type so ``int`` fields don't end up
    holding floats (QSpinBox → ``int``; QDoubleSpinBox → ``float``).
    """
    spec_by_key = {f"{fs.section}.{fs.attr}": fs for fs in SPEC}
    for key, raw in values.items():
        fs = spec_by_key.get(key)
        if fs is None:
            continue   # forward-compat
        if fs.coerce is not None:
            raw = fs.coerce(raw)
        section = _get_section(cfg, fs.section)
        current = getattr(section, fs.attr)
        # Preserve native type on numeric fields — bool passes through.
        if fs.kind == "int":
            raw = int(raw)
        elif fs.kind == "float":
            raw = float(raw)
        elif fs.kind == "bool":
            raw = bool(raw)
        elif fs.kind == "choice":
            # Accept the raw string; AnalysisConfig validates method names
            # lazily inside the pipeline (downstream errors are readable).
            raw = str(raw)
        # Defensive: if the current attribute is a tuple (some configs
        # use tuples for ranges), refuse to silently break it.
        if isinstance(current, tuple):
            raise TypeError(
                f"Field {key} is a tuple on AnalysisConfig; the dialog "
                f"must not expose it through a scalar widget."
            )
        setattr(section, fs.attr, raw)
    return cfg


def fields_for_page(page: str) -> tuple[FieldSpec, ...]:
    """Return all SPEC entries whose ``page`` matches ``page``.

    Used by ``SettingsDialog`` to build one tab per page.  Preserves the
    declaration order in ``SPEC`` so the UI order mirrors the source.
    """
    return tuple(fs for fs in SPEC if fs.page == page)


# The list of page labels, in declaration order — used by the dialog
# to build its ``QTabWidget`` without a hard-coded tab list.
def page_order() -> tuple[str, ...]:
    seen: list[str] = []
    for fs in SPEC:
        if fs.page not in seen:
            seen.append(fs.page)
    return tuple(seen)
