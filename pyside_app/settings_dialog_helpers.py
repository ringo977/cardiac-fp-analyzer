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

PAGE_SIGNAL = "Signal"
PAGE_DETECTION = "Detection"
PAGE_REPOL = "Repolarisation"
PAGE_QC = "QC"
PAGE_INCLUSION = "Inclusion"


SPEC: tuple[FieldSpec, ...] = (
    # ── Segnale ─────────────────────────────────────────────────────
    FieldSpec(
        attr="amplifier_gain", section="_top", kind="float",
        page=PAGE_SIGNAL, label="Amplifier gain",
        tooltip=(
            "Hardware gain factor (raw / gain = real volts). Default µECG-Pharma Digilent: 10000 "
            "(×10⁴).  Set to 1 if the CSV is already in physical volts."
        ),
        minimum=1.0, maximum=1e7, step=100.0, decimals=1,
    ),
    FieldSpec(
        attr="notch_freq_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Notch (Hz)",
        tooltip="Mains frequency for the notch filter (50 Hz EU, 60 Hz US).",
        minimum=0.0, maximum=500.0, step=1.0, decimals=1, suffix=" Hz",
    ),
    FieldSpec(
        attr="bandpass_low_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Band-pass — low",
        tooltip="Lower cutoff of the band-pass filter.",
        minimum=0.0, maximum=1000.0, step=0.1, decimals=2, suffix=" Hz",
    ),
    FieldSpec(
        attr="bandpass_high_hz", section="filtering", kind="float",
        page=PAGE_SIGNAL, label="Band-pass — high",
        tooltip="Upper cutoff of the band-pass filter.",
        minimum=1.0, maximum=5000.0, step=10.0, decimals=1, suffix=" Hz",
    ),
    FieldSpec(
        attr="bandpass_order", section="filtering", kind="int",
        page=PAGE_SIGNAL, label="Band-pass — order",
        tooltip="Order of the Butterworth filter (1–8).",
        minimum=1, maximum=8, step=1,
    ),

    # ── Detection ───────────────────────────────────────────────────
    FieldSpec(
        attr="method", section="beat_detection", kind="choice",
        page=PAGE_DETECTION, label="Method",
        tooltip=(
            "Picking algorithm: 'auto' chooses the best one by physiological scoring; 'derivative' "
            "is the CiPA standard (dV/dt); 'prominence' is more sensitive but may mistake the T "
            "wave for a depolarisation in 3D constructs."
        ),
        choices=("auto", "prominence", "derivative", "peak"),
    ),
    FieldSpec(
        attr="min_distance_ms", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Minimum distance between beats",
        tooltip="Minimum allowed inter-beat interval (ms).",
        minimum=50.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="threshold_factor", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Adaptive threshold (× noise)",
        tooltip=(
            "Multiplier of the noise standard deviation that sets the detection threshold. Higher "
            "→ less sensitive."
        ),
        minimum=0.5, maximum=20.0, step=0.5, decimals=2,
    ),
    FieldSpec(
        attr="enable_morphology_validation", section="beat_detection",
        kind="bool", page=PAGE_DETECTION,
        label="Morphology validation (template)",
        tooltip=(
            "Builds a template from the strongest candidates and rejects the beats with low "
            "correlation. Disable for very irregular or bradycardic signals."
        ),
    ),
    FieldSpec(
        attr="morphology_min_corr", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Minimum template correlation",
        tooltip=(
            "Template correlation threshold to accept a beat (default 0.7; CardioMDA uses 0.95 on "
            "planar MEAs)."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="enable_amplitude_cluster_filter", section="beat_detection",
        kind="bool", page=PAGE_DETECTION,
        label="Amplitude cluster filter",
        tooltip=(
            "Rejects the low-amplitude cluster of beats when the gap from the dominant cluster "
            "exceeds the threshold (useful on signals where a residual T wave masquerades as a "
            "beat)."
        ),
    ),
    FieldSpec(
        attr="min_distance_floor_ms", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Minimum distance — lower limit (fast rhythms)",
        tooltip=(
            "On a regular rhythm faster than the minimum distance the software lowers the distance "
            "to half the measured period; it never goes below this value (100 ms = behaviour up to "
            "v3.16)."
        ),
        minimum=50.0, maximum=1000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="min_rr_physiological_ms", section="beat_detection", kind="float",
        page=PAGE_DETECTION, label="Minimum physiological interval between depolarisations",
        tooltip=(
            "Final check on the beat train: two detections closer than this cannot both be "
            "depolarisations; the one with the larger spike stays. 0 = off; default 200 ms. The "
            "lab's physiological limit: 250 ms (fastest rhythms observed: high-dose verapamil, "
            "250–350 ms)."
        ),
        minimum=0.0, maximum=2000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="enable_weak_population_split", section="beat_detection",
        kind="bool", page=PAGE_DETECTION,
        label="Remove weak detections (experimental)",
        tooltip=(
            "On slow, noisy recordings the detector also picks repolarisation waves and noise "
            "peaks. Splits the detections into two groups (ratio to the local noise and shape) and "
            "removes the weak group only when the groups are well separated. Off by default "
            "(v3.17)."
        ),
    ),
    FieldSpec(
        attr="enable_burst_veto", section="beat_detection",
        kind="bool", page=PAGE_DETECTION,
        label="Remove false detections in the incubator bursts (experimental)",
        tooltip=(
            "Finds the noise bursts (10 s cycle of the incubator) and removes the detections "
            "inside the bursts and at their edges that do not resemble the clean beats (shape, "
            "slope, amplitude on a 30 Hz low-pass). Off by default (v3.17)."
        ),
    ),

    # ── Repolarizzazione ────────────────────────────────────────────
    FieldSpec(
        attr="fpd_method", section="repolarization", kind="choice",
        page=PAGE_REPOL, label="FPD method",
        tooltip=(
            "peak = peak of the repolarisation wave (default since v3.6.0: convention of the "
            "manual reference measurement); tangent = tangent at the max-downslope point (end of "
            "the wave, a few % longer; default up to v3.5); max_slope = max-downslope point; 50pct "
            "= 50% of the descending amplitude; baseline_return = zero crossing; consensus = vote "
            "among methods."
        ),
        choices=("peak", "tangent", "max_slope", "50pct",
                 "baseline_return", "consensus"),
    ),
    FieldSpec(
        attr="correction", section="repolarization", kind="choice",
        page=PAGE_REPOL, label="FPD correction",
        tooltip=(
            "fridericia = FPDcF = FPD / RR^(1/3); bazett = FPDcB = FPD / √RR; none = raw FPD."
        ),
        choices=("fridericia", "bazett", "none"),
    ),
    FieldSpec(
        attr="search_start_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="T-wave search window start",
        tooltip="Starts searching for the repolarisation N ms after the spike.",
        minimum=0.0, maximum=1000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="search_end_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="T-wave search window end",
        tooltip="End of the T-wave search (extended adaptively via %RR).",
        minimum=100.0, maximum=5000.0, step=50.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="search_end_pct_rr", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Adaptive extension (%RR)",
        tooltip=(
            "On bradycardic signals extends the window end up to this fraction of RR. 0 disables."
        ),
        minimum=0.0, maximum=0.95, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="min_fpd_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Minimum FPD (fixed floor)",
        tooltip=(
            "Physiological floor: an FPD below this threshold is almost certainly an "
            "afterpotential. Default 120 ms."
        ),
        minimum=0.0, maximum=1000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="min_fpd_pct_rr", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Adaptive minimum FPD (%RR)",
        tooltip=(
            "Adaptive floor: the FPD must be at least this fraction of the RR interval. 0 disables."
        ),
        minimum=0.0, maximum=0.95, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="max_adaptive_min_fpd_ms", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Cap on the adaptive floor",
        tooltip=(
            "Cap on the adaptive part of the floor (ms). Prevents the floor from jumping past the "
            "real T wave on extreme bradycardia. 0 disables the cap."
        ),
        minimum=0.0, maximum=2000.0, step=50.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="min_valid_fpd_ratio", section="repolarization", kind="float",
        page=PAGE_REPOL, label="Minimum fraction of valid FPDs",
        tooltip=(
            "If the fraction of beats with a valid FPD falls below this threshold, the report "
            "flags 'fpd_reliable=False'."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),

    # ── QC ─────────────────────────────────────────────────────────
    FieldSpec(
        attr="use_morphology", section="quality", kind="bool",
        page=PAGE_QC, label="Enable morphology QC",
        tooltip=(
            "Post-segmentation QC filter based on the correlation with the template. Disable to "
            "accept all beats."
        ),
    ),
    FieldSpec(
        attr="morphology_threshold", section="quality", kind="float",
        page=PAGE_QC, label="Morphology correlation threshold",
        tooltip="Minimum correlation to accept a beat in QC.",
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="morphology_marginal", section="quality", kind="float",
        page=PAGE_QC, label="Marginal morphology threshold",
        tooltip=(
            "Below this threshold the beat is rejected for sure (not recoverable via strict "
            "rhythm)."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="max_rejection_rate", section="quality", kind="float",
        page=PAGE_QC, label="Maximum fraction of rejected beats",
        tooltip=(
            "Above this fraction of rejections the QC grade is lowered to D."
        ),
        minimum=0.0, maximum=1.0, step=0.05, decimals=2,
    ),
    FieldSpec(
        attr="min_beats_for_analysis", section="quality", kind="int",
        page=PAGE_QC, label="Minimum beats for analysis",
        tooltip="Fewer than N accepted beats → grade F.",
        minimum=1, maximum=50, step=1,
    ),
    FieldSpec(
        attr="not_analysable_gap_tolerant", section="quality",
        kind="bool", page=PAGE_QC,
        label="Rhythm with missing beats stays analysable (experimental)",
        tooltip=(
            "A recording with fewer than 16 beats and CV > 40 % stays analysable if the intervals "
            "are multiples of a common period (missing beats, not an irregular rhythm). To be used "
            "with the two detection options. Off by default (v3.17)."
        ),
    ),

    # ── Inclusione ─────────────────────────────────────────────────
    FieldSpec(
        attr="enabled_cv", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="CV(BP) — enable criterion",
        tooltip="Exclude baselines with too high a beat-period CV.",
    ),
    FieldSpec(
        attr="max_cv_bp", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="CV(BP) — maximum (%)",
        tooltip="Upper limit of the beat-period CV%.",
        minimum=0.0, maximum=100.0, step=1.0, decimals=1, suffix=" %",
    ),
    FieldSpec(
        attr="enabled_fpdc_range", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="FPDcF — enable plausible range",
        tooltip="Exclude baselines with FPDcF outside the safety range.",
    ),
    FieldSpec(
        attr="fpdc_range_min", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="FPDcF — minimum",
        tooltip="Lower limit of the FPDcF safety range.",
        minimum=0.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="fpdc_range_max", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="FPDcF — maximum",
        tooltip="Upper limit of the FPDcF safety range.",
        minimum=100.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="enabled_confidence", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="Confidence — enable",
        tooltip="Exclude baselines with low FPD confidence.",
    ),
    FieldSpec(
        attr="min_fpd_confidence", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="Minimum FPD confidence",
        tooltip="Confidence threshold to include a baseline.",
        minimum=0.0, maximum=1.0, step=0.01, decimals=2,
    ),
    FieldSpec(
        attr="enabled_fpdc_physiol", section="inclusion", kind="bool",
        page=PAGE_INCLUSION, label="Physiological range — enable",
        tooltip="Exclude baselines with FPDcF outside the physiological range.",
    ),
    FieldSpec(
        attr="fpdc_physiol_min", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="Physiological range — minimum",
        tooltip="Lower physiological limit of FPDcF (literature: ~350 ms).",
        minimum=0.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),
    FieldSpec(
        attr="fpdc_physiol_max", section="inclusion", kind="float",
        page=PAGE_INCLUSION, label="Physiological range — maximum",
        tooltip="Upper physiological limit of FPDcF (literature: ~800 ms).",
        minimum=100.0, maximum=5000.0, step=10.0, decimals=0, suffix=" ms",
    ),

    # ── Use overrides (cross-cutting, mostrato nel footer del dialog) ──
    FieldSpec(
        attr="use_overrides", section="_top", kind="bool",
        page=PAGE_DETECTION,  # semantically detection-related
        label="Apply manual corrections (.overrides.json sidecar)",
        tooltip=(
            "Loads and applies at every analysis the manual-corrections sidecar saved next to the "
            "CSV. Switch off for a clean automatic re-read."
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
