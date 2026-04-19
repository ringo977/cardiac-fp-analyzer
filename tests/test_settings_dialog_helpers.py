"""Tests for ``pyside_app.settings_dialog_helpers`` (task #74).

These tests pin the config ↔ dict round-trip used by the PySide6
Settings dialog.  They do NOT touch PySide6 — the pure helpers live in
a separate module precisely so they are testable in environments
without a Qt display (our headless sandbox, CI containers, etc.).

Contracts covered:

* Every ``FieldSpec`` in ``SPEC`` points to a real attribute on
  ``AnalysisConfig`` — no typos slip through to runtime.
* ``values_from_config`` snapshots every exposed field; the returned
  dict has one entry per ``FieldSpec``.
* ``apply_values_to_config(cfg, values_from_config(cfg))`` is a no-op
  on the observed fields (identity round-trip).
* Partial updates (``use_overrides`` off, ``fpd_method`` → ``peak``)
  are reflected on the sub-config AND don't touch unrelated fields.
* Apply mutates the passed-in config instance (returning the same
  object), so callers holding a reference see the change.
* Unknown keys are silently dropped — forward-compat for future
  dialog versions that expose extra fields.
* Type coercion for int/float/bool kinds prevents QSpinBox-returns-int
  widgets from leaking ``float`` into int fields and vice versa.
"""

from __future__ import annotations

import pytest
from pyside_app.settings_dialog_helpers import (
    SPEC,
    apply_values_to_config,
    fields_for_page,
    page_order,
    values_from_config,
)

from cardiac_fp_analyzer.config import AnalysisConfig

# ═══════════════════════════════════════════════════════════════════════
#   SPEC sanity
# ═══════════════════════════════════════════════════════════════════════


def test_every_spec_field_exists_on_analysis_config():
    """Guard against typos / attribute drift in ``SPEC``.

    Every ``FieldSpec.attr`` must resolve on the correct section of a
    default ``AnalysisConfig``.  If this test fails, either the attr
    name is wrong or ``AnalysisConfig`` changed shape — bump both.
    """
    cfg = AnalysisConfig()
    for fs in SPEC:
        if fs.section == "_top":
            assert hasattr(cfg, fs.attr), (
                f"AnalysisConfig has no top-level attribute {fs.attr!r}"
            )
        else:
            sub = getattr(cfg, fs.section, None)
            assert sub is not None, (
                f"AnalysisConfig has no sub-config {fs.section!r}"
            )
            assert hasattr(sub, fs.attr), (
                f"AnalysisConfig.{fs.section} has no attribute "
                f"{fs.attr!r}"
            )


def test_spec_kind_values_are_supported():
    """``FieldSpec.kind`` must be one of the four widget kinds.

    Fail loud here so ``SettingsDialog``'s widget factory doesn't hit
    a silent fallback at runtime.
    """
    allowed = {"float", "int", "bool", "choice"}
    for fs in SPEC:
        assert fs.kind in allowed, (
            f"FieldSpec({fs.attr}).kind={fs.kind!r} not in {allowed}"
        )
        if fs.kind == "choice":
            assert fs.choices, (
                f"FieldSpec({fs.attr}) is a choice but has no choices"
            )


# ═══════════════════════════════════════════════════════════════════════
#   values_from_config
# ═══════════════════════════════════════════════════════════════════════


def test_values_from_config_covers_every_spec_field():
    """Snapshot must contain one key per ``FieldSpec``.

    Keys use the ``{section}.{attr}`` dotted form to avoid collisions.
    """
    cfg = AnalysisConfig()
    values = values_from_config(cfg)
    assert len(values) == len(SPEC)
    for fs in SPEC:
        key = f"{fs.section}.{fs.attr}"
        assert key in values, f"missing snapshot key {key!r}"


def test_values_from_config_returns_current_values():
    """Values must reflect the config's CURRENT state, not defaults.

    We mutate a representative field per section and verify the
    snapshot tracks it.  Snapshot is a plain dict — no lazy binding.
    """
    cfg = AnalysisConfig()
    cfg.amplifier_gain = 1e4                       # _top
    cfg.filtering.notch_freq_hz = 60.0             # filtering
    cfg.beat_detection.method = "derivative"       # beat_detection
    cfg.repolarization.fpd_method = "peak"         # repolarization
    cfg.quality.morphology_threshold = 0.55        # quality
    cfg.inclusion.max_cv_bp = 30.0                 # inclusion

    values = values_from_config(cfg)
    assert values["_top.amplifier_gain"] == pytest.approx(1e4)
    assert values["filtering.notch_freq_hz"] == pytest.approx(60.0)
    assert values["beat_detection.method"] == "derivative"
    assert values["repolarization.fpd_method"] == "peak"
    assert values["quality.morphology_threshold"] == pytest.approx(0.55)
    assert values["inclusion.max_cv_bp"] == pytest.approx(30.0)


# ═══════════════════════════════════════════════════════════════════════
#   apply_values_to_config — round-trip
# ═══════════════════════════════════════════════════════════════════════


def test_round_trip_is_identity_on_observed_fields():
    """snapshot → apply → snapshot must be stable.

    This is the core contract: opening the dialog and clicking OK
    without changing anything must not perturb the config.
    """
    cfg = AnalysisConfig()
    cfg.amplifier_gain = 1e4
    cfg.repolarization.fpd_method = "peak"
    cfg.inclusion.enabled_cv = False

    snap = values_from_config(cfg)
    cfg2 = AnalysisConfig()        # fresh defaults
    cfg2 = apply_values_to_config(cfg2, snap)

    # Every exposed field is now aligned; the snapshot of cfg2 matches
    # the snapshot of cfg.
    assert values_from_config(cfg2) == snap


# ═══════════════════════════════════════════════════════════════════════
#   apply_values_to_config — partial updates + side-effects
# ═══════════════════════════════════════════════════════════════════════


def test_apply_mutates_in_place_and_returns_same_object():
    """The dialog's MainWindow holds a reference to ``_config``.

    We must mutate the instance, not replace it, or the window keeps
    using the pre-Applica object.  This is the bug we want to never
    write: ``self._config = apply_values_to_config(self._config, v)``
    would work but relying on return alone hides re-plumbing bugs.
    """
    cfg = AnalysisConfig()
    out = apply_values_to_config(cfg, {"_top.amplifier_gain": 5.0})
    assert out is cfg
    assert cfg.amplifier_gain == pytest.approx(5.0)


def test_partial_update_leaves_unspecified_fields_alone():
    """Passing only a couple of keys must not reset everything else.

    Ensures ``apply_values_to_config`` is an UPDATE, not a REPLACE.
    """
    cfg = AnalysisConfig()
    cfg.amplifier_gain = 1e4           # pre-existing non-default
    cfg.repolarization.fpd_method = "tangent"
    cfg.inclusion.max_cv_bp = 25.0

    apply_values_to_config(cfg, {
        "repolarization.fpd_method": "peak",
    })

    assert cfg.repolarization.fpd_method == "peak"
    # Everything else unchanged.
    assert cfg.amplifier_gain == pytest.approx(1e4)
    assert cfg.inclusion.max_cv_bp == pytest.approx(25.0)


def test_apply_ignores_unknown_keys():
    """Unknown keys must be silently dropped (forward-compat).

    A newer UI version can ship extra keys; older pipelines must keep
    working on the same project file.
    """
    cfg = AnalysisConfig()
    apply_values_to_config(cfg, {
        "_top.amplifier_gain": 42.0,
        "_top.never_heard_of_this": 7,
        "bogus_section.whatever": "x",
    })
    assert cfg.amplifier_gain == pytest.approx(42.0)


def test_apply_coerces_numeric_types():
    """QSpinBox returns int, QDoubleSpinBox returns float — don't cross.

    If the user's widget backend ever returns a ``float`` for an
    ``int`` field (Qt does when step is non-integer), we must cast so
    downstream code that does ``isinstance(v, int)`` keeps working.
    """
    cfg = AnalysisConfig()

    # int field: beat_detection.method needs a string anyway; use
    # quality.min_beats_for_analysis which is a genuine int.
    apply_values_to_config(cfg, {
        "quality.min_beats_for_analysis": 5.0,          # float → int
        "filtering.notch_freq_hz": 50,                  # int → float
        "beat_detection.enable_morphology_validation": 0,   # 0 → False
    })
    assert cfg.quality.min_beats_for_analysis == 5
    assert isinstance(cfg.quality.min_beats_for_analysis, int)
    assert cfg.filtering.notch_freq_hz == pytest.approx(50.0)
    assert isinstance(cfg.filtering.notch_freq_hz, float)
    assert cfg.beat_detection.enable_morphology_validation is False


def test_spec_never_points_at_tuple_fields():
    """Invariant: no ``FieldSpec`` targets a tuple-typed attribute.

    ``AnalysisConfig`` uses tuples for a few range fields (e.g.
    ``beat_detection.bp_ideal_range_s``).  A scalar widget would
    silently stringify them and break the pipeline, so the design rule
    is "tuple fields are never in ``SPEC``" — checked here so a future
    well-meaning PR can't sneak one in.
    """
    cfg = AnalysisConfig()
    for fs in SPEC:
        section = cfg if fs.section == "_top" else getattr(cfg, fs.section)
        current = getattr(section, fs.attr)
        assert not isinstance(current, tuple), (
            f"FieldSpec({fs.attr}) points at a tuple attribute — "
            f"tuples must not be exposed through the scalar dialog"
        )


def test_apply_raises_on_tuple_fields_at_runtime():
    """Defensive guard inside ``apply_values_to_config`` fires when a
    caller bypasses ``SPEC`` and crafts a key pointing at a tuple.

    We monkey-patch ``SPEC`` to temporarily add a FieldSpec that points
    at ``beat_detection.bp_ideal_range_s`` (a real tuple field), then
    assert the guard raises.  The patch is undone in the finally block
    so other tests see the normal SPEC.
    """
    from pyside_app import settings_dialog_helpers as H

    bad = H.FieldSpec(
        attr="bp_ideal_range_s", section="beat_detection", kind="choice",
        page="dummy", label="dummy",
    )
    original = H.SPEC
    H.SPEC = H.SPEC + (bad,)
    try:
        cfg = AnalysisConfig()
        with pytest.raises(TypeError):
            apply_values_to_config(
                cfg, {"beat_detection.bp_ideal_range_s": "oops"},
            )
    finally:
        H.SPEC = original


# ═══════════════════════════════════════════════════════════════════════
#   Page layout helpers
# ═══════════════════════════════════════════════════════════════════════


def test_page_order_matches_spec_declaration_order():
    """Tab order in the dialog must match ``SPEC`` declaration order.

    This is the only ordering contract: the dialog relies on
    ``page_order()`` to build its ``QTabWidget`` without a hard-coded
    tab list.
    """
    order = page_order()
    # Every page label referenced in SPEC must appear exactly once.
    seen_in_spec: list[str] = []
    for fs in SPEC:
        if fs.page not in seen_in_spec:
            seen_in_spec.append(fs.page)
    assert list(order) == seen_in_spec


def test_fields_for_page_returns_only_matching_fields():
    """Fields are routed to the right page; no cross-page leaks."""
    for page in page_order():
        fields = fields_for_page(page)
        assert all(f.page == page for f in fields)
        # And every SPEC entry is reachable from SOME page.
    all_reached = sum(len(fields_for_page(p)) for p in page_order())
    assert all_reached == len(SPEC)
