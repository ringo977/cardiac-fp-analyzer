"""Regression tests for the FPDc rate-correction selection.

Background
----------
``RepolarizationConfig.correction`` and the ``--correction`` CLI flag were
exposed from the start, but nothing consumed them: ``parameters.py`` always
wrote the Fridericia value into ``fpdc_ms`` and ``rc.correction`` was only
stamped into ``summary['correction']`` as a label.  Running with
``--correction bazett`` therefore produced Fridericia values labelled
"bazett", including in the CDISC SEND export.

These tests pin the contract that fixes it:

  * ``fpdc_ms``            → the *configured* correction
  * ``fpdc_fridericia_ms`` → always Fridericia, whatever the config says
  * ``fpdc_bazett_ms``     → always Bazett, whatever the config says
  * an unknown correction  → falls back to Fridericia (does not raise)

Plus the invariant that matters most in practice: with the default config
(``correction='fridericia'``) the reported ``fpdc_ms`` is bit-identical to
the pre-fix behaviour, so no existing result changes.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.parameters import (
    VALID_CORRECTIONS,
    _select_corrected_fpd,
)

# ── Reference values ────────────────────────────────────────────────────
# FPD = 300 ms, RR = 1.0 s → both corrections collapse onto the raw value,
# which is useless for telling them apart.  Use RR = 0.8 s so that
# Fridericia (÷ 0.8^(1/3) = 0.9283) and Bazett (÷ sqrt(0.8) = 0.8944)
# give clearly distinct numbers.
_FPD_S = 0.300
_RR_S = 0.8
_FPD_MS = _FPD_S * 1000
_FRIDERICIA_MS = (_FPD_S / (_RR_S ** (1 / 3))) * 1000   # ≈ 323.2 ms
_BAZETT_MS = (_FPD_S / np.sqrt(_RR_S)) * 1000           # ≈ 335.4 ms


def test_reference_values_are_actually_distinct():
    """Guard the fixture itself: the two formulas must differ here.

    If this fails the other tests below become vacuous — they would pass
    whichever formula was selected.
    """
    assert abs(_FRIDERICIA_MS - _BAZETT_MS) > 10.0
    assert abs(_FRIDERICIA_MS - _FPD_MS) > 10.0


# ── _select_corrected_fpd ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "correction, expected",
    [
        ('fridericia', _FRIDERICIA_MS),
        ('bazett', _BAZETT_MS),
        ('none', _FPD_MS),
    ],
)
def test_selects_the_configured_formula(correction, expected):
    got = _select_corrected_fpd(
        fpd_ms=_FPD_MS,
        fpdc_fridericia_ms=_FRIDERICIA_MS,
        fpdc_bazett_ms=_BAZETT_MS,
        correction=correction,
    )
    assert got == pytest.approx(expected)


def test_bazett_is_not_silently_fridericia():
    """The exact bug this fix addresses, stated as an assertion."""
    got = _select_corrected_fpd(
        fpd_ms=_FPD_MS,
        fpdc_fridericia_ms=_FRIDERICIA_MS,
        fpdc_bazett_ms=_BAZETT_MS,
        correction='bazett',
    )
    assert got != pytest.approx(_FRIDERICIA_MS)


def test_unknown_correction_falls_back_to_fridericia_with_warning(caplog):
    """A typo in a hand-edited sidecar must degrade, not abort a batch.

    ``AnalysisConfig.from_dict`` does not validate this field, so an
    unknown value is reachable in practice.
    """
    with caplog.at_level('WARNING'):
        got = _select_corrected_fpd(
            fpd_ms=_FPD_MS,
            fpdc_fridericia_ms=_FRIDERICIA_MS,
            fpdc_bazett_ms=_BAZETT_MS,
            correction='fridericai',   # transposed letters
        )
    assert got == pytest.approx(_FRIDERICIA_MS)
    assert any('fridericai' in r.getMessage() for r in caplog.records), (
        "expected a warning naming the bad value"
    )


def test_valid_corrections_matches_config_docstring():
    """The tuple must stay in sync with what the config/CLI accept."""
    assert set(VALID_CORRECTIONS) == {'fridericia', 'bazett', 'none'}


# ── Default-config invariant ────────────────────────────────────────────

def test_default_config_is_fridericia():
    """Default results must be unchanged by this fix.

    Everyone's existing data was produced with the (only reachable)
    Fridericia behaviour, so the default must still select it.
    """
    cfg = AnalysisConfig()
    assert cfg.repolarization.correction == 'fridericia'

    got = _select_corrected_fpd(
        fpd_ms=_FPD_MS,
        fpdc_fridericia_ms=_FRIDERICIA_MS,
        fpdc_bazett_ms=_BAZETT_MS,
        correction=cfg.repolarization.correction,
    )
    assert got == pytest.approx(_FRIDERICIA_MS)


# ── Per-beat integration ────────────────────────────────────────────────

def _extract_with_correction(correction):
    """Run ``extract_beat_parameters`` on a synthetic beat.

    Returns the params dict, or ``None`` when repolarization was not
    detected on this synthetic (in which case the caller skips — the
    point of these tests is the correction arithmetic, not detection).
    """
    from cardiac_fp_analyzer.parameters import extract_beat_parameters

    fs = 2000.0
    # ``extract_beat_parameters`` takes the RepolarizationConfig sub-config
    # directly (see ``_get_repol_cfg``), not the full AnalysisConfig.
    cfg = AnalysisConfig().repolarization
    cfg.correction = correction

    # Synthetic beat: sharp depolarization spike at t=0, Gaussian
    # repolarization bump ~300 ms later.
    t = np.arange(-0.05, 0.60, 1.0 / fs)
    data = np.zeros_like(t)
    data += -1.0e-3 * np.exp(-((t - 0.0) ** 2) / (2 * 0.002 ** 2))   # spike
    data += 0.3e-3 * np.exp(-((t - 0.30) ** 2) / (2 * 0.030 ** 2))   # T wave

    params = extract_beat_parameters(
        data, t, fs,
        rr_interval=_RR_S,
        cfg=cfg,
    )
    if params is None or np.isnan(params.get('fpd_ms', np.nan)):
        return None
    return params


@pytest.mark.parametrize("correction", ['fridericia', 'bazett', 'none'])
def test_per_beat_fpdc_ms_follows_config(correction):
    params = _extract_with_correction(correction)
    if params is None:
        pytest.skip("repolarization not detected on this synthetic beat")

    fpd_ms = params['fpd_ms']
    expected = {
        'fridericia': (fpd_ms / 1000 / (_RR_S ** (1 / 3))) * 1000,
        'bazett': (fpd_ms / 1000 / np.sqrt(_RR_S)) * 1000,
        'none': fpd_ms,
    }[correction]

    assert params['fpdc_ms'] == pytest.approx(expected, rel=1e-9)


def test_per_beat_named_keys_are_formula_stable():
    """``fpdc_fridericia_ms``/``fpdc_bazett_ms`` ignore the config.

    A consumer that needs a specific formula (the CDISC ``FPDCF`` code)
    must be able to read it without inspecting the config.
    """
    p_fri = _extract_with_correction('fridericia')
    p_baz = _extract_with_correction('bazett')
    if p_fri is None or p_baz is None:
        pytest.skip("repolarization not detected on this synthetic beat")

    # Same signal → same measured FPD regardless of the correction setting.
    assert p_fri['fpd_ms'] == pytest.approx(p_baz['fpd_ms'])

    # ...and therefore the named keys must agree across the two runs.
    assert p_fri['fpdc_fridericia_ms'] == pytest.approx(
        p_baz['fpdc_fridericia_ms'])
    assert p_fri['fpdc_bazett_ms'] == pytest.approx(p_baz['fpdc_bazett_ms'])

    # But the *selected* value must differ, since the configs differ.
    assert p_fri['fpdc_ms'] != pytest.approx(p_baz['fpdc_ms'])

    # And each selected value equals its own named key.
    assert p_fri['fpdc_ms'] == pytest.approx(p_fri['fpdc_fridericia_ms'])
    assert p_baz['fpdc_ms'] == pytest.approx(p_baz['fpdc_bazett_ms'])
