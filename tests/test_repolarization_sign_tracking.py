"""Regression tests for per-beat repolarization polarity tracking.

Background
----------
``find_repolarization_per_beat`` searches both polarities:

    for sign in [template_repol_sign, -template_repol_sign]:
        pks, props = sig.find_peaks(sign * seg_det, ...)
        ...
        if score > best_score:
            best_score = score
            best_idx = best_pk        # ← which peak
                                      # ← but NOT which sign

and then called ``apply_fpd_method(..., template_repol_sign, ...)``, always
passing the *template* polarity.  The tangent / 50% / baseline-return maths
is polarity-dependent, so for any beat whose T-wave is inverted relative to
the template the endpoint was computed on the wrong slope and silently
reported as a normal FPD.

The template path (``find_repolarization_on_template``) already passed its
own ``best_sign`` — that asymmetry was the tell.

This is not a theoretical path: ``parameters.py`` deliberately flips the
sign for inverted beats, so mixed-polarity recordings exercise it.

These tests pin two things:
  1. the winning polarity is what reaches ``apply_fpd_method``;
  2. an inverted-T beat measured against an upright template yields
     essentially the same FPD as the upright case (the physiology is
     mirror-symmetric; the measurement should be too).
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.repolarization import find_repolarization_per_beat

FS = 2000.0
SPIKE_T = 0.0
REPOL_T = 0.300          # T-wave centre, 300 ms after the spike
REPOL_SIGMA = 0.030


def _make_beat(repol_sign):
    """Synthetic FP beat: sharp depol spike + Gaussian T-wave.

    Parameters
    ----------
    repol_sign : int
        ``+1`` for an upright T-wave, ``-1`` for an inverted one.

    Returns
    -------
    (data, t, spike_idx)
    """
    t = np.arange(-0.05, 0.60, 1.0 / FS)
    data = np.zeros_like(t)
    # Depolarization spike — always the same polarity, as in real data.
    data += -1.0e-3 * np.exp(-((t - SPIKE_T) ** 2) / (2 * 0.002 ** 2))
    # Repolarization wave — polarity under test.
    data += repol_sign * 0.3e-3 * np.exp(
        -((t - REPOL_T) ** 2) / (2 * REPOL_SIGMA ** 2)
    )
    spike_idx = int(np.argmin(np.abs(t)))
    return data, t, spike_idx


def _measure(data, t, spike_idx, template_repol_sign, cfg=None):
    """Run per-beat repolarization detection, return FPD in ms or None."""
    cfg = cfg if cfg is not None else AnalysisConfig().repolarization
    fpd, _amp, _peak_i, _end_i = find_repolarization_per_beat(
        data, t, spike_idx, FS,
        template_repol_sign=template_repol_sign,
        cfg=cfg,
    )
    return None if fpd is None else fpd * 1000.0


# ── Sanity: the fixture produces a measurable, correct FPD ──────────────

def test_upright_beat_with_matching_template_sign():
    """Baseline case — template and beat agree. Must have always worked."""
    data, t, spike_idx = _make_beat(repol_sign=+1)
    fpd_ms = _measure(data, t, spike_idx, template_repol_sign=+1)

    assert fpd_ms is not None, "repolarization not detected on a clean beat"
    # Tangent method lands before the T-wave peak, so expect somewhat less
    # than 300 ms; the window is wide because the exact endpoint depends on
    # the configured fpd_method.
    assert 150.0 < fpd_ms < 400.0, f"implausible FPD {fpd_ms:.1f} ms"


def test_inverted_beat_with_matching_template_sign():
    """Mirror of the above — both flipped. Also a pre-existing happy path."""
    data, t, spike_idx = _make_beat(repol_sign=-1)
    fpd_ms = _measure(data, t, spike_idx, template_repol_sign=-1)

    assert fpd_ms is not None
    assert 150.0 < fpd_ms < 400.0, f"implausible FPD {fpd_ms:.1f} ms"


# ── The regression this fix addresses ───────────────────────────────────

def test_inverted_beat_against_upright_template():
    """An inverted T-wave measured with an upright template.

    This is the mixed-polarity case.  Before the fix the endpoint was
    computed with the template's (wrong) polarity and the resulting FPD
    was meaningless.
    """
    data, t, spike_idx = _make_beat(repol_sign=-1)
    fpd_ms = _measure(data, t, spike_idx, template_repol_sign=+1)

    assert fpd_ms is not None, (
        "inverted T-wave not detected at all — the dual-polarity search "
        "should have found it"
    )
    assert 150.0 < fpd_ms < 400.0, (
        f"FPD {fpd_ms:.1f} ms is outside the plausible range: the endpoint "
        "was probably computed on the wrong polarity"
    )


def test_polarity_mismatch_does_not_change_the_measurement():
    """The core invariant.

    Measuring the same inverted beat with a matching template sign and
    with a mismatched one must give the same answer: the search finds the
    same peak either way, so the endpoint maths must also agree.

    Before the fix these two diverged, because only the second call ran
    apply_fpd_method on the wrong slope.
    """
    data, t, spike_idx = _make_beat(repol_sign=-1)

    fpd_matched = _measure(data, t, spike_idx, template_repol_sign=-1)
    fpd_mismatched = _measure(data, t, spike_idx, template_repol_sign=+1)

    assert fpd_matched is not None and fpd_mismatched is not None

    # Allow a small tolerance: the two calls search the polarities in a
    # different order, which can shift the winner by a sample or two on a
    # noiseless synthetic.
    assert fpd_mismatched == pytest.approx(fpd_matched, abs=5.0), (
        f"matched={fpd_matched:.1f} ms vs mismatched={fpd_mismatched:.1f} ms "
        "— the template sign must not influence the measured FPD"
    )


def test_upright_and_inverted_beats_measure_the_same_fpd():
    """Mirror-symmetric signals must yield mirror-identical FPDs.

    The two beats differ only in the sign of the T-wave, so any
    polarity-dependent leakage in the endpoint maths shows up here.
    """
    up_data, up_t, up_spike = _make_beat(repol_sign=+1)
    dn_data, dn_t, dn_spike = _make_beat(repol_sign=-1)

    fpd_up = _measure(up_data, up_t, up_spike, template_repol_sign=+1)
    fpd_dn = _measure(dn_data, dn_t, dn_spike, template_repol_sign=-1)

    assert fpd_up is not None and fpd_dn is not None
    assert fpd_dn == pytest.approx(fpd_up, abs=5.0), (
        f"upright={fpd_up:.1f} ms vs inverted={fpd_dn:.1f} ms"
    )


@pytest.mark.parametrize("fpd_method", ['tangent', 'peak', '50pct',
                                        'baseline_return', 'max_slope'])
def test_all_fpd_methods_are_polarity_stable(fpd_method):
    """Every endpoint method must be immune to the template sign.

    ``apply_fpd_method`` branches on the method, and each branch does its
    own polarity-dependent arithmetic — so the invariant is checked per
    method rather than only on the default.
    """
    cfg = AnalysisConfig().repolarization
    cfg.fpd_method = fpd_method

    data, t, spike_idx = _make_beat(repol_sign=-1)
    fpd_matched = _measure(data, t, spike_idx,
                           template_repol_sign=-1, cfg=cfg)
    fpd_mismatched = _measure(data, t, spike_idx,
                              template_repol_sign=+1, cfg=cfg)

    if fpd_matched is None or fpd_mismatched is None:
        pytest.skip(f"method {fpd_method!r} produced no FPD on this synthetic")

    assert fpd_mismatched == pytest.approx(fpd_matched, abs=5.0), (
        f"method={fpd_method}: matched={fpd_matched:.1f} ms vs "
        f"mismatched={fpd_mismatched:.1f} ms"
    )
