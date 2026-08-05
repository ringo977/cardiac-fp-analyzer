"""Tests for the FPD/RR plausibility criterion.

Background
----------
Repolarization cannot occupy the whole cardiac cycle: an FPD equal to or
longer than the beat interval would mean repolarization finishing after
the next depolarization has already begun. Ratios near or above 100% are
detection failures — an afterpotential, or the following depolarization,
mistaken for the T wave.

This replaces the absolute ``fpdc_physiol`` window [350, 800] ms, which
misfires at the extremes of beat rate because Fridericia does not fully
remove the rate dependence. Measured on the 36 calibration baselines with
corrected RR, the absolute window rejected two QC-grade-B recordings for
being bradycardic (27 bpm, FPD/RR 51%) or fast (70 bpm, FPD/RR 31%)
rather than for anything being wrong — and rejecting a baseline removes
its entire dose-response group.

Distribution over those baselines: p25 32%, median 41%, p75 55%, p90 83%,
max 109%. Only 4 of 36 exceed 80%, and all four are implausible.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import InclusionConfig
from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria


class _QC:
    def __init__(self, grade):
        self.grade = grade


def _baseline(*, fpd_ms=400.0, bp_ms=1000.0, qc='A', conf=0.90,
              cv=10.0, filename='bl.csv'):
    return {
        'metadata': {'filename': filename},
        'file_info': {'drug': 'baseline', 'experiment': 'E1', 'chip': 'c1',
                      'chamber': 'ch1', 'electrode': 'el1',
                      'concentration': ''},
        'summary': {
            'fpd_ms_median': fpd_ms,
            'fpd_ms_mean': fpd_ms,
            'beat_period_ms_median': bp_ms,
            'beat_period_ms_mean': bp_ms,
            'beat_period_ms_cv': cv,
            'fpd_confidence': conf,
            'fpdc_ms_mean': 500.0,
            'bpm_mean': (60000.0 / bp_ms) if bp_ms else np.nan,
            'pct_beats_no_repol': 0.0,
        },
        'qc_report': _QC(qc),
    }


def _cfg(**over):
    cfg = InclusionConfig()
    cfg.enabled_fpdc_range = False        # isolate the ratio criterion
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _passes(result, cfg):
    apply_inclusion_criteria([result], verbose=False, cfg=cfg)
    return result['inclusion']['passed']


# ── Defaults ────────────────────────────────────────────────────────────

def test_ratio_criterion_is_on_and_absolute_window_is_off():
    cfg = InclusionConfig()
    assert cfg.enabled_fpd_rr_ratio is True
    assert cfg.enabled_fpdc_physiol is False
    assert cfg.max_fpd_rr_ratio == pytest.approx(0.80)


def test_combined_qc_rule_stays_opt_in():
    """The QC-grade cutoff is not calibrated; see config.py.

    At grade 'C' it re-excludes the dofetilide baseline (QC=D) and loses
    the whole dose-response, which is precisely what Sprint 2 set out to
    recover.
    """
    assert InclusionConfig().enabled_combined_rule is False


# ── The physically impossible cases ─────────────────────────────────────

@pytest.mark.parametrize("fpd_ms, bp_ms, ratio_pct", [
    (740.0, 678.0, 109),   # chipE_ch2_baseline — FPD longer than RR
    (581.0, 555.0, 105),   # chip1_ch3_baseline_nosignal
    (903.0, 924.0, 98),    # chipA_ch1_basline
    (818.0, 866.0, 94),    # chipA_ch2_baseline
])
def test_rejects_repolarization_that_fills_the_cycle(fpd_ms, bp_ms, ratio_pct):
    """The four real baselines that motivated this criterion."""
    bl = _baseline(fpd_ms=fpd_ms, bp_ms=bp_ms)
    assert _passes(bl, _cfg()) is False
    assert f'{ratio_pct}%' in bl['inclusion']['reason']


def test_reason_names_the_criterion_and_both_operands():
    bl = _baseline(fpd_ms=740.0, bp_ms=678.0)
    report = {}
    apply_inclusion_criteria([bl], verbose=False, cfg=_cfg(),
                             report_out=report)
    (info,) = report['excluded_groups'].values()
    assert info['criterion'] == 'fpd_rr_ratio'
    assert '740' in info['reason'] and '678' in info['reason']


# ── The rate extremes the absolute window got wrong ─────────────────────

def test_bradycardic_preparation_with_proportionate_fpd_passes():
    """chipD_ch3_baseline: 27 bpm, FPDc 871 ms, FPD/RR 51%, QC B.

    The absolute [350, 800] ms window rejected it for being slow.
    """
    bl = _baseline(fpd_ms=1135.0, bp_ms=2208.0, qc='B')
    assert _passes(bl, _cfg()) is True


def test_fast_preparation_with_short_fpd_passes():
    """chipD_ch1_baseline: 70 bpm, FPDc 277 ms, FPD/RR 31%, QC B."""
    bl = _baseline(fpd_ms=270.0, bp_ms=860.0, qc='B')
    assert _passes(bl, _cfg()) is True


def test_absolute_window_would_have_rejected_both():
    """Pins the contrast that motivated the replacement."""
    brady = _baseline(fpd_ms=1135.0, bp_ms=2208.0, qc='B')
    brady['summary']['fpdc_ms_mean'] = 871.0
    fast = _baseline(fpd_ms=270.0, bp_ms=860.0, qc='B')
    fast['summary']['fpdc_ms_mean'] = 277.0

    old_cfg = _cfg(enabled_fpd_rr_ratio=False, enabled_fpdc_physiol=True)
    assert _passes(brady, old_cfg) is False
    assert _passes(fast, old_cfg) is False


# ── Boundary and degenerate ─────────────────────────────────────────────

@pytest.mark.parametrize("ratio, expected", [
    (0.40, True), (0.79, True), (0.80, True), (0.81, False), (1.20, False),
])
def test_threshold_boundary(ratio, expected):
    bl = _baseline(fpd_ms=1000.0 * ratio, bp_ms=1000.0)
    assert _passes(bl, _cfg()) is expected


def test_threshold_is_configurable():
    bl = _baseline(fpd_ms=700.0, bp_ms=1000.0)          # 70%
    assert _passes(bl, _cfg(max_fpd_rr_ratio=0.60)) is False
    assert _passes(_baseline(fpd_ms=700.0, bp_ms=1000.0),
                   _cfg(max_fpd_rr_ratio=0.90)) is True


def test_missing_values_do_not_trip_the_criterion():
    bl = _baseline()
    bl['summary']['fpd_ms_median'] = np.nan
    bl['summary']['fpd_ms_mean'] = np.nan
    assert _passes(bl, _cfg()) is True


def test_zero_beat_period_does_not_divide_by_zero():
    bl = _baseline(bp_ms=0.0)
    assert _passes(bl, _cfg()) is True


def test_disabling_the_criterion_admits_the_impossible_case():
    bl = _baseline(fpd_ms=740.0, bp_ms=678.0)
    assert _passes(bl, _cfg(enabled_fpd_rr_ratio=False)) is True


def test_median_is_preferred_over_mean():
    """Median is the more robust statistic; mean is only the fallback."""
    bl = _baseline(fpd_ms=400.0, bp_ms=1000.0)
    bl['summary']['fpd_ms_median'] = 900.0      # 90% → reject
    bl['summary']['fpd_ms_mean'] = 400.0        # 40% → would pass
    assert _passes(bl, _cfg()) is False
