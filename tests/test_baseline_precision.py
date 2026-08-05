"""Tests for the baseline FPDc reference-precision criterion.

Background
----------
A baseline exists to supply a reference FPDc. What matters is how
precisely that reference is determined — the relative standard error,
SD/(mean·sqrt(n)) — not how regular the rhythm happened to be.

CV(RR) was standing in for this. Measured on the 36 calibration
baselines with corrected RR it explains only ~38% of the variance in
FPDc dispersion (r = 0.61), and end-to-end over 201 recordings the two
criteria compare as:

    CV < 25%                134/201 included, 11 drugs
    rSEM <= 3%              169/201 included, 18 drugs
    both                    134/201 included, 11 drugs

Adding CV on top of the precision criterion costs 35 files and 7 drugs
and gains nothing — it is strictly dominated.

Caveat encoded below: precision guards against *random* error only. A
noisy recording whose detector consistently locks onto an afterpotential
yields a precise estimate of the wrong quantity, so this criterion is
not sufficient on its own and is not enabled by default.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import InclusionConfig
from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria


class _QC:
    def __init__(self, grade):
        self.grade = grade


def _baseline(*, fpdc=500.0, sd=50.0, n=100, cv=10.0, qc='A', conf=0.90,
              fpd_ms=400.0, bp_ms=1000.0):
    return {
        'metadata': {'filename': 'bl.csv'},
        'file_info': {'drug': 'baseline', 'experiment': 'E1', 'chip': 'c1',
                      'chamber': 'ch1', 'electrode': 'el1',
                      'concentration': ''},
        'summary': {
            'fpdc_ms_mean': fpdc,
            'fpdc_ms_std': sd,
            'fpd_ms_n': n,
            'beat_period_ms_cv': cv,
            'fpd_confidence': conf,
            'fpd_ms_median': fpd_ms,
            'fpd_ms_mean': fpd_ms,
            'beat_period_ms_median': bp_ms,
            'beat_period_ms_mean': bp_ms,
            'bpm_mean': 60000.0 / bp_ms,
            'pct_beats_no_repol': 0.0,
        },
        'qc_report': _QC(qc),
    }


def _cfg(**over):
    cfg = InclusionConfig()
    cfg.enabled_fpdc_range = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _passes(result, cfg):
    apply_inclusion_criteria([result], verbose=False, cfg=cfg)
    return result['inclusion']['passed']


def _rsem(sd, mean, n):
    return (sd / mean * 100.0) / np.sqrt(n)


# ── Defaults ────────────────────────────────────────────────────────────

def test_criterion_is_opt_in():
    """Not enabled by default: precision alone is not sufficient.

    It admits QC-grade-D/F recordings whose reference happens to be
    precisely determined, and precision does not protect against a
    systematically mis-detected T wave.
    """
    cfg = InclusionConfig()
    assert cfg.enabled_baseline_precision is False
    assert cfg.max_baseline_fpdc_rsem == pytest.approx(3.0)


# ── The statistic ───────────────────────────────────────────────────────

def test_precise_reference_passes():
    # SD 50 / mean 500 = CV 10%, over 100 beats → rSEM 1.0%
    bl = _baseline(fpdc=500.0, sd=50.0, n=100)
    assert _rsem(50.0, 500.0, 100) == pytest.approx(1.0)
    assert _passes(bl, _cfg(enabled_baseline_precision=True)) is True


def test_imprecise_reference_fails():
    # CV 33% over 27 beats → rSEM 6.4%, the worst real baseline
    bl = _baseline(fpdc=500.0, sd=165.0, n=27)
    assert _rsem(165.0, 500.0, 27) == pytest.approx(6.35, abs=0.05)
    assert _passes(bl, _cfg(enabled_baseline_precision=True)) is False


def test_beat_count_matters_not_just_dispersion():
    """The whole point of using SEM rather than CV.

    Same 30% dispersion; 9 beats is imprecise, 400 beats is not.
    """
    on = _cfg(enabled_baseline_precision=True)
    few = _baseline(fpdc=500.0, sd=150.0, n=9)     # rSEM 10%
    many = _baseline(fpdc=500.0, sd=150.0, n=400)  # rSEM 1.5%
    assert _passes(few, on) is False
    assert _passes(many, on) is True


@pytest.mark.parametrize("sd, n, expect", [
    (150.0, 100, True),    # rSEM 3.00% — boundary, inclusive
    (151.0, 100, False),   # rSEM 3.02%
    (100.0, 100, True),    # rSEM 2.00%
])
def test_threshold_boundary(sd, n, expect):
    bl = _baseline(fpdc=500.0, sd=sd, n=n)
    assert _passes(bl, _cfg(enabled_baseline_precision=True)) is expect


def test_threshold_is_configurable():
    bl = _baseline(fpdc=500.0, sd=125.0, n=100)    # rSEM 2.5%
    assert _passes(bl, _cfg(enabled_baseline_precision=True,
                            max_baseline_fpdc_rsem=2.0)) is False
    assert _passes(_baseline(fpdc=500.0, sd=125.0, n=100),
                   _cfg(enabled_baseline_precision=True,
                        max_baseline_fpdc_rsem=5.0)) is True


# ── Against the CV gate ─────────────────────────────────────────────────

def test_irregular_but_well_sampled_baseline():
    """chipD_ch3_baseline: CV(RR) 29.2%, rSEM 1.25%, n=65, QC B.

    The CV gate rejects it and takes its whole dose-response group with
    it; the averaging absorbs the rhythm irregularity and the reference
    is perfectly usable.
    """
    bl = _baseline(cv=29.2, fpdc=871.0, sd=81.0, n=65, qc='B', conf=0.89,
                   fpd_ms=1135.0, bp_ms=2208.0)
    assert _passes(bl, _cfg(enabled_cv=True,
                            enabled_baseline_precision=False)) is False
    assert _passes(_baseline(cv=29.2, fpdc=871.0, sd=81.0, n=65, qc='B',
                             conf=0.89, fpd_ms=1135.0, bp_ms=2208.0),
                   _cfg(enabled_cv=False,
                        enabled_baseline_precision=True)) is True


def test_regular_rhythm_does_not_guarantee_a_precise_reference():
    """The converse case, and why CV is a poor proxy.

    chipD_ch1_baseline: CV(RR) 24.9% passes the gate, but FPDc dispersion
    of 33% over 116 beats gives rSEM 3.04% — just over the bound.
    """
    args = dict(cv=24.9, fpdc=277.0, sd=91.0, n=116, qc='B')
    assert _passes(_baseline(**args), _cfg(enabled_cv=True,
                                           enabled_baseline_precision=False)) is True
    assert _passes(_baseline(**args), _cfg(enabled_cv=False,
                                           enabled_baseline_precision=True)) is False


# ── Reporting and degenerate input ──────────────────────────────────────

def test_reason_names_the_statistic_and_its_operands():
    bl = _baseline(fpdc=500.0, sd=165.0, n=27)
    report = {}
    apply_inclusion_criteria([bl], verbose=False,
                             cfg=_cfg(enabled_baseline_precision=True),
                             report_out=report)
    (info,) = report['excluded_groups'].values()
    assert info['criterion'] == 'baseline_precision'
    assert 'rSEM' in info['reason']
    assert 'n=27' in info['reason']


@pytest.mark.parametrize("field", ['fpdc_ms_std', 'fpd_ms_n', 'fpdc_ms_mean'])
def test_missing_inputs_do_not_trip_the_criterion(field):
    bl = _baseline()
    bl['summary'][field] = np.nan
    assert _passes(bl, _cfg(enabled_baseline_precision=True)) is True


def test_single_beat_does_not_divide_by_zero():
    bl = _baseline(fpdc=500.0, sd=0.0, n=1)
    assert _passes(bl, _cfg(enabled_baseline_precision=True)) is True
