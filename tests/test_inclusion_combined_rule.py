"""Tests for the combined inclusion rule and the plausibility guardrails.

Background
----------
Criterion 1 was a single gate on CV(BP) at 25%. Measured on the 36
calibration baselines, CV correlates with quality (QC-A median 7.4%,
QC-C median 56.4%) — so the threshold is not absurd — but it is a poor
discriminator in both directions:

  * it excludes QC-B/C baselines with confidence 0.78-0.88 whose only
    sin is irregular spontaneous beating, which is normal in hiPSC-CM;
  * a low CV is evidence of *regularity*, not quality — periodic noise
    scores a better CV than a real preparation.

Two opt-in additions:

  * ``enabled_combined_rule`` — QC grade AND a much wider CV bound,
    superseding (not stacking with) the CV-only gate;
  * ``enabled_plausibility`` — BPM range and %-beats-without-repolarization,
    catching what CV structurally cannot.

Both default OFF: changing which recordings enter an analysis must be an
explicit decision.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import InclusionConfig
from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria


class _QC:
    def __init__(self, grade):
        self.grade = grade


def _baseline(*, cv=5.0, qc='A', conf=0.90, fpdc=500.0, bpm=40.0,
              pct_no_repol=0.0, filename='bl.csv'):
    return {
        'metadata': {'filename': filename},
        'file_info': {'drug': 'baseline', 'experiment': 'E1', 'chip': 'c1',
                      'chamber': 'ch1', 'electrode': 'el1',
                      'concentration': ''},
        'summary': {
            'beat_period_ms_cv': cv,
            'fpd_confidence': conf,
            'fpdc_ms_mean': fpdc,
            'bpm_mean': bpm,
            'pct_beats_no_repol': pct_no_repol,
        },
        'qc_report': _QC(qc),
    }


def _cfg(**over):
    """Config with the FPDc gates off, to isolate criterion 0/1."""
    cfg = InclusionConfig()
    cfg.enabled_fpdc_range = False
    cfg.enabled_fpdc_physiol = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _passes(result, cfg):
    apply_inclusion_criteria([result], verbose=False, cfg=cfg)
    return result['inclusion']['passed']


# ── Defaults must not change ────────────────────────────────────────────

def test_both_additions_are_off_by_default():
    cfg = InclusionConfig()
    assert cfg.enabled_combined_rule is False
    assert cfg.enabled_plausibility is False


def test_default_behaviour_is_still_cv_only():
    """CV 30% fails under the default rule, as before."""
    assert _passes(_baseline(cv=20.0), _cfg()) is True
    assert _passes(_baseline(cv=30.0), _cfg()) is False


# ── Combined rule ───────────────────────────────────────────────────────

def test_combined_rule_admits_irregular_but_clean_baseline():
    """The false-positive class: QC-B, high confidence, irregular rhythm.

    Mirrors ChipE/chipE_ch1_baseline (CV 30.8%, QC B, conf 0.86).
    """
    bl = _baseline(cv=30.8, qc='B', conf=0.86)
    assert _passes(bl, _cfg()) is False
    assert _passes(_baseline(cv=30.8, qc='B', conf=0.86),
                   _cfg(enabled_combined_rule=True)) is True


def test_combined_rule_rejects_low_cv_but_poor_quality():
    """The mirror case: excellent CV, high confidence, QC grade D.

    Mirrors day7/chipA_ch2_baseline (CV 20.1%, QC D, conf 0.86) — the
    CV-only gate lets it through.
    """
    assert _passes(_baseline(cv=20.1, qc='D', conf=0.86), _cfg()) is True
    assert _passes(_baseline(cv=20.1, qc='D', conf=0.86),
                   _cfg(enabled_combined_rule=True)) is False


def test_combined_rule_supersedes_rather_than_stacks():
    """CV 40% must pass when the combined rule is on.

    If the two gates stacked, the CV-only 25% bound would still reject
    it and the combined rule could never admit anything it was designed
    to recover.
    """
    bl = _baseline(cv=40.0, qc='B', conf=0.80)
    assert _passes(bl, _cfg(enabled_combined_rule=True)) is True


def test_combined_rule_still_rejects_extreme_cv():
    """The wide bound is 60%, not infinity."""
    assert _passes(_baseline(cv=70.0, qc='A', conf=0.90),
                   _cfg(enabled_combined_rule=True)) is False


@pytest.mark.parametrize("grade, expected", [
    ('A', True), ('B', True), ('C', True), ('D', False), ('F', False),
])
def test_combined_rule_qc_grade_boundary(grade, expected):
    assert _passes(_baseline(cv=10.0, qc=grade, conf=0.90),
                   _cfg(enabled_combined_rule=True)) is expected


def test_combined_rule_reports_which_gate_fired():
    bl = _baseline(cv=10.0, qc='F', conf=0.90)
    report = {}
    apply_inclusion_criteria([bl], verbose=False,
                             cfg=_cfg(enabled_combined_rule=True),
                             report_out=report)
    (info,) = report['excluded_groups'].values()
    assert info['criterion'] == 'qc_grade'
    assert 'QC grade=F' in info['reason']


def test_confidence_gate_still_applies_under_combined_rule():
    """The combined rule replaces criterion 1 only, not the whole chain."""
    assert _passes(_baseline(cv=10.0, qc='A', conf=0.50),
                   _cfg(enabled_combined_rule=True)) is False


# ── Plausibility guardrails ─────────────────────────────────────────────

@pytest.mark.parametrize("bpm, expected", [
    (5.0, False),     # dying preparation
    (10.0, True),     # lower bound is inclusive
    (40.0, True),
    (120.0, True),    # upper bound is inclusive
    (150.0, False),   # implausible for spontaneous hiPSC-CM
])
def test_bpm_guardrail(bpm, expected):
    assert _passes(_baseline(bpm=bpm),
                   _cfg(enabled_plausibility=True)) is expected


@pytest.mark.parametrize("pct, expected", [
    (0.0, True), (33.0, True), (50.0, True), (68.0, False),
])
def test_no_repol_guardrail(pct, expected):
    assert _passes(_baseline(pct_no_repol=pct),
                   _cfg(enabled_plausibility=True)) is expected


def test_guardrail_catches_what_cv_cannot():
    """Periodic noise: good CV, but implausible rate and no repolarization.

    Mirrors chip1_ch3_baseline_nosignal (CV 24.8%, BPM 106, 33% no-repol).
    Here the rate is pushed past the bound to isolate the guardrail —
    in the real file it is the confidence gate that catches it.
    """
    noise = _baseline(cv=24.8, qc='C', conf=0.90, bpm=140.0,
                      pct_no_repol=33.0)
    assert _passes(noise, _cfg()) is True, (
        "precondition: the CV gate alone does not stop it"
    )
    assert _passes(_baseline(cv=24.8, qc='C', conf=0.90, bpm=140.0,
                             pct_no_repol=33.0),
                   _cfg(enabled_plausibility=True)) is False


def test_guardrail_runs_before_cv_and_wins():
    """A plausibility failure must be reported as such, not as a CV failure."""
    report = {}
    apply_inclusion_criteria([_baseline(cv=99.0, bpm=5.0)], verbose=False,
                             cfg=_cfg(enabled_plausibility=True),
                             report_out=report)
    (info,) = report['excluded_groups'].values()
    assert info['criterion'] == 'bpm_plausible'


def test_nan_values_do_not_trip_the_guardrails():
    """Missing metrics must not be treated as failures."""
    bl = _baseline()
    bl['summary']['bpm_mean'] = np.nan
    bl['summary']['pct_beats_no_repol'] = np.nan
    assert _passes(bl, _cfg(enabled_plausibility=True)) is True
