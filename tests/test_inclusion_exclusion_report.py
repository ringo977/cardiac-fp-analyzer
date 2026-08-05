"""Tests for the removed-groups provenance in apply_inclusion_criteria.

Background
----------
When a baseline fails an inclusion criterion, its entire dose-response
group is removed — every concentration of that drug on that chip and
channel. The re-analysis of the calibration dataset found the whole
dofetilide dose-response in EXP 8/day6 gone this way: the baseline
failed on CV=31.8% >= 25.0%, taking 7 concentrations with it.

Dofetilide is the canonical CiPA positive control, and nothing in the
output said so. That failure mode produces *absence* of numbers rather
than wrong ones, which is much harder to notice.

These tests pin the provenance contract: which group, which baseline,
which criterion, which measured value, and which drug recordings were
lost with it.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import InclusionConfig
from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria


class _QC:
    def __init__(self, grade):
        self.grade = grade


def _result(filename, *, drug, cv=5.0, conf=0.90, fpdc=500.0, qc='A',
            fpd_ms=400.0, bp_ms=1000.0,
            chip='chipD', chamber='ch1', experiment='EXP8'):
    """Minimal result dict shaped like analyze_single_file output.

    Carries a QC grade and the FPD/beat-period pair because the default
    inclusion chain reads both (combined QC+CV rule, FPD/RR ratio).
    """
    return {
        'metadata': {'filename': filename},
        'file_info': {
            'drug': drug,
            'concentration': '',
            'experiment': experiment,
            'chip': chip,
            'chamber': chamber,
            'electrode': 'el1',
        },
        'summary': {
            'beat_period_ms_cv': cv,
            'fpd_confidence': conf,
            'fpdc_ms_mean': fpdc,
            'beat_period_ms_mean': bp_ms,
            'beat_period_ms_median': bp_ms,
            'fpd_ms_mean': fpd_ms,
            'fpd_ms_median': fpd_ms,
            'bpm_mean': 60000.0 / bp_ms if bp_ms else np.nan,
            'pct_beats_no_repol': 0.0,
        },
        'qc_report': _QC(qc),
        'beat_periods': np.array([1.0, 1.0, 1.0]),
    }


def _dofetilide_group(baseline_cv):
    """A baseline plus 7 concentrations, mirroring EXP 8/day6."""
    results = [_result('chipD_ch1_baseline.csv', drug='baseline',
                       cv=baseline_cv)]
    for c in ['0.3nM', '1nM', '2nM', '3nM', '6nM', '10nM', '30nM']:
        results.append(_result(f'chipD_ch1_Dofe_{c}.csv', drug='dofe'))
    return results


# ── Provenance content ──────────────────────────────────────────────────

def test_failing_baseline_records_full_provenance():
    results = _dofetilide_group(baseline_cv=31.8)
    cfg = InclusionConfig()
    report = {}

    apply_inclusion_criteria(results, verbose=False, cfg=cfg,
                             report_out=report)

    assert report['n_groups_removed'] == 1
    assert report['n_baselines_failed'] == 1
    assert report['n_drug_recordings_excluded'] == 7

    (info,) = report['excluded_groups'].values()
    assert info['baseline_file'] == 'chipD_ch1_baseline.csv'
    assert info['criterion'] == 'cv_bp'
    assert info['cv_bp'] == pytest.approx(31.8)
    assert '31.8' in info['reason']
    assert info['n_drug_recordings_lost'] == 7
    assert len(info['drug_recordings_lost']) == 7
    assert 'chipD_ch1_Dofe_0.3nM.csv' in info['drug_recordings_lost']


def test_healthy_baseline_produces_empty_report():
    results = _dofetilide_group(baseline_cv=5.0)
    report = {}

    apply_inclusion_criteria(results, verbose=False, cfg=InclusionConfig(),
                             report_out=report)

    assert report['n_groups_removed'] == 0
    assert report['n_drug_recordings_excluded'] == 0
    assert report['excluded_groups'] == {}
    assert all(r['inclusion']['passed'] for r in results)


def test_drug_recordings_carry_the_root_cause():
    """A single excluded recording must be self-explanatory.

    'Baseline of group X failed inclusion' alone forces the reader to go
    hunting for the baseline; the root cause travels with the record.
    """
    results = _dofetilide_group(baseline_cv=31.8)

    apply_inclusion_criteria(results, verbose=False, cfg=InclusionConfig())

    drug = results[1]
    assert drug['inclusion']['passed'] is False
    assert drug['inclusion']['criterion'] == 'group_removed'
    assert 'excluded_group' in drug['inclusion']
    assert '31.8' in drug['inclusion']['root_cause']


@pytest.mark.parametrize(
    "kwargs, expected_criterion",
    [
        (dict(cv=99.0), 'cv_bp'),
        (dict(fpdc=5.0), 'fpdc_range'),
        (dict(conf=0.10), 'fpd_confidence'),
    ],
)
def test_criterion_label_identifies_which_gate_fired(kwargs, expected_criterion):
    """The label must be machine-readable, not parsed out of prose."""
    results = _dofetilide_group(baseline_cv=5.0)
    results[0]['summary'].update({
        'beat_period_ms_cv': kwargs.get('cv', 5.0),
        'fpd_confidence': kwargs.get('conf', 0.90),
        'fpdc_ms_mean': kwargs.get('fpdc', 500.0),
    })
    report = {}

    apply_inclusion_criteria(results, verbose=False, cfg=InclusionConfig(),
                             report_out=report)

    (info,) = report['excluded_groups'].values()
    assert info['criterion'] == expected_criterion


def test_report_out_is_cleared_before_filling():
    """Reusing a dict across batches must not accumulate stale entries."""
    report = {'stale': 'value', 'n_groups_removed': 999}

    apply_inclusion_criteria(_dofetilide_group(baseline_cv=5.0),
                             verbose=False, cfg=InclusionConfig(),
                             report_out=report)

    assert 'stale' not in report
    assert report['n_groups_removed'] == 0


def test_report_out_is_optional():
    """Existing callers pass no report_out and must keep working."""
    results = _dofetilide_group(baseline_cv=31.8)
    out = apply_inclusion_criteria(results, verbose=False,
                                   cfg=InclusionConfig())
    assert out is results
    assert results[0]['inclusion']['passed'] is False


# ── Multiple groups ─────────────────────────────────────────────────────

def test_only_the_failing_group_is_removed():
    """A bad baseline must not take down an unrelated chamber."""
    bad = _dofetilide_group(baseline_cv=31.8)
    good = [
        _result('chipD_ch2_baseline.csv', drug='baseline', cv=5.0,
                chamber='ch2'),
        _result('chipD_ch2_Quin_1uM.csv', drug='quin', chamber='ch2'),
    ]
    report = {}

    apply_inclusion_criteria(bad + good, verbose=False,
                             cfg=InclusionConfig(), report_out=report)

    assert report['n_groups_removed'] == 1
    assert report['n_drug_recordings_excluded'] == 7
    assert all(r['inclusion']['passed'] for r in good)


def test_verbose_block_names_the_group_and_the_loss(capsys):
    """The printed block is the thing a user actually sees."""
    apply_inclusion_criteria(_dofetilide_group(baseline_cv=31.8),
                             verbose=True, cfg=InclusionConfig())

    out = capsys.readouterr().out
    assert 'GRUPPI RIMOSSI' in out
    assert 'chipD_ch1_baseline.csv' in out
    assert '31.8' in out
    assert '7 registrazioni' in out
