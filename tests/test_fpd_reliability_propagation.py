"""Regression tests for the FPD reliability gate reaching normalization.

Background
----------
``parameters.apply_fpd_reliability_gate`` sets ``summary['fpd_reliable']``
to False when the repolarization wave was measurable on fewer than
``min_valid_fpd_ratio`` of the beats.  The flag exists because of an
observed case: 1 of 7 beats valid, reported as "FPDcF 318.8 ± 0.0".

The flag was computed and then read by nobody on the active code path —
only ``ui/single_file.py`` (the retired Streamlit UI) consulted it.
``compute_normalized_parameters`` and ``classify_drug`` never did, so such
a recording still produced a %ΔFPDcF and could flip a drug call.

Contract pinned here:

  * the flag is ALWAYS propagated into the normalization dict, as the
    per-side values and their AND — visibility is not opt-in;
  * excluding unreliable recordings from drug classification IS opt-in
    (``norm_require_fpd_reliable``), defaulting off, so no existing
    analysis changes silently;
  * a %ΔFPDcF is only as reliable as the weaker of its two operands.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.normalization import (
    classify_drug,
    compute_normalized_parameters,
)


def _make_result(*, filename, fpdc_mean, reliable, drug='', conc='',
                 is_baseline=False):
    """Build a minimal pipeline-result dict for normalization."""
    return {
        'metadata': {'filename': filename},
        'file_info': {
            'drug': 'baseline' if is_baseline else drug,
            'concentration': conc,
            'experiment': 'Exp1',
            'chip': 'chipA',
            'chamber': 'ch1',
        },
        'summary': {
            'beat_period_ms_mean': 1000.0,
            'fpdc_ms_mean': fpdc_mean,
            'spike_amplitude_mV_mean': 1.0,
            'fpd_reliable': reliable,
        },
        'inclusion': {'passed': True},
        'beat_periods': np.array([1.0, 1.0, 1.0]),
    }


# ── Propagation is unconditional ────────────────────────────────────────

@pytest.mark.parametrize(
    "bl_reliable, dr_reliable, expected_pair",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_pair_reliability_is_the_and_of_both_sides(
        bl_reliable, dr_reliable, expected_pair):
    """%ΔFPDcF is only as reliable as its weaker operand."""
    baseline = _make_result(filename='bl.csv', fpdc_mean=300.0,
                            reliable=bl_reliable, is_baseline=True)
    drug = _make_result(filename='drug.csv', fpdc_mean=345.0,
                        reliable=dr_reliable, drug='ti12', conc='B')

    norm = compute_normalized_parameters(drug, baseline)

    assert norm['baseline_fpd_reliable'] is bl_reliable
    assert norm['drug_fpd_reliable'] is dr_reliable
    assert norm['fpd_reliable'] is expected_pair


def test_flag_present_even_without_baseline():
    """The keys must always exist so consumers can read them blindly."""
    drug = _make_result(filename='drug.csv', fpdc_mean=345.0,
                        reliable=True, drug='ti12', conc='B')

    norm = compute_normalized_parameters(drug, None)

    assert norm['has_baseline'] is False
    for key in ('baseline_fpd_reliable', 'drug_fpd_reliable', 'fpd_reliable'):
        assert key in norm, f"missing key {key!r}"


def test_missing_flag_defaults_to_reliable():
    """Results produced before the gate existed must stay usable."""
    baseline = _make_result(filename='bl.csv', fpdc_mean=300.0, reliable=True,
                            is_baseline=True)
    drug = _make_result(filename='drug.csv', fpdc_mean=345.0, reliable=True,
                        drug='ti12', conc='B')
    # Simulate an old result: no 'fpd_reliable' key at all.
    del baseline['summary']['fpd_reliable']
    del drug['summary']['fpd_reliable']

    norm = compute_normalized_parameters(drug, baseline)

    assert norm['fpd_reliable'] is True


def test_unreliable_pair_still_computes_the_percentage():
    """Propagation must not silently change the arithmetic.

    Visibility and exclusion are separate concerns: the number is still
    produced, it is just now flagged.
    """
    baseline = _make_result(filename='bl.csv', fpdc_mean=300.0,
                            reliable=False, is_baseline=True)
    drug = _make_result(filename='drug.csv', fpdc_mean=345.0,
                        reliable=True, drug='ti12', conc='B')

    norm = compute_normalized_parameters(drug, baseline)

    assert norm['fpd_reliable'] is False
    assert norm['pct_fpdc_change'] == pytest.approx(15.0)


def test_unreliable_pair_logs_a_warning(caplog):
    baseline = _make_result(filename='bl.csv', fpdc_mean=300.0,
                            reliable=False, is_baseline=True)
    drug = _make_result(filename='drug.csv', fpdc_mean=345.0,
                        reliable=True, drug='ti12', conc='B')

    with caplog.at_level('WARNING'):
        compute_normalized_parameters(drug, baseline)

    assert any('unreliable' in r.getMessage().lower()
               for r in caplog.records), "expected a warning"


# ── Exclusion is opt-in ─────────────────────────────────────────────────

def _classified_concentrations(results, cfg):
    """Return the concentrations that survived filtering for drug 'ti12'.

    ``classify_drug`` returns ``{drug: {..., 'concentrations': [(conc, pct),
    ...]}}``.
    """
    entry = classify_drug(results, cfg=cfg).get('ti12')
    if entry is None:
        return []
    return [c for c, _pct in entry.get('concentrations', [])]


def _build_mixed_results():
    """One reliable drug point, one unreliable, sharing a baseline."""
    baseline = _make_result(filename='bl.csv', fpdc_mean=300.0,
                            reliable=True, is_baseline=True)
    good = _make_result(filename='good.csv', fpdc_mean=312.0, reliable=True,
                        drug='ti12', conc='A')
    bad = _make_result(filename='bad.csv', fpdc_mean=400.0, reliable=False,
                       drug='ti12', conc='B')

    for r in (good, bad):
        r['normalization'] = compute_normalized_parameters(r, baseline)
    baseline['normalization'] = {'has_baseline': False}

    return [baseline, good, bad]


def test_exclusion_is_off_by_default():
    """Default config must not change any existing analysis."""
    cfg = AnalysisConfig()
    assert cfg.normalization.norm_require_fpd_reliable is False


def test_unreliable_recording_is_kept_when_filter_off():
    results = _build_mixed_results()
    cfg = AnalysisConfig().normalization
    cfg.norm_require_fpd_reliable = False

    concs = _classified_concentrations(results, cfg)
    assert 'B' in concs, "unreliable point should be kept with filter off"


def test_unreliable_recording_is_dropped_when_filter_on():
    results = _build_mixed_results()
    cfg = AnalysisConfig().normalization
    cfg.norm_require_fpd_reliable = True

    concs = _classified_concentrations(results, cfg)
    assert 'B' not in concs, "unreliable point should be dropped with filter on"
    assert 'A' in concs, "reliable point must survive the filter"


def test_unreliable_recording_can_flip_the_drug_call():
    """Why this gate matters, stated as an assertion.

    The fixture is built so the unreliable point (B, +33%) is above the
    15% classification threshold while the reliable one (A, +4%) is well
    below it.  Because ``classification_method`` defaults to ``'max'``, a
    single unreliable recording decides the whole drug call.

    This is the ``max``-of-N behaviour flagged separately; here it is used
    to show the concrete cost of ignoring ``fpd_reliable``.
    """
    results = _build_mixed_results()
    cfg = AnalysisConfig().normalization

    cfg.norm_require_fpd_reliable = False
    off = classify_drug(results, cfg=cfg)['ti12']

    cfg.norm_require_fpd_reliable = True
    on = classify_drug(results, cfg=cfg)['ti12']

    assert off['positive'] is True, (
        "fixture precondition: the unreliable point should drive a positive "
        "call when it is not filtered"
    )
    assert on['positive'] is False, (
        "filtering the unreliable point should remove the positive call"
    )
    assert off['max_pct_change'] > on['max_pct_change']
