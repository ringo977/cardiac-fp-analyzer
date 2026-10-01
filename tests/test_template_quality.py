"""Core template-representativity policy (shared by both UIs)."""

from cardiac_fp_analyzer.template_quality import (
    FPD_CV_TEMPLATE_WARN,
    TEMPLATE_RISKY_RHYTHM_TYPES,
    template_representativity,
)


def test_constants():
    assert FPD_CV_TEMPLATE_WARN == 0.20
    assert {'chaotic', 'ambiguous', 'alternans_2_to_1', 'trimodal'} == set(TEMPLATE_RISKY_RHYTHM_TYPES)


def test_regular_low_dispersion_is_representative():
    ok = {'detection_info': {'rhythm_classification': {'rhythm_type': 'regular'}},
          'summary': {'fpd_ms_mean': 500.0, 'fpd_ms_std': 20.0}}
    r = template_representativity(ok)
    assert r['representative'] and not r['risky_rhythm'] and not r['dispersive_fpd']
    assert abs(r['fpd_cv'] - 0.04) < 1e-9


def test_risky_rhythm_triggers():
    risky = {'detection_info': {'rhythm_classification': {'rhythm_type': 'alternans_2_to_1'}},
             'summary': {'fpd_ms_mean': 500.0, 'fpd_ms_std': 20.0}}
    r = template_representativity(risky)
    assert not r['representative'] and r['risky_rhythm'] and not r['dispersive_fpd']


def test_dispersive_fpd_triggers():
    disp = {'detection_info': {}, 'summary': {'fpd_ms_mean': 500.0, 'fpd_ms_std': 150.0}}
    r = template_representativity(disp)
    assert not r['representative'] and r['dispersive_fpd'] and not r['risky_rhythm']


def test_none_and_garbage_are_representative():
    assert template_representativity(None)['representative']
    assert template_representativity({'summary': {'fpd_ms_mean': 'n/a'}})['representative']
