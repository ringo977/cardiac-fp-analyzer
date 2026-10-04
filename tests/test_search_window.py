"""Repolarisation search window (v3.15): up to 85 % of the cycle on a
regular rhythm, bounded by where the next beat can start; the per-beat
window is bounded by the beat's own next depolarisation."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from cardiac_fp_analyzer.analyze import analyze_single_file
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.repolarization import find_repolarization_per_beat, search_window_end_ms
from tests.golden_signals import generate_regular_fp

FS = 2000.0


def test_window_end_rule():
    rc = AnalysisConfig().repolarization
    assert rc.search_end_pct_rr == 0.85 and rc.search_end_pct_rr_safe == 0.70
    assert search_window_end_ms(rc) == 900.0                       # no rhythm: fixed end
    assert search_window_end_ms(rc, 1.0) == 900.0                  # 85 % of 1 s < 900 ms
    assert search_window_end_ms(rc, 1.4) == pytest.approx(1190.0)  # regular: 85 % of the cycle
    assert search_window_end_ms(rc, 1.4, rr_low_s=1.38) == pytest.approx(1190.0)   # next beat far enough
    assert search_window_end_ms(rc, 2.0, rr_low_s=1.2) == pytest.approx(1400.0)    # irregular: back to 70 %
    assert search_window_end_ms(rc, 2.0, rr_low_s=1.6) == pytest.approx(1540.0)    # between: RR_low − 60 ms
    off = dataclasses.replace(rc, search_end_pct_rr=0.0)
    assert search_window_end_ms(off, 3.0, rr_low_s=2.0) == 900.0   # extension disabled


def _df(sig):
    return pd.DataFrame({'time': np.arange(len(sig)) / FS, 'el1': sig})


def _run(sig, pct=None):
    cfg = AnalysisConfig()
    cfg.amplifier_gain = 1.0
    if pct is not None:
        cfg.repolarization = dataclasses.replace(cfg.repolarization, search_end_pct_rr=pct)
    return analyze_single_file('x.csv', channel='el1', verbose=False, config=cfg,
                               preloaded=({'sample_rate': FS, 'format': 'csv'}, _df(sig)))['summary']


def test_late_t_wave_on_a_slow_regular_tissue_is_measured():
    """FPD at 71 % of a 1.4 s cycle (a strong hERG blocker on a slow
    µHeart tissue): out of reach of the 70 % window, inside the 85 % one."""
    sig, _t, _m = generate_regular_fp(fs=FS, duration_s=60, beat_period_ms=1400, fpd_ms=1000,
                                      depol_amp=60e-6, repol_amp=10e-6, noise_std=0.5e-6, seed=7)
    new = _run(sig)
    assert new['beat_period_ms_median'] == pytest.approx(1400, abs=3)
    assert new['fpd_ms_median'] == pytest.approx(1000, abs=25) and new['fpd_reliable']
    old = _run(sig, pct=0.70)
    assert not (abs(old['fpd_ms_median'] - 1000) <= 25 and old['fpd_reliable'])


def test_regular_tissue_with_a_normal_t_wave_is_unchanged():
    sig, _t, _m = generate_regular_fp(fs=FS, duration_s=40, beat_period_ms=1400, fpd_ms=600,
                                      depol_amp=60e-6, repol_amp=10e-6, noise_std=0.5e-6, seed=8)
    new, old = _run(sig), _run(sig, pct=0.70)
    assert new['fpd_ms_median'] == pytest.approx(600, abs=15)
    assert new['fpd_ms_median'] == pytest.approx(old['fpd_ms_median'], abs=3)


def test_per_beat_window_stops_before_the_beats_own_next_spike():
    """A beat whose next depolarisation falls inside the template-guided
    window: without the bound the spike (far more prominent than the
    T-wave) wins; with it the T-wave is measured."""
    rc = AnalysisConfig().repolarization
    n = int(2.5 * FS)
    t_ms = np.arange(n) / FS * 1000 - 50
    x = np.zeros(n)
    x += 60e-6 * np.exp(-0.5 * (t_ms / 1.0) ** 2)                          # spike at 0
    x += 8e-6 * np.exp(-0.5 * ((t_ms - 900) / 20) ** 2)                    # T-wave at 900 ms
    # next beat at 1150 ms: a 3D-tissue spike is wide (tens of ms), so it
    # survives the 20 Hz low-pass of the repolarisation search
    x += 60e-6 * np.exp(-0.5 * ((t_ms - 1150) / 15.0) ** 2)
    x -= 36e-6 * np.exp(-0.5 * ((t_ms - 1180) / 22.0) ** 2)
    x += np.random.default_rng(0).normal(0, 0.3e-6, n)
    spike = int(0.05 * FS)
    peak = int(0.9 * FS)                                                   # template says ~900 ms
    unbounded = find_repolarization_per_beat(x, t_ms / 1000, spike, FS, template_fpd_samples=peak,
                                             template_peak_samples=peak, template_repol_sign=1, cfg=rc,
                                             beat_period_s=1.4)
    bounded = find_repolarization_per_beat(x, t_ms / 1000, spike, FS, template_fpd_samples=peak,
                                           template_peak_samples=peak, template_repol_sign=1, cfg=rc,
                                           beat_period_s=1.4, next_rr_s=1.15)
    assert bounded[0] is not None and bounded[0] * 1000 == pytest.approx(900, abs=15)
    assert unbounded[0] is None or abs(unbounded[0] * 1000 - 900) > 100
