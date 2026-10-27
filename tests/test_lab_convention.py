"""The operators' depolarisation convention (their MATLAB analyzer v1.5) as an option (v3.24).

``BeatDetectionConfig.dep_time_rule = 'steepest_fall'``: time = steepest downstroke of the complex (minimum of
dV/dt of the 100 Hz low-pass of the mains-notched signal, ±12 ms around the 20–300 Hz |peak|), amplitude =
highest − lowest point of the complex on the raw signal with the lab's windows. On the Exp1 baseline it
reproduces the operators' times to the sample in 98–99 % of the beats (test below, run where the data is).
"""
from __future__ import annotations

import contextlib
import glob
import io
from pathlib import Path

import numpy as np
import pytest

from cardiac_fp_analyzer import lab_convention as L
from cardiac_fp_analyzer.config import AnalysisConfig

FS = 2000.0


def _complex_train(n_s=20.0, period_s=1.5, overshoot=0.8, pre=0.6, seed=0, negative_first=False):
    """Beats with a leading peak, a steep fall to a trough and a slower overshoot (positive-first), or the
    mirror image of the leading part (negative-first). Returns (x, true downstroke samples)."""
    rng = np.random.default_rng(seed)
    n = int(n_s * FS)
    t = np.arange(n) / FS
    x = 0.005 * rng.standard_normal(n)
    beats = np.arange(int(1.0 * FS), n - int(1.0 * FS), int(period_s * FS))
    for b in beats:
        tb = t - t[b]
        if negative_first:
            x += -1.0 * np.exp(-((tb - 0.002) / 0.0015) ** 2 / 2) + 0.9 * np.exp(-((tb + 0.004) / 0.003) ** 2 / 2) * 0
            x += overshoot * np.exp(-((tb - 0.015) / 0.004) ** 2 / 2)
        else:
            x += pre * np.exp(-((tb + 0.004) / 0.0015) ** 2 / 2) - 1.0 * np.exp(-((tb - 0.003) / 0.0015) ** 2 / 2)
            x += overshoot * np.exp(-((tb - 0.015) / 0.004) ** 2 / 2)
    return x, beats


def test_default_is_the_software_peak_and_rule_of_reads_any_config():
    cfg = AnalysisConfig()
    assert L.rule_of(cfg) == 'peak' and L.rule_of(cfg.beat_detection) == 'peak' and L.rule_of(None) == 'peak'
    cfg.beat_detection.dep_time_rule = 'steepest_fall'
    assert L.rule_of(cfg) == 'steepest_fall'
    cfg.beat_detection.dep_time_rule = 'nonsense'
    assert L.rule_of(cfg) == 'peak'


@pytest.mark.parametrize('negative_first', [False, True])
def test_steepest_fall_is_the_minimum_of_dvdt_whatever_the_order_of_the_lobes(negative_first):
    x, beats = _complex_train(negative_first=negative_first)
    tr = L.lab_traces(x, FS)
    for b in beats:
        a = L.steepest_fall(tr, b + int(0.008 * FS))            # a detection 8 ms late: same anchor
        lo, hi = b - int(0.012 * FS), b + int(0.012 * FS)
        assert a == lo + int(np.argmin(tr['dv'][lo:hi]))
        assert tr['dv'][a] < 0 and abs(a - L.steepest_fall(tr, b - int(0.008 * FS))) <= 1
    anc = L.anchors(beats + 5, x, FS)
    assert np.all(np.diff(anc) > 0) and len(anc) == len(beats)


def test_extrema_lowest_point_and_the_taller_of_leading_peak_and_overshoot():
    for overshoot, pre, expect in ((0.8, 0.4, 'post'), (0.3, 0.9, 'pre')):
        x, beats = _complex_train(overshoot=overshoot, pre=pre, seed=1)
        tr = L.lab_traces(x, FS)
        b = beats[3]
        i_dep = L.steepest_fall(tr, b)
        i_lo, i_hi = L.extrema(x, tr, i_dep)
        assert abs(i_lo - (b + int(0.003 * FS))) <= 2                 # the trough
        want = b + int(0.015 * FS) if expect == 'post' else b - int(0.004 * FS)
        assert abs(i_hi - want) <= (6 if expect == 'post' else 2)
        amp, hi_v, lo_v, _, _ = L.amplitude(x, tr, i_dep)
        assert amp == pytest.approx(x[i_hi] - x[i_lo]) and hi_v == x[i_hi] and lo_v == x[i_lo]


def test_pipeline_review_and_sidecar_follow_the_rule(tmp_path):
    from cardiac_fp_analyzer.analyze import analyze_single_file
    from cardiac_fp_analyzer.review import ReviewState
    x, beats = _complex_train(n_s=40.0, period_s=1.5, seed=2)
    t = np.arange(len(x)) / FS
    p = tmp_path / 'Exp1_chipA_ch1_baseline.csv'
    with open(p, 'w') as f:
        f.write('#Sample rate: 2000Hz\nTime (s),Channel 1 (V)\n')
        for ti, xi in zip(t, x * 1e-4 * 1e4 * 1e-4):
            f.write(f'{ti:.6f},{xi:.9e}\n')
    cfg = AnalysisConfig()
    cfg.beat_detection.dep_time_rule = 'steepest_fall'
    with contextlib.redirect_stdout(io.StringIO()):
        r = analyze_single_file(str(p), 'el1', verbose=False, config=cfg)
    bi = np.asarray(r['beat_indices_raw'])
    assert r['summary']['dep_time_rule'] == 'steepest_fall'
    assert 'steepest downstroke' in r['metadata']['processing']['testo']
    assert all(np.min(np.abs(beats - b)) <= 3 for b in bi)       # on the downstrokes (± filtering)
    assert all('dep_min_idx' in q and 'dep_max_idx' in q for q in r['all_params'])
    st = ReviewState.from_result(r, cfg)
    assert st.lab_rule and st.dep_rule == 'steepest_fall'
    d0 = st.beats[2].dep
    assert st._snap_to_spike(d0 + int(0.010 * FS)) == d0          # a click 10 ms late snaps back on the downstroke
    amp, up, down = st._amp_uV(d0)
    lo, hi = st.dep_extrema(d0)
    assert amp == pytest.approx((st.raw[hi] - st.raw[lo]) * 1e6) and up == pytest.approx(st.raw[hi] * st.gain)
    st.save(str(p))
    fresh = ReviewState.from_result(r, AnalysisConfig())          # opened again with the peak convention
    assert ReviewState.load_into(fresh, str(p))
    assert fresh.saved_dep_rule == 'steepest_fall' and fresh.dep_rule == 'peak'


EXP1 = Path(__file__).resolve().parents[1] / 'Italfarmaco' / 'Exp1'


@pytest.mark.skipif(not (EXP1 / 'Baseline_fp_results.xlsx').exists(), reason='Exp1 data not here (local only)')
def test_reproduces_the_operators_times_on_exp1():
    """Fully automatic from the software's own detections: same sample as the lab's 'Dep. time' and the same
    'Dep. Voltage up' on two tissues (A1: sharp spikes; G1: wide complex where the peak was 11 ms off)."""
    import openpyxl

    from cardiac_fp_analyzer.analyze import analyze_single_file
    wb = openpyxl.load_workbook(EXP1 / 'Baseline_fp_results.xlsx', data_only=True)
    cfg = AnalysisConfig()
    cfg.beat_detection.dep_time_rule = 'steepest_fall'
    for chip_file, ch, sheet in (('CHIPA_Ch1', 'el1', 'CHIP Exp1 A Ch1'), ('CHIPE_Ch1', 'el2', 'CHIP Exp1 G Ch1 raw')):
        rows, started = [], False
        for row in wb[sheet].iter_rows(values_only=True):
            if row[0] == 'baseline' and row[1] and str(row[1]).startswith('Dep. time'):
                started = True
                continue
            if started and row[0] in ('Mean', 'St. dev'):
                break
            if started and isinstance(row[1], (int, float)):
                rows.append((row[1], row[5]))
        path = glob.glob(str(EXP1 / 'baseline' / f'*_{chip_file}_baseline_*.csv'))[0]
        with contextlib.redirect_stdout(io.StringIO()):
            r = analyze_single_file(path, ch, verbose=False, config=cfg)
        fs = r['metadata']['sample_rate']
        raw = np.asarray(r['raw_signal'], float) * 1e4
        tr = L.lab_traces(raw, fs, cfg.beat_detection)
        bi = np.asarray(r['beat_indices_raw'])
        same, up_ok = 0, 0
        for t_dep, up in rows:
            s = int(round(t_dep * fs))
            j = int(np.argmin(np.abs(bi - s)))
            same += int(bi[j] == s)
            up_ok += int(abs(L.amplitude(raw, tr, int(bi[j]), cfg.beat_detection)[1] - up) < 1e-6)
        assert same / len(rows) >= 0.95 and up_ok / len(rows) >= 0.95, (sheet, same, up_ok, len(rows))
