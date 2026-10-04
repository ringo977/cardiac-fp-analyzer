"""Fixes of the parameter-sheet audit (v3.14.1): rhythm filter matches
beats by sample, retry and channel selection honour the configuration, QC
jitter is two-sided, pairing respects inclusion under both rules, the
batch's second arrhythmia pass keeps the first pass's verdict, per-beat
'consensus' no longer falls through to 'tangent'."""

import dataclasses

import numpy as np
import pytest

from cardiac_fp_analyzer import beat_detection as BD
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.normalization import pair_with_baselines, recording_key
from cardiac_fp_analyzer.quality_control import morphology_correlation
from cardiac_fp_analyzer.rhythm_integration import apply_rhythm_filter
from tests.golden_signals import generate_regular_fp
from tests.test_drug_call_and_reference import _res as _timed_res

# ── rhythm filter: cluster membership by sample index ──────────────────

def test_rhythm_filter_keeps_the_right_beats_after_the_beat_set_changed():
    # classification on 6 beats: dominant = even positions; later the first
    # beat was dropped (noise gate), so positional indices are off by one
    bi_at_classification = np.array([100, 200, 300, 400, 500, 600])
    dominant_samples = [100, 300, 500]
    rc = {'rhythm_type': 'alternans_2_to_1', 'n_beats': 6,
          'clusters': [{'role': 'dominant', 'indices_in_bi': [0, 2, 4], 'n': 3,
                        'sample_indices': dominant_samples},
                       {'role': 'secondary', 'indices_in_bi': [1, 3, 5], 'n': 3,
                        'sample_indices': [200, 400, 600]}]}
    bi_now = bi_at_classification[1:]                    # 200..600
    bd = np.ones((len(bi_now), 10)); btm = np.ones((len(bi_now), 10))
    _, _, bi_f, info = apply_rhythm_filter(bd, btm, bi_now, bi_now, rc, min_retention_ratio=0.0,
                                           min_retention_beats=1)
    assert info['filter_applied'] and sorted(bi_f.tolist()) == [300, 500]
    # old classification without sample indices and a changed beat count: passthrough, not wrong beats
    rc_old = {'rhythm_type': 'alternans_2_to_1', 'n_beats': 6,
              'clusters': [{'role': 'dominant', 'indices_in_bi': [0, 2, 4], 'n': 3}]}
    _, _, bi_f2, info2 = apply_rhythm_filter(bd, btm, bi_now, bi_now, rc_old, min_retention_ratio=0.0,
                                             min_retention_beats=1)
    assert not info2['filter_applied'] and np.array_equal(bi_f2, bi_now)


def test_classifier_stores_sample_indices():
    fs = 2000.0
    sig, _t, _m = generate_regular_fp(fs=fs, duration_s=20, beat_period_ms=800, fpd_ms=300,
                                      depol_amp=60e-6, repol_amp=10e-6, noise_std=1e-6, seed=0)
    bi, _bt, det = BD.detect_beats(sig, fs, cfg=AnalysisConfig().beat_detection)
    rc = det['rhythm_classification']
    assert rc['clusters'], rc['rhythm_type']
    for cl in rc['clusters']:
        assert 'sample_indices' in cl and len(cl['sample_indices']) == cl['n']
        assert set(cl['sample_indices']) <= set(int(v) for v in bi) or rc['n_beats'] != len(bi)


# ── retry and channel selection pass the configuration ────────────────

def test_retry_keeps_the_configuration(monkeypatch):
    from cardiac_fp_analyzer import analyze as A
    seen = []
    real = A.detect_beats

    def spy(data, fs, *args, **kw):
        seen.append(kw.get('cfg'))
        bi, bt, det = real(data, fs, *args, **kw)
        if len(seen) == 1:                               # force the retry
            det = dict(det); det['n_beats'] = 1
            return bi[:1], bt[:1], det
        return bi, bt, det
    monkeypatch.setattr(A, 'detect_beats', spy)
    fs = 2000.0
    sig, _t, _m = generate_regular_fp(fs=fs, duration_s=20, beat_period_ms=800, fpd_ms=300,
                                      depol_amp=60e-6, repol_amp=10e-6, noise_std=1e-6, seed=1)
    import pandas as pd
    df = pd.DataFrame({'time': np.arange(len(sig)) / fs, 'el1': sig, 'el2': sig * 0.5})
    cfg = AnalysisConfig(); cfg.amplifier_gain = 1.0
    cfg.beat_detection.enable_morphology_validation = False
    r = A.analyze_single_file('x.csv', channel='el1', verbose=False, config=cfg,
                              preloaded=({'sample_rate': fs, 'format': 'csv'}, df))
    assert len(seen) >= 2 and seen[1] is not None
    assert seen[1].enable_morphology_validation is False          # user's setting kept
    assert seen[1].min_distance_ms == cfg.beat_detection.retry_min_distance_ms
    assert seen[1].threshold_factor == cfg.beat_detection.retry_threshold_factor
    assert r['detection_info'].get('retry') is True


def test_channel_selection_passes_the_configuration(monkeypatch):
    from cardiac_fp_analyzer import channel_selection as CS
    seen = []
    real = CS.detect_beats

    def spy(*args, **kw):
        seen.append(kw.get('cfg'))
        return real(*args, **kw)
    monkeypatch.setattr(CS, 'detect_beats', spy)
    fs = 2000.0
    sig, _t, _m = generate_regular_fp(fs=fs, duration_s=15, beat_period_ms=800, fpd_ms=300,
                                      depol_amp=60e-6, repol_amp=10e-6, noise_std=1e-6, seed=2)
    import pandas as pd
    df = pd.DataFrame({'time': np.arange(len(sig)) / fs, 'el1': sig, 'el2': sig * 0.3})
    cfg = AnalysisConfig(); cfg.amplifier_gain = 1.0
    cfg.beat_detection.min_distance_ms = 350.0
    CS.select_best_channel(df, fs, cfg=cfg)
    assert seen and all(c is cfg.beat_detection for c in seen)


# ── QC morphology correlation: both shift directions ───────────────────

def test_morphology_correlation_is_symmetric_in_the_shift():
    fs = 2000.0
    n = 400
    t = np.arange(n) / fs
    template = np.exp(-((t - 0.05) / 0.004) ** 2)             # spike at 50 ms
    early = np.roll(template, -6)                             # beat 6 samples early
    late = np.roll(template, 6)
    c_early = morphology_correlation(early, template, max_samples=300, jitter_max=8)
    c_late = morphology_correlation(late, template, max_samples=300, jitter_max=8)
    assert c_early > 0.99 and c_late > 0.99
    assert c_early == pytest.approx(c_late, abs=1e-3)
    # without jitter both are poor, and equally so
    assert morphology_correlation(early, template, max_samples=300, jitter_max=0) < 0.9


# ── pairing respects inclusion in both rules ───────────────────────────

def test_timed_rule_skips_a_baseline_that_failed_inclusion():
    bad = _timed_res('D/Exp5/Day7/chipB_ch2_baseline.csv', when='10:30:00', passed=False, fpdc=300.0)
    good = _timed_res('D/Exp5/Day7/chipB_ch2_t0.csv', when='10:00:00', fpdc=500.0)
    dose = _timed_res('D/Exp5/Day7/chipB_ch2_terfe_10nM.csv', when='11:00:00', fpdc=550.0)
    details = {}
    m = pair_with_baselines([bad, good, dose], details=details)
    assert m[recording_key(dose)] is good            # the later baseline failed inclusion
    only_bad = _timed_res('D/Exp6/Day7/chipB_ch2_baseline.csv', when='10:30:00', passed=False)
    dose6 = _timed_res('D/Exp6/Day7/chipB_ch2_terfe_10nM.csv', when='11:00:00')
    m = pair_with_baselines([only_bad, dose6], details=details)
    assert m[recording_key(dose6)] is None
    assert 'failed inclusion' in details[recording_key(dose6)]['reason']


def test_untimed_rule_picks_another_baseline_when_the_best_failed_inclusion():
    bad = _timed_res('D/Exp5/Day7/chipC_ch1_baseline.csv', grade='A', passed=False, fpdc=300.0)
    good = _timed_res('D/Exp5/Day7/chipC_ch1_t0.csv', grade='C', fpdc=500.0)
    good['file_info']['reference_kind'] = 'baseline'   # no t0 preference: grade would pick 'bad'
    dose = _timed_res('D/Exp5/Day7/chipC_ch1_terfe_10nM.csv', fpdc=550.0)
    m = pair_with_baselines([bad, good, dose])
    assert m[recording_key(dose)] is good


# ── per-beat consensus is consensus ────────────────────────────────────

def test_per_beat_consensus_matches_the_template_method():
    import pandas as pd

    from cardiac_fp_analyzer.analyze import analyze_single_file
    fs = 2000.0
    sig, _t, _m = generate_regular_fp(fs=fs, duration_s=30, beat_period_ms=900, fpd_ms=350,
                                      depol_amp=60e-6, repol_amp=12e-6, noise_std=0.5e-6, seed=3)
    df = pd.DataFrame({'time': np.arange(len(sig)) / fs, 'el1': sig})
    out = {}
    for method in ('consensus', 'tangent', 'peak'):
        cfg = AnalysisConfig(); cfg.amplifier_gain = 1.0
        cfg.repolarization = dataclasses.replace(cfg.repolarization, fpd_method=method)
        r = analyze_single_file('x.csv', channel='el1', verbose=False, config=cfg,
                                preloaded=({'sample_rate': fs, 'format': 'csv'}, df))
        out[method] = (r['summary']['template_fpd_ms'], r['summary']['fpd_ms_median'])
    # per-beat FPD follows the template's method within a few ms
    for method, (tmpl, med) in out.items():
        assert np.isfinite(tmpl) and np.isfinite(med), method
        assert abs(tmpl - med) < 15, (method, tmpl, med)
