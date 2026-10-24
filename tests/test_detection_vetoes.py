"""Opt-in post-detection vetoes and gap-tolerant rhythm criterion (v3.17)."""
import dataclasses

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig, BeatDetectionConfig, QualityConfig
from cardiac_fp_analyzer.detection_vetoes import (
    apply_post_detection_vetoes,
    burst_veto,
    burst_windows,
    lowpass_30,
    weak_population_split,
)
from cardiac_fp_analyzer.quality_control import assess_analysability, common_period

FS = 2000.0


def _spike(n, fs, width_ms=6.0, amp=1.0):
    # asymmetric depolarisation: sharp negative deflection, smaller and broader positive rebound
    t = np.arange(-n // 2, n // 2) / fs
    s = width_ms / 1000 / 3
    return amp * (-np.exp(-(t / s) ** 2 / 2) + 0.4 * np.exp(-((t - 2.5 * s) / (1.5 * s)) ** 2 / 2))


def _trace(dur_s=60.0, period_s=2.0, amp=1.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    x = noise * rng.standard_normal(n)
    sp = _spike(int(0.04 * FS), FS, amp=amp)
    beats = np.arange(int(0.5 * FS), n - int(0.5 * FS), int(period_s * FS))
    for b in beats:
        x[b - len(sp) // 2:b + len(sp) // 2] += sp
    return x, beats


def test_defaults_are_off():
    bd, q = BeatDetectionConfig(), QualityConfig()
    assert bd.enable_weak_population_split is False
    assert bd.enable_burst_veto is False
    assert q.not_analysable_gap_tolerant is False


def test_weak_population_split_removes_weak_bumps():
    x, beats = _trace()
    rng = np.random.default_rng(1)
    weak = beats[:-1] + int(1.0 * FS)                       # broad, small waves half-way (repolarisation-like)
    t = np.arange(-int(.15 * FS), int(.15 * FS)) / FS
    wave = 0.06 * np.exp(-(t / 0.05) ** 2 / 2)          # local SNR ~4 (< 5)
    for w in weak:
        x[w - len(wave) // 2:w + len(wave) // 2] += wave + 0.01 * rng.standard_normal(len(wave))
    det = np.sort(np.r_[beats, weak])
    kept, info = weak_population_split(x, FS, det, BeatDetectionConfig())
    assert info['applied']
    assert set(kept) == set(beats)


def test_weak_population_split_keeps_weak_group_above_snr_limit():
    x, beats = _trace()
    weak = beats[:-1] + int(1.0 * FS)
    t = np.arange(-int(.15 * FS), int(.15 * FS)) / FS
    wave = 0.12 * np.exp(-(t / 0.05) ** 2 / 2)           # local SNR ~8: could be real beats, not removed
    for w in weak:
        x[w - len(wave) // 2:w + len(wave) // 2] += wave
    det = np.sort(np.r_[beats, weak])
    kept, info = weak_population_split(x, FS, det, BeatDetectionConfig())
    assert not info['applied']
    assert np.array_equal(kept, det)


def test_weak_population_split_leaves_clean_train_alone():
    x, beats = _trace()
    kept, info = weak_population_split(x, FS, beats, BeatDetectionConfig())
    assert not info['applied']
    assert np.array_equal(kept, beats)


def test_weak_population_split_needs_six_detections():
    x, beats = _trace(dur_s=10.0)
    kept, info = weak_population_split(x, FS, beats[:5], BeatDetectionConfig())
    assert np.array_equal(kept, beats[:5])
    assert not info['applied']


def _bursty_trace(seed=2):
    x, beats = _trace(seed=seed)
    rng = np.random.default_rng(seed)
    n = len(x)
    from scipy import signal as sg
    b, a = sg.butter(4, [150 / (FS / 2), 400 / (FS / 2)], 'band')
    windows = []
    for s in range(int(3 * FS), n - int(4 * FS), int(10 * FS)):
        e = s + int(3 * FS)
        x[s:e] += 0.25 * sg.lfilter(b, a, rng.standard_normal(e - s)) / 0.1
        windows.append((s, e))
    return x, beats, windows


def test_burst_windows_found():
    x, _, windows = _bursty_trace()
    win = burst_windows(x, FS, BeatDetectionConfig())
    assert len(win) == len(windows)
    for (s, e), (s0, e0) in zip(win, windows):
        assert abs(s - s0) <= 0.2 * FS and abs(e - e0) <= 0.2 * FS


def test_burst_veto_removes_noise_detections_keeps_beats():
    x, beats, windows = _bursty_trace()
    noise_det = np.array([s + int(1.2 * FS) + 7 for s, _ in windows])     # detections on burst noise
    noise_det = np.array([d for d in noise_det if np.min(np.abs(beats - d)) > 0.3 * FS])
    det = np.sort(np.r_[beats, noise_det])
    kept, info = burst_veto(x, FS, det, BeatDetectionConfig())
    assert info['n_refs'] >= 3
    assert not set(noise_det) & set(kept)
    lost = set(beats) - set(kept)
    assert len(lost) <= 1


def test_burst_veto_nothing_in_scope_without_bursts():
    x, beats = _trace()
    kept, info = burst_veto(x, FS, beats, BeatDetectionConfig())
    assert info['status'] == 'nothing_in_scope'
    assert np.array_equal(kept, beats)


def test_burst_veto_min_kept_guard():
    x, beats, windows = _bursty_trace()
    cfg = dataclasses.replace(BeatDetectionConfig(), post_veto_min_kept=10_000)
    noise_det = np.array([s + int(1.2 * FS) + 7 for s, _ in windows])
    det = np.sort(np.r_[beats, noise_det])
    kept, info = burst_veto(x, FS, det, cfg)
    assert info['status'] == 'abstain_min_kept'
    assert np.array_equal(kept, det)


def test_lowpass_30_removes_high_frequency():
    t = np.arange(int(4 * FS)) / FS
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 300 * t)
    y = lowpass_30(x, FS)
    core = slice(int(FS), int(3 * FS))
    assert np.std(y[core] - np.sin(2 * np.pi * 5 * t[core])) < 0.05


def test_apply_vetoes_respects_flags():
    x, beats = _trace()
    kept, info = apply_post_detection_vetoes(x, FS, beats, BeatDetectionConfig())
    assert np.array_equal(kept, beats)
    assert 'weak_population_split' not in info and 'burst_veto' not in info


def test_common_period_with_missing_beats():
    P = 2.0
    rr = np.array([P, P, 2 * P, P, 3 * P, P, P, 2 * P, P]) * (1 + 0.02 * np.sin(np.arange(9)))
    per, folded, info = common_period(rr)
    assert per == pytest.approx(P, rel=0.05)
    assert info['missing'] == 1 + 2 + 1
    assert info['cv_folded'] < 10


def test_common_period_irregular_returns_none():
    rr = np.array([0.7, 2.9, 1.3, 4.1, 0.9, 2.2, 3.6, 1.7])
    per, _, _ = common_period(rr)
    assert per is None


def test_gap_tolerant_verdict_only_when_enabled():
    x, beats = _trace(dur_s=40.0, period_s=2.0)
    keep = np.ones(len(beats), bool)
    keep[[2, 5, 6, 9, 12, 15]] = False                          # missing beats → < 16 left, CV > 40 %
    bi = beats[keep]
    assert len(bi) < 16
    q = QualityConfig()
    out = assess_analysability(x, FS, bi, cfg=q)
    assert out['not_analysable'] and 'sparse' in out['reason']
    out2 = assess_analysability(x, FS, bi, cfg=dataclasses.replace(q, not_analysable_gap_tolerant=True))
    assert not out2['not_analysable']
    assert out2['common_period_ms'] == pytest.approx(2000, rel=0.02)


def test_analyze_single_file_with_options(tmp_path):
    from cardiac_fp_analyzer.analyze import analyze_single_file
    x, _, _ = _bursty_trace()
    t = np.arange(len(x)) / FS
    p = tmp_path / 'synthetic_burst.csv'
    with open(p, 'w') as f:
        f.write('#Sample rate: 2000Hz\nTime (s),Channel 1 (V),Channel 2 (V)\n')
        for ti, xi in zip(t, x * 1e-4):
            f.write(f'{ti:.6f},{xi:.9e},0\n')
    cfg = AnalysisConfig()
    cfg.beat_detection = dataclasses.replace(cfg.beat_detection, enable_weak_population_split=True, enable_burst_veto=True)
    cfg.quality = dataclasses.replace(cfg.quality, not_analysable_gap_tolerant=True)
    r = analyze_single_file(str(p), channel='el1', verbose=False, config=cfg)
    assert r is not None
    assert 'post_detection_vetoes' in r['detection_info']
