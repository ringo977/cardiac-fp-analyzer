"""Rhythm train among the detected beats (Oct 2026).

On the GG recordings (slow beating, bursts of noise, sharp artefacts, large
repolarisation waves) the detector returns the beats plus many other events:
against the analyst's marks, 32 % more detections than beats on the
development experiments. The CV of the detected train then exceeded 25 % on
49 of 114 recordings (analyst: 4), and inclusion removed the tissue. The
rhythm train is the most regular sequence among the detections; beat
period, CV and the local RR of the rate correction come from it. These
tests pin it on synthetic signals.
"""

import contextlib
import io

import numpy as np
import pytest

from cardiac_fp_analyzer.analyze import analyze_single_file
from cardiac_fp_analyzer.beat_detection import _regular_subsequence, dominant_period, rhythm_train
from cardiac_fp_analyzer.config import AnalysisConfig, BeatDetectionConfig
from tests.golden_signals import _single_fp_beat, generate_regular_fp

FS = 2000.0


def _cfg(**kw):
    c = BeatDetectionConfig()
    c.enable_rhythm_train = True
    c.rhythm_min_cv = 0.0          # exercise the selection itself, whatever the CV
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_on_by_default_for_trains_that_would_fail_inclusion():
    c = BeatDetectionConfig()
    assert c.enable_rhythm_train is True and c.rhythm_min_cv == AnalysisConfig().inclusion.max_cv_bp == 25.0


def _train_with_extras(period=3.0, n=20, n_extra=12, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    true = np.arange(n) * period + 1.0
    true = np.delete(true, list(drop))
    extra = []
    while len(extra) < n_extra:
        x = rng.uniform(true[0], true[-1])
        if np.min(np.abs(true - x)) > 0.25 * period and all(abs(x - e) > 0.3 for e in extra):
            extra.append(x)
    return true, np.sort(np.concatenate([true, extra]))


# ── period ──────────────────────────────────────────────────────────────


def test_period_of_a_train_with_extra_detections_and_gaps():
    true, t = _train_with_extras(period=3.0, n_extra=12, drop=(5, 11))
    assert abs(dominant_period(t) - 3.0) < 0.3


def test_period_is_not_a_multiple_of_the_true_one():
    t = np.arange(25) * 1.2
    assert abs(dominant_period(t) - 1.2) < 0.1


def test_period_needs_a_few_detections():
    assert np.isnan(dominant_period(np.array([0.0, 1.0, 2.0])))


# ── sequence ────────────────────────────────────────────────────────────


def test_regular_subsequence_keeps_the_beats_and_drops_the_rest():
    true, t = _train_with_extras(period=2.5, n=24, n_extra=15, seed=3)
    keep = _regular_subsequence(t, 2.5)
    kept = t[keep]
    assert len(kept) >= len(true) - 1
    assert np.all([np.min(np.abs(true - k)) < 1e-9 for k in kept])


def test_irregular_but_beat_only_train_is_kept_whole():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(1.6, 2.4, 25))          # ±20 % around 2 s, nothing else
    assert len(_regular_subsequence(t, dominant_period(t))) == len(t)


# ── rhythm_train on a signal ────────────────────────────────────────────


def _slow_recording_with_artefacts(n_art=15, seed=3):
    sig, t, exp = generate_regular_fp(fs=FS, duration_s=60.0, beat_period_ms=3000.0, fpd_ms=900.0,
                                      noise_std=0.0005, seed=1)
    rng = np.random.default_rng(seed)
    spike = _single_fp_beat(FS, depol_amp=0.04, repol_amp=0.0, fpd_ms=50)[:int(0.02 * FS)]
    true = exp['beat_indices']
    extra = []
    while len(extra) < n_art:
        p = int(rng.uniform(1, 58) * FS)
        if np.min(np.abs(true - p)) > 0.4 * FS and all(abs(p - e) > 0.5 * FS for e in extra):
            extra.append(p)
    for p in extra:
        sig[p:p + len(spike)] += spike * rng.uniform(0.7, 1.2)
    return sig, t, true, np.array(sorted(extra))


def test_clean_train_is_left_unchanged():
    sig, _, true, _ = _slow_recording_with_artefacts(n_art=0)
    out, info = rhythm_train(sig, FS, true, cfg=_cfg())
    assert info['rhythm_train'] == 'unchanged' and np.array_equal(out, true)


def test_regular_enough_train_is_not_touched():
    sig, _, true, _ = _slow_recording_with_artefacts(n_art=0)
    rng = np.random.default_rng(2)
    bi = true + rng.integers(-300, 300, len(true))     # ±150 ms jitter on 3 s beats
    out, info = rhythm_train(sig, FS, bi, cfg=BeatDetectionConfig())
    assert info['rhythm_train'] == 'regular_enough' and info['cv_detected'] < 25
    assert np.array_equal(out, bi)


def test_option_off_returns_the_detections():
    sig, _, true, extra = _slow_recording_with_artefacts()
    bi = np.sort(np.concatenate([true, extra]))
    out, info = rhythm_train(sig, FS, bi, cfg=BeatDetectionConfig(enable_rhythm_train=False))
    assert info['rhythm_train'] == 'disabled' and np.array_equal(out, bi)


def test_artefacts_are_left_out_of_the_rhythm():
    sig, _, true, extra = _slow_recording_with_artefacts()
    bi = np.sort(np.concatenate([true, extra]))
    out, info = rhythm_train(sig, FS, bi, cfg=_cfg())
    assert info['rhythm_train'] == 'applied'
    assert info['n_dropped'] >= len(extra) - 3          # an artefact can extend the chain at an end
    assert sum(1 for b in out if np.min(np.abs(true - b)) < 0.02 * FS) >= len(true) - 1
    assert sum(1 for b in out if np.min(np.abs(extra - b)) < 0.02 * FS) <= 3


@pytest.mark.parametrize('on', [False, True])
def test_beat_period_and_cv_come_from_the_rhythm_train(tmp_path, on):
    sig, t, true, extra = _slow_recording_with_artefacts()
    path = tmp_path / 'chipA_ch1_baseline.csv'
    with open(path, 'w') as f:
        f.write('#Digilent WaveForms Oscilloscope Acquisition\n#Sample rate: 2000Hz\n\n'
                'Time (s),Channel 1 (V),Channel 2 (V)\n')
        np.savetxt(f, np.column_stack([t, sig, sig]), delimiter=',', fmt='%.7g')
    cfg = AnalysisConfig()
    cfg.beat_detection.enable_rhythm_train = on
    with contextlib.redirect_stdout(io.StringIO()):
        r = analyze_single_file(path, channel='el1', verbose=False, config=cfg)
    s = r['summary']
    # the detector picks up the artefacts; they stay in the raw train (arrhythmia analysis)
    assert len(r['beat_indices_raw']) > len(true) + len(extra) // 2
    if on:
        assert s['beat_period_ms_cv'] < 15 and abs(s['beat_period_ms_median'] - 3000) < 150
        assert len(r['beat_indices_rhythm']) < len(r['beat_indices_raw'])
    else:
        assert s['beat_period_ms_cv'] > 30
        assert np.array_equal(r['beat_indices_rhythm'], r['beat_indices_raw'])
