"""Unit tests for the minor-amplitude-population rejection in beat_detection."""

import numpy as np

from cardiac_fp_analyzer.beat_detection import _reject_minor_amplitude_population
from cardiac_fp_analyzer.config import BeatDetectionConfig

FS = 2000.0


def _signal(duration_s=60.0, noise_std=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, noise_std, int(duration_s * FS)), rng


def _spike(x, idx, amp, width_ms=4.0):
    w = int(width_ms / 1000 * FS)
    t = np.arange(-3 * w, 3 * w)
    shape = -np.exp(-0.5 * (t / w) ** 2) + 0.6 * np.exp(-0.5 * ((t - w) / w) ** 2)
    lo, hi = idx - 3 * w, idx + 3 * w
    if lo >= 0 and hi <= len(x):
        x[lo:hi] += amp * shape


def test_irregular_small_deflections_are_removed():
    """Regular big beats + irregular small bursts (3-4× smaller): drop the small ones."""
    x, rng = _signal()
    big = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(2.0 * FS))
    for b in big:
        _spike(x, b, 1.0)
    small = np.sort(rng.integers(int(1.5 * FS), len(x) - int(1.5 * FS), 14))
    small = small[np.all(np.abs(small[:, None] - big[None, :]) > int(0.3 * FS), axis=1)]
    for s_ in small:
        _spike(x, s_, 0.3)
    bi = np.sort(np.concatenate([big, small]))
    kept, info = _reject_minor_amplitude_population(x, FS, bi, cfg=BeatDetectionConfig())
    assert info['minor_pop'] == 'applied', info
    assert set(kept.tolist()) == set(big.tolist())


def test_amplitude_alternans_is_kept():
    """Big/small/big/small at fixed phase 0.5: real beats, must survive."""
    x, _ = _signal()
    beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(1.0 * FS))
    for k, b in enumerate(beats):
        _spike(x, b, 1.0 if k % 2 == 0 else 0.3)
    kept, info = _reject_minor_amplitude_population(x, FS, beats, cfg=BeatDetectionConfig())
    assert info['minor_pop'] == 'alternans_like', info
    assert len(kept) == len(beats)


def test_uniform_amplitude_untouched():
    x, rng = _signal()
    beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(1.5 * FS))
    for b in beats:
        _spike(x, b, 1.0 * rng.uniform(0.85, 1.15))
    kept, info = _reject_minor_amplitude_population(x, FS, beats, cfg=BeatDetectionConfig())
    assert len(kept) == len(beats)
    assert info['minor_pop'] in ('ratio_too_small', 'no_split', 'rhythm_not_improved')


def test_small_beats_that_regularise_rhythm_are_kept():
    """Small beats filling the gaps of a regular rhythm (big beats are the
    irregular subset): removing them would make RR worse → keep."""
    x, _ = _signal()
    beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(1.0 * FS))
    rng = np.random.default_rng(3)
    amps = np.where(rng.uniform(size=len(beats)) < 0.4, 1.0, 0.3)  # random 40 % big
    for b, a in zip(beats, amps):
        _spike(x, b, a)
    kept, info = _reject_minor_amplitude_population(x, FS, beats, cfg=BeatDetectionConfig())
    assert len(kept) == len(beats), info


def test_disabled():
    x, _ = _signal()
    beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(1.5 * FS))
    cfg = BeatDetectionConfig()
    cfg.enable_minor_population_reject = False
    kept, info = _reject_minor_amplitude_population(x, FS, beats, cfg=cfg)
    assert info['minor_pop'] == 'disabled' and len(kept) == len(beats)
