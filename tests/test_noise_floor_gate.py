"""Unit tests for the noise-floor SNR gate in beat_detection.

Real-signal behaviour is covered by tests/test_real_signal_regression.py;
these synthetic cases pin down the rule's edges:
  - noise-level insertions at rhythmically expected positions are removed;
  - small but real beats (amplitude alternans) are NOT removed, even when
    they form a distinct lower cluster;
  - T-wave-sized secondary peaks are left to the other filters;
  - the gate never wipes a recording and can be disabled.
"""

import numpy as np

from cardiac_fp_analyzer.beat_detection import (
    _reject_below_noise_floor,
    estimate_noise_floor,
)
from cardiac_fp_analyzer.config import BeatDetectionConfig

FS = 2000.0


def _signal(duration_s=60.0, noise_std=0.03, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    return rng.normal(0.0, noise_std, n), rng


def _add_spike(x, idx, amp, width_ms=4.0):
    w = int(width_ms / 1000 * FS)
    t = np.arange(-3 * w, 3 * w)
    shape = -np.exp(-0.5 * (t / w) ** 2) + 0.6 * np.exp(-0.5 * ((t - w) / w) ** 2)
    lo, hi = idx - 3 * w, idx + 3 * w
    if lo >= 0 and hi <= len(x):
        x[lo:hi] += amp * shape


class TestNoiseFloorEstimate:
    def test_noise_floor_tracks_noise_not_beats(self):
        x, _ = _signal(noise_std=0.03)
        beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(2.0 * FS))
        for b in beats:
            _add_spike(x, b, 1.0)
        nf = estimate_noise_floor(x, FS)
        # ptp of Gaussian noise over 80 samples ≈ 4–5 σ
        assert 0.08 < nf < 0.20, nf

    def test_short_signal_returns_zero(self):
        assert estimate_noise_floor(np.zeros(50), FS) == 0.0


class TestGateRules:
    def test_noise_insertions_removed_real_beats_kept(self):
        x, _ = _signal()
        real = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(2.0 * FS))
        for b in real:
            _add_spike(x, b, 0.5)
        # One "detection" on pure noise at the midpoint of every RR interval.
        fake = real[:-1] + int(1.0 * FS)
        bi = np.sort(np.concatenate([real, fake]))
        kept, info = _reject_below_noise_floor(x, FS, bi, cfg=BeatDetectionConfig())
        assert info['noise_gate'] == 'applied'
        assert set(kept.tolist()) == set(real.tolist()), info

    def test_amplitude_alternans_is_preserved(self):
        """Big/small/big/small real beats: the small ones are a distinct
        lower cluster but far above noise — the gate must keep them."""
        x, _ = _signal(noise_std=0.03)
        beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(1.5 * FS))
        for k, b in enumerate(beats):
            _add_spike(x, b, 1.0 if k % 2 == 0 else 0.35)   # small ≈ 3× noise ptp
        kept, info = _reject_below_noise_floor(x, FS, beats, cfg=BeatDetectionConfig())
        assert len(kept) == len(beats), info

    def test_t_wave_sized_peaks_not_touched(self):
        """Secondary peaks at ~40 % of spike amplitude are above the noise
        caps: they are the bimodal-BP / cluster filter's job, not the gate's."""
        x, _ = _signal(noise_std=0.02)
        beats = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(2.0 * FS))
        for b in beats:
            _add_spike(x, b, 1.0)
            _add_spike(x, b + int(0.4 * FS), 0.4, width_ms=20.0)
        bi = np.sort(np.concatenate([beats, beats + int(0.4 * FS)]))
        kept, info = _reject_below_noise_floor(x, FS, bi, cfg=BeatDetectionConfig())
        assert len(kept) == len(bi), info

    def test_never_wipes_recording(self):
        x, _ = _signal()
        bi = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(2.0 * FS))  # all noise
        kept, info = _reject_below_noise_floor(x, FS, bi, cfg=BeatDetectionConfig())
        assert info['noise_gate'] == 'aborted_all_below'
        assert len(kept) == len(bi)

    def test_disabled_is_identity(self):
        x, _ = _signal()
        bi = np.arange(int(1.0 * FS), len(x) - int(1.0 * FS), int(2.0 * FS))
        cfg = BeatDetectionConfig()
        cfg.enable_noise_floor_gate = False
        kept, info = _reject_below_noise_floor(x, FS, bi, cfg=cfg)
        assert info['noise_gate'] == 'disabled'
        assert np.array_equal(kept, bi)

    def test_empty_input(self):
        x, _ = _signal()
        kept, info = _reject_below_noise_floor(x, FS, np.array([], dtype=int), cfg=BeatDetectionConfig())
        assert len(kept) == 0 and info['noise_gate'] == 'no_beats'
