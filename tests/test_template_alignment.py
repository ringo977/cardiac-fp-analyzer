"""Template construction: beat alignment and per-beat polarity inversion.

1. ``_align_beats_xcorr`` was off by ``alignment_max_shift_ms``: already
   aligned beats moved by 50 ms and jitter was amplified, so every template
   was 50 ms late relative to its beats. These tests fail on the old code.
2. Per-beat polarity inversion is decided by anti-correlation with the
   template spike (``beat_spike_inverted``), not by comparing deflections.
"""

import numpy as np

from cardiac_fp_analyzer.config import RepolarizationConfig
from cardiac_fp_analyzer.parameters import (
    _align_beats_xcorr,
    beat_spike_inverted,
    build_beat_template,
)

FS = 2000.0
PRE_MS = 50.0


def _gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def _beat(post_ms=1450.0, t_pos=None, a_pos=0.0, t_neg=None, a_neg=0.0, spike=1.0,
          noise=0.0, rng=None, invert=False):
    """Synthetic FP beat: biphasic spike at PRE_MS + optional ±T lobes."""
    n = int((PRE_MS + post_ms) / 1000 * FS)
    t = np.arange(n) / FS * 1000 - PRE_MS            # ms from spike
    x = spike * (_gauss(t, 0, 2.0) - 0.7 * _gauss(t, 6, 3.0))
    if t_pos is not None:
        x = x + a_pos * _gauss(t, t_pos, 40.0)
    if t_neg is not None:
        x = x - a_neg * _gauss(t, t_neg, 40.0)
    if noise and rng is not None:
        x = x + rng.normal(0, noise, n)
    return -x if invert else x


def _beat_time(n):
    return np.arange(n) / FS - PRE_MS / 1000


# ── 1. alignment ─────────────────────────────────────────────────────────

class TestTemplateAlignment:
    def test_identical_beats_are_not_moved(self):
        b = _beat(t_pos=600, a_pos=0.2)
        aligned = _align_beats_xcorr([b.copy() for _ in range(6)], FS, RepolarizationConfig())
        assert int(np.argmax(aligned[0])) == int(np.argmax(b))

    def test_jitter_is_removed_not_amplified(self):
        rng = np.random.default_rng(1)
        b = _beat(t_pos=600, a_pos=0.2)
        shifts = rng.integers(-30, 31, 10)                 # up to ±15 ms
        beats = [np.roll(b, s) + rng.normal(0, 0.01, len(b)) for s in shifts]
        aligned = _align_beats_xcorr(beats, FS, RepolarizationConfig())
        spread_in = np.ptp([np.argmax(x) for x in beats])
        spread_out = np.ptp([np.argmax(x) for x in aligned])
        # residual ≤ 2 ms on noisy beats; the old code amplified it
        assert spread_out <= 4 and spread_out < spread_in / 10

    def test_template_spike_stays_at_segment_pre(self):
        """The template must describe the beats on THEIR time axis: its spike
        at segment_pre_ms (the old bug put it 50 ms later)."""
        rng = np.random.default_rng(2)
        beats = [_beat(t_pos=600, a_pos=0.2, noise=0.01, rng=rng) for _ in range(12)]
        tpl = build_beat_template(beats, FS, RepolarizationConfig())
        pre = int(PRE_MS / 1000 * FS)
        assert abs(int(np.argmax(tpl)) - pre) <= 2


# ── 2. inversion ─────────────────────────────────────────────────────────

class TestInversionTest:
    def _windows(self, beat):
        pre = int(PRE_MS / 1000 * FS)
        return beat[pre - int(0.010 * FS):pre + int(0.020 * FS)]

    def test_same_shape_biphasic_spike_is_not_inverted(self):
        """Lobes of similar size: comparing deflections is a coin toss,
        correlation is not."""
        tpl = _beat(spike=1.0)
        k = self._windows(tpl); k = k - k.mean()
        rng = np.random.default_rng(3)
        for _ in range(20):
            b = _beat(spike=1.0) + rng.normal(0, 0.05, len(tpl))
            assert not beat_spike_inverted(k, self._windows(b))

    def test_truly_inverted_beat_is_detected(self):
        tpl = _beat(spike=1.0)
        k = self._windows(tpl); k = k - k.mean()
        assert beat_spike_inverted(k, self._windows(_beat(spike=1.0, invert=True)))

    def test_no_template_means_not_inverted(self):
        assert not beat_spike_inverted(None, np.ones(10))
