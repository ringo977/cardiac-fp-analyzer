"""
test_repolarization_bounds.py — Regression guard for the out-of-bounds
fallback bug in ``repolarization.py``.

Context
-------
Both ``find_repolarization_on_template`` and ``find_repolarization_per_beat``
contained a fallback branch of the form::

    valid_region = seg_det[min_pk_idx:] if min_pk_idx < len(seg_det) else seg_det
    best_idx = min_pk_idx + np.argmax(np.abs(valid_region))

When ``min_pk_idx >= len(seg_det)`` (the minimum-FPD floor exceeds the
search segment length — happens on slow rhythms / short templates),
``valid_region`` degenerated to the full ``seg_det`` but the code still
added ``min_pk_idx`` to the argmax offset, producing an index far past
``len(seg_det)``. The next access ``seg_det[best_idx]`` then raised
``IndexError``.

Observed in production on ``Exp6_chipD_ch1_chipB_ch1_baseline.csv``:

    IndexError: index 10773 is out of bounds for axis 0 with size 1600

These tests exercise the guarded code path directly by driving the
repolarization search onto a segment shorter than the ``min_fpd_ms``
floor, so the fallback branch is reached without needing the full
beat-detection pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from cardiac_fp_analyzer.config import RepolarizationConfig
from cardiac_fp_analyzer.repolarization import (
    find_repolarization_on_template,
    find_repolarization_per_beat,
)


def _build_template(fs=2000.0, duration_s=0.4, pre_ms=50, amp=0.5, seed=0):
    """Build a template with a sharp spike at `pre_ms` from the start and
    no repolarization bump — the fallback branch will be exercised."""
    n = int(fs * duration_s)
    pre_samples = int(pre_ms / 1000 * fs)
    tpl = np.random.default_rng(seed).standard_normal(n) * 0.001
    x = np.arange(n) - pre_samples
    tpl += amp * np.exp(-0.5 * (x / 4.0) ** 2)
    return tpl


class TestTemplateFallbackBounds:
    def test_min_fpd_past_template_does_not_crash(self):
        """min_fpd_ms so large it exceeds the search segment — must not
        raise IndexError. Acceptable returns: either a valid result or the
        graceful ``(None, 1, 0.0, 0.0, None, None)`` tuple."""
        fs = 2000.0
        tpl = _build_template(fs=fs, duration_s=0.4, pre_ms=50)
        cfg = RepolarizationConfig()
        # Force min_fpd floor to 800 ms — longer than any plausible segment
        cfg.min_fpd_ms = 800
        cfg.search_end_ms = 400
        # Aggressive min_fpd_pct_rr to stress the pct_rr adaptive path too
        cfg.min_fpd_pct_rr = 0.8

        # Should NOT raise
        result = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg, median_bp_s=1.5,
        )
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_very_short_template_does_not_crash(self):
        """Pathological short template: barely longer than the minimum
        search window."""
        fs = 2000.0
        tpl = _build_template(fs=fs, duration_s=0.22, pre_ms=30)
        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 300
        cfg.search_end_ms = 400
        cfg.min_fpd_pct_rr = 0.7

        result = find_repolarization_on_template(
            tpl, fs, pre_ms=30, cfg=cfg, median_bp_s=2.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_normal_template_still_works(self):
        """Sanity: a template with a real T-wave should still produce a
        valid FPD measurement after the fix (no regression)."""
        fs = 2000.0
        n = int(fs * 0.8)
        pre = int(0.05 * fs)
        tpl = np.random.default_rng(3).standard_normal(n) * 0.0005
        # Spike
        x = np.arange(n) - pre
        tpl += 0.8 * np.exp(-0.5 * (x / 4.0) ** 2)
        # T-wave ~ 300 ms after the spike (biologically plausible)
        t_idx = pre + int(0.3 * fs)
        xt = np.arange(n) - t_idx
        tpl += 0.3 * np.exp(-0.5 * (xt / 80.0) ** 2)

        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 150
        cfg.search_end_ms = 600
        result = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg, median_bp_s=1.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 6
        # The first element (fpd_samples) should be populated for a normal
        # template with a clear T-wave.
        fpd_samples = result[0]
        assert fpd_samples is not None


class TestPerBeatFallbackBounds:
    def _make_signal(self, fs=2000.0, total_s=1.0, spike_s=0.7, seed=1):
        n = int(fs * total_s)
        data = np.random.default_rng(seed).standard_normal(n) * 0.001
        spike_idx = int(spike_s * fs)
        x = np.arange(n) - spike_idx
        data += 0.6 * np.exp(-0.5 * (x / 4.0) ** 2)
        t_vec = np.arange(n) / fs
        return data, t_vec, spike_idx

    def test_per_beat_min_fpd_past_segment_does_not_crash(self):
        """Spike late in the signal → little room for repolarization
        search → fallback branch with min_pk_idx >= len(seg_det)."""
        fs = 2000.0
        data, t, spike_idx = self._make_signal(fs=fs, total_s=1.0, spike_s=0.7)

        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 500
        cfg.search_end_ms = 600
        cfg.min_fpd_pct_rr = 0.6

        # Must not raise
        result = find_repolarization_per_beat(
            data, t, spike_idx, fs,
            template_fpd_samples=None,
            template_peak_samples=None,
            template_repol_sign=1,
            cfg=cfg,
            beat_period_s=1.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    @pytest.mark.parametrize('min_fpd_ms,bp_s', [
        (900, 1.5),    # pct_rr (0.6) × bp (1.5s) = 900 ms threshold
        (1200, 2.0),   # even larger floor
        (300, 0.5),    # normal short-rhythm case
    ])
    def test_per_beat_various_min_fpd(self, min_fpd_ms, bp_s):
        fs = 2000.0
        data, t, spike_idx = self._make_signal(
            fs=fs, total_s=1.2, spike_s=0.5, seed=min_fpd_ms,
        )

        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = min_fpd_ms
        cfg.search_end_ms = 700

        result = find_repolarization_per_beat(
            data, t, spike_idx, fs,
            template_fpd_samples=None,
            template_peak_samples=None,
            template_repol_sign=1,
            cfg=cfg,
            beat_period_s=bp_s,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_per_beat_normal_case_still_works(self):
        """Regression sanity: a beat with a clear T-wave after the spike
        should still produce a valid FPD."""
        fs = 2000.0
        n = int(fs * 1.5)
        data = np.random.default_rng(7).standard_normal(n) * 0.0005
        spike_idx = int(0.3 * fs)
        x = np.arange(n) - spike_idx
        data += 0.8 * np.exp(-0.5 * (x / 4.0) ** 2)
        # T-wave 280 ms after spike
        t_idx = spike_idx + int(0.28 * fs)
        xt = np.arange(n) - t_idx
        data += 0.25 * np.exp(-0.5 * (xt / 80.0) ** 2)
        t_vec = np.arange(n) / fs

        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 150
        cfg.search_end_ms = 500

        result = find_repolarization_per_beat(
            data, t_vec, spike_idx, fs,
            template_fpd_samples=None,
            template_peak_samples=None,
            template_repol_sign=1,
            cfg=cfg,
            beat_period_s=1.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        # Normal case should produce a real FPD
        fpd = result[0]
        assert fpd is not None
