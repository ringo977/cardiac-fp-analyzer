"""Depolarisation times re-read on the unfiltered signal (BeatDetectionConfig.refine_on_raw_ms, v3.23, off).

Measured against the operators (GG DEV, Exp1 baseline, 26 Oct 2026): the detections already sit on the spike
peak (median 0.7 ms from the operator's time on GG, 1.5 ms on Exp1); with refinement 0.4 ms, but FPD ±10 %
64.8 → 63.4 % on DEV, so it stays an option. The operator's time is closer to the spike's steepest fall
(median −0.2 ms) than to its peak; on a few wide complexes (Exp1 G1, E2, F1) the operator marks a later hump.
"""
from __future__ import annotations

import numpy as np

from cardiac_fp_analyzer.analyze import refine_beats_on_raw
from cardiac_fp_analyzer.config import AnalysisConfig

FS = 2000.0


def _signals(sign=1):
    n = int(10 * FS)
    t = np.arange(n) / FS
    beats = np.arange(int(1 * FS), n - int(1 * FS), int(1.5 * FS))
    raw = np.zeros(n)
    filt = np.zeros(n)
    for b in beats:
        raw += sign * np.exp(-((t - t[b] - 0.004) / 0.001) ** 2 / 2)      # raw peak 4 ms after the detection
        filt += sign * np.exp(-((t - t[b]) / 0.004) ** 2 / 2)             # smoothed peak on the detection
    return raw, filt, beats


def test_off_by_default_and_identity_when_off():
    assert AnalysisConfig().beat_detection.refine_on_raw_ms == 0.0
    raw, filt, beats = _signals()
    out, side = refine_beats_on_raw(beats, raw, filt, FS, 0.0)
    assert np.array_equal(out, beats) and side == 0


def test_moves_to_the_raw_peak_on_the_spike_side():
    for sign in (1, -1):
        raw, filt, beats = _signals(sign)
        out, side = refine_beats_on_raw(beats, raw, filt, FS, 8.0)
        assert side == sign and np.all(np.abs(out - (beats + int(0.004 * FS))) <= 1)


def test_not_applied_when_raw_and_filtered_are_not_aligned():
    raw, filt, beats = _signals()
    out, side = refine_beats_on_raw(beats, raw[::2], filt, FS, 8.0)      # decimated filtered vs full-rate raw
    assert np.array_equal(out, beats) and side == 0
