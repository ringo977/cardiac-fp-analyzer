"""Per-beat repolarisation gate: the noise it divides by (v3.22).

The search window is the complex itself (±150 ms around the template latency, +200 ms). With the default
up to v3.21 the gate's noise was the std of that window, so a second, larger lobe of the repolarisation wave
next to the chosen one counted as noise: on Exp1 chip E ch2 (RR 3.6 s, 75 µV wave with a 140 µV lobe 170 ms
later) every beat got SNR ≈ 1 and no FPD at all, while the template had found the wave. 'diastolic' reads
the noise where there is no waveform.
"""
from __future__ import annotations

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.repolarization import _diastolic_noise, find_repolarization_per_beat

FS = 2000.0


def _beat(noise=2e-6, pos_uv=75.0, neg_uv=140.0, lat_s=1.10, gap_s=0.17, rr_s=3.6, seed=0):
    """One beat segment like the pipeline's: spike at 50 ms, segment to 85 % of the RR."""
    rng = np.random.default_rng(seed)
    n = int((0.05 + 0.85 * rr_s) * FS)
    t = np.arange(n) / FS - 0.05
    x = noise * rng.standard_normal(n)
    x += -300e-6 * np.exp(-(t / 0.003) ** 2 / 2)
    x += pos_uv * 1e-6 * np.exp(-((t - lat_s) / 0.04) ** 2 / 2)
    x += -neg_uv * 1e-6 * np.exp(-((t - lat_s - gap_s) / 0.04) ** 2 / 2)
    return x, int(0.05 * FS)


def _run(mode, **kw):
    x, spike = _beat(**kw)
    rc = AnalysisConfig().repolarization
    rc.repol_gate_noise = mode
    fpd, amp, pk, end = find_repolarization_per_beat(
        x, np.arange(len(x)) / FS, spike, FS, template_peak_samples=int(1.10 * FS), template_repol_sign=1,
        cfg=rc, beat_period_s=3.6, next_rr_s=3.6)
    return fpd


def test_second_lobe_no_longer_hides_the_wave():
    assert _run('std') is None                          # the old gate: SNR ≈ 1 → not detectable
    fpd = _run('diastolic')
    assert fpd is not None and abs(fpd - 1.10) < 0.03   # the template's wave, on this beat
    assert AnalysisConfig().repolarization.repol_gate_noise == 'diastolic'


def test_pure_noise_not_more_accepted_than_before():
    """Neither per-beat gate is a strong protection against a noise bump (both accept one on about two
    thirds of pure-noise windows — the template gate, SNR 2 on the averaged beat, and the valid-beat
    fraction are what protects a recording); the diastolic gate must not be noticeably more permissive."""
    std = sum(_run('std', pos_uv=0.0, neg_uv=0.0, noise=4e-6, seed=k) is not None for k in range(30))
    dia = sum(_run('diastolic', pos_uv=0.0, neg_uv=0.0, noise=4e-6, seed=k) is not None for k in range(30))
    assert dia <= std + 5


def test_diastolic_noise_estimate():
    x, spike = _beat(noise=5e-6)
    n = _diastolic_noise(x, int((0.05 + 1.5) * FS), int((0.05 + 1.10) * FS), FS, AnalysisConfig().repolarization)
    assert n is not None and 2e-7 < n < 5e-6                  # below the white noise: the 20 Hz low-pass smooths it
    assert _diastolic_noise(x, len(x) - 10, len(x) - 20, FS, AnalysisConfig().repolarization) is None   # too short


@pytest.mark.parametrize('rr', [0.6, 1.0])
def test_fast_rhythm_falls_back_to_the_window(rr):
    """Little or no diastole in the segment: the estimate is None and the gate uses the window's MAD."""
    x, spike = _beat(rr_s=rr, lat_s=0.3, gap_s=0.1)
    assert _diastolic_noise(x, int((0.05 + 0.45) * FS), int((0.05 + 0.3) * FS), FS, AnalysisConfig().repolarization) is None or rr > 0.6
