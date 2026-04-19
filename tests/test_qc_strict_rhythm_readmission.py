"""Unit tests for strict-rhythm QC re-admission (v3.4.x).

Motivation
----------
On bradycardic 3D constructs (e.g. Exp6 Ti08 EL2), a fraction of real beats
show decorrelated / inverted morphology but retain robust amplitude and
fall exactly on the expected rhythm.  The original period-aware re-admission
gated on ``corr ≥ morphology_marginal`` and lost these beats.

A second pathway now admits a morph-rejected beat when:

  * it is amplitude-passing (``amp_ratio ≥ amplitude_reject_fraction``),
  * its distance to the nearest accepted beat rounds cleanly to an integer
    number of median RR periods with residual below
    ``strict_rhythm_residual_ratio × median RR``,
  * its amp_ratio is ≥ ``strict_rhythm_amp_ratio``.

These tests use synthetic fixtures — no real-signal CSVs — so they are
deterministic and fast.
"""
from __future__ import annotations

import numpy as np
import pytest

from cardiac_fp_analyzer.config import QualityConfig
from cardiac_fp_analyzer.quality_control import validate_beats

# ─────────────────────────────────────────────────────────────────────────
#   Fixture helpers
# ─────────────────────────────────────────────────────────────────────────

def _biphasic_spike(fs: float, amp: float = 1.0, invert: bool = False) -> np.ndarray:
    """Short biphasic depolarisation segment (~20 ms)."""
    n = int(0.020 * fs)
    t_ms = np.arange(n) / fs * 1000.0
    w = (amp * np.exp(-0.5 * ((t_ms - 2.0) / 1.0) ** 2)
         - 0.6 * amp * np.exp(-0.5 * ((t_ms - 5.0) / 1.5) ** 2))
    return -w if invert else w


def _make_bradycardic_signal(
    fs: float = 2000.0,
    duration_s: float = 40.0,
    rr_s: float = 2.5,
    miss_indices: tuple[int, ...] = (),
    miss_amp_ratio: float = 0.7,
    miss_invert: bool = True,
    noise_std: float = 0.0,
):
    """Build a bradycardic synthetic signal.

    Parameters
    ----------
    fs, duration_s, rr_s : signal timing
    miss_indices : 0-based positions of beats that should be "degraded"
                   (reduced amplitude and optionally inverted polarity).
    miss_amp_ratio : amplitude of degraded beats relative to full-amp beats.
    miss_invert : if True, flip the polarity of degraded beats (so morph
                  correlation against the template goes very negative).
    noise_std : Gaussian noise (set to 0 for deterministic spike positions).

    Returns
    -------
    signal : 1-D array
    beat_indices : 1-D int array of spike sample positions
    """
    n = int(duration_s * fs)
    signal = np.zeros(n)
    rng = np.random.default_rng(seed=0)
    if noise_std > 0:
        signal += rng.normal(0.0, noise_std, n)

    beat_indices = []
    spike = _biphasic_spike(fs, amp=1.0, invert=False)
    spike_miss = _biphasic_spike(fs, amp=miss_amp_ratio, invert=miss_invert)
    sl = len(spike)

    n_beats = int(duration_s / rr_s)
    for k in range(1, n_beats + 1):
        idx = int(k * rr_s * fs)
        if idx + sl >= n:
            break
        if (k - 1) in miss_indices:
            signal[idx:idx + sl] += spike_miss
        else:
            signal[idx:idx + sl] += spike
        beat_indices.append(idx)
    return signal, np.asarray(beat_indices, dtype=int)


def _segment(signal: np.ndarray, bi: np.ndarray, fs: float,
             pre_ms: float = 50.0, post_ms: float = 650.0):
    """Minimal segmentation wrapper (no edge dropping for the fixture)."""
    pre = int(pre_ms / 1000.0 * fs)
    post = int(post_ms / 1000.0 * fs)
    bd, bt, valid = [], [], []
    for i, idx in enumerate(bi):
        s, e = idx - pre, idx + post
        if s < 0 or e >= len(signal):
            continue
        bd.append(signal[s:e].copy())
        bt.append(np.arange(-pre, post) / fs)
        valid.append(i)
    return bd, bt, np.asarray(valid, dtype=int)


# ─────────────────────────────────────────────────────────────────────────
#   Tests
# ─────────────────────────────────────────────────────────────────────────

class TestStrictRhythmReadmission:
    """The strict-rhythm path must recover degraded-but-on-rhythm beats."""

    def test_strict_path_recovers_inverted_on_rhythm_beat(self):
        """An inverted on-rhythm beat with amp ≥ 0.5 must be re-admitted."""
        fs = 2000.0
        signal, bi = _make_bradycardic_signal(
            fs=fs, duration_s=40.0, rr_s=2.5,
            miss_indices=(5,),           # beat #6 (index 5) is inverted
            miss_amp_ratio=0.7,
            miss_invert=True,
        )
        bd, bt, valid = _segment(signal, bi, fs)
        bi_seg = bi[valid]

        cfg = QualityConfig()
        qc, acc_bi, _, _ = validate_beats(
            signal, bi_seg, bd, bt, fs, cfg=cfg)

        # The inverted beat must be in the accepted set
        degraded_sample = int(bi[5])
        assert degraded_sample in set(int(v) for v in acc_bi), (
            "Strict-rhythm path failed to rescue inverted on-rhythm beat")
        # At least one re-admission should have occurred
        assert qc.n_readmitted >= 1

    def test_strict_path_disabled_by_threshold_one(self):
        """Setting strict_rhythm_amp_ratio=1 disables the strict path."""
        fs = 2000.0
        signal, bi = _make_bradycardic_signal(
            fs=fs, duration_s=40.0, rr_s=2.5,
            miss_indices=(5,),
            miss_amp_ratio=0.7,
            miss_invert=True,
        )
        bd, bt, valid = _segment(signal, bi, fs)
        bi_seg = bi[valid]

        cfg = QualityConfig()
        cfg.strict_rhythm_amp_ratio = 1.0  # disable strict path
        qc, acc_bi, _, _ = validate_beats(
            signal, bi_seg, bd, bt, fs, cfg=cfg)

        # The inverted beat must NOT be re-admitted
        degraded_sample = int(bi[5])
        assert degraded_sample not in set(int(v) for v in acc_bi), (
            "Disabling strict path must prevent rescue of inverted beat")

    def test_strict_path_respects_amplitude_floor(self):
        """A degraded beat below amp_ratio floor must stay rejected."""
        fs = 2000.0
        # Degraded amp_ratio = 0.30 — above default amplitude_reject (0.25)
        # but below strict_rhythm_amp_ratio (0.50)
        signal, bi = _make_bradycardic_signal(
            fs=fs, duration_s=40.0, rr_s=2.5,
            miss_indices=(5,),
            miss_amp_ratio=0.30,
            miss_invert=True,
        )
        bd, bt, valid = _segment(signal, bi, fs)
        bi_seg = bi[valid]

        cfg = QualityConfig()
        qc, acc_bi, _, _ = validate_beats(
            signal, bi_seg, bd, bt, fs, cfg=cfg)

        degraded_sample = int(bi[5])
        assert degraded_sample not in set(int(v) for v in acc_bi), (
            "Beats below strict_rhythm_amp_ratio must not be re-admitted")

    def test_strict_path_respects_timing(self):
        """A robust-amplitude beat NOT on-rhythm must stay rejected."""
        fs = 2000.0
        duration_s = 40.0
        rr_s = 2.5
        signal = np.zeros(int(duration_s * fs))
        spike = _biphasic_spike(fs, amp=1.0, invert=False)
        spike_off = _biphasic_spike(fs, amp=0.7, invert=True)
        sl = len(spike)

        bi = []
        n_beats = int(duration_s / rr_s)
        for k in range(1, n_beats + 1):
            idx = int(k * rr_s * fs)
            if idx + sl >= len(signal):
                break
            signal[idx:idx + sl] += spike
            bi.append(idx)
        # Insert an OFF-RHYTHM degraded beat midway between beats 5 and 6
        off_idx = int((5 * rr_s + rr_s * 0.4) * fs)
        signal[off_idx:off_idx + sl] += spike_off
        bi.insert(5, off_idx)  # keep sorted-ish
        bi = sorted(set(bi))
        bi = np.asarray(bi, dtype=int)

        bd, bt, valid = _segment(signal, bi, fs)
        bi_seg = bi[valid]

        cfg = QualityConfig()
        qc, acc_bi, _, _ = validate_beats(
            signal, bi_seg, bd, bt, fs, cfg=cfg)

        assert off_idx not in set(int(v) for v in acc_bi), (
            "Off-rhythm beat must not be re-admitted regardless of amplitude")

    def test_clean_regular_signal_unaffected(self):
        """On a clean regular signal, no beats should need re-admission."""
        fs = 2000.0
        signal, bi = _make_bradycardic_signal(
            fs=fs, duration_s=40.0, rr_s=2.5,
            miss_indices=(),  # no degraded beats
        )
        bd, bt, valid = _segment(signal, bi, fs)
        bi_seg = bi[valid]

        cfg = QualityConfig()
        qc, acc_bi, _, _ = validate_beats(
            signal, bi_seg, bd, bt, fs, cfg=cfg)

        assert qc.n_readmitted == 0
        # Almost all beats should be accepted on a clean fixture
        assert len(acc_bi) >= len(bi_seg) - 1


class TestStrictRhythmConfig:
    """Config wiring + backwards compatibility."""

    def test_config_has_new_fields(self):
        cfg = QualityConfig()
        assert hasattr(cfg, 'strict_rhythm_residual_ratio')
        assert hasattr(cfg, 'strict_rhythm_amp_ratio')
        assert 0.0 < cfg.strict_rhythm_residual_ratio <= 0.5
        assert 0.0 < cfg.strict_rhythm_amp_ratio <= 1.0

    def test_cfg_without_new_attrs_still_works(self):
        """Code path should tolerate a cfg-like object missing the new attrs.

        Uses a minimal stand-in for QualityConfig with only the attrs
        validate_beats strictly requires; the strict-rhythm path must fall
        back to defaults via getattr().
        """

        class _MinimalCfg:
            snr_excellent = 8.0
            snr_good = 5.0
            snr_fair = 3.0
            snr_poor = 2.0
            amplitude_reject_fraction = 0.25
            morphology_threshold = 0.40
            morphology_marginal = 0.20
            use_morphology = True
            max_rejection_rate = 0.40
            rejection_high_note = 0.50
            rejection_grade_c = 0.20
            rejection_grade_b = 0.05
            morphology_max_beats = 30
            morphology_window_ms = 20.0
            morphology_corr_region_ms = 150.0
            min_beats_for_analysis = 3

        fs = 2000.0
        signal, bi = _make_bradycardic_signal(
            fs=fs, duration_s=40.0, rr_s=2.5, miss_indices=())
        bd, bt, valid = _segment(signal, bi, fs)
        bi_seg = bi[valid]

        # Should not raise
        validate_beats(signal, bi_seg, bd, bt, fs, cfg=_MinimalCfg())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
