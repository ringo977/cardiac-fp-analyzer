"""
test_amplitude_cluster_filter.py — Regression tests for Sprint 2 #1.

Background
----------
The Exp6_ChipD_ch2 recording (hiPSC-CM microtissue, 60 s @ 2000 Hz, slow
rhythm ~17 bpm) exposed a gap in the detection pipeline:

  * 16 real depolarisation spikes at ~1.4 V
  * 33 spurious bumps at ~0.15 V (T-wave residuals / baseline features)
    occurring between the real beats

`_fix_bimodal_bp` only catches strictly alternating short/long RR
patterns (ratio ≈ 1.5–2.8×) — here the pattern is "1 big + 2-3 small +
1 big", not bimodal but plurimodal. Morphology validation is skipped
on mixed-polarity signals, falling back to an amplitude-only gate
whose reference (`ref_amplitude`) is the median of *all* detected
peaks — contaminated by the 33 false positives.

Sprint 2 #1 adds `_reject_amplitude_cluster`, an amplitude-based,
polarity-agnostic filter that runs between `_fix_bimodal_bp` and
`validate_beats_morphology`. When sorted peak amplitudes have a
largest-adjacent-ratio ≥ ``cluster_gap_ratio`` (default 3.0) AND the
dominant (high-amplitude) cluster contains ≥
``cluster_min_dominant_count`` peaks (default 3), the low-amp cluster
is dropped.

These tests verify:

1. The filter is a no-op on unimodal recordings (synthetic clean FP);
2. The filter fires on the Exp6_ChipD_ch2 topology, reducing 16+33
   detections to 16;
3. The filter declines to fire when the dominant cluster is too small
   (single-artefact guard);
4. The filter is polarity-agnostic (inverted signal → same behaviour);
5. Disabling the filter via config restores pre-Sprint-2 behaviour;
6. Diagnostics round-trip through ``info['amplitude_cluster']``.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#   Synthetic signal generators
# ═══════════════════════════════════════════════════════════════════════

def _biphasic_spike(n_samples, center_idx, amp, width_samples):
    """Add a sharp positive-then-negative spike centred at ``center_idx``."""
    t = np.arange(n_samples) - center_idx
    sigma = max(1.0, width_samples / 3.0)
    pos = amp * np.exp(-0.5 * (t / sigma) ** 2)
    neg = -0.4 * amp * np.exp(-0.5 * ((t - width_samples) / sigma) ** 2)
    return pos + neg


def _bumpettino(n_samples, center_idx, amp, width_samples):
    """Small symmetric bipolar wavelet — a T-wave-like residual."""
    t = np.arange(n_samples) - center_idx
    sigma = max(2.0, width_samples / 2.0)
    # Derivative of Gaussian → biphasic symmetric bump
    return amp * (-t / (sigma ** 2)) * np.exp(-0.5 * (t / sigma) ** 2) * sigma


def make_bimodal_signal(fs=2000.0, duration_s=60.0,
                        big_rr_s=3.6, big_amp=1.4,
                        small_per_rr=2, small_amp=0.15,
                        noise_std=0.01, seed=123):
    """Replicate the Exp6_ChipD_ch2 topology.

    * 1 big spike (``big_amp``) every ``big_rr_s`` seconds
    * ``small_per_rr`` small bumps (``small_amp``) uniformly distributed
      between consecutive big spikes

    Returns signal, expected big-spike sample indices.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)

    big_rr_samp = int(big_rr_s * fs)
    first_big = big_rr_samp  # leave a gap at the start
    big_indices = []
    idx = first_big
    while idx < n - int(0.5 * fs):
        big_indices.append(idx)
        sig += _biphasic_spike(n, idx, big_amp, width_samples=int(0.005 * fs))
        # Small bumps uniformly between this and the next big spike
        step = big_rr_samp / (small_per_rr + 1)
        for k in range(1, small_per_rr + 1):
            bump_idx = int(idx + k * step)
            if 0 < bump_idx < n - int(0.1 * fs):
                sig += _bumpettino(n, bump_idx, small_amp,
                                   width_samples=int(0.05 * fs))
        idx += big_rr_samp

    sig += rng.standard_normal(n) * noise_std
    return sig, np.array(big_indices)


def make_unimodal_signal(fs=2000.0, duration_s=10.0, rr_s=0.8,
                         amp=0.05, noise_std=0.001, seed=7):
    """Clean unimodal FP — all beats at the same amplitude."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    rr_samp = int(rr_s * fs)
    idx = rr_samp
    indices = []
    while idx < n - int(0.2 * fs):
        indices.append(idx)
        sig += _biphasic_spike(n, idx, amp, width_samples=int(0.003 * fs))
        idx += rr_samp
    sig += rng.standard_normal(n) * noise_std
    return sig, np.array(indices)


# ═══════════════════════════════════════════════════════════════════════
#   1. Filter fires on bimodal Exp6-like signal
# ═══════════════════════════════════════════════════════════════════════

class TestFilterOnBimodalSignal:
    """End-to-end: detect_beats on a bimodal synthetic signal must return
    only the big-amplitude cluster."""

    def test_exp6_like_signal_filtered_to_big_cluster(self):
        from cardiac_fp_analyzer.beat_detection import detect_beats

        fs = 2000.0
        sig, expected_big = make_bimodal_signal(
            fs=fs, duration_s=60.0,
            big_rr_s=3.6, big_amp=1.4,
            small_per_rr=2, small_amp=0.15,
        )

        idxs, times, info = detect_beats(sig, fs, method='auto')

        # Must report the cluster filter in diagnostics
        assert 'amplitude_cluster' in info
        cinfo = info['amplitude_cluster']
        assert cinfo['cluster_filter'] == 'applied', (
            f"expected filter to fire; got {cinfo}"
        )
        assert cinfo['n_rejected'] >= 10, (
            f"expected filter to reject many peaks; got {cinfo}"
        )
        assert cinfo['max_gap_ratio'] >= 3.0

        # Final count should be close to the number of big spikes (allow
        # ±2 for boundary effects at the edges of the recording).
        n_big = len(expected_big)
        assert abs(len(idxs) - n_big) <= 2, (
            f"expected ≈{n_big} big spikes; got {len(idxs)}"
        )

        # RR must be extremely regular on the big-cluster signal.
        if len(times) > 1:
            rr = np.diff(times) * 1000
            cv = np.std(rr) / np.mean(rr)
            assert cv < 0.05, f"CV(RR) too high: {cv:.3f}"

    def test_detected_peaks_align_with_big_cluster(self):
        """The detected peaks must be near the injected big-spike
        locations, not the bumpettini."""
        from cardiac_fp_analyzer.beat_detection import detect_beats

        fs = 2000.0
        sig, expected_big = make_bimodal_signal(fs=fs, duration_s=60.0)
        idxs, _times, _info = detect_beats(sig, fs, method='auto')

        tol = int(0.05 * fs)  # ±50 ms
        matched = 0
        for exp in expected_big:
            if np.any(np.abs(idxs - exp) <= tol):
                matched += 1
        match_rate = matched / len(expected_big)
        assert match_rate > 0.85, (
            f"only {matched}/{len(expected_big)} big spikes matched "
            f"(rate={match_rate:.2%})"
        )


# ═══════════════════════════════════════════════════════════════════════
#   2. Filter is a no-op on unimodal signals
# ═══════════════════════════════════════════════════════════════════════

class TestUnimodalNoOp:
    """Clean signals must pass through the filter unchanged."""

    def test_unimodal_signal_reports_unimodal(self):
        from cardiac_fp_analyzer.beat_detection import detect_beats

        fs = 2000.0
        sig, _ = make_unimodal_signal(fs=fs, duration_s=10.0, rr_s=0.8)

        idxs, _times, info = detect_beats(sig, fs, method='auto')

        cinfo = info.get('amplitude_cluster', {})
        # Either unimodal, too-few-peaks, or the filter decided the
        # dominant cluster is too small — in all cases, the filter
        # must NOT have rejected a large fraction of the peaks.
        assert cinfo.get('cluster_filter') in (
            'unimodal', 'too_few_peaks', 'dominant_too_small',
            'disabled', 'zero_amp_present',
        ), f"unexpected cluster_filter state: {cinfo}"
        # And the peak count must be reasonable (~10–13 for 10 s @ 0.8 s RR).
        assert 8 <= len(idxs) <= 15, f"unexpected count {len(idxs)}"


# ═══════════════════════════════════════════════════════════════════════
#   3. Direct unit tests on _reject_amplitude_cluster
# ═══════════════════════════════════════════════════════════════════════

class TestRejectAmplitudeClusterUnit:

    def test_dominant_too_small_guard(self):
        """One giant artefact spike + many small ones → filter must
        refuse to eat the whole recording."""
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(12.0 * fs)  # room for 10 peaks at 1 Hz + buffer
        data = np.zeros(n)
        # 10 small peaks + 1 huge artefact (distinct location)
        small_idxs = np.array([int(i * fs) for i in range(1, 11)])
        big_idx = int(11.5 * fs)  # past the small peaks
        for i in small_idxs:
            data[i] = 0.1
        data[big_idx] = 5.0
        bi = np.sort(np.concatenate([small_idxs, [big_idx]]))

        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        # Gap ratio is massive, but the dominant cluster has only 1
        # peak — below the default min_dominant=3 → filter must NOT fire.
        assert info['cluster_filter'] == 'dominant_too_small'
        assert len(bi_out) == len(bi)

    def test_unimodal_skipped_gracefully(self):
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(12.0 * fs)
        data = np.zeros(n)
        bi = np.array([int(i * fs) for i in range(1, 11)])
        for i in bi:
            data[i] = 0.1 + 0.01 * (i / n)  # tiny gradient, not a cluster
        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['cluster_filter'] == 'unimodal'
        assert len(bi_out) == len(bi)

    def test_too_few_peaks_skipped(self):
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        data = np.ones(4000) * 0.1
        bi = np.array([500, 1500, 2500])  # only 3 peaks
        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['cluster_filter'] == 'too_few_peaks'
        assert len(bi_out) == len(bi)

    def test_bimodal_two_clusters_applied(self):
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(10.0 * fs)
        data = np.zeros(n)
        # 5 big peaks + 8 small peaks
        big_idxs = np.array([1000, 3000, 5000, 7000, 9000])
        small_idxs = np.array([500, 1500, 2500, 3500, 4500,
                               5500, 6500, 7500])
        for i in big_idxs:
            data[i] = 1.4
        for i in small_idxs:
            data[i] = 0.15
        bi = np.sort(np.concatenate([big_idxs, small_idxs]))

        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['cluster_filter'] == 'applied'
        assert info['n_kept'] == 5
        assert info['n_rejected'] == 8
        assert set(bi_out) == set(big_idxs)

    def test_polarity_agnostic(self):
        """Flipping the sign of the signal must not change the filter
        behaviour (uses max |x|)."""
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(12.0 * fs)
        data = np.zeros(n)
        # 5 big + 8 small interlaced (same as test_bimodal_two_clusters_applied
        # topology — safely in the "apply" regime: ratio 8/5=1.6 avoids the
        # alternans band, and smalls are between bigs → interlaced).
        big_idxs = np.array([1000, 3000, 5000, 7000, 9000])
        small_idxs = np.array([1500, 2500, 3500, 4500,
                               5500, 6500, 7500, 8500])
        for i in big_idxs:
            data[i] = 1.4
        for i in small_idxs:
            data[i] = 0.15
        bi = np.sort(np.concatenate([big_idxs, small_idxs]))

        _, info_pos = _reject_amplitude_cluster(data, fs, bi)
        _, info_neg = _reject_amplitude_cluster(-data, fs, bi)
        assert info_pos['cluster_filter'] == 'applied'
        assert info_pos['cluster_filter'] == info_neg['cluster_filter']
        assert info_pos['n_kept'] == info_neg['n_kept']
        assert info_pos['n_rejected'] == info_neg['n_rejected']


# ═══════════════════════════════════════════════════════════════════════
#   4. Config toggle disables the filter
# ═══════════════════════════════════════════════════════════════════════

class TestConfigToggle:

    def test_disable_restores_old_behaviour(self):
        """With enable_amplitude_cluster_filter=False the filter reports
        'disabled' and leaves the beat set untouched."""
        from cardiac_fp_analyzer.beat_detection import detect_beats
        from cardiac_fp_analyzer.config import BeatDetectionConfig

        fs = 2000.0
        sig, _ = make_bimodal_signal(fs=fs, duration_s=60.0)

        cfg = BeatDetectionConfig()
        cfg.enable_amplitude_cluster_filter = False
        idxs, _times, info = detect_beats(sig, fs, cfg=cfg)

        cinfo = info['amplitude_cluster']
        assert cinfo['cluster_filter'] == 'disabled'
        # With the filter off we expect many more peaks (the 33 false
        # positives survive).
        assert len(idxs) > 20

    def test_custom_gap_ratio_makes_filter_stricter(self):
        """Raising cluster_gap_ratio above the actual gap makes the
        filter see the signal as unimodal."""
        from cardiac_fp_analyzer.beat_detection import detect_beats
        from cardiac_fp_analyzer.config import BeatDetectionConfig

        fs = 2000.0
        sig, _ = make_bimodal_signal(fs=fs, duration_s=60.0,
                                     big_amp=1.4, small_amp=0.15)
        cfg = BeatDetectionConfig()
        cfg.cluster_gap_ratio = 50.0  # much higher than the ~3.6× gap
        idxs, _times, info = detect_beats(sig, fs, cfg=cfg)
        cinfo = info['amplitude_cluster']
        assert cinfo['cluster_filter'] == 'unimodal'
        assert len(idxs) > 20  # false positives survive


# ═══════════════════════════════════════════════════════════════════════
#   5. Diagnostics round-trip
# ═══════════════════════════════════════════════════════════════════════

class TestDiagnostics:

    def test_applied_info_has_all_keys(self):
        from cardiac_fp_analyzer.beat_detection import detect_beats

        fs = 2000.0
        sig, _ = make_bimodal_signal(fs=fs, duration_s=60.0)
        _idxs, _times, info = detect_beats(sig, fs, method='auto')
        c = info['amplitude_cluster']
        assert c['cluster_filter'] == 'applied'
        for k in ('n_input', 'n_kept', 'n_rejected', 'max_gap_ratio',
                  'gap_low_v', 'gap_high_v', 'threshold_v', 'n_dominant',
                  'n_low_cluster', 'n_high_cluster', 'interlaced_frac'):
            assert k in c, f"missing diagnostic key {k!r} in {c}"

    def test_threshold_between_gap_boundaries(self):
        """Returned threshold must fall strictly between the low and
        high amplitudes flanking the gap."""
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(10.0 * fs)
        data = np.zeros(n)
        big_idxs = np.array([1000, 3000, 5000, 7000])
        small_idxs = np.array([500, 1500, 2500, 3500])
        for i in big_idxs:
            data[i] = 2.0
        for i in small_idxs:
            data[i] = 0.1
        bi = np.sort(np.concatenate([big_idxs, small_idxs]))
        _, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['gap_low_v'] < info['threshold_v'] < info['gap_high_v']


# ═══════════════════════════════════════════════════════════════════════
#   6. Safeguards — filter must abort on these patterns
# ═══════════════════════════════════════════════════════════════════════

class TestSafeguardAlternans:
    """Severe amplitude alternans (ratio >3× between odd and even beats)
    must NOT be filtered — the "low" cluster is the small beat of the
    alternance, not an artefact. Safeguard: n_low ≈ n_high triggers
    ``aborted_alternans``."""

    def test_severe_alternans_aborts(self):
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(22.0 * fs)
        data = np.zeros(n)
        # 20 regular beats, big/small alternating (ratio 4×)
        indices = np.arange(1, 21) * int(fs)
        for k, idx in enumerate(indices):
            data[idx] = 1.0 if k % 2 == 0 else 0.25
        bi_out, info = _reject_amplitude_cluster(data, fs, indices)
        assert info['cluster_filter'] == 'aborted_alternans'
        assert len(bi_out) == len(indices)
        assert 0.85 <= info['low_over_high_ratio'] <= 1.15

    def test_alternans_boundary_still_applies(self):
        """If n_low / n_high just outside the alternans band (e.g. 1.3:1),
        the filter must still be able to apply."""
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        # 5 big interlaced with 7 small (ratio 7/5 = 1.4, outside band)
        big_idxs = np.array([2000, 4000, 6000, 8000, 10000])
        small_idxs = np.array([1000, 3000, 5000, 5500, 7000, 8500, 9500])
        n = 12000
        data = np.zeros(n)
        for i in big_idxs:
            data[i] = 1.5
        for i in small_idxs:
            data[i] = 0.2
        bi = np.sort(np.concatenate([big_idxs, small_idxs]))
        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['cluster_filter'] == 'applied', info


class TestSafeguardTopology:
    """Low-amplitude peaks that are TEMPORALLY contiguous at the edges
    (weak beats at start/end, fade-out) must NOT be filtered — they are
    real beats in a low-SNR region, not interspersed artefacts.
    Safeguard: interlaced_frac < 0.7 triggers ``aborted_topology``."""

    def test_weak_beats_at_start_aborts(self):
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(25.0 * fs)
        data = np.zeros(n)
        # 3 weak beats at start (amp 0.03) + 17 normal beats (amp 0.15)
        times = [1.0 + i * 1.0 for i in range(20)]
        for k, t in enumerate(times):
            idx = int(t * fs)
            data[idx] = 0.03 if k < 3 else 0.15
        bi = np.array([int(t * fs) for t in times])
        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['cluster_filter'] == 'aborted_topology'
        assert info['interlaced_frac'] == 0.0
        assert len(bi_out) == len(bi)

    def test_weak_beats_at_end_aborts(self):
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        n = int(25.0 * fs)
        data = np.zeros(n)
        times = [1.0 + i * 1.0 for i in range(20)]
        for k, t in enumerate(times):
            idx = int(t * fs)
            data[idx] = 0.03 if k >= 17 else 0.15  # weak at end
        bi = np.array([int(t * fs) for t in times])
        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        assert info['cluster_filter'] == 'aborted_topology'
        assert len(bi_out) == len(bi)

    def test_single_weak_beat_in_middle_still_applies(self):
        """A lone weak peak surrounded by many strong ones is likely an
        actual artefact/glitch — the filter may remove it. This pins
        that the topology safeguard does NOT over-fire."""
        from cardiac_fp_analyzer.beat_detection import _reject_amplitude_cluster

        fs = 2000.0
        # 10 strong peaks at 1 s intervals, 1 weak peak inserted in the middle
        big_idxs = np.array([int(i * fs) for i in range(1, 11)])
        weak_idx = int(5.5 * fs)  # between 5th and 6th big peak
        n = int(12.0 * fs)
        data = np.zeros(n)
        for i in big_idxs:
            data[i] = 1.5
        data[weak_idx] = 0.2
        bi = np.sort(np.concatenate([big_idxs, [weak_idx]]))
        bi_out, info = _reject_amplitude_cluster(data, fs, bi)
        # Single low peak, surrounded by bigs → interlaced = 1/1 = 1.0
        # But n_dominant = 10, n_low = 1 → ratio 0.1 → not alternans
        # → filter applies, removes 1 peak
        assert info['cluster_filter'] == 'applied', info
        assert info['n_kept'] == 10
        assert weak_idx not in bi_out

    def test_weak_beats_end_to_end_via_detect_beats(self):
        """Full pipeline: 3 weak beats at start must survive through
        detect_beats (i.e. the cluster filter must abort, not silently
        eat the low cluster)."""
        from cardiac_fp_analyzer.beat_detection import detect_beats

        fs = 2000.0
        n = int(21.0 * fs)
        data = np.zeros(n)
        from tests.test_amplitude_cluster_filter import _biphasic_spike
        for k, t in enumerate([1.0 + i * 1.0 for i in range(20)]):
            idx = int(t * fs)
            amp = 0.03 if k < 3 else 0.15
            data += _biphasic_spike(n, idx, amp, int(0.005 * fs))
        data += np.random.default_rng(0).standard_normal(n) * 0.001
        idxs, _, info = detect_beats(data, fs, method='auto')
        ci = info.get('amplitude_cluster', {})
        # The filter must NOT delete the 3 weak beats — either abort or
        # leave all peaks through.
        assert ci.get('cluster_filter') in (
            'aborted_topology', 'aborted_alternans', 'unimodal',
            'dominant_too_small', 'too_few_peaks',
        ), f"filter applied when it should have aborted: {ci}"


# ═══════════════════════════════════════════════════════════════════════
#   7. Does not interfere with slow-rhythm detection
# ═══════════════════════════════════════════════════════════════════════

class TestSlowRhythmCompatibility:
    """Slow rhythms (big_rr ~ 3.6 s) are well within the detector's
    enabled range post-Fix #2 — verify the new filter doesn't
    accidentally reject legitimate slow rhythms."""

    def test_slow_regular_rhythm_preserved(self):
        from cardiac_fp_analyzer.beat_detection import detect_beats

        fs = 2000.0
        sig, expected = make_bimodal_signal(
            fs=fs, duration_s=60.0, big_rr_s=3.6,
            big_amp=1.4, small_amp=0.15,
        )
        idxs, times, _info = detect_beats(sig, fs, method='auto')
        # At least half of the expected big spikes must be detected.
        tol = int(0.05 * fs)
        matched = sum(
            1 for e in expected if np.any(np.abs(idxs - e) <= tol)
        )
        assert matched >= len(expected) * 0.85
        # And the RR should be close to 3.6 s.
        if len(times) > 1:
            rr = np.diff(times)
            assert 3.3 < np.median(rr) < 3.9, (
                f"slow rhythm not preserved: median RR = {np.median(rr):.2f} s"
            )
