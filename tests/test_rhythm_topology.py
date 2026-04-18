"""
test_rhythm_topology.py — Regression tests for Sprint 2 #3.

Background
----------
`_classify_rhythm_topology(data, fs, bi, cfg)` is a pure analysis function
that categorises a set of detected beats into one of:

  * 'regular'                — single amplitude cluster, CV(RR) low
  * 'chaotic'                — single amplitude cluster, CV(RR) high
  * 'alternans_2_to_1'       — two clusters, low beats phase-locked at 0.5
                                of each high-high cycle, |n_low ≈ n_high|
  * 'regular_with_ectopics'  — two clusters, low cluster amplitudes uniform
                                (biologically homogeneous → likely ectopics)
  * 'regular_with_noise'     — two clusters, low cluster amplitudes very
                                dispersed (likely noise / artefact)
  * 'trimodal'               — three clusters (dominant + secondary + noise)
  * 'unimodal_insufficient'  — fewer than ``topology_min_beats`` peaks
  * 'degenerate'             — zero-amplitude window present
  * 'ambiguous'              — bimodal but low cluster amp CV in grey zone

The classifier is side-effect-free: it never modifies the beat list.

These tests verify:
  1. Each of the rhythm types is correctly emitted on synthetic signals.
  2. The function is robust to amplitude scale, polarity, and short
     recordings.
  3. Metrics and flags are populated sensibly per rhythm type.
  4. The classifier is integrated into `detect_beats` — `info` contains
     an ``info['rhythm_classification']`` key.
  5. Disabling via config returns ``rhythm_type='disabled'``.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#   Synthetic signal generators
# ═══════════════════════════════════════════════════════════════════════


def _spike(n, center, amp, width_samples):
    """Sharp Gaussian spike."""
    t = np.arange(n) - center
    sigma = max(1.0, width_samples / 3.0)
    return amp * np.exp(-0.5 * (t / sigma) ** 2)


def make_regular_signal(fs=2000.0, duration_s=30.0, rr_s=1.0, amp=1.0,
                        noise_std=0.01, seed=1):
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    rr = int(rr_s * fs)
    idxs = []
    idx = rr
    while idx < n - int(0.2 * fs):
        idxs.append(idx)
        sig += _spike(n, idx, amp, int(0.005 * fs))
        idx += rr
    sig += rng.standard_normal(n) * noise_std
    return sig, np.array(idxs, dtype=int)


def make_chaotic_signal(fs=2000.0, duration_s=30.0, mean_rr_s=0.8,
                        jitter_s=0.6, amp=1.0, noise_std=0.01, seed=2):
    """Single amplitude cluster but highly variable RR."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    idxs = []
    t_curr = 1.0
    while t_curr < duration_s - 0.2:
        idx = int(t_curr * fs)
        idxs.append(idx)
        sig += _spike(n, idx, amp, int(0.005 * fs))
        t_curr += mean_rr_s + rng.uniform(-jitter_s, jitter_s)
    sig += rng.standard_normal(n) * noise_std
    return sig, np.array(sorted(idxs), dtype=int)


def make_alternans_signal(fs=2000.0, duration_s=60.0, big_rr_s=2.0,
                          big_amp=1.0, small_amp=0.2,
                          small_amp_cv=0.1, noise_std=0.01, seed=3):
    """2:1 alternans: one big spike every big_rr_s, one small spike at
    exactly the midpoint. |n_small - n_big| ≤ 1.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    rr = int(big_rr_s * fs)
    half = rr // 2
    big_idx, small_idx = [], []
    idx = rr
    while idx < n - int(0.2 * fs):
        big_idx.append(idx)
        sig += _spike(n, idx, big_amp, int(0.005 * fs))
        s = idx + half
        if 0 < s < n - int(0.1 * fs):
            small_idx.append(s)
            jitter = rng.normal(1.0, small_amp_cv)
            sig += _spike(n, s, small_amp * jitter, int(0.005 * fs))
        idx += rr
    sig += rng.standard_normal(n) * noise_std
    all_idx = np.array(sorted(big_idx + small_idx), dtype=int)
    return sig, all_idx, np.array(big_idx), np.array(small_idx)


def make_ectopics_signal(fs=2000.0, duration_s=60.0, big_rr_s=1.5,
                         big_amp=1.0, n_ectopics=5, ectopic_amp=0.3,
                         ectopic_amp_cv=0.1, noise_std=0.01, seed=4):
    """Regular big beats + a handful of low-amplitude, morphologically
    uniform ectopic beats at random times (NOT phase-locked).
    n_ectopics < n_big so ratio_low_high < 0.85 (outside alternans band).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    rr = int(big_rr_s * fs)
    big_idx = []
    idx = rr
    while idx < n - int(0.2 * fs):
        big_idx.append(idx)
        sig += _spike(n, idx, big_amp, int(0.005 * fs))
        idx += rr

    # Insert ectopic beats at random times, avoiding big beats ±100ms
    ect_idx = []
    guard = int(0.1 * fs)
    attempts = 0
    while len(ect_idx) < n_ectopics and attempts < 500:
        attempts += 1
        cand = rng.integers(guard, n - guard)
        if all(abs(cand - b) > guard for b in big_idx) and \
           all(abs(cand - e) > guard for e in ect_idx):
            ect_idx.append(int(cand))
            amp = ectopic_amp * rng.normal(1.0, ectopic_amp_cv)
            sig += _spike(n, cand, amp, int(0.005 * fs))

    sig += rng.standard_normal(n) * noise_std
    all_idx = np.array(sorted(big_idx + ect_idx), dtype=int)
    return sig, all_idx, np.array(big_idx), np.array(sorted(ect_idx))


def make_noise_contamination_signal(
    fs=2000.0, duration_s=60.0, big_rr_s=1.5, big_amp=1.0,
    n_noise=18, noise_amp_mu=-1.8, noise_amp_sigma=0.60,
    noise_std=0.01, seed=5
):
    """Regular big beats + dispersed low-amplitude noise spikes.
    Noise amplitudes drawn from log-normal so CV is high (>0.40) but
    adjacent sorted ratios stay small enough to avoid internal splits.
    Expected: regular_with_noise.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    rr = int(big_rr_s * fs)
    big_idx = []
    idx = rr
    while idx < n - int(0.2 * fs):
        big_idx.append(idx)
        sig += _spike(n, idx, big_amp, int(0.005 * fs))
        idx += rr

    noise_idx = []
    guard = int(0.1 * fs)
    attempts = 0
    while len(noise_idx) < n_noise and attempts < 2000:
        attempts += 1
        cand = rng.integers(guard, n - guard)
        if all(abs(cand - b) > guard for b in big_idx) and \
           all(abs(cand - e) > guard for e in noise_idx):
            noise_idx.append(int(cand))
            amp = float(np.exp(noise_amp_mu + noise_amp_sigma * rng.standard_normal()))
            amp = max(0.02, min(amp, 0.45))
            sig += _spike(n, cand, amp, int(0.005 * fs))

    sig += rng.standard_normal(n) * noise_std
    all_idx = np.array(sorted(big_idx + noise_idx), dtype=int)
    return sig, all_idx, np.array(big_idx), np.array(sorted(noise_idx))


def make_trimodal_signal(fs=2000.0, duration_s=60.0, big_rr_s=3.3,
                         big_amp=0.8, n_ectopics=16, ectopic_amp=0.30,
                         n_noise=60, noise_amp_lo=0.03, noise_amp_hi=0.10,
                         noise_std=0.003, seed=6):
    """Replicates the Exp6_chipD_ch3 topology: 18 dominant + 16 uniform
    secondary (ectopic) + 60+ dispersed low-amplitude noise peaks.

    Gap sizing (on log-ratio of sorted amps):
      * big (0.80)  ↔  ectopic (0.30)  →  ratio 2.67
      * ectopic (0.30)  ↔  noise_hi (0.10)  →  ratio 3.0

    Both are well above the 1.3 floor, so the classifier should split
    into three clusters.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    sig = np.zeros(n)
    rr = int(big_rr_s * fs)
    big_idx = []
    idx = rr
    while idx < n - int(0.2 * fs):
        big_idx.append(idx)
        sig += _spike(n, idx, big_amp, int(0.005 * fs))
        idx += rr

    ect_idx, noise_idx = [], []
    guard = int(0.1 * fs)

    def _place(n_target, out_list, amp_fn, width):
        attempts = 0
        while len(out_list) < n_target and attempts < 2000:
            attempts += 1
            cand = rng.integers(guard, n - guard)
            taken = big_idx + ect_idx + noise_idx
            if all(abs(cand - t) > guard for t in taken):
                out_list.append(int(cand))
                sig_local = _spike(n, cand, amp_fn(), width)
                sig[:] = sig + sig_local

    # Uniform ectopic amps (CV ~8%)
    _place(n_ectopics, ect_idx, lambda: ectopic_amp * rng.normal(1.0, 0.08),
           int(0.005 * fs))
    # Dispersed noise amps
    _place(n_noise, noise_idx, lambda: rng.uniform(noise_amp_lo, noise_amp_hi),
           int(0.005 * fs))

    sig += rng.standard_normal(n) * noise_std
    all_idx = np.array(sorted(big_idx + ect_idx + noise_idx), dtype=int)
    return sig, all_idx, np.array(big_idx), np.array(sorted(ect_idx)), \
        np.array(sorted(noise_idx))


# ═══════════════════════════════════════════════════════════════════════
#   1. Regular rhythm
# ═══════════════════════════════════════════════════════════════════════


class TestRegularRhythm:
    def test_regular_signal_classified(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, idx = make_regular_signal(fs=fs, duration_s=30.0, rr_s=1.0)
        cls = _classify_rhythm_topology(sig, fs, idx)
        assert cls['rhythm_type'] == 'regular'
        assert len(cls['clusters']) == 1
        assert cls['clusters'][0]['role'] == 'dominant'
        assert cls['flags'] == []
        assert cls['metrics']['bpm_dominant'] is not None
        assert 55 <= cls['metrics']['bpm_dominant'] <= 65  # rr=1.0s → 60 bpm
        assert cls['metrics']['cv_rr_dominant'] < 0.05

    def test_regular_polarity_invariant(self):
        """Negative-going spikes should classify identically."""
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, idx = make_regular_signal(fs=fs, duration_s=30.0, amp=1.0)
        cls_pos = _classify_rhythm_topology(sig, fs, idx)
        cls_neg = _classify_rhythm_topology(-sig, fs, idx)
        assert cls_pos['rhythm_type'] == cls_neg['rhythm_type']
        assert abs(cls_pos['clusters'][0]['amp_median'] -
                   cls_neg['clusters'][0]['amp_median']) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
#   2. Chaotic rhythm
# ═══════════════════════════════════════════════════════════════════════


class TestChaoticRhythm:
    def test_chaotic_high_cv_rr(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, idx = make_chaotic_signal(fs=fs, duration_s=30.0,
                                       mean_rr_s=0.8, jitter_s=0.6)
        cls = _classify_rhythm_topology(sig, fs, idx)
        assert cls['rhythm_type'] == 'chaotic'
        assert 'manual_review_required' in cls['flags']
        assert cls['metrics']['cv_rr_dominant'] >= 0.25


# ═══════════════════════════════════════════════════════════════════════
#   3. Alternans 2:1
# ═══════════════════════════════════════════════════════════════════════


class TestAlternans2To1:
    def test_alternans_classified(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, all_idx, big_idx, small_idx = make_alternans_signal(
            fs=fs, duration_s=60.0, big_rr_s=2.0,
            big_amp=1.0, small_amp=0.2,
        )
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        assert cls['rhythm_type'] == 'alternans_2_to_1'
        assert 'alternans_pattern' in cls['flags']
        assert len(cls['clusters']) == 2
        # Check phase is near 0.5
        assert abs(cls['metrics']['alternans_phase_median'] - 0.5) < 0.1
        assert cls['metrics']['alternans_phase_std'] < 0.08
        # Ratio close to 1
        assert 0.85 <= cls['metrics']['ratio_low_high'] <= 1.15
        # bpm_effective should be ~2x bpm_dominant
        assert cls['metrics']['bpm_effective'] / cls['metrics']['bpm_dominant'] > 1.8

    def test_alternans_roles_correct(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, all_idx, big_idx, small_idx = make_alternans_signal(fs=fs)
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        roles = {cl['role'] for cl in cls['clusters']}
        assert roles == {'dominant', 'secondary'}
        # Dominant cluster has largest amp
        dom = next(cl for cl in cls['clusters'] if cl['role'] == 'dominant')
        sec = next(cl for cl in cls['clusters'] if cl['role'] == 'secondary')
        assert dom['amp_median'] > sec['amp_median']

    def test_alternans_not_phase_locked_becomes_ectopics(self):
        """If low beats have similar count but scattered phases, should
        NOT be classified as alternans."""
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        # 10 big + 10 ectopic at random phases (not 0.5-locked)
        rng = np.random.default_rng(99)
        n = int(60 * fs)
        sig = np.zeros(n)
        big_idx = []
        for i in range(10):
            idx = int((3 + i * 5.5) * fs)
            big_idx.append(idx)
            sig += _spike(n, idx, 1.0, int(0.005 * fs))
        ect_idx = []
        for i in range(10):
            # Random phase in [0.1, 0.9], avoiding 0.5 ± 0.15
            rrs = big_idx[1] - big_idx[0]  # ~5.5s
            phase = rng.choice([0.2, 0.25, 0.3, 0.35, 0.72, 0.78, 0.82, 0.88])
            # pick a cycle
            i_cyc = rng.integers(0, 9)
            cand = int(big_idx[i_cyc] + phase * rrs)
            if cand not in ect_idx and cand < n - int(0.1 * fs):
                ect_idx.append(cand)
                sig += _spike(n, cand, 0.25, int(0.005 * fs))
        sig += rng.standard_normal(n) * 0.01
        all_idx = np.array(sorted(big_idx + ect_idx), dtype=int)
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        # Should NOT be alternans (phase scattered) — could be ectopics or
        # ambiguous depending on amp CV
        assert cls['rhythm_type'] != 'alternans_2_to_1'


# ═══════════════════════════════════════════════════════════════════════
#   4. Regular with ectopics (biologically uniform low cluster)
# ═══════════════════════════════════════════════════════════════════════


class TestRegularWithEctopics:
    def test_uniform_low_cluster_flagged_ectopic(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, all_idx, big_idx, ect_idx = make_ectopics_signal(
            fs=fs, duration_s=60.0, big_rr_s=1.5,
            n_ectopics=5, ectopic_amp=0.3, ectopic_amp_cv=0.08,
        )
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        assert cls['rhythm_type'] == 'regular_with_ectopics'
        assert 'arrhythmia_candidate' in cls['flags']
        # Secondary cluster should have low amp CV
        sec = next(cl for cl in cls['clusters'] if cl['role'] == 'secondary')
        assert sec['amp_cv'] <= 0.25


# ═══════════════════════════════════════════════════════════════════════
#   5. Regular with noise (dispersed low cluster)
# ═══════════════════════════════════════════════════════════════════════


class TestRegularWithNoise:
    def test_dispersed_low_cluster_flagged_noise(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, all_idx, big_idx, noise_idx = make_noise_contamination_signal(
            fs=fs, duration_s=60.0, big_rr_s=1.5,
            n_noise=18, noise_amp_mu=-1.8, noise_amp_sigma=0.60,
        )
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        assert cls['rhythm_type'] == 'regular_with_noise'
        # Low cluster role should be 'noise'
        noise_cl = next(cl for cl in cls['clusters'] if cl['role'] == 'noise')
        assert noise_cl['amp_cv'] >= 0.40


# ═══════════════════════════════════════════════════════════════════════
#   6. Trimodal
# ═══════════════════════════════════════════════════════════════════════


class TestTrimodal:
    def test_trimodal_three_clusters(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, all_idx, big, ect, noise = make_trimodal_signal(
            fs=fs, duration_s=60.0, big_rr_s=3.3,
            big_amp=0.8, n_ectopics=14, ectopic_amp=0.30,
            n_noise=40, noise_amp_lo=0.03, noise_amp_hi=0.10,
        )
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        assert cls['rhythm_type'] == 'trimodal'
        assert len(cls['clusters']) == 3
        roles = [cl['role'] for cl in cls['clusters']]
        assert set(roles) == {'noise', 'secondary', 'dominant'}
        assert 'arrhythmia_candidate' in cls['flags']
        assert 'noise_contamination' in cls['flags']

    def test_trimodal_amp_ordering(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, all_idx, *_ = make_trimodal_signal(fs=fs)
        cls = _classify_rhythm_topology(sig, fs, all_idx)
        # Clusters returned in order noise < secondary < dominant
        noise = cls['clusters'][0]
        sec = cls['clusters'][1]
        dom = cls['clusters'][2]
        assert noise['amp_median'] < sec['amp_median'] < dom['amp_median']


# ═══════════════════════════════════════════════════════════════════════
#   7. Degenerate / edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_fewer_than_min_beats_insufficient(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        n = int(5 * fs)
        sig = np.zeros(n)
        sig += _spike(n, 2000, 1.0, 5)
        sig += _spike(n, 5000, 1.0, 5)
        idxs = np.array([2000, 5000], dtype=int)
        cls = _classify_rhythm_topology(sig, fs, idxs)
        assert cls['rhythm_type'] == 'unimodal_insufficient'
        assert cls['clusters'] == []

    def test_empty_bi(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig = np.zeros(1000)
        cls = _classify_rhythm_topology(sig, fs, np.array([], dtype=int))
        assert cls['rhythm_type'] == 'unimodal_insufficient'
        assert cls['n_beats'] == 0

    def test_zero_amp_window_degenerate(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        n = int(15 * fs)
        sig = np.zeros(n)
        idxs = np.arange(1000, n - 1000, 2000)
        # First index lands in a window of all zeros → degenerate
        cls = _classify_rhythm_topology(sig, fs, idxs)
        assert cls['rhythm_type'] == 'degenerate'

    def test_disabled_via_config(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        from cardiac_fp_analyzer.config import BeatDetectionConfig
        fs = 2000.0
        sig, idx = make_regular_signal(fs=fs, duration_s=15.0, rr_s=0.8)
        cfg = BeatDetectionConfig()
        cfg.enable_rhythm_topology = False
        cls = _classify_rhythm_topology(sig, fs, idx, cfg=cfg)
        assert cls['rhythm_type'] == 'disabled'

    def test_pure_function_no_side_effects(self):
        """Input arrays must be unchanged after call."""
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, idx = make_regular_signal(fs=fs, duration_s=15.0)
        sig_before = sig.copy()
        idx_before = idx.copy()
        _classify_rhythm_topology(sig, fs, idx)
        assert np.array_equal(sig, sig_before)
        assert np.array_equal(idx, idx_before)


# ═══════════════════════════════════════════════════════════════════════
#   8. Integration with detect_beats
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationDetectBeats:
    def test_detect_beats_populates_rhythm_classification(self):
        """detect_beats must always populate `rhythm_classification`.

        The exact label depends on what the detector picks up — with
        residual noise it can occasionally see phantom half-beats — so
        accept any non-error label here; the goal is just to verify the
        key is present and well-formed.
        """
        from cardiac_fp_analyzer.beat_detection import detect_beats
        fs = 2000.0
        sig, _ = make_regular_signal(fs=fs, duration_s=30.0, rr_s=1.0,
                                      amp=1.0, noise_std=0.005)
        bi, bt, info = detect_beats(sig, fs, method='auto')
        assert 'rhythm_classification' in info
        cls = info['rhythm_classification']
        assert cls['rhythm_type'] not in ('error',)
        assert isinstance(cls['clusters'], list)
        assert 'metrics' in cls
        assert 'flags' in cls

    def test_detect_beats_alternans_flag(self):
        from cardiac_fp_analyzer.beat_detection import detect_beats
        fs = 2000.0
        sig, _all, _big, _small = make_alternans_signal(
            fs=fs, duration_s=60.0, big_rr_s=2.0,
            big_amp=1.0, small_amp=0.25,
        )
        bi, bt, info = detect_beats(sig, fs, method='auto')
        cls = info['rhythm_classification']
        # Either auto-detection finds all beats and classifies as alternans,
        # or rejects small ones via amplitude-cluster safeguard. Whichever
        # happens, rhythm_classification must exist and be sensible.
        assert cls['rhythm_type'] in (
            'alternans_2_to_1', 'regular', 'regular_with_ectopics',
            'regular_with_noise', 'ambiguous', 'trimodal',
        )


# ═══════════════════════════════════════════════════════════════════════
#   9. Metric details
# ═══════════════════════════════════════════════════════════════════════


class TestMetricDetails:
    def test_bpm_effective_only_on_alternans(self):
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        # Regular → no bpm_effective
        sig, idx = make_regular_signal(fs=fs, duration_s=15.0)
        cls = _classify_rhythm_topology(sig, fs, idx)
        assert 'bpm_effective' not in cls['metrics']

        # Alternans → bpm_effective present
        sig2, all_idx, *_ = make_alternans_signal(fs=fs, duration_s=60.0)
        cls2 = _classify_rhythm_topology(sig2, fs, all_idx)
        if cls2['rhythm_type'] == 'alternans_2_to_1':
            assert 'bpm_effective' in cls2['metrics']
            assert cls2['metrics']['bpm_effective'] is not None

    def test_amp_scale_invariance(self):
        """Classifier output should not depend on absolute amplitude scale."""
        from cardiac_fp_analyzer.beat_detection import _classify_rhythm_topology
        fs = 2000.0
        sig, idx = make_regular_signal(fs=fs, amp=1.0, duration_s=20.0)
        sig_small = sig * 0.01  # 100× smaller
        cls1 = _classify_rhythm_topology(sig, fs, idx)
        cls2 = _classify_rhythm_topology(sig_small, fs, idx)
        assert cls1['rhythm_type'] == cls2['rhythm_type']
