"""
test_bradycardia_fpd_robustness.py — Regression guard for the
bradycardia-driven FPD over-shoot discovered on
``Exp6_chipD_ch1_chipB_ch1_baseline.csv`` (v3.3.0 → v3.3.1).

Context
-------
On strongly bradycardic signals (RR > 2.5 s, e.g. quiescent chambers or
dofetilide-overdosed microtissues) the adaptive minimum-FPD floor

    adaptive_min_fpd_ms = min_fpd_pct_rr × RR

could exceed the physiological upper bound of the hiPSC-CM T-wave
latency (~500 ms), pushing the search start past the real
repolarisation peak. Every beat then fell into the argmax fallback and
the pipeline reported a single, spurious FPD for a ~24 s RR-outlier
artefact (FPDcF 318.8 ± 0, QC grade F) because:

  1. The adaptive floor on RR ≈ 3 s became 600 ms, hiding the real
     T-wave at 300–500 ms.
  2. A 24 s dropout gap passed beat QC and the rhythm filter and
     contributed the only "valid" FPD.
  3. The summary reported the lone artefact as if it were the mean,
     with no indication that 6/7 beats had no measurable FPD.

Three fixes were introduced:

  * **Fix A-bis** — cap ``max_adaptive_min_fpd_ms`` (default 500 ms)
    clamps the adaptive contribution so it never dominates on slow
    rhythms. Tests in :class:`TestAdaptiveFloorCap`.
  * **Fix B** — RR sanity filter drops beats whose preceding or
    following RR interval exceeds ``max_rr_outlier_ratio × median``.
    Tests in :class:`TestRROutlierFilter`.
  * **Fix C** — ``min_valid_fpd_ratio`` gate on the summary flags
    recordings where too few beats produced a valid FPD. Tests in
    :class:`TestFPDReliabilityGate`.
"""
from __future__ import annotations

import numpy as np
import pytest

from cardiac_fp_analyzer.config import RepolarizationConfig
from cardiac_fp_analyzer.repolarization import (
    find_repolarization_on_template,
    find_repolarization_per_beat,
)

# ═══════════════════════════════════════════════════════════════════════
#   Synthetic signal helpers
# ═══════════════════════════════════════════════════════════════════════


def _build_bradycardic_template(fs=2000.0, duration_s=0.9, pre_ms=50,
                                t_wave_latency_ms=400, seed=0):
    """Build a synthetic FP template with:
      * sharp Gaussian spike at ``pre_ms``
      * Gaussian T-wave at ``t_wave_latency_ms`` after the spike (within
        the physiological 200–500 ms window)
    """
    n = int(fs * duration_s)
    pre = int(pre_ms / 1000 * fs)
    rng = np.random.default_rng(seed)
    tpl = rng.standard_normal(n) * 0.001
    x = np.arange(n) - pre
    tpl += 0.8 * np.exp(-0.5 * (x / 4.0) ** 2)  # spike
    t_idx = pre + int(t_wave_latency_ms / 1000 * fs)
    xt = np.arange(n) - t_idx
    tpl += 0.25 * np.exp(-0.5 * (xt / 80.0) ** 2)  # T-wave bump
    return tpl, pre, t_idx


def _build_per_beat_signal(fs=2000.0, total_s=1.0,
                            spike_s=0.3, t_wave_latency_ms=400,
                            seed=1):
    """Single-beat signal with a clearly visible T-wave."""
    n = int(fs * total_s)
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(n) * 0.0005
    spike_idx = int(spike_s * fs)
    x = np.arange(n) - spike_idx
    data += 0.8 * np.exp(-0.5 * (x / 4.0) ** 2)
    t_idx = spike_idx + int(t_wave_latency_ms / 1000 * fs)
    xt = np.arange(n) - t_idx
    data += 0.3 * np.exp(-0.5 * (xt / 80.0) ** 2)
    t_vec = np.arange(n) / fs
    return data, t_vec, spike_idx, t_idx


# ═══════════════════════════════════════════════════════════════════════
#   Fix A-bis — Cap on the adaptive floor
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptiveFloorCap:
    """Verify that ``max_adaptive_min_fpd_ms`` prevents the adaptive
    floor from overshooting the real T-wave on bradycardic signals."""

    def test_config_field_has_sensible_default(self):
        cfg = RepolarizationConfig()
        assert cfg.max_adaptive_min_fpd_ms == 350.0

    def test_template_bradycardia_recovers_t_wave(self):
        """RR = 3 s + T-wave at 400 ms. Without the cap the floor is
        0.20 × 3000 = 600 ms → T-wave at 400 ms is excluded. With the
        cap (350 ms) the T-wave becomes reachable again."""
        fs = 2000.0
        tpl, pre, t_idx = _build_bradycardic_template(
            fs=fs, duration_s=0.9, pre_ms=50, t_wave_latency_ms=400,
        )
        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 120.0
        cfg.min_fpd_pct_rr = 0.20
        cfg.max_adaptive_min_fpd_ms = 350.0  # default
        cfg.search_end_ms = 700.0
        cfg.enable_repol_gate = False  # test the bound, not the SNR gate

        fpd_samples, *_rest = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg, median_bp_s=3.0,  # bradycardia
        )
        assert fpd_samples is not None
        fpd_ms = fpd_samples / fs * 1000
        # T-wave latency = 400 ms; tangent method adds some overhead.
        # With cap active the floor is 350 ms and the T-wave is
        # reachable → FPD must land near the T-wave (350–550 ms band).
        assert 350 <= fpd_ms <= 550, f"FPD={fpd_ms:.0f} ms (expected 350–550)"

    def test_template_without_cap_misses_t_wave(self):
        """Baseline: with the cap disabled (pre-fix behaviour) the same
        bradycardic signal pushes the floor past the T-wave and the
        result is outside the physiological band (or None)."""
        fs = 2000.0
        tpl, pre, t_idx = _build_bradycardic_template(
            fs=fs, duration_s=0.9, pre_ms=50, t_wave_latency_ms=400,
        )
        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 120.0
        cfg.min_fpd_pct_rr = 0.20
        cfg.max_adaptive_min_fpd_ms = 0.0  # disable cap → pre-fix
        cfg.search_end_ms = 700.0
        cfg.enable_repol_gate = False

        fpd_samples, *_rest = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg, median_bp_s=3.0,
        )
        # Either detection fails outright, or it lands past the real
        # T-wave peak (> 500 ms). Either way, the T-wave is NOT
        # correctly reported around 400 ms.
        if fpd_samples is not None:
            fpd_ms = fpd_samples / fs * 1000
            assert fpd_ms >= 500, (
                f"Pre-fix behaviour should overshoot or fail; got {fpd_ms:.0f} ms"
            )

    def test_cap_zero_restores_pre_fix_behaviour(self):
        """max_adaptive_min_fpd_ms=0 must be equivalent to the pre-fix
        code path (no cap applied)."""
        fs = 2000.0
        tpl, *_ = _build_bradycardic_template(
            fs=fs, duration_s=0.9, pre_ms=50, t_wave_latency_ms=400,
        )
        cfg_capped = RepolarizationConfig()
        cfg_capped.max_adaptive_min_fpd_ms = 0.0
        cfg_capped.min_fpd_pct_rr = 0.20
        cfg_capped.enable_repol_gate = False

        cfg_huge = RepolarizationConfig()
        cfg_huge.max_adaptive_min_fpd_ms = 10_000.0  # effectively no cap
        cfg_huge.min_fpd_pct_rr = 0.20
        cfg_huge.enable_repol_gate = False

        r1 = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg_capped, median_bp_s=3.0,
        )
        r2 = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg_huge, median_bp_s=3.0,
        )
        # Identical floor → identical result
        assert r1[0] == r2[0]

    def test_per_beat_bradycardia_recovers_t_wave(self):
        fs = 2000.0
        data, t_vec, spike_idx, t_idx = _build_per_beat_signal(
            fs=fs, total_s=1.0, spike_s=0.3, t_wave_latency_ms=400,
        )
        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 120.0
        cfg.min_fpd_pct_rr = 0.20
        cfg.max_adaptive_min_fpd_ms = 350.0
        cfg.search_end_ms = 700.0
        cfg.enable_repol_gate = False

        # Per-beat returns fpd in SECONDS, not samples.
        fpd_s, _, _, _ = find_repolarization_per_beat(
            data, t_vec, spike_idx, fs,
            template_fpd_samples=None,
            template_peak_samples=None,
            template_repol_sign=1,
            cfg=cfg,
            beat_period_s=3.0,  # bradycardic
        )
        assert fpd_s is not None
        fpd_ms = fpd_s * 1000
        assert 350 <= fpd_ms <= 550, f"per-beat FPD={fpd_ms:.0f} ms"

    def test_physiological_rhythm_unaffected(self):
        """A normal 1-Hz rhythm (RR = 1 s) yields adaptive floor
        = 200 ms, well below the 500 ms cap — results must be identical
        with and without the cap enabled."""
        fs = 2000.0
        tpl, *_ = _build_bradycardic_template(
            fs=fs, duration_s=0.8, pre_ms=50, t_wave_latency_ms=300,
        )
        cfg_capped = RepolarizationConfig()
        cfg_capped.max_adaptive_min_fpd_ms = 500.0
        cfg_capped.min_fpd_pct_rr = 0.20
        cfg_capped.enable_repol_gate = False

        cfg_nocap = RepolarizationConfig()
        cfg_nocap.max_adaptive_min_fpd_ms = 0.0
        cfg_nocap.min_fpd_pct_rr = 0.20
        cfg_nocap.enable_repol_gate = False

        r1 = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg_capped, median_bp_s=1.0,
        )
        r2 = find_repolarization_on_template(
            tpl, fs, pre_ms=50, cfg=cfg_nocap, median_bp_s=1.0,
        )
        # Adaptive floor = 200 ms on both paths → same output
        assert r1[0] == r2[0]


# ═══════════════════════════════════════════════════════════════════════
#   Fix B — RR outlier filter  (tests added once function is in place)
# ═══════════════════════════════════════════════════════════════════════


class TestRROutlierFilter:
    """Tests pinning ``apply_rr_outlier_filter`` behaviour."""

    def test_rejects_gap_beat(self):
        from cardiac_fp_analyzer.rhythm_integration import apply_rr_outlier_filter

        fs = 2000.0
        # 10 regular beats (RR = 1000 samples = 0.5 s) + 1 outlier at
        # +12000 samples (RR = 6.0 s, 12× median)
        bi = np.concatenate([np.arange(10) * 1000,
                              np.array([9000 + 12000])]).astype(int)
        bd = np.zeros((len(bi), 400))
        btm = np.zeros_like(bd)

        bd_f, btm_f, bi_f, info = apply_rr_outlier_filter(
            bd, btm, bi, fs, max_rr_ratio=5.0, enable=True,
        )
        assert info['filter_applied'] is True
        assert info['n_dropped'] == 1
        assert len(bi_f) == len(bi) - 1
        assert int(bi[-1]) not in set(int(v) for v in bi_f)

    def test_passthrough_on_regular_rhythm(self):
        from cardiac_fp_analyzer.rhythm_integration import apply_rr_outlier_filter

        fs = 2000.0
        bi = (np.arange(12) * 1000).astype(int)
        bd = np.zeros((len(bi), 400))
        btm = np.zeros_like(bd)

        _, _, bi_f, info = apply_rr_outlier_filter(
            bd, btm, bi, fs, max_rr_ratio=5.0, enable=True,
        )
        assert info['n_dropped'] == 0
        assert len(bi_f) == len(bi)

    def test_disabled_is_noop(self):
        from cardiac_fp_analyzer.rhythm_integration import apply_rr_outlier_filter

        fs = 2000.0
        bi = np.concatenate([np.arange(5) * 1000,
                              np.array([20000])]).astype(int)
        bd = np.zeros((len(bi), 400))
        btm = np.zeros_like(bd)

        _, _, bi_f, info = apply_rr_outlier_filter(
            bd, btm, bi, fs, max_rr_ratio=5.0, enable=False,
        )
        assert info['filter_applied'] is False
        assert len(bi_f) == len(bi)

    def test_tolerates_short_inputs(self):
        """< 3 beats → cannot compute a meaningful median RR; must
        passthrough without raising."""
        from cardiac_fp_analyzer.rhythm_integration import apply_rr_outlier_filter

        fs = 2000.0
        bi = np.array([1000, 2500], dtype=int)
        bd = np.zeros((len(bi), 400))
        btm = np.zeros_like(bd)

        _, _, bi_f, info = apply_rr_outlier_filter(
            bd, btm, bi, fs, max_rr_ratio=5.0, enable=True,
        )
        assert len(bi_f) == len(bi)
        assert info['n_dropped'] == 0

    @pytest.mark.parametrize('ratio', [3.0, 5.0, 8.0])
    def test_threshold_respected(self, ratio):
        """Outlier at exactly 4× median → dropped at ratio=3, kept at ratio=5 / 8."""
        from cardiac_fp_analyzer.rhythm_integration import apply_rr_outlier_filter

        fs = 2000.0
        # 8 regular beats (RR = 1000) + 1 at +4000 (RR = 4× median)
        bi = np.concatenate([np.arange(8) * 1000,
                              np.array([7000 + 4000])]).astype(int)
        bd = np.zeros((len(bi), 400))
        btm = np.zeros_like(bd)

        _, _, bi_f, info = apply_rr_outlier_filter(
            bd, btm, bi, fs, max_rr_ratio=ratio, enable=True,
        )
        expected_dropped = 1 if ratio < 4.0 else 0
        assert info['n_dropped'] == expected_dropped


# ═══════════════════════════════════════════════════════════════════════
#   Fix C — Valid-FPD-ratio gate on the summary
# ═══════════════════════════════════════════════════════════════════════


class TestFPDReliabilityGate:
    """Tests pinning the ``fpd_reliable`` / ``fpd_valid_ratio`` flags
    added to the summary dict by ``apply_fpd_reliability_gate``.

    Tested at the gate level with hand-crafted per-beat params so the
    ratio is deterministic regardless of upstream detection noise.
    ``test_pipeline_integration_smoke`` checks that the gate fires via
    the end-to-end ``extract_all_parameters`` path too.
    """

    @staticmethod
    def _mk_params(n_total, n_valid, fpd_valid_ms=300.0):
        """Construct ``all_params`` with ``n_valid`` entries carrying a
        numeric FPD and the rest carrying NaN (simulates the repol-gate
        rejections)."""
        params = []
        for i in range(n_total):
            if i < n_valid:
                params.append({'fpd_ms': float(fpd_valid_ms),
                               'fpdc_ms': float(fpd_valid_ms)})
            else:
                params.append({'fpd_ms': float('nan'),
                               'fpdc_ms': float('nan')})
        return params

    def test_gate_marks_reliable_when_ratio_high(self):
        from cardiac_fp_analyzer.parameters import apply_fpd_reliability_gate

        summary = {}
        params = self._mk_params(n_total=10, n_valid=8)
        cfg = RepolarizationConfig()
        apply_fpd_reliability_gate(summary, params, cfg)
        assert summary['fpd_reliable'] is True
        assert summary['fpd_valid_ratio'] == pytest.approx(0.8)
        assert summary['fpd_note'] is None

    def test_gate_marks_unreliable_when_ratio_low(self):
        from cardiac_fp_analyzer.parameters import apply_fpd_reliability_gate

        summary = {}
        # 2/10 valid → ratio = 0.2 < default 0.5
        params = self._mk_params(n_total=10, n_valid=2)
        cfg = RepolarizationConfig()
        apply_fpd_reliability_gate(summary, params, cfg)
        assert summary['fpd_reliable'] is False
        assert summary['fpd_valid_ratio'] == pytest.approx(0.2)
        note = summary['fpd_note']
        assert note
        # Must mention the percentage and the Italian warning
        assert '20%' in note
        assert 'affidabile' in note.lower()

    def test_gate_at_exact_threshold_is_reliable(self):
        from cardiac_fp_analyzer.parameters import apply_fpd_reliability_gate

        summary = {}
        params = self._mk_params(n_total=10, n_valid=5)  # exactly 50%
        cfg = RepolarizationConfig()  # min_valid_fpd_ratio=0.5
        apply_fpd_reliability_gate(summary, params, cfg)
        assert summary['fpd_reliable'] is True  # ≥ threshold passes

    def test_ratio_threshold_configurable(self):
        """Lowering ``min_valid_fpd_ratio`` to 0.1 must flip a
        20 %-valid recording from unreliable to reliable."""
        from cardiac_fp_analyzer.parameters import apply_fpd_reliability_gate

        cfg_strict = RepolarizationConfig()
        cfg_strict.min_valid_fpd_ratio = 0.50
        cfg_loose = RepolarizationConfig()
        cfg_loose.min_valid_fpd_ratio = 0.10

        params = self._mk_params(n_total=10, n_valid=2)

        s_strict = {}
        apply_fpd_reliability_gate(s_strict, params, cfg_strict)
        assert s_strict['fpd_reliable'] is False

        s_loose = {}
        apply_fpd_reliability_gate(s_loose, params, cfg_loose)
        assert s_loose['fpd_reliable'] is True

    def test_gate_disabled_when_threshold_zero(self):
        """``min_valid_fpd_ratio <= 0`` → always reliable (pre-v3.3.1
        behaviour), as documented in the config comment."""
        from cardiac_fp_analyzer.parameters import apply_fpd_reliability_gate

        cfg = RepolarizationConfig()
        cfg.min_valid_fpd_ratio = 0.0

        params = self._mk_params(n_total=10, n_valid=1)

        summary = {}
        apply_fpd_reliability_gate(summary, params, cfg)
        assert summary['fpd_reliable'] is True
        assert summary['fpd_note'] is None

    def test_gate_empty_input_is_unreliable(self):
        from cardiac_fp_analyzer.parameters import apply_fpd_reliability_gate

        cfg = RepolarizationConfig()
        summary = {}
        apply_fpd_reliability_gate(summary, [], cfg)
        assert summary['fpd_reliable'] is False
        assert summary['fpd_valid_ratio'] == 0.0
        assert summary['fpd_note']  # diagnostic string present

    def test_pipeline_integration_smoke(self):
        """Smoke-test: a clean 10-beat recording run through
        ``extract_all_parameters`` must carry the new fields with
        ``fpd_reliable=True``."""
        from cardiac_fp_analyzer.parameters import extract_all_parameters

        fs = 2000.0
        bp_s = 1.0
        n_samp = int(fs * bp_s)
        pre = int(0.05 * fs)
        rng = np.random.default_rng(0)
        bd, btm, bi = [], [], []
        for i in range(10):
            beat = rng.standard_normal(n_samp) * 0.0005
            x = np.arange(n_samp) - pre
            beat += 0.8 * np.exp(-0.5 * (x / 4.0) ** 2)
            t_idx = pre + int(0.28 * fs)
            xt = np.arange(n_samp) - t_idx
            beat += 0.30 * np.exp(-0.5 * (xt / 80.0) ** 2)
            bd.append(beat)
            btm.append(i * bp_s + np.arange(n_samp) / fs)
            bi.append(int(i * bp_s * fs) + pre)
        bd = np.asarray(bd)
        btm = np.asarray(btm)
        bi = np.asarray(bi, dtype=int)
        cfg = RepolarizationConfig()
        cfg.min_fpd_ms = 120.0
        cfg.search_end_ms = 500.0
        _all_p, summary = extract_all_parameters(bd, btm, bi, fs, cfg=cfg)
        assert 'fpd_valid_ratio' in summary
        assert 'fpd_reliable' in summary
        assert 'fpd_note' in summary
        assert summary['fpd_reliable'] is True
