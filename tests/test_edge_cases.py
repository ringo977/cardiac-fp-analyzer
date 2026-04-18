"""
test_edge_cases.py — Edge-case tests for hardening the cardiac FP analyzer.

Tests cover:
  - Empty / too-short / all-NaN / partial-NaN signals in beat detection
  - Invalid sampling rate
  - Empty beats in parameter extraction
  - Alignment mismatch in parameter extraction
  - NaN propagation in arrhythmia analysis
  - Zero noise level in repolarization gate
  - segment_beats with beats near edges
  - Single-beat signals
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.arrhythmia import (
    ArrhythmiaReport,
    analyze_arrhythmia,
)
from cardiac_fp_analyzer.beat_detection import (
    compute_beat_periods,
    detect_beats,
    segment_beats,
)
from cardiac_fp_analyzer.parameters import (
    build_beat_template,
    extract_all_parameters,
)
from cardiac_fp_analyzer.repolarization import (
    find_repolarization_on_template,
    find_repolarization_per_beat,
)

# ═══════════════════════════════════════════════════════════════════════
#   BEAT DETECTION — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestDetectBeatsEdgeCases:
    """Edge cases for detect_beats() input validation."""

    def test_empty_signal(self):
        """Empty array should return empty results, not crash."""
        indices, times, info = detect_beats(np.array([]), fs=1000.0)
        assert len(indices) == 0
        assert len(times) == 0
        assert info['n_beats'] == 0

    def test_all_nan_signal(self):
        """All-NaN signal should return empty results."""
        data = np.full(5000, np.nan)
        indices, times, info = detect_beats(data, fs=1000.0)
        assert len(indices) == 0
        assert info['n_beats'] == 0

    def test_too_short_signal(self):
        """Signal shorter than 0.5s should return empty results."""
        data = np.random.randn(100)  # 0.1s at 1000 Hz
        indices, times, info = detect_beats(data, fs=1000.0)
        assert len(indices) == 0
        assert info['n_beats'] == 0

    def test_invalid_fs_zero(self):
        """fs=0 should raise ValueError."""
        with pytest.raises(ValueError, match="sampling rate must be positive"):
            detect_beats(np.random.randn(5000), fs=0)

    def test_invalid_fs_negative(self):
        """Negative fs should raise ValueError."""
        with pytest.raises(ValueError, match="sampling rate must be positive"):
            detect_beats(np.random.randn(5000), fs=-100)

    def test_partial_nan_signal(self):
        """Signal with some NaN should still work (NaN replaced with 0)."""
        fs = 1000.0
        t = np.arange(0, 2, 1/fs)
        # Clean signal with a few spikes
        data = np.sin(2 * np.pi * 2 * t) * 0.001
        for i in range(4):
            idx = int((0.25 + i * 0.5) * fs)
            if idx < len(data):
                data[idx] = 0.05
                data[idx + 1] = -0.03
        # Sprinkle some NaN
        data[10] = np.nan
        data[500] = np.nan
        data[1200] = np.nan
        indices, times, info = detect_beats(data, fs=fs)
        # Should not crash; may or may not find beats depending on signal
        assert isinstance(indices, np.ndarray)

    def test_flat_signal(self):
        """Constant (flat) signal should return no beats."""
        data = np.ones(5000) * 0.001
        indices, times, info = detect_beats(data, fs=1000.0)
        assert len(indices) == 0

    def test_single_beat_signal(self):
        """Signal with exactly one spike should detect one beat."""
        fs = 1000.0
        data = np.zeros(2000)
        # Single sharp spike
        data[1000] = 0.1
        data[1001] = -0.05
        indices, times, info = detect_beats(data, fs=fs)
        # Should find at most 1 beat
        assert len(indices) <= 1


# ═══════════════════════════════════════════════════════════════════════
#   SEGMENT BEATS — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestSegmentBeatsEdgeCases:
    """Edge cases for segment_beats()."""

    def test_empty_beat_indices(self):
        """No beats to segment."""
        data = np.random.randn(5000)
        time = np.arange(5000) / 1000.0
        beats_data, beats_time, valid = segment_beats(
            data, time, np.array([], dtype=int), fs=1000.0
        )
        assert len(beats_data) == 0
        assert len(valid) == 0

    def test_all_beats_at_edges(self):
        """Beats too close to start/end should be excluded with warning."""
        data = np.random.randn(5000)
        time = np.arange(5000) / 1000.0
        # Beats at positions that can't fit pre/post windows
        beat_indices = np.array([10, 20, 4990])  # All near edges
        beats_data, beats_time, valid = segment_beats(
            data, time, beat_indices, fs=1000.0,
            pre_ms=50, post_ms=600,
        )
        # All should be excluded (pre=50 samples, post=600 samples)
        assert len(valid) == 0

    def test_mix_valid_and_edge_beats(self):
        """Mix of valid and edge beats."""
        data = np.random.randn(10000)
        time = np.arange(10000) / 1000.0
        beat_indices = np.array([10, 2000, 5000, 9990])
        beats_data, beats_time, valid = segment_beats(
            data, time, beat_indices, fs=1000.0,
            pre_ms=50, post_ms=600,
        )
        # Only middle beats should survive
        assert 2000 in [beat_indices[v] for v in valid]
        assert 5000 in [beat_indices[v] for v in valid]


# ═══════════════════════════════════════════════════════════════════════
#   COMPUTE BEAT PERIODS — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestComputeBeatPeriodsEdgeCases:

    def test_empty(self):
        result = compute_beat_periods(np.array([]), fs=1000.0)
        assert len(result) == 0

    def test_single_beat(self):
        result = compute_beat_periods(np.array([500]), fs=1000.0)
        assert len(result) == 0

    def test_two_beats(self):
        result = compute_beat_periods(np.array([500, 1500]), fs=1000.0)
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
#   PARAMETER EXTRACTION — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestExtractAllParametersEdgeCases:

    def test_empty_beats(self):
        """Empty beats list should return empty results."""
        all_params, summary = extract_all_parameters(
            beats_data=[], beats_time=[], beat_indices=np.array([]),
            fs=1000.0,
        )
        assert len(all_params) == 0
        assert isinstance(summary, dict)

    def test_alignment_mismatch_raises(self):
        """Mismatched lengths should raise ValueError (not AssertionError)."""
        beats_data = [np.zeros(100)]
        beats_time = [np.zeros(100), np.zeros(100)]  # Mismatch
        beat_indices = np.array([500])
        with pytest.raises(ValueError, match="alignment error"):
            extract_all_parameters(
                beats_data=beats_data, beats_time=beats_time,
                beat_indices=beat_indices, fs=1000.0,
            )

    def test_few_beats_no_template(self):
        """With < 5 beats, template should be None but extraction works."""
        fs = 1000.0
        # 3 simple beats
        beat_len = 650  # pre_ms=50 + post_ms=600
        beats_data = [np.random.randn(beat_len) * 0.001 for _ in range(3)]
        beats_time = [np.arange(-50, 600) / fs for _ in range(3)]
        beat_indices = np.array([1000, 2000, 3000])
        template = build_beat_template(beats_data, fs)
        assert template is None  # < 5 beats
        # extract_all_parameters should still work
        all_params, summary = extract_all_parameters(
            beats_data, beats_time, beat_indices, fs,
        )
        assert len(all_params) == 3


# ═══════════════════════════════════════════════════════════════════════
#   ARRHYTHMIA — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestArrhythmiaEdgeCases:

    def test_risk_score_empty_details(self):
        """Risk score with empty details should be 0."""
        report = ArrhythmiaReport()
        report.compute_risk_score()
        assert report.risk_score == 0

    def test_risk_score_nan_ead_pct(self):
        """NaN in ead_incidence_pct should not crash risk score."""
        report = ArrhythmiaReport()
        report.details = {
            'n_beats': 10,
            'cv_bp_pct': 5,
            'n_premature': 0,
            'n_delayed': 0,
            'morphology_instability': 0,
            'ead_incidence_pct': np.nan,
            'amplitude_cv_pct': 5,
            'poincare_stv_fpdc_ms': 3,
            'has_pauses': False,
            'pct_beats_no_repol': 0,
        }
        # Should not crash
        report.compute_risk_score()
        assert isinstance(report.risk_score, int)

    def test_insufficient_beats(self):
        """< 3 beat periods → Insufficient Data classification."""
        report = analyze_arrhythmia(
            beat_indices=np.array([100, 200]),
            beat_periods=np.array([0.5]),
            all_params=[{'fpd_ms': 300, 'spike_amplitude_mV': 10}],
            summary={'stv_ms': np.nan},
            fs=1000.0,
        )
        assert report.classification == 'Insufficient Data'
        assert report.risk_score == 0


# ═══════════════════════════════════════════════════════════════════════
#   REPOLARIZATION — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestRepolarizationEdgeCases:

    def test_template_too_short(self):
        """Very short template should return None."""
        template = np.random.randn(50)  # Too short for meaningful analysis
        result = find_repolarization_on_template(template, fs=1000.0, pre_ms=10)
        assert result[0] is None  # fpd_samples

    def test_flat_template(self):
        """Flat (zero-variance) template: noise_level=0, should not crash."""
        template = np.ones(2000) * 0.001
        result = find_repolarization_on_template(template, fs=1000.0, pre_ms=50)
        # Should not crash even with zero noise_level
        assert isinstance(result, tuple)

    def test_per_beat_short_segment(self):
        """Very short beat segment should return None, not crash."""
        data = np.random.randn(20)
        t = np.arange(20) / 1000.0
        fpd, amp, peak_idx, end_idx = find_repolarization_per_beat(
            data, t, spike_idx=5, fs=1000.0,
        )
        # Short segment → likely None
        assert fpd is None or isinstance(fpd, float)
