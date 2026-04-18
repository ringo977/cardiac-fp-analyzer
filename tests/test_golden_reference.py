"""
test_golden_reference.py — Golden-reference regression tests.

Runs the full analysis pipeline on synthetic signals with known ground truth
and verifies that key outputs stay within expected ranges. These tests catch
regressions when modifying the signal processing logic.

Each test is deterministic (fixed seed) so results are reproducible.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia
from cardiac_fp_analyzer.beat_detection import compute_beat_periods, detect_beats, segment_beats
from cardiac_fp_analyzer.cessation import detect_cessation
from cardiac_fp_analyzer.config import (
    BeatDetectionConfig,
    RepolarizationConfig,
)
from cardiac_fp_analyzer.parameters import extract_all_parameters
from tests.golden_signals import (
    generate_cessation_fp,
    generate_irregular_fp,
    generate_regular_fp,
)


def _run_pipeline(signal, fs, bd_cfg=None, repol_cfg=None):
    """Run the core pipeline: detect → segment → extract → arrhythmia."""
    if bd_cfg is None:
        bd_cfg = BeatDetectionConfig()
    if repol_cfg is None:
        repol_cfg = RepolarizationConfig()

    beat_indices, beat_times, info = detect_beats(signal, fs, cfg=bd_cfg)
    if len(beat_indices) < 2:
        return None

    beats_data, beats_time, valid = segment_beats(
        signal, np.arange(len(signal)) / fs, beat_indices, fs, cfg=repol_cfg,
    )
    bi_clean = beat_indices[valid]
    beat_periods = compute_beat_periods(bi_clean, fs)

    all_params, summary = extract_all_parameters(
        beats_data, beats_time, bi_clean, fs, cfg=repol_cfg,
    )

    ar = analyze_arrhythmia(
        bi_clean, beat_periods, all_params, summary, fs,
        beats_data=beats_data,
    )

    return {
        'beat_indices': bi_clean,
        'beat_periods': beat_periods,
        'all_params': all_params,
        'summary': summary,
        'arrhythmia': ar,
        'info': info,
    }


# ═══════════════════════════════════════════════════════════════════════
#   GOLDEN REFERENCE: Regular signal
# ═══════════════════════════════════════════════════════════════════════

class TestGoldenRegular:
    """Regular rhythm signal — the most basic 'happy path'."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fs = 2000.0
        self.signal, self.time, self.expected = generate_regular_fp(
            fs=self.fs, duration_s=10.0, beat_period_ms=800.0,
            depol_amp=0.05, repol_amp=0.008, fpd_ms=300.0,
        )
        self.result = _run_pipeline(self.signal, self.fs)

    def test_pipeline_runs(self):
        assert self.result is not None

    def test_beat_count(self):
        """Should detect the expected number of beats (±2)."""
        n_detected = len(self.result['beat_indices'])
        n_expected = self.expected['n_beats']
        assert abs(n_detected - n_expected) <= 2, (
            f"Detected {n_detected} beats, expected ~{n_expected}"
        )

    def test_beat_period(self):
        """Mean beat period should match ground truth (±10%)."""
        bp_ms = self.result['beat_periods'] * 1000
        mean_bp = np.mean(bp_ms)
        expected_bp = self.expected['beat_period_ms']
        assert abs(mean_bp - expected_bp) / expected_bp < 0.10, (
            f"Mean BP = {mean_bp:.0f} ms, expected {expected_bp:.0f} ms"
        )

    def test_beat_regularity(self):
        """CV of beat period should be < 5% for regular signal."""
        bp_ms = self.result['beat_periods'] * 1000
        cv = np.std(bp_ms) / np.mean(bp_ms) * 100
        assert cv < 5.0, f"CV = {cv:.1f}%, expected < 5% for regular signal"

    def test_classification_normal(self):
        """Regular signal should be classified as Normal Sinus Rhythm."""
        assert self.result['arrhythmia'].classification == 'Normal Sinus Rhythm'

    def test_risk_score_low(self):
        """Regular signal should have low risk score."""
        assert self.result['arrhythmia'].risk_score <= self.expected['risk_score_max']

    def test_spike_amplitude_consistent(self):
        """All beats should have similar spike amplitude (CV < 20%)."""
        amps = [p['spike_amplitude_mV'] for p in self.result['all_params']]
        amps = np.array(amps)
        amps = amps[~np.isnan(amps)]
        if len(amps) > 3:
            cv = np.std(amps) / np.mean(amps) * 100
            assert cv < 20, f"Amplitude CV = {cv:.1f}%, expected < 20%"

    def test_summary_has_required_keys(self):
        """Summary should contain all essential statistics."""
        s = self.result['summary']
        required = [
            'beat_period_ms_mean', 'fpd_ms_mean', 'fpdc_ms_mean',
            'spike_amplitude_mV_mean', 'bpm_mean', 'stv_ms',
            'fpd_confidence', 'n_beats_no_repol',
        ]
        for key in required:
            assert key in s, f"Missing summary key: {key}"


# ═══════════════════════════════════════════════════════════════════════
#   GOLDEN REFERENCE: Irregular rhythm
# ═══════════════════════════════════════════════════════════════════════

class TestGoldenIrregular:
    """Irregular rhythm signal — should flag rhythm abnormalities."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fs = 2000.0
        self.signal, self.time, self.expected = generate_irregular_fp(
            fs=self.fs, duration_s=10.0, cv_period_pct=25.0,
        )
        self.result = _run_pipeline(self.signal, self.fs)

    def test_pipeline_runs(self):
        assert self.result is not None

    def test_beat_count_in_range(self):
        n = len(self.result['beat_indices'])
        assert self.expected['n_beats_min'] <= n <= self.expected['n_beats_max']

    def test_not_classified_normal(self):
        """Irregular signal should NOT be classified as normal."""
        classification = self.result['arrhythmia'].classification
        assert classification != self.expected['classification_not'], (
            f"Irregular signal classified as '{classification}', expected abnormal"
        )

    def test_risk_score_elevated(self):
        """Irregular signal should have elevated risk score."""
        assert self.result['arrhythmia'].risk_score >= self.expected['risk_score_min']


# ═══════════════════════════════════════════════════════════════════════
#   GOLDEN REFERENCE: Cessation
# ═══════════════════════════════════════════════════════════════════════

class TestGoldenCessation:
    """Signal with cessation — beats stop midway."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fs = 2000.0
        self.signal, self.time, self.expected = generate_cessation_fp(
            fs=self.fs, duration_s=10.0, cessation_at_s=5.0,
        )

    def test_beat_detection(self):
        """Should detect beats only in the first half."""
        bd_cfg = BeatDetectionConfig()
        indices, _, info = detect_beats(self.signal, self.fs, cfg=bd_cfg)
        # All detected beats should be in the first ~5s
        late_beats = indices[indices > int(5.5 * self.fs)]
        assert len(late_beats) == 0, (
            f"Found {len(late_beats)} beats after cessation at 5s"
        )

    def test_cessation_detection(self):
        """Cessation module should flag the silence."""
        bd_cfg = BeatDetectionConfig()
        indices, _, _ = detect_beats(self.signal, self.fs, cfg=bd_cfg)
        report = detect_cessation(self.signal, self.fs, indices)
        # Should detect terminal silence or cessation
        assert report.terminal_silence_s > 3.0 or report.has_cessation, (
            f"Expected cessation detection: terminal_silence={report.terminal_silence_s:.1f}s, "
            f"has_cessation={report.has_cessation}"
        )


# ═══════════════════════════════════════════════════════════════════════
#   DETERMINISM
# ═══════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Running the same signal twice should produce identical results."""

    def test_same_results(self):
        fs = 2000.0
        sig1, _, _ = generate_regular_fp(fs=fs, seed=99)
        sig2, _, _ = generate_regular_fp(fs=fs, seed=99)
        np.testing.assert_array_equal(sig1, sig2)

        r1 = _run_pipeline(sig1, fs)
        r2 = _run_pipeline(sig2, fs)
        np.testing.assert_array_equal(r1['beat_indices'], r2['beat_indices'])
        assert r1['arrhythmia'].risk_score == r2['arrhythmia'].risk_score
