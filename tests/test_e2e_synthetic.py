"""End-to-end numeric tests on synthetic cardiac field-potential signals.

These tests verify that the full pipeline (filtering → beat detection →
segmentation → parameter extraction → arrhythmia analysis) produces
quantitatively correct results on signals with known ground truth.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia
from cardiac_fp_analyzer.beat_detection import (
    compute_beat_periods,
    detect_beats,
    segment_beats,
)
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.residual_analysis import (
    analyze_residual,
    compute_template,
    poincare_stv,
)

# ══════════════════════════════════════════════════════════════════════════
#   SYNTHETIC SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════

def _make_single_beat(fs, beat_period_s=1.0, fpd_s=0.35, spike_amp=0.002,
                      repol_amp=0.0003):
    """Generate one synthetic FP beat waveform.

    The waveform consists of:
      - A sharp negative-then-positive spike at t=0 (depolarisation)
      - A slow positive repolarisation peak at t=fpd_s
      - Return to baseline

    Parameters
    ----------
    fs : float — sampling rate (Hz)
    beat_period_s : float — total beat duration (s)
    fpd_s : float — field potential duration (s)
    spike_amp : float — spike amplitude (V)
    repol_amp : float — repolarisation amplitude (V)

    Returns
    -------
    1-D array of shape (int(beat_period_s * fs),)
    """
    n_samples = int(beat_period_s * fs)
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)

    # Depolarisation spike: Gaussian derivative shape
    spike_width = 0.003  # 3 ms
    spike = -spike_amp * np.exp(-0.5 * (t / spike_width) ** 2)
    spike += spike_amp * 0.6 * np.exp(-0.5 * ((t - 0.005) / spike_width) ** 2)
    signal += spike

    # Repolarisation: broader Gaussian peak at fpd_s
    repol_width = 0.025  # 25 ms
    signal += repol_amp * np.exp(-0.5 * ((t - fpd_s) / repol_width) ** 2)

    return signal


def make_synthetic_signal(fs=1000, n_beats=15, beat_period_s=1.0,
                           fpd_s=0.35, spike_amp=0.002, repol_amp=0.0003,
                           noise_std=0.0, beat_period_jitter=0.0):
    """Generate a multi-beat synthetic FP signal with known parameters.

    Returns
    -------
    signal : 1-D array — the full recording
    time : 1-D array — time vector in seconds
    true_beat_indices : list of int — exact spike locations
    true_beat_period_s : float — nominal beat period
    true_fpd_s : float — nominal FPD
    """
    rng = np.random.default_rng(42)
    beats = []
    true_indices = []
    offset = int(0.5 * fs)  # 500 ms pre-padding

    # Pre-padding
    beats.append(np.zeros(offset))

    for i in range(n_beats):
        bp = beat_period_s + (rng.normal(0, beat_period_jitter) if beat_period_jitter > 0 else 0)
        bp = max(0.3, bp)  # clamp
        beat = _make_single_beat(fs, beat_period_s=bp, fpd_s=fpd_s,
                                  spike_amp=spike_amp, repol_amp=repol_amp)
        true_indices.append(offset)
        beats.append(beat)
        offset += len(beat)

    # Post-padding
    beats.append(np.zeros(int(0.5 * fs)))

    signal = np.concatenate(beats)
    if noise_std > 0:
        signal += rng.normal(0, noise_std, len(signal))

    time = np.arange(len(signal)) / fs
    return signal, time, true_indices, beat_period_s, fpd_s


# ══════════════════════════════════════════════════════════════════════════
#   TEST: BEAT DETECTION ACCURACY
# ══════════════════════════════════════════════════════════════════════════

class TestBeatDetectionE2E:
    """Verify beat detection finds the correct number and locations."""

    @pytest.fixture
    def synth(self):
        return make_synthetic_signal(fs=1000, n_beats=15, beat_period_s=1.0,
                                      spike_amp=0.002, noise_std=1e-5)

    def test_correct_beat_count(self, synth):
        signal, time, true_indices, _, _ = synth
        bi, bt, info = detect_beats(signal, fs=1000)
        # Should find all 15 beats (±1 tolerance for edge beats)
        assert abs(len(bi) - 15) <= 1, f"Expected ~15 beats, got {len(bi)}"

    def test_beat_locations_within_tolerance(self, synth):
        signal, time, true_indices, _, _ = synth
        bi, bt, info = detect_beats(signal, fs=1000)
        # Each detected beat should be within 10 ms of a true beat
        tolerance = 10  # samples = 10 ms at 1 kHz
        for detected in bi:
            dists = np.abs(np.array(true_indices) - detected)
            assert np.min(dists) <= tolerance, (
                f"Detected beat at {detected} not near any true beat "
                f"(min dist = {np.min(dists)} samples)"
            )


# ══════════════════════════════════════════════════════════════════════════
#   TEST: BEAT PERIOD ACCURACY
# ══════════════════════════════════════════════════════════════════════════

class TestBeatPeriodE2E:
    """Verify computed beat periods match the expected value."""

    def test_mean_beat_period(self):
        signal, time, true_indices, true_bp, _ = make_synthetic_signal(
            fs=1000, n_beats=15, beat_period_s=0.8
        )
        bi, bt, _ = detect_beats(signal, fs=1000)
        bp = compute_beat_periods(bi, fs=1000)
        mean_bp = np.mean(bp)
        # Mean beat period should be within 5% of ground truth
        assert abs(mean_bp - true_bp) / true_bp < 0.05, (
            f"Mean BP = {mean_bp:.3f}s, expected {true_bp:.3f}s"
        )

    def test_cv_low_for_regular_rhythm(self):
        signal, time, true_indices, _, _ = make_synthetic_signal(
            fs=1000, n_beats=20, beat_period_s=1.0, noise_std=1e-5
        )
        bi, bt, _ = detect_beats(signal, fs=1000)
        bp = compute_beat_periods(bi, fs=1000)
        cv = np.std(bp) / np.mean(bp) * 100
        # CV should be very low for a perfectly regular signal
        assert cv < 5.0, f"CV = {cv:.1f}%, expected < 5% for regular rhythm"


# ══════════════════════════════════════════════════════════════════════════
#   TEST: PARAMETER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

class TestParameterExtractionE2E:
    """Verify spike amplitude and beat period extraction are numerically correct."""

    @pytest.fixture
    def pipeline_result(self):
        """Run the full pipeline on synthetic data and return results."""
        fs = 1000
        signal, time, true_indices, true_bp, true_fpd = make_synthetic_signal(
            fs=fs, n_beats=15, beat_period_s=1.0, spike_amp=0.002,
            fpd_s=0.35, noise_std=1e-5
        )
        bi, bt, info = detect_beats(signal, fs=fs)
        beats_data, beats_time, valid = segment_beats(signal, time, bi, fs)
        bp = compute_beat_periods(bi, fs)
        all_params, summary = extract_all_parameters(
            beats_data, beats_time, bi, fs
        )
        return {
            'all_params': all_params, 'summary': summary,
            'beat_indices': bi, 'beat_periods': bp,
            'beats_data': beats_data, 'beats_time': beats_time,
            'true_bp': true_bp, 'true_fpd': true_fpd,
            'fs': fs,
        }

    def test_spike_amplitude_positive(self, pipeline_result):
        """All spike amplitudes should be positive and non-zero."""
        for p in pipeline_result['all_params']:
            amp = p.get('spike_amplitude_mV', 0)
            if not np.isnan(amp):
                assert amp > 0, f"Spike amplitude should be > 0, got {amp}"

    def test_mean_beat_period_in_summary(self, pipeline_result):
        summary = pipeline_result['summary']
        true_bp = pipeline_result['true_bp']
        # Summary uses 'beat_period_ms_mean' key
        mean_bp_ms = summary.get('beat_period_ms_mean', 0)
        expected_ms = true_bp * 1000
        assert abs(mean_bp_ms - expected_ms) / expected_ms < 0.10, (
            f"Summary mean BP = {mean_bp_ms:.0f} ms, "
            f"expected ~{expected_ms:.0f} ms"
        )

    def test_n_beats_matches(self, pipeline_result):
        """Number of extracted parameter sets should match detected beats."""
        n_params = len(pipeline_result['all_params'])
        n_beats_data = len(pipeline_result['beats_data'])
        assert n_params == n_beats_data


# ══════════════════════════════════════════════════════════════════════════
#   TEST: RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

class TestResidualAnalysisE2E:
    """Verify residual analysis on synthetic regular rhythm → low instability."""

    def test_low_morphology_instability_for_clean_signal(self):
        """A perfectly regular signal should have near-zero instability."""
        fs = 1000
        signal, _, _, _, _ = make_synthetic_signal(
            fs=fs, n_beats=15, beat_period_s=1.0, noise_std=1e-6
        )
        bi, _, _ = detect_beats(signal, fs=fs)
        beats_data, beats_time, valid = segment_beats(signal, np.arange(len(signal)) / fs, bi, fs)
        compute_beat_periods(bi, fs)
        all_params, summary = extract_all_parameters(beats_data, beats_time, bi, fs)

        res = analyze_residual(beats_data, fs, all_params)
        assert res['morphology_instability'] < 0.15, (
            f"Expected low instability for clean signal, "
            f"got {res['morphology_instability']:.3f}"
        )

    def test_template_shape(self):
        """Template from identical beats should closely match the beat shape."""
        fs = 1000
        beat = _make_single_beat(fs, beat_period_s=1.0, spike_amp=0.002)
        # Create list of identical beats
        beats_data = [beat.copy() for _ in range(10)]
        template = compute_template(beats_data)
        assert template is not None
        # Template should be almost identical to original beat
        corr = np.corrcoef(template, beat[:len(template)])[0, 1]
        assert corr > 0.99, f"Template correlation = {corr:.4f}, expected > 0.99"

    def test_zero_eads_for_clean_signal(self):
        """A clean regular signal should have zero EADs detected."""
        fs = 1000
        signal, _, _, _, _ = make_synthetic_signal(
            fs=fs, n_beats=15, noise_std=1e-6
        )
        bi, _, _ = detect_beats(signal, fs=fs)
        time = np.arange(len(signal)) / fs
        beats_data, beats_time, valid = segment_beats(signal, time, bi, fs)
        all_params, summary = extract_all_parameters(beats_data, beats_time, bi, fs)

        res = analyze_residual(beats_data, fs, all_params)
        assert res['n_ead_beats'] == 0, (
            f"Expected 0 EAD beats for clean signal, got {res['n_ead_beats']}"
        )


# ══════════════════════════════════════════════════════════════════════════
#   TEST: ARRHYTHMIA CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

class TestArrhythmiaClassificationE2E:
    """Verify arrhythmia classification on known-good and known-bad signals."""

    def test_regular_rhythm_classified_normal(self):
        """A perfectly regular rhythm should be classified as Normal."""
        fs = 1000
        signal, time, _, _, _ = make_synthetic_signal(
            fs=fs, n_beats=15, beat_period_s=1.0, noise_std=1e-5
        )
        bi, _, _ = detect_beats(signal, fs=fs)
        beats_data, beats_time, valid = segment_beats(signal, time, bi, fs)
        bp = compute_beat_periods(bi, fs)
        all_params, summary = extract_all_parameters(beats_data, beats_time, bi, fs)

        report = analyze_arrhythmia(
            bi, bp, all_params, summary, fs,
            beats_data=beats_data
        )
        # Should be normal or borderline — not a severe arrhythmia
        assert report.classification in (
            'Normal Sinus Rhythm', 'Borderline / Mild Abnormalities'
        ), f"Expected normal classification, got '{report.classification}'"
        assert report.risk_score < 30, (
            f"Risk score = {report.risk_score}, expected < 30 for regular rhythm"
        )

    def test_irregular_rhythm_flagged(self):
        """A signal with jittery beat periods should flag irregularity."""
        fs = 1000
        signal, time, _, _, _ = make_synthetic_signal(
            fs=fs, n_beats=20, beat_period_s=1.0,
            spike_amp=0.002, noise_std=1e-5,
            beat_period_jitter=0.3,  # 30% jitter → high CV
        )
        bi, _, _ = detect_beats(signal, fs=fs)
        beats_data, beats_time, valid = segment_beats(signal, time, bi, fs)
        bp = compute_beat_periods(bi, fs)

        if len(bp) < 3:
            pytest.skip("Not enough beats detected")

        all_params, summary = extract_all_parameters(beats_data, beats_time, bi, fs)
        report = analyze_arrhythmia(bi, bp, all_params, summary, fs)

        # With 30% jitter, should see irregularity flags or non-normal classification
        assert report.classification != 'Normal Sinus Rhythm' or report.risk_score > 0, (
            f"Expected non-normal classification for jittery rhythm, "
            f"got '{report.classification}' with score {report.risk_score}"
        )


# ══════════════════════════════════════════════════════════════════════════
#   TEST: POINCARÉ STV
# ══════════════════════════════════════════════════════════════════════════

class TestPoincareSTV:
    """Verify Poincaré STV computation on known data."""

    def test_constant_values_zero_stv(self):
        """STV of identical values should be 0."""
        values = [350.0] * 10
        stv = poincare_stv(values)
        assert stv == pytest.approx(0.0, abs=1e-10)

    def test_alternating_values_known_stv(self):
        """STV of alternating 300/400 should equal 100/√2 ≈ 70.71."""
        values = [300.0, 400.0] * 5
        stv = poincare_stv(values)
        expected = 100.0 / np.sqrt(2)
        assert stv == pytest.approx(expected, rel=1e-6), (
            f"STV = {stv:.2f}, expected {expected:.2f}"
        )

    def test_too_few_values_returns_nan(self):
        """STV with < 3 values should return NaN."""
        assert np.isnan(poincare_stv([100.0, 200.0]))
        assert np.isnan(poincare_stv([]))


# ══════════════════════════════════════════════════════════════════════════
#   TEST: FULL PIPELINE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """Run the complete pipeline end-to-end and verify consistency."""

    def test_pipeline_produces_complete_output(self):
        """Full pipeline should produce all expected keys."""
        fs = 1000
        signal, time, _, _, _ = make_synthetic_signal(
            fs=fs, n_beats=12, beat_period_s=1.0, noise_std=1e-5
        )

        # 1. Detect beats
        bi, bt, info = detect_beats(signal, fs=fs)
        assert len(bi) >= 5, f"Expected ≥5 beats, got {len(bi)}"

        # 2. Segment
        beats_data, beats_time, valid = segment_beats(signal, time, bi, fs)
        assert len(beats_data) >= 5

        # 3. Beat periods
        bp = compute_beat_periods(bi, fs)
        assert len(bp) == len(bi) - 1

        # 4. Parameters
        all_params, summary = extract_all_parameters(
            beats_data, beats_time, bi, fs
        )
        assert len(all_params) == len(beats_data)
        assert 'beat_period_ms_mean' in summary

        # 5. Arrhythmia analysis
        report = analyze_arrhythmia(
            bi, bp, all_params, summary, fs,
            beats_data=beats_data
        )
        assert hasattr(report, 'classification')
        assert hasattr(report, 'risk_score')
        assert 0 <= report.risk_score <= 100

    def test_beat_period_denominator_consistency(self):
        """Verify beat_periods length == len(beat_indices) - 1 always."""
        fs = 1000
        signal, time, _, _, _ = make_synthetic_signal(
            fs=fs, n_beats=15, noise_std=1e-5
        )
        bi, _, _ = detect_beats(signal, fs=fs)
        bp = compute_beat_periods(bi, fs)
        assert len(bp) == len(bi) - 1, (
            f"len(bp)={len(bp)} != len(bi)-1={len(bi)-1}"
        )
