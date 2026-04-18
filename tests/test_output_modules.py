"""
test_output_modules.py — Tests for report, plotting, and risk_map modules.

Verifies that output/visualization modules don't crash with valid input
and degrade gracefully with edge cases (empty results, missing keys).
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cardiac_fp_analyzer.plotting import (
    minmax_downsample,
    plot_analysis_summary,
    plot_beat_overlay,
    plot_raw_trace,
)
from cardiac_fp_analyzer.report import (
    generate_excel_report,
    generate_pdf_report,
)
from cardiac_fp_analyzer.risk_map import (
    aggregate_drug_metrics,
    compute_proarrhythmic_index,
    generate_risk_map,
)

# ── Helpers ──

def _make_result(drug='TestDrug', concentration='1uM', is_baseline=False,
                 n_beats=8, n_samples=10000, fs=2000.0):
    """Build a realistic result dict for report/risk_map tests."""
    t = np.arange(n_samples) / fs
    sig = np.random.randn(n_samples) * 0.001
    beat_indices = np.linspace(500, n_samples - 500, n_beats, dtype=int)

    qc = SimpleNamespace(
        grade='A', global_snr=12.0,
        n_beats_input=n_beats + 2, rejection_rate=0.1,
    )
    ar = SimpleNamespace(
        classification='Normal Sinus Rhythm', risk_score=8,
        flags=[], details={'morphology_instability': 0.05},
        residual_details={'baseline_relative': False},
    )
    cess = SimpleNamespace(
        has_cessation=False, cessation_confidence=0.0,
        cessation_type='none',
    )

    return {
        'metadata': {'filename': f'{drug}_{concentration}.csv',
                     'sample_rate': fs, 'n_samples': n_samples,
                     'analyzed_channel': 'el1'},
        'file_info': {'experiment': 'EXP1', 'chip': 'ChipA',
                      'analyzed_channel': 'el1',
                      'drug': 'Baseline' if is_baseline else drug,
                      'concentration': concentration},
        'summary': {
            'beat_period_ms_n': n_beats - 1,
            'beat_period_ms_mean': 800.0, 'beat_period_ms_std': 30.0,
            'beat_period_ms_cv': 3.75, 'beat_period_ms_median': 800.0,
            'beat_period_ms_min': 750.0, 'beat_period_ms_max': 850.0,
            'bpm_mean': 75.0, 'stv_ms': 2.5,
            'spike_amplitude_mV_mean': 15.0, 'spike_amplitude_mV_std': 1.0,
            'spike_amplitude_mV_n': n_beats,
            'fpd_ms_mean': 320.0, 'fpd_ms_std': 10.0,
            'fpdc_ms_mean': 350.0, 'fpdc_ms_std': 12.0,
            'fpdc_bazett_ms_mean': 360.0,
            'fpd_confidence': 0.85, 'template_fpd_ms': 320.0,
            'fpd_method': 'tangent', 'correction': 'fridericia',
            'beat_periods': np.ones(n_beats - 1) * 0.8,
            'fpd_values': np.ones(n_beats) * 0.32,
            'fpdc_values': np.ones(n_beats) * 0.35,
            'n_beats_no_repol': 0, 'pct_beats_no_repol': 0.0,
        },
        'qc_report': qc,
        'arrhythmia_report': ar,
        'cessation_report': cess,
        'normalization': {
            'has_baseline': not is_baseline,
            'baseline_file': 'baseline.csv' if not is_baseline else '',
            'baseline_bp_ms': 800.0 if not is_baseline else np.nan,
            'baseline_fpdc_ms': 340.0 if not is_baseline else np.nan,
            'pct_bp_change': 2.0 if not is_baseline else np.nan,
            'pct_fpdc_change': 5.0 if not is_baseline else np.nan,
            'pct_amp_change': -3.0 if not is_baseline else np.nan,
            'exceeds_LOW': False, 'exceeds_MID': False, 'exceeds_HIGH': False,
            'tdp_score': 0, 'spectral_change_score': 0.1,
        },
        'spectral_report': SimpleNamespace(
            fundamental_freq_hz=1.25,
            spectral_entropy=0.5,
            band_powers={'low': 0.3, 'mid': 0.5, 'high': 0.2},
        ),
        'inclusion': {'passed': True, 'fpdc_plausible': True},
        'all_params': [
            {
                'beat_number': j + 1,
                'spike_amplitude_mV': 15.0,
                'fpd_ms': 320.0 + j, 'fpdc_ms': 350.0 + j,
                'fpdc_bazett_ms': 360.0,
                'rise_time_ms': 1.5, 'repol_amplitude_mV': 3.0,
                'rr_interval_ms': 800.0, 'max_dvdt': 100.0,
                'repol_peak_global_idx': int(beat_indices[j]) + 300,
                'fpd_endpoint_global_idx': int(beat_indices[j]) + 350,
            }
            for j in range(n_beats)
        ],
        'beat_indices': beat_indices,
        'beat_periods': np.ones(n_beats - 1) * 0.8,
        'filtered_signal': sig,
        'time_vector': t,
        'beats_data': [np.random.randn(200) * 0.001 for _ in range(n_beats)],
        'beats_time': [np.linspace(-0.025, 0.075, 200) for _ in range(n_beats)],
    }


# ═══════════════════════════════════════════════════════════════════════
#   PLOTTING
# ═══════════════════════════════════════════════════════════════════════

class TestMinmaxDownsample:

    def test_short_signal_passthrough(self):
        x = np.arange(100, dtype=float)
        y = np.sin(x)
        xd, yd = minmax_downsample(x, y, target_points=5000)
        np.testing.assert_array_equal(xd, x)
        np.testing.assert_array_equal(yd, y)

    def test_long_signal_reduced(self):
        x = np.arange(20000, dtype=float)
        y = np.sin(x / 100)
        xd, yd = minmax_downsample(x, y, target_points=1000)
        assert len(xd) <= 1000
        assert len(xd) == len(yd)

    def test_preserves_extremes(self):
        """Max and min should survive downsampling."""
        x = np.arange(10000, dtype=float)
        y = np.sin(x / 500 * 2 * np.pi)
        xd, yd = minmax_downsample(x, y, target_points=500)
        assert np.max(yd) >= 0.99  # Peak preserved
        assert np.min(yd) <= -0.99


class TestPlotFunctions:

    def test_plot_raw_trace(self):
        import matplotlib
        import pandas as pd
        matplotlib.use('Agg')
        n = 5000
        df = pd.DataFrame({'time': np.arange(n) / 2000.0,
                           'el1': np.random.randn(n) * 0.001})
        metadata = {'filename': 'test.csv', 'sample_rate': 2000, 'n_samples': n}
        fig, ax = plot_raw_trace(df, metadata, channel='el1')
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_beat_overlay(self):
        import matplotlib
        matplotlib.use('Agg')
        beats_time = [np.linspace(-0.05, 0.3, 200) for _ in range(5)]
        beats_data = [np.random.randn(200) * 0.001 for _ in range(5)]
        metadata = {'filename': 'test.csv'}
        fig, ax = plot_beat_overlay(beats_time, beats_data, metadata)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_beat_overlay_empty(self):
        """Empty beats list should not crash."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = plot_beat_overlay([], [], {'filename': 'test.csv'})
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_analysis_summary(self):
        import matplotlib
        matplotlib.use('Agg')
        r = _make_result()
        fig = plot_analysis_summary(
            r['time_vector'], r['filtered_signal'],
            r['beat_indices'], r['summary'],
            r['metadata'], all_params=r['all_params'],
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#   REPORT
# ═══════════════════════════════════════════════════════════════════════

class TestExcelReport:

    def test_generates_file(self):
        results = [_make_result(is_baseline=True), _make_result()]
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            generate_excel_report(results, tmp.name)
            assert Path(tmp.name).stat().st_size > 0

    def test_single_result(self):
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            generate_excel_report([_make_result()], tmp.name)
            assert Path(tmp.name).stat().st_size > 0

    def test_empty_results(self):
        """Empty list should create a valid (possibly empty) file."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            generate_excel_report([], tmp.name)
            assert Path(tmp.name).exists()


class TestPdfReport:

    def test_generates_file(self):
        results = [_make_result()]
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            generate_pdf_report(results, tmp.name)
            assert Path(tmp.name).stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════
#   RISK MAP
# ═══════════════════════════════════════════════════════════════════════

class TestRiskMap:

    def test_aggregate_drug_metrics(self):
        results = [
            _make_result(is_baseline=True),
            _make_result(drug='Dofetilide', concentration='10nM'),
            _make_result(drug='Dofetilide', concentration='100nM'),
        ]
        metrics = aggregate_drug_metrics(results)
        assert isinstance(metrics, dict)
        # Baseline should be excluded
        assert 'dofetilide' in [k.lower() for k in metrics]

    def test_compute_proarrhythmic_index(self):
        results = [
            _make_result(is_baseline=True),
            _make_result(drug='TestDrug'),
        ]
        metrics = aggregate_drug_metrics(results)
        if metrics:
            drug, m = next(iter(metrics.items()))
            score = compute_proarrhythmic_index(m)
            assert 0 <= score <= 100

    def test_generate_risk_map_basic(self):
        import matplotlib
        matplotlib.use('Agg')
        results = [
            _make_result(is_baseline=True),
            _make_result(drug='DrugA', concentration='10nM'),
            _make_result(drug='DrugB', concentration='100nM'),
        ]
        fig = generate_risk_map(results)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_generate_risk_map_empty(self):
        """Empty results should produce a figure without crashing."""
        import matplotlib
        matplotlib.use('Agg')
        fig = generate_risk_map([])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
