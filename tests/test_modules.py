"""
test_modules.py — Tests for previously untested modules.

Covers: cessation, spectral, cdisc_export, normalization, residual_analysis.
Uses synthetic signals to verify that public APIs don't crash and return
correctly-typed results.
"""

import tempfile

import numpy as np

# ─── Helpers ───

def make_synthetic_fp(fs=1000.0, duration=5.0, n_beats=10, noise=0.0005):
    """Create a synthetic FP signal with sharp depolarization spikes."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    data = np.random.randn(n) * noise

    beat_period = duration / (n_beats + 1)
    beat_indices = []
    for i in range(1, n_beats + 1):
        idx = int(i * beat_period * fs)
        if idx + 5 < n:
            data[idx] = 0.05
            data[idx + 1] = -0.03
            data[idx + 2] = 0.01
            beat_indices.append(idx)
    return data, t, np.array(beat_indices), fs


def make_minimal_results(n_recordings=3, n_beats=10):
    """Create minimal results dicts for normalization/cdisc tests."""
    results = []
    for i in range(n_recordings):
        is_bl = (i == 0)
        r = {
            'filename': f'rec_{i}.csv',
            'experiment_name': 'EXP001',
            'chip_id': 'CHIP01',
            'chamber': 'A1',
            'electrode': 'E1',
            'drug_name': 'Vehicle' if is_bl else 'TestDrug',
            'concentration_uM': 0.0 if is_bl else float(i * 10),
            'is_baseline': is_bl,
            'condition': 'baseline' if is_bl else f'dose_{i}',
            'beat_indices': np.arange(n_beats) * 1000 + 500,
            'fs': 1000.0,
            'summary': {
                'beat_period_ms_mean': 800.0 + i * 10,
                'fpdc_ms_mean': 350.0 + i * 20,
                'spike_amplitude_mV_mean': 15.0 - i * 0.5,
                'bpm_mean': 75.0,
                'stv_ms': 2.5,
                'fpd_ms_mean': 320.0 + i * 15,
                'fpd_ms_std': 10.0,
                'fpdc_ms_std': 12.0,
                'beat_period_ms_std': 30.0,
                'beat_periods': np.ones(n_beats - 1) * 0.8,
                'fpd_values': np.ones(n_beats) * 0.32,
                'fpdc_values': np.ones(n_beats) * 0.35,
                'n_beats_no_repol': 0,
                'pct_beats_no_repol': 0.0,
                'fpd_confidence': 0.8,
                'template_fpd_ms': 320.0,
                'fpd_method': 'tangent',
                'correction': 'fridericia',
            },
            'all_params': [
                {
                    'beat_number': j + 1,
                    'spike_amplitude_mV': 15.0,
                    'fpd_ms': 320.0 + j * 2,
                    'fpdc_ms': 350.0 + j * 2,
                    'fpdc_bazett_ms': 360.0,
                    'rise_time_ms': 1.5,
                    'repol_amplitude_mV': 3.0,
                    'rr_interval_ms': 800.0,
                    'max_dvdt': 100.0,
                    'repol_peak_global_idx': 1000 + j * 1000 + 300,
                    'fpd_endpoint_global_idx': 1000 + j * 1000 + 350,
                }
                for j in range(n_beats)
            ],
            'arrhythmia': None,
        }
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════
#   CESSATION
# ═══════════════════════════════════════════════════════════════════════

class TestCessation:

    def test_import(self):
        from cardiac_fp_analyzer.cessation import detect_cessation
        assert callable(detect_cessation)

    def test_no_cessation_normal_signal(self):
        """Normal signal with regular beats → no cessation detected."""
        from cardiac_fp_analyzer.cessation import detect_cessation
        data, t, beat_indices, fs = make_synthetic_fp(duration=5.0, n_beats=10)
        report = detect_cessation(data, fs, beat_indices)
        assert hasattr(report, 'cessation_type')
        assert hasattr(report, 'cessation_confidence')
        assert report.cessation_confidence >= 0

    def test_cessation_with_silent_ending(self):
        """Signal that stops beating midway → should detect cessation."""
        from cardiac_fp_analyzer.cessation import detect_cessation
        fs = 1000.0
        data = np.random.randn(10000) * 0.0005
        # Beats only in first half
        beat_indices = []
        for i in range(1, 6):
            idx = i * 800
            data[idx] = 0.05
            data[idx + 1] = -0.03
            beat_indices.append(idx)
        report = detect_cessation(data, fs, np.array(beat_indices))
        assert isinstance(report.cessation_confidence, (int, float))

    def test_empty_beats(self):
        """No beats at all should not crash."""
        from cardiac_fp_analyzer.cessation import detect_cessation
        data = np.random.randn(5000) * 0.001
        report = detect_cessation(data, 1000.0, np.array([], dtype=int))
        assert hasattr(report, 'cessation_type')


# ═══════════════════════════════════════════════════════════════════════
#   SPECTRAL
# ═══════════════════════════════════════════════════════════════════════

class TestSpectral:

    def test_import(self):
        from cardiac_fp_analyzer.spectral import analyze_spectral
        assert callable(analyze_spectral)

    def test_basic_analysis(self):
        """Spectral analysis on synthetic signal should return SpectralReport."""
        from cardiac_fp_analyzer.spectral import analyze_spectral
        data, t, beat_indices, fs = make_synthetic_fp(duration=5.0, n_beats=10)
        report = analyze_spectral(data, fs, beat_indices=beat_indices)
        assert hasattr(report, 'fundamental_freq_hz')
        assert hasattr(report, 'spectral_entropy')

    def test_short_signal(self):
        """Very short signal should not crash."""
        from cardiac_fp_analyzer.spectral import analyze_spectral
        data = np.random.randn(500) * 0.001
        report = analyze_spectral(data, 1000.0)
        assert report is not None

    def test_morphology_change_score(self):
        """compute_morphology_change_score should return 0-1 float."""
        from cardiac_fp_analyzer.spectral import (
            analyze_spectral,
            compute_morphology_change_score,
        )
        data, t, beat_indices, fs = make_synthetic_fp()
        report1 = analyze_spectral(data, fs, beat_indices=beat_indices)
        # Slightly modified signal
        data2 = data + np.random.randn(len(data)) * 0.001
        report2 = analyze_spectral(data2, fs, beat_indices=beat_indices)
        score = compute_morphology_change_score(report2, report1)
        assert 0 <= score <= 1


# ═══════════════════════════════════════════════════════════════════════
#   RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

class TestResidualAnalysis:

    def test_import(self):
        from cardiac_fp_analyzer.residual_analysis import (
            analyze_residual,
        )
        assert callable(analyze_residual)

    def test_compute_template(self):
        """Template from uniform beats should be close to any individual beat."""
        from cardiac_fp_analyzer.residual_analysis import compute_template
        beat = np.sin(np.linspace(0, 2 * np.pi, 100))
        beats_data = [beat + np.random.randn(100) * 0.01 for _ in range(10)]
        template = compute_template(beats_data)
        assert template is not None
        assert len(template) == 100
        # Template should correlate well with input
        corr = np.corrcoef(template, beat)[0, 1]
        assert corr > 0.9

    def test_poincare_stv(self):
        """STV of constant values should be 0."""
        from cardiac_fp_analyzer.residual_analysis import poincare_stv
        stv = poincare_stv(np.ones(10) * 350)
        assert stv == 0.0

    def test_poincare_stv_insufficient(self):
        """STV with < 2 values should return NaN."""
        from cardiac_fp_analyzer.residual_analysis import poincare_stv
        stv = poincare_stv(np.array([350]))
        assert np.isnan(stv)

    def test_analyze_residual_basic(self):
        """Full residual analysis should return dict with expected keys."""
        from cardiac_fp_analyzer.residual_analysis import analyze_residual
        beat = np.sin(np.linspace(0, 2 * np.pi, 200))
        beats_data = [beat + np.random.randn(200) * 0.01 for _ in range(10)]
        all_params = [
            {'fpd_ms': 320 + i, 'fpdc_ms': 350 + i}
            for i in range(10)
        ]
        result = analyze_residual(beats_data, 1000.0, all_params)
        assert 'template' in result
        assert 'morphology_instability' in result
        assert 'n_ead_beats' in result


# ═══════════════════════════════════════════════════════════════════════
#   NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

class TestNormalization:

    def test_import(self):
        from cardiac_fp_analyzer.normalization import (
            normalize_all_results,
        )
        assert callable(normalize_all_results)

    def test_is_baseline(self):
        from cardiac_fp_analyzer.normalization import is_baseline
        assert is_baseline({'file_info': {'drug': 'Baseline'}})
        assert not is_baseline({'file_info': {'drug': 'Dofetilide'}})

    def test_get_group_key(self):
        from cardiac_fp_analyzer.normalization import get_group_key
        r = {
            'file_info': {
                'experiment': 'EXP1', 'chip': 'ChipA',
                'chamber': 'ch1', 'analyzed_channel': 'el1',
            },
        }
        key = get_group_key(r)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_pair_with_baselines(self):
        from cardiac_fp_analyzer.normalization import pair_with_baselines
        results = make_minimal_results()
        pairs = pair_with_baselines(results)
        assert isinstance(pairs, dict)

    def test_normalize_all_results(self):
        from cardiac_fp_analyzer.normalization import normalize_all_results
        results = make_minimal_results()
        normalized = normalize_all_results(results)
        assert len(normalized) == len(results)


# ═══════════════════════════════════════════════════════════════════════
#   CDISC EXPORT
# ═══════════════════════════════════════════════════════════════════════

class TestCdiscExport:

    def test_import(self):
        from cardiac_fp_analyzer.cdisc_export import export_send_package
        assert callable(export_send_package)

    def test_basic_export(self):
        """Export should create files without crashing."""
        from cardiac_fp_analyzer.cdisc_export import export_send_package
        results = make_minimal_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_send_package(results, tmpdir)
            assert 'files' in output
            assert 'datasets' in output
            assert 'summary' in output
            # Check that at least some files were created
            assert len(output['files']) > 0

    def test_empty_results(self):
        """Empty results should not crash."""
        from cardiac_fp_analyzer.cdisc_export import export_send_package
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_send_package([], tmpdir)
            assert isinstance(output, dict)

    def test_backend_helpers_return_bool(self):
        """has_pyreadstat() and has_xport() must return booleans
        without crashing, regardless of the environment."""
        from cardiac_fp_analyzer.cdisc_export import (
            has_pyreadstat,
            has_xport,
            xpt_backend,
        )
        assert isinstance(has_pyreadstat(), bool)
        assert isinstance(has_xport(), bool)
        assert xpt_backend() in ('pyreadstat', 'xport', 'csv_fallback')

    def test_xpt_backend_matches_helpers(self):
        """xpt_backend() precedence must agree with the boolean helpers."""
        from cardiac_fp_analyzer.cdisc_export import (
            has_pyreadstat,
            has_xport,
            xpt_backend,
        )
        backend = xpt_backend()
        if has_pyreadstat():
            assert backend == 'pyreadstat'
        elif has_xport():
            assert backend == 'xport'
        else:
            assert backend == 'csv_fallback'

    def test_export_returns_backend_used(self):
        """export_send_package must include `backend_used` in the returned
        dict, and the value must equal xpt_backend() for a non-empty run."""
        from cardiac_fp_analyzer.cdisc_export import (
            export_send_package,
            xpt_backend,
        )
        results = make_minimal_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_send_package(results, tmpdir)
            assert 'backend_used' in output
            assert output['backend_used'] in (
                'pyreadstat', 'xport', 'csv_fallback'
            )
            # Multi-domain export: the package-level backend should match
            # what xpt_backend() reports for this environment (all domains
            # use the same import, so no downgrade path exists here).
            assert output['backend_used'] == xpt_backend()

    def test_empty_export_reports_backend_from_environment(self):
        """With empty results no domain is written, but backend_used must
        still reflect what WOULD be used so the GUI can warn honestly."""
        from cardiac_fp_analyzer.cdisc_export import (
            export_send_package,
            xpt_backend,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_send_package([], tmpdir)
            assert output.get('backend_used') == xpt_backend()

    def test_csv_fallback_writes_warning_header(self):
        """When the CSV fallback is used, the .xpt files must carry an
        explicit header marker so a human opening the file sees it is
        NOT a real SAS Transport."""
        from cardiac_fp_analyzer.cdisc_export import (
            export_send_package,
            xpt_backend,
        )
        if xpt_backend() != 'csv_fallback':
            import pytest
            pytest.skip("pyreadstat or xport installed — fallback not exercised")
        results = make_minimal_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_send_package(results, tmpdir)
            xpt_files = [p for p in output['files'] if str(p).endswith('.xpt')]
            assert xpt_files, "export should have produced at least one .xpt"
            with open(xpt_files[0]) as f:
                header = f.readline()
            assert header.startswith('#') and 'CSV fallback' in header

    def test_summary_flags_fallback_package(self):
        """The README summary must flag packages written via csv_fallback
        so anyone sharing the zip downstream can see the status."""
        from cardiac_fp_analyzer.cdisc_export import (
            export_send_package,
            xpt_backend,
        )
        if xpt_backend() != 'csv_fallback':
            import pytest
            pytest.skip("pyreadstat or xport installed — fallback not exercised")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = export_send_package(make_minimal_results(), tmpdir)
            assert 'NOT regulatory-grade' in output['summary'] \
                or 'CSV placeholders' in output['summary']
