"""Regression tests on REAL recordings against published reference values.

Corpus: tests/fixtures/real_signals/ — 60 s windows (full sampling rate) of
baseline recordings from Visone et al. 2023, on the electrode the authors
used, with their published RR and FPD. Built by
tools/build_real_signal_fixtures.py; see manifest.json for provenance.

These are the only tests in the suite where a measured FPD / RR is compared
with an independent ground truth on a real signal. Tolerances are loose on
purpose (the reference is a population summary over a different window and
a manual pipeline); the point is to catch gross failures — the 2× beat
over-detection on Exp8 was invisible to every synthetic test.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from cardiac_fp_analyzer.analyze import analyze_single_file
from cardiac_fp_analyzer.config import AnalysisConfig

FIXTURES = Path(__file__).parent / 'fixtures' / 'real_signals'
MANIFEST = json.loads((FIXTURES / 'manifest.json').read_text())
BY_LABEL = {e['label']: e for e in MANIFEST}

# Per-file RR tolerance. Default ±12 %; exp7_chipC_ch1 is wider because the
# published value is ~12 % below what both electrodes show on every beat
# (reference-window discrepancy, see manifest notes).
RR_TOL = {'exp7_chipC_ch1': 0.16}
RR_TOL_DEFAULT = 0.12
FPD_TOL = 0.12


def _write_waveforms_csv(entry, path):
    """Rebuild a Digilent WaveForms CSV (header + 2 columns) from a fixture."""
    d = np.load(FIXTURES / entry['file'])
    sig = d['signal'].astype(np.float64)
    fs = float(d['fs'])
    t = np.arange(len(sig)) / fs
    header = (
        '#Digilent WaveForms Oscilloscope Acquisition\n'
        f'#Sample rate: {fs:.0f}Hz\n'
        f'#Samples: {len(sig)}\n'
        '\n'
        'Time (s),Channel 1 (V),Channel 2 (V)\n'
    )
    with open(path, 'w') as f:
        f.write(header)
        np.savetxt(f, np.column_stack([t, sig, sig]), delimiter=',', fmt='%.6f')
    return fs


def _run(entry, tmp_path, **cfg_overrides):
    """Run the full single-file pipeline on a fixture (authors' electrode forced)."""
    csv = tmp_path / f"{entry['label']}_baseline.csv"
    _write_waveforms_csv(entry, csv)
    cfg = AnalysisConfig()
    cfg.amplifier_gain = 1e4  # hardware gain; raw files are in amplified volts
    for k, v in cfg_overrides.items():
        obj, attr = k.rsplit('.', 1) if '.' in k else (None, k)
        target = getattr(cfg, obj) if obj else cfg
        setattr(target, attr, v)
    logging.disable(logging.CRITICAL)
    try:
        # Fixture stores only the authors' electrode (duplicated into both
        # columns), so 'el1' always addresses it.
        return analyze_single_file(str(csv), channel='el1', verbose=False, config=cfg)
    finally:
        logging.disable(logging.NOTSET)


def _rr_median_ms(result):
    fs = result['metadata']['sample_rate']
    bi = np.asarray(result.get('beat_indices_raw', result['beat_indices']))
    return float(np.median(np.diff(bi)) / fs * 1000.0)


@pytest.mark.parametrize('label', sorted(BY_LABEL))
class TestReferenceAgreement:
    """RR and FPD within tolerance of the published values, per recording."""

    def test_rr_matches_reference(self, label, tmp_path):
        e = BY_LABEL[label]
        r = _run(e, tmp_path)
        rr = _rr_median_ms(r)
        ref = e['reference']['rr_ms']
        tol = RR_TOL.get(label, RR_TOL_DEFAULT)
        assert abs(rr - ref) / ref <= tol, (
            f"{label}: RR median {rr:.0f} ms vs reference {ref:.0f} ms "
            f"({(rr / ref - 1) * 100:+.0f} %, tol ±{tol * 100:.0f} %)")

    def test_fpd_matches_reference(self, label, tmp_path):
        e = BY_LABEL[label]
        r = _run(e, tmp_path)
        s = r['summary']
        fpd = s.get('fpd_ms_median', s.get('fpd_ms_mean'))
        ref = e['reference']['fpd_ms']
        assert fpd is not None and not np.isnan(fpd), f"{label}: no FPD measured"
        assert abs(fpd - ref) / ref <= FPD_TOL, (
            f"{label}: FPD median {fpd:.0f} ms vs reference {ref:.0f} ms "
            f"({(fpd / ref - 1) * 100:+.0f} %, tol ±{FPD_TOL * 100:.0f} %)")

    def test_no_over_detection(self, label, tmp_path):
        """Raw beat count must not exceed what the reference rhythm allows.

        This is the assertion that would have caught the Exp8 2× problem:
        n_raw ≈ 2 × duration / RR_ref while FPD was still correct.
        """
        e = BY_LABEL[label]
        r = _run(e, tmp_path)
        n_raw = len(r.get('beat_indices_raw', r['beat_indices']))
        expected = e['duration_s'] * 1000.0 / e['reference']['rr_ms']
        assert n_raw <= 1.25 * expected + 2, (
            f"{label}: {n_raw} raw detections, reference rhythm allows ~{expected:.0f}")
        assert n_raw >= 0.70 * expected - 2, (
            f"{label}: only {n_raw} raw detections, reference rhythm implies ~{expected:.0f}")


class TestNoiseFloorGateMechanism:
    """The Exp8 doubling is caused by noise-level insertions and fixed by the gate."""

    LABEL = 'exp8_d6_chipD_ch1'

    def test_gate_halves_detections_on_exp8(self, tmp_path):
        e = BY_LABEL[self.LABEL]
        with_gate = _run(e, tmp_path)
        without = _run(e, tmp_path, **{'beat_detection.enable_noise_floor_gate': False})
        n_with = len(with_gate['beat_indices_raw'])
        n_without = len(without['beat_indices_raw'])
        # Without the gate: ~2× the true count. With: within reference.
        assert n_without >= 1.6 * n_with, (
            f"expected the ungated detector to roughly double the count, "
            f"got {n_without} vs {n_with}")
        rr_without = _rr_median_ms(without)
        rr_with = _rr_median_ms(with_gate)
        ref = e['reference']['rr_ms']
        assert rr_without < 0.65 * ref, f"ungated RR {rr_without:.0f} should be ~half of {ref:.0f}"
        assert abs(rr_with - ref) / ref <= RR_TOL_DEFAULT

    def test_rejected_detections_are_noise_level(self, tmp_path):
        """Everything the gate rejects is noise-compatible; everything it
        keeps is well above the noise floor — a gap, not a tuned threshold."""
        e = BY_LABEL[self.LABEL]
        r = _run(e, tmp_path)
        info = r['detection_info']['noise_floor_gate']['post_detection']
        assert info['noise_gate'] == 'applied'
        assert info['rule'] == 'noise_cluster'
        assert info['snr_max_rejected'] <= 3.5     # noise_cluster_max_snr
        assert info['snr_min_kept'] >= 3.0
        assert info['cluster_gap'] >= 1.5

    def test_fpd_unaffected_by_gate(self, tmp_path):
        """QC already discarded the noise beats before FPD; the gate must not
        move FPD, only RR/CV (which use the raw train)."""
        e = BY_LABEL[self.LABEL]
        with_gate = _run(e, tmp_path)
        without = _run(e, tmp_path, **{'beat_detection.enable_noise_floor_gate': False})
        f1 = with_gate['summary']['fpd_ms_median']
        f0 = without['summary']['fpd_ms_median']
        assert abs(f1 - f0) / f0 < 0.05


    def test_exp6_window_doubling_also_fixed(self, tmp_path):
        """Exp6 chipC shows the same failure on 90 s windows (RR ~830 vs 1835)."""
        e = BY_LABEL['exp6_chipC_ch1']
        with_gate = _run(e, tmp_path)
        without = _run(e, tmp_path, **{'beat_detection.enable_noise_floor_gate': False})
        ref = e['reference']['rr_ms']
        rr_without, rr_with = _rr_median_ms(without), _rr_median_ms(with_gate)
        assert rr_without < 0.6 * ref, f"expected ungated doubling, RR={rr_without:.0f}"
        assert abs(rr_with - ref) / ref <= RR_TOL_DEFAULT, f"gated RR {rr_with:.0f} vs {ref:.0f}"


class TestGateIsConservativeOnCleanSignals:
    """On recordings without noise insertions the gate removes (almost) nothing."""

    @pytest.mark.parametrize('label', ['exp8_d7_chipE_ch2', 'exp5_chipA_ch1', 'exp7_chipC_ch1'])
    def test_no_true_beats_lost(self, label, tmp_path):
        e = BY_LABEL[label]
        r = _run(e, tmp_path)
        gate = r['detection_info']['noise_floor_gate']
        assert gate['n_rejected_total'] <= 1, f"{label}: gate rejected {gate['n_rejected_total']}"
