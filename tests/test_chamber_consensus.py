"""Chamber-level measurements (v3.14): rhythm status from the electrodes'
common beats, consensus FPD on a reference, same wave followed on a dose,
and how they replace the single-electrode values in the summary."""

import numpy as np
import pandas as pd
import pytest

from cardiac_fp_analyzer import chamber as CH
from cardiac_fp_analyzer.chambers import UHEART_MVP_64
from cardiac_fp_analyzer.config import AnalysisConfig
from tests.golden_signals import generate_regular_fp

FS = 2000.0
CHAMBER = UHEART_MVP_64['A']


def _chamber_df(beat_ms=800.0, fpd_ms=300.0, seed=0, silent=False, irregular=False, split=False, n_s=20.0,
                jitter_ms=0.0):
    """Electrodes of chamber A at 2 kHz. Every recording electrode sees the
    same beats (small conduction delays); stimulation electrodes noise only.
    irregular: beat times with 30 % random jitter; split: half of the
    electrodes beat at another time (conduction lost); silent: noise only."""
    rng = np.random.default_rng(seed)
    n = int(n_s * FS)
    sig, _t, _m = generate_regular_fp(fs=FS, duration_s=n_s, beat_period_ms=beat_ms, fpd_ms=fpd_ms,
                                      depol_amp=60e-6, repol_amp=10e-6, noise_std=0.0, seed=seed)
    if irregular:
        # rebuild with jittered beats
        beat = sig[int(beat_ms / 1000 * FS) - 100:int(beat_ms / 1000 * FS) + int(0.9 * beat_ms / 1000 * FS)]
        sig = np.zeros(n)
        t = 0.5
        while t < n_s - 1.5:
            k = int(t * FS)
            sig[k:k + len(beat)] += beat
            t += beat_ms / 1000 * rng.uniform(0.6, 1.4)
    df = {}
    for j, e in enumerate(CHAMBER.electrodes):
        noise = rng.normal(0, 3e-6, n)
        if silent or e in CHAMBER.stimulation:
            df[e] = noise
            continue
        delay = int((j % 6) * 2)                       # 0..10 samples of conduction delay
        x = np.roll(sig, delay)
        if split and j % 2 == 0:
            x = np.roll(sig, int(0.3 * FS))             # another timing for half of the electrodes
        df[e] = x * rng.uniform(0.6, 1.0) + noise
    return pd.DataFrame(df)


def test_regular_consensus_and_same_wave():
    cfg = AnalysisConfig()
    base = CH.analyze_chamber(_chamber_df(seed=1), FS, CHAMBER.electrodes, CHAMBER.stimulation, cfg)
    assert base['rhythm_status'] == 'regular' and base['ok']
    assert base['bp_ms'] == pytest.approx(800, abs=3)
    assert base['cv_robust_pct'] < 5 and base['synchrony'] > 0.9
    assert len(base['electrodes_usable']) >= 10
    assert base['fpd_ms'] == pytest.approx(300, abs=40) and base['fpd_n'] >= 3
    assert base['fpd_method'] == 'consensus' and base['reference'] is not None
    assert set(base['reference']['electrodes']) <= set(CHAMBER.recording)
    # dose: shorter FPD, faster rhythm; the same wave is followed
    dose = CH.analyze_chamber(_chamber_df(beat_ms=650.0, fpd_ms=240.0, seed=2), FS, CHAMBER.electrodes,
                              CHAMBER.stimulation, cfg, reference=base['reference'])
    assert dose['rhythm_status'] == 'regular' and dose['ok'] and dose['fpd_method'] == 'same wave'
    assert dose['bp_ms'] == pytest.approx(650, abs=3)
    assert dose['fpd_ms'] == pytest.approx(base['fpd_ms'] - 60, abs=25)
    assert dose['fpdc_ms'] == pytest.approx(dose['fpd_ms'] / 0.65 ** (1 / 3), rel=1e-6)
    assert all(c >= CH.MIN_CORR for c in dose['electrode_corr'].values() if np.isfinite(c)) or dose['fpd_n'] >= 3


def test_silent_irregular_conduction_lost():
    cfg = AnalysisConfig()
    silent = CH.analyze_chamber(_chamber_df(silent=True), FS, CHAMBER.electrodes, CHAMBER.stimulation, cfg)
    assert silent['rhythm_status'] == 'insufficient' and not silent['ok']
    irr = CH.analyze_chamber(_chamber_df(irregular=True, seed=3), FS, CHAMBER.electrodes, CHAMBER.stimulation, cfg)
    assert irr['rhythm_status'] == 'irregular' and irr['cv_robust_pct'] > CH.CV_IRREGULAR_PCT and not irr['ok']
    assert np.isnan(irr['fpd_ms']) and np.isfinite(irr['bp_ms'])
    lost = CH.analyze_chamber(_chamber_df(split=True, seed=4), FS, CHAMBER.electrodes, CHAMBER.stimulation, cfg)
    assert lost['rhythm_status'] == 'conduction_lost' and lost['synchrony'] < CH.SYNC_LOST


def test_agreeing_groups():
    assert CH.agreeing({'a': 100.0, 'b': 105.0, 'c': 200.0, 'd': 210.0, 'e': 205.0}) == {'c': 200.0, 'd': 210.0, 'e': 205.0}
    assert set(CH.agreeing({'a': 100.0, 'b': 110.0, 'c': 300.0})) == {'a', 'b'}
    assert CH.agreeing({}) == {}


def test_summary_override_and_reference_through_batch(tmp_path):
    """analyze_single_file on a chamber: chamber values replace the electrode's;
    an irregular dose becomes not analysable; the batch passes the baseline
    templates to the doses."""
    h5py = pytest.importorskip('h5py')  # noqa: F841
    from cardiac_fp_analyzer import mcs_hdf5 as M
    from cardiac_fp_analyzer.analyze import analyze_single_file, batch_analyze
    from tests.test_chambers import NAME

    def write(path, dfc):
        labels = [f'E{k}' for k in range(1, 65)]
        X = np.zeros((64, len(dfc)), dtype=np.int32)
        for e in dfc.columns:
            X[labels.index(e)] = np.round(dfc[e].values / 1e-9)
        M.write_mcs_h5(path, X, labels, tick_us=500, conversion_factor=1, exponent=-9, mea_layout='uHeart MVP 64')
        return path

    base = write(tmp_path / ('2026-01-27T12-00-41' + NAME.format('baseline')), _chamber_df(seed=11))
    write(tmp_path / ('2026-01-27T12-58-51' + NAME.format('A')), _chamber_df(beat_ms=650.0, fpd_ms=240.0, seed=12))
    irr = write(tmp_path / ('2026-01-27T13-41-31' + NAME.format('B')), _chamber_df(irregular=True, seed=13))
    upd = {'chamber': 'A', 'channel': 'A', 'electrodes': list(CHAMBER.electrodes),
           'stimulation_electrodes': list(CHAMBER.stimulation), 'tissue': '-/-/chipPM01001_chA', 'two_tissue_file': True}
    cfg = AnalysisConfig()
    r = analyze_single_file(base, channel='auto', verbose=False, config=cfg, file_info_update=upd)
    s = r['summary']
    assert r['chamber']['rhythm_status'] == 'regular' and s['chamber_status'] == 'regular'
    assert s['fpd_source'].startswith('chamber consensus')
    assert s['fpd_ms_median'] == pytest.approx(r['chamber']['fpd_ms']) and 'fpd_ms_median_electrode' in s
    assert s['beat_period_ms_median'] == pytest.approx(800, abs=3)
    # consensus off: single-electrode values untouched
    cfg_off = AnalysisConfig()
    cfg_off.chamber_consensus = False
    r0 = analyze_single_file(base, channel='auto', verbose=False, config=cfg_off, file_info_update=upd)
    assert 'chamber' not in r0 and 'fpd_source' not in r0['summary']
    # irregular dose: not analysable with the chamber's reason
    ri = analyze_single_file(irr, channel='auto', verbose=False, config=cfg,
                             file_info_update={**upd, 'chamber_reference': r['chamber']['reference']})
    assert ri['summary']['not_analysable'] and 'irregolare' in ri['summary']['not_analysable_reason']
    assert np.isnan(ri['summary']['fpdc_ms_mean']) and ri['summary']['fpd_reliable'] is False
    # batch: the dose follows the baseline's wave
    out = batch_analyze(tmp_path, channel='auto', output_dir=tmp_path / 'out', verbose=False, config=cfg, n_workers=1)
    by = {(r['file_info']['chamber'], r['file_info'].get('concentration')): r for r in out}
    d = by[('A', 'A')]
    assert d['chamber']['fpd_method'] == 'same wave' and d['summary']['fpd_source'].startswith('chamber same wave')
    assert d['summary']['fpdc_ms_mean'] < by[('A', '0')]['summary']['fpdc_ms_mean']
    assert by[('A', 'B')]['summary']['not_analysable']
