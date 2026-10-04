"""Multi-chamber chips: layout, MCS file names, sample sheet rows per chamber,
one recording per chamber in the batch, quick electrode choice."""

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

from cardiac_fp_analyzer import mcs_hdf5 as M  # noqa: E402
from cardiac_fp_analyzer.chambers import (  # noqa: E402
    LAYOUTS,
    UHEART_MVP_64,
    detect_layout,
    layout_for,
)
from cardiac_fp_analyzer.loader import (  # noqa: E402
    describe_recording,
    parse_filename,
    recording_channels,
    tissue_key,
)
from tests.golden_signals import generate_regular_fp  # noqa: E402

FS = 2000.0
NAME = 'McsRecording_PM01001_{}_Recording-0_(Data Acquisition (1);MEAS4-USB64; Electrode Raw Data1)_Analog.h5'


class TestLayout:
    def test_uheart_chambers(self):
        lay = UHEART_MVP_64
        assert lay.names == ('A', 'B', 'C', 'D')
        assert lay['A'].electrodes[0] == 'E16' and lay['A'].electrodes[-1] == 'E62'
        assert lay['C'].electrodes[:2] == ('E1', 'E2') and lay['C'].electrodes[-1] == 'E61'
        assert lay['B'].stimulation == ('E31', 'E32', 'E44', 'E45')
        assert len(lay['D'].recording) == 12 and 'E64' in lay['D'].row and len(lay['D'].row) == 12
        assert lay['D'].pitch_mm == pytest.approx(0.4)
        assert lay.chamber_of('E62').name == 'A' and lay.chamber_of('E99') is None
        assert {e for c in lay.chambers for e in c.electrodes} == set(lay.channels)

    def test_detection_and_choice(self):
        labels = [f'E{k}' for k in range(1, 65)]
        assert detect_layout(labels) is UHEART_MVP_64
        assert detect_layout(labels[:8]) is None
        assert layout_for(labels, 'auto') is UHEART_MVP_64
        assert layout_for(labels, 'none') is None
        assert layout_for(labels[:8], 'uheart_mvp_64') is UHEART_MVP_64
        with pytest.raises(ValueError):
            layout_for(labels, 'no_such_layout')
        assert 'uheart_mvp_64' in LAYOUTS


class TestNames:
    def test_mcs_file_name(self):
        info = parse_filename('2026-01-27T12-00-41' + NAME.format('baseline'))
        assert info['chip'] == 'PM01001' and info['is_baseline'] and info['format'] == 'mcs_hdf5'
        info = parse_filename('2026-01-27T15-04-21' + NAME.format('D'))
        assert info['chip'] == 'PM01001' and info['concentration'] == 'D' and not info['is_baseline']
        info = parse_filename('2026-01-26T19-36-01' + NAME.format('D_heater'))
        assert info['concentration'] == 'D_heater'
        d = describe_recording('/data/Exp7/' + NAME.format('A'))
        assert d['experiment'] == 'Exp7' and d['chip'] == 'PM01001' and 'tissue' not in d

    def test_tissue_key_letters(self):
        assert tissue_key('Exp 5', 'Day7', 'B', 1) == 'exp5/day7/chipB_ch1'
        assert tissue_key(None, None, 'PM01001', 'A') == '-/-/chipPM01001_chA'
        assert tissue_key('Exp3', None, 'PM01001', 'c') == 'exp3/-/chipPM01001_chC'


def _chip_file(path, seed=0, beat_ms=(800.0, 1000.0, 650.0, 900.0), silent=()):
    """A 64-channel µHeart-like file at 2 kHz: each chamber beats at its own
    period on every recording electrode (stimulation electrodes: noise only);
    chambers in ``silent`` carry noise only. Returns the written path."""
    rng = np.random.default_rng(seed)
    n = int(12 * FS)
    X = np.zeros((64, n), dtype=np.int32)
    labels = [f'E{k}' for k in range(1, 65)]
    for c, bp in zip(UHEART_MVP_64.chambers, beat_ms):
        sig, _t, _m = generate_regular_fp(fs=FS, duration_s=12.0, beat_period_ms=bp, fpd_ms=0.35 * bp,
                                          depol_amp=60e-6, repol_amp=8e-6, noise_std=0.0, seed=seed)
        for e in c.electrodes:
            noise = rng.normal(0, 3e-6, n)
            v = noise if (e in c.stimulation or c.name in silent) else sig * rng.uniform(0.6, 1.0) + noise
            X[labels.index(e)] = np.round(v / 1e-9)
    M.write_mcs_h5(path, X, labels, tick_us=500, conversion_factor=1, exponent=-9, mea_layout='uHeart MVP 64')
    return path


@pytest.fixture(scope='module')
def plate(tmp_path_factory):
    d = tmp_path_factory.mktemp('plate')
    base = _chip_file(d / ('2026-01-27T12-00-41' + NAME.format('baseline')), seed=1)
    dose = _chip_file(d / ('2026-01-27T12-58-51' + NAME.format('A')), seed=2, beat_ms=(700.0, 900.0, 600.0, 850.0), silent=('D',))
    return d, base, dose


class TestPlan:
    def test_channels_and_units_from_layout(self, plate):
        from cardiac_fp_analyzer.sample_sheet import plan_batch
        d, base, dose = plate
        assert recording_channels(base) == [f'E{k}' for k in range(1, 65)]
        units, report = plan_batch([base, dose], d)
        assert len(units) == 8 and report['multi_chamber_files'] == 2 and report['from_layout'] == 8
        u = next(u for u in units if u['file'] == base and u['update']['chamber'] == 'B')
        assert u['electrode'] == 'auto' and u['update']['tissue'] == '-/-/chipPM01001_chB'
        assert u['update']['electrodes'] == list(UHEART_MVP_64['B'].electrodes)
        assert u['update']['stimulation_electrodes'] == list(UHEART_MVP_64['B'].stimulation)
        assert u['update']['is_baseline'] and u['update']['two_tissue_file']
        ud = next(u for u in units if u['file'] == dose and u['update']['chamber'] == 'A')['update']
        assert ud['concentration'] == 'A' and not ud['is_baseline']

    def test_units_from_sheet(self, plate, tmp_path):
        from cardiac_fp_analyzer.sample_sheet import plan_batch, read_sample_sheet
        d, base, dose = plate
        sheet = d / 'samples.csv'
        # MCS names contain ';' and ',': the file column must be quoted (the draft writer does it)
        sheet.write_text('file;electrode;chip;chamber;item;dose;exclude\n'
                         f'"{base.name}";A;PM01001;A;;baseline;\n'
                         f'"{base.name}";B;PM01001;;;baseline;\n'
                         f'"{dose.name}";A;PM01001;A;TI01;A;\n'
                         f'"{dose.name}";E40;PM01001;B;TI03;A;\n'
                         f'"{dose.name}";C;PM01001;C;TI02;A;bad tissue\n', encoding='utf-8')
        try:
            rows, problems = read_sample_sheet(sheet)
            assert not problems and [r.chamber for r in rows] == ['A', 'B', 'A', 'B', 'C']
            units, report = plan_batch([base, dose], d)
            assert report['from_sheet'] == 4 and len(units) == 4
            ub = next(u for u in units if u['file'] == dose and u['update']['chamber'] == 'B')
            assert ub['electrode'] == 'E40' and ub['update']['item'] == 'TI03' and ub['update']['concentration'] == 'A'
            assert ('E40' in ub['update']['electrodes'])
            assert any(e[1] == 'C' for e in report['excluded'])
        finally:
            sheet.unlink()

    def test_layout_none(self, plate):
        from cardiac_fp_analyzer.sample_sheet import plan_batch
        d, base, dose = plate
        units, report = plan_batch([base, dose], d, layout='none')
        assert len(units) == 2 and all(u['update'] is None for u in units)

    def test_draft(self, plate):
        from cardiac_fp_analyzer.sample_sheet import draft_sample_sheet
        d, base, dose = plate
        rows, path = draft_sample_sheet(d, write=False)
        assert len(rows) == 8 and {r['chamber'] for r in rows} == {'A', 'B', 'C', 'D'}
        assert all(r['dose'] == 'baseline' for r in rows if r['file'] == base.name)


class TestAnalysis:
    def test_chamber_recording(self, plate):
        from cardiac_fp_analyzer.analyze import analyze_single_file
        from cardiac_fp_analyzer.config import AnalysisConfig
        from cardiac_fp_analyzer.sample_sheet import plan_batch
        d, base, dose = plate
        units, _ = plan_batch([base], d)
        cfg = AnalysisConfig()                      # default gain 1e4 must not touch MCS data
        res = {}
        for u in units:
            r = analyze_single_file(base, channel='auto', verbose=False, config=cfg, file_info_update=u['update'])
            assert r is not None
            fi = r['file_info']
            assert fi['analyzed_channel'] in UHEART_MVP_64[fi['chamber']].recording
            assert fi['tissue'] == f"-/-/chipPM01001_ch{fi['chamber']}"
            assert set(fi['electrode_scores']) <= set(UHEART_MVP_64[fi['chamber']].recording)
            res[fi['chamber']] = r['summary']['beat_period_ms_median']
        assert res['A'] == pytest.approx(800, abs=5) and res['B'] == pytest.approx(1000, abs=5)
        assert res['C'] == pytest.approx(650, abs=5) and res['D'] == pytest.approx(900, abs=5)
        upd = next(u['update'] for u in units if u['update']['chamber'] == 'C')
        r = analyze_single_file(base, channel='E9', verbose=False, config=cfg, file_info_update=upd)
        assert r['file_info']['analyzed_channel'] == 'E9'
        assert analyze_single_file(base, channel='E40', verbose=False, config=cfg, file_info_update=upd) is None

    def test_quick_scores_fast_rhythm(self, tmp_path):
        from cardiac_fp_analyzer.analyze import analyze_single_file
        from cardiac_fp_analyzer.config import AnalysisConfig
        path = _chip_file(tmp_path / ('2026-01-27T12-00-41' + NAME.format('baseline')), seed=3,
                          beat_ms=(320.0, 1000.0, 650.0, 900.0))
        upd = {'chamber': 'A', 'electrodes': list(UHEART_MVP_64['A'].electrodes),
               'stimulation_electrodes': list(UHEART_MVP_64['A'].stimulation), 'tissue': '-/-/chipPM01001_chA'}
        r = analyze_single_file(path, channel='auto', verbose=False, config=AnalysisConfig(), file_info_update=upd)
        assert r is not None
        assert r['file_info']['spike_period_ms'] == pytest.approx(320, abs=5)
        assert r['file_info']['min_distance_ms'] == pytest.approx(160, abs=5)
        assert r['summary']['beat_period_ms_median'] == pytest.approx(320, abs=5)

    def test_batch_per_chamber(self, plate):
        from cardiac_fp_analyzer.analyze import batch_analyze
        from cardiac_fp_analyzer.config import AnalysisConfig
        d, base, dose = plate
        out = batch_analyze(d, channel='auto', output_dir=d / 'out', verbose=False, config=AnalysisConfig(), n_workers=1)
        got = {(r['file_info']['chamber'], r['file_info'].get('is_baseline', False)) for r in out}
        assert {('A', True), ('B', True), ('C', True), ('D', True), ('A', False), ('B', False), ('C', False)} <= got
        for r in out:
            fi = r['file_info']
            if not fi.get('is_baseline') and fi['chamber'] != 'D':
                assert fi['tissue_electrode'] == fi['analyzed_channel'] and fi['tissue_electrode'] in UHEART_MVP_64[fi['chamber']].recording
