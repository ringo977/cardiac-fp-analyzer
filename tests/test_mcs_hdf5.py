"""MCS HDF5 raw-data files: reader, decimation, loader dispatch, pipeline.

A small file in the MCS RawData layout is written with ``write_mcs_h5``
(three electrodes at 20 kHz, a synthetic field potential on two of them,
a stimulus event stream) and read back through every entry point. When the
MCS test file ``2014-07-09T10-17-35W8 Standard all 500 Hz.h5`` (McsPyDataTools
TestData) is present in ``tests/fixtures`` or ``/tmp/mcs/TestData`` it is
read as well.
"""

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

from cardiac_fp_analyzer import mcs_hdf5 as M  # noqa: E402
from cardiac_fp_analyzer.loader import (  # noqa: E402
    electrode_columns,
    is_recording_file,
    load_recording,
    recording_datetime,
)
from tests.golden_signals import generate_regular_fp  # noqa: E402

FS = 20000.0
STEP_CODE = 866973400          # ADC step of the µHeart export: 866973400 × 1e-17 V = 8.67 nV
EXP = -17


def _codes(x_volt):
    return np.round(x_volt / (STEP_CODE * 10.0 ** EXP)).astype(np.int32)


@pytest.fixture(scope='module')
def mcs_file(tmp_path_factory):
    """Three electrodes, 12 s at 20 kHz: E1 and E2 beat (period 800 ms,
    FPD 300 ms, 60 µV spike), E61 is noise; one stimulus event stream."""
    rng = np.random.default_rng(0)
    sig, _t, meta = generate_regular_fp(fs=FS, duration_s=12.0, beat_period_ms=800.0, fpd_ms=300.0,
                                    depol_amp=60e-6, repol_amp=8e-6, noise_std=2e-6, seed=0)
    noise = rng.normal(0, 3e-6, len(sig))
    X = np.stack([_codes(sig), _codes(0.8 * sig + 0.5 * noise), _codes(noise)])
    path = tmp_path_factory.mktemp('mcs') / 'Exp1_chipA_ch1_baseline.h5'
    M.write_mcs_h5(path, X, ['E1', 'E2', 'E61'], tick_us=50, conversion_factor=STEP_CODE, exponent=EXP,
                   mea_layout='uHeart MVP 64',
                   events=[{'label': 'STG 1', 'subtype': 'StgSideband', 'times_s': [1.0, 2.0, 3.0],
                            'durations_s': [0.001, 0.001, 0.001]}])
    return path, X, meta


class TestFormat:
    def test_detects_protocol(self, mcs_file, tmp_path):
        path, _, _ = mcs_file
        assert M.is_mcs_hdf5(path)
        assert is_recording_file(path)
        plain = tmp_path / 'other.h5'
        with h5py.File(plain, 'w') as h:
            h.create_dataset('x', data=np.arange(3))
        assert not M.is_mcs_hdf5(plain)
        assert not is_recording_file(plain)
        assert not M.is_mcs_hdf5(tmp_path / 'missing.h5')

    def test_inspect(self, mcs_file):
        path, X, _ = mcs_file
        info = M.inspect(path)
        assert info['protocol_version'] == 3
        rec = info['recordings'][0]
        a = rec['analog'][0]
        assert a['channels'] == ['E1', 'E2', 'E61']
        assert a['sample_rate'] == pytest.approx(FS)
        assert a['n_samples'] == X.shape[1]
        assert a['subtype'] == 'Electrode' and a['unit'] == 'V'
        assert rec['events'][0]['subtype'] == 'StgSideband'

    def test_recording_datetime(self, mcs_file):
        path, _, _ = mcs_file
        dt = recording_datetime(path)
        assert dt is not None and dt.year >= 2026


class TestLoad:
    def test_values_without_decimation(self, mcs_file):
        path, X, _ = mcs_file
        meta, df = M.load_mcs_h5(path, max_sample_rate=None)
        assert meta['sample_rate'] == pytest.approx(FS)
        assert electrode_columns(df) == ['E1', 'E2', 'E61']
        expected = X[2].astype(float) * STEP_CODE * 10.0 ** EXP
        assert np.allclose(df['E61'].values, expected, rtol=1e-5, atol=1e-12)
        assert meta['unit'] == 'V' and meta['conversion']['E1'] == pytest.approx(8.66973400e-9)
        assert meta['paced'] is True
        assert meta['events'][0]['times_s'] == pytest.approx([1.0, 2.0, 3.0])

    def test_decimation_on_load(self, mcs_file):
        path, X, _ = mcs_file
        meta, df = load_recording(path)
        assert meta['format'] == 'mcs_hdf5'
        assert meta['sample_rate'] == pytest.approx(2000.0)
        assert meta['decimation_factor'] == 10 and meta['original_sample_rate'] == pytest.approx(FS)
        assert len(df) == -(-X.shape[1] // 10)
        assert df['time'].iloc[0] == 0 and df['time'].iloc[1] == pytest.approx(0.0005)
        # no delay: the spike of beat k sits at the same time in both versions
        full = X[0].astype(float) * STEP_CODE * 10.0 ** EXP
        k_full = int(np.argmin(full[int(2.5 * FS):int(3.5 * FS)])) + int(2.5 * FS)
        k_dec = int(np.argmin(df['E1'].values[5000:7000])) + 5000
        assert abs(k_full / FS - k_dec / 2000.0) < 0.6e-3

    def test_channel_subset_and_errors(self, mcs_file):
        path, _, _ = mcs_file
        meta, df = M.load_mcs_h5(path, channels=['E61', 'E1'])
        assert electrode_columns(df) == ['E61', 'E1'] and meta['channels'] == ['E61', 'E1']
        with pytest.raises(ValueError, match='not in the stream'):
            M.load_mcs_h5(path, channels=['E99'])
        with pytest.raises(ValueError, match='not found'):
            M.load_mcs_h5(path, stream='Filter (3)')

    def test_block_decimator_matches_whole_signal(self):
        from scipy.signal import upfirdn
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 123457))
        dec = M._Decimator(10, FS, 2)
        parts = [dec.push(x[:, a:a + 7000]) for a in range(0, x.shape[1], 7000)] + [dec.flush()]
        y = np.concatenate(parts, axis=1)[:, :-(-x.shape[1] // 10)]
        ext = np.concatenate([np.repeat(x[:, :1], dec.P // 2, 1), x, np.repeat(x[:, -1:], dec.P, 1)], 1)
        ref = upfirdn(dec.h, ext, down=10, axis=1)[:, dec.P // 10:dec.P // 10 + y.shape[1]]
        assert y.shape[1] == -(-x.shape[1] // 10)
        assert np.abs(y - ref).max() < 1e-6
        d = np.zeros((1, 50001))
        d[0, 30000] = 1.0
        dec = M._Decimator(10, FS, 1)
        z = np.concatenate([dec.push(d[:, :20000]), dec.push(d[:, 20000:]), dec.flush()], axis=1)
        assert int(np.argmax(z[0])) == 3000

    def test_segments_empty(self, mcs_file):
        path, _, _ = mcs_file
        assert M.read_segments(path) == []


class TestPipeline:
    def test_single_file_auto_and_label(self, mcs_file):
        from cardiac_fp_analyzer.analyze import analyze_single_file
        from cardiac_fp_analyzer.config import AnalysisConfig
        path, _, meta = mcs_file
        cfg = AnalysisConfig()
        r = analyze_single_file(path, channel='auto', verbose=False, config=cfg)
        assert r is not None
        fi = r['file_info']
        assert fi['analyzed_channel'] in ('E1', 'E2')
        assert fi['electrodes'] == ['E1', 'E2', 'E61']
        assert fi['paced'] is True and fi['stimulus_times_s'] == pytest.approx([1.0, 2.0, 3.0])
        assert fi['tissue'] == '-/-/chipA_ch1' and fi['is_baseline']   # experiment comes from the folder
        s = r['summary']
        assert s['beat_period_ms_median'] == pytest.approx(800.0, abs=5.0)
        assert s['fpd_ms_median'] == pytest.approx(300.0, abs=40.0)
        r2 = analyze_single_file(path, channel='E2', verbose=False, config=cfg)
        assert r2['file_info']['analyzed_channel'] == 'E2'
        assert r2['summary']['beat_period_ms_median'] == pytest.approx(800.0, abs=5.0)
        # a column that is not in the file is an error, caught like any other bad input
        assert analyze_single_file(path, channel='el1', verbose=False, config=cfg) is None

    def test_find_recordings(self, mcs_file, tmp_path):
        from cardiac_fp_analyzer.analyze import find_recordings
        path, _, _ = mcs_file
        (tmp_path / 'a.csv').write_text('#Device Name: x\ntime,el1,el2\n0,0,0\n')
        (tmp_path / 'samples.csv').write_text('file;chip\n')
        with h5py.File(tmp_path / 'plain.h5', 'w') as h:
            h.create_dataset('x', data=[1])
        (tmp_path / 'rec.h5').write_bytes(path.read_bytes())
        from cardiac_fp_analyzer.sample_sheet import is_sheet_file
        found = [p.name for p in find_recordings(tmp_path, is_sheet_file)]
        assert found == ['a.csv', 'rec.h5']


REAL = [p for p in (Path(__file__).parent / 'fixtures' / '2014-07-09T10-17-35W8 Standard all 500 Hz.h5',
                    Path('/tmp/mcs/TestData/2014-07-09T10-17-35W8 Standard all 500 Hz.h5')) if p.is_file()]


@pytest.mark.skipif(not REAL, reason='MCS test file not available')
def test_real_mcs_test_file():
    path = REAL[0]
    info = M.inspect(path)
    assert info['file']['ProgramName'] == 'Multi Channel Experimenter'
    rec = info['recordings'][0]
    assert [a['label'] for a in rec['analog']][:2] == ['Filter (1) Filter Data', 'Data Acquisition (1) Electrode Raw Data']
    meta, df = M.load_mcs_h5(path)                        # default: the raw electrode stream
    assert meta['stream'] == 'Data Acquisition (1) Electrode Raw Data'
    assert meta['sample_rate'] == 500.0 and len(df) == 9800 and electrode_columns(df)[:2] == ['E1', 'E2']
    with h5py.File(path, 'r') as h:
        raw = h['Data/Recording_0/AnalogStream/Stream_1/ChannelData'][0, :].astype(float) * 381470e-9
    assert np.abs(raw - df['E1'].values).max() < 1e-7
    assert meta['paced'] is True                           # 'Digital Events 1', DigitalPort, 12 pulses
    assert len(meta['events'][0]['times_s']) == 12
    ts = meta['spike_timestamps']
    assert set(ts) == {f'E{k}' for k in range(1, 9)} and ts['E8'] == sorted(ts['E8']) and 0.9 < ts['E8'][0] < 1.0
    segs = M.read_segments(path)
    assert segs and segs[0]['subtype'] == 'Spike' and segs[0]['waveforms'].shape[1] == 2
    assert recording_datetime(path).year == 2014
