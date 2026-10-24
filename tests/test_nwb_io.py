"""NWB 2 files (cardiac_fp_analyzer.nwb_io): CSV → .nwb with the WaveForms header and the tissue identity,
reading back through the loader, planning and analysis identical to the CSV."""
from __future__ import annotations

import numpy as np
import pytest

from cardiac_fp_analyzer.analyze import analyze_single_file
from cardiac_fp_analyzer.loader import (
    describe_recording,
    input_columns,
    is_recording_file,
    load_recording,
    recording_channels,
    recording_datetime,
)
from cardiac_fp_analyzer.nwb_io import csv_to_nwb, inputs_of_csv, read_header, read_nwb, validate
from cardiac_fp_analyzer.results_export import plan_recordings
from tests.test_review_model import FS, _trace

pytest.importorskip('pynwb')

HEADER = ("#Digilent WaveForms Oscilloscope Acquisition\n#Device Name: Discovery2\n#Serial Number: SN:210321A80B62\n"
          "#Date Time: 2026-04-15 15:36:06.081\n#Sample rate: 2000Hz\n#Samples: {n}\n"
          "#Trigger: Source: Channel 1 Type: Edge Condition: Rising Level: 0 V Hyst.: Auto HoldOff: 0 s\n"
          "#Channel 1: Range: 500 mV/div Offset: 0 V Sample Mode: Average\n"
          "#Channel 2: Range: 500 mV/div Offset: 0 V Sample Mode: Average\n\n")


def _write_csv(path, raw, two_inputs=True, t0=-10.0):
    t = t0 + np.arange(len(raw)) / FS
    with open(path, 'w') as f:
        f.write(HEADER.format(n=len(raw)))
        if two_inputs:
            f.write('Time (s),Channel 1 (V),Channel 2 (V)\n')
            for ti, xi in zip(t, raw):
                f.write(f'{ti:.6f},{xi:.9e},{0.5 * xi:.9e}\n')
        else:
            f.write('Time (s),Channel 1 (V)\n')
            for ti, xi in zip(t, raw):
                f.write(f'{ti:.6f},{xi:.9e}\n')


@pytest.fixture(scope='module')
def two_tissue_csv(tmp_path_factory):
    x, raw, beats, reps = _trace(dur_s=30.0, amp=1.0, noise=0.02, pos_wave=0.3)
    d = tmp_path_factory.mktemp('nwb') / 'Exp 7' / 'Day8' / 'B'
    d.mkdir(parents=True)
    p = d / 'Exp7_chipC_ch3_Ti05_chipA_ch3_Ti06_B.csv'
    _write_csv(p, raw * 1e-4)
    return p


def test_header_and_inputs(two_tissue_csv):
    lines, fields = read_header(two_tissue_csv)
    assert fields['device'] == 'Discovery2' and fields['sample_rate'] == 2000.0 and fields['channels']['1']['sample_mode'] == 'Average'
    assert len(lines) == 9
    ins = inputs_of_csv(two_tissue_csv)
    assert [(i['label'], i['chip'], i['chamber'], i['item'], i['condition']) for i in ins] == \
        [('el1', 'C', '3', 'Ti05', 'B'), ('el2', 'A', '3', 'Ti06', 'B')]
    assert ins[0]['tissue'] == 'exp7/day8/chipC_ch3' and ins[0]['experiment'] == 'Exp 7'


def test_round_trip_two_inputs(two_tissue_csv, tmp_path):
    out = csv_to_nwb(two_tissue_csv, tmp_path / 'rec.nwb', experimenter=['Rasponi, Marco'], institution='PoliMi')
    assert out.exists() and is_recording_file(out) and not is_recording_file(tmp_path / 'nope.nwb')
    md0, df0 = load_recording(two_tissue_csv, max_sample_rate=None)
    md, df = read_nwb(out)
    assert md['format'] == 'nwb' and md['sample_rate'] == 2000.0 and md['n_inputs'] == 2 and md['gain'] == 10000.0
    assert df['time'].iloc[0] == 0.0 and len(df) == len(df0)
    np.testing.assert_allclose(df['el1'].values, df0['el1'].values / 1e4, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(df['el2'].values, df0['el2'].values / 1e4, rtol=1e-6, atol=1e-12)
    n = md['nwb']
    assert n['session_id'] == 'Exp7_Day8_chipC-ch3+chipA-ch3_B' and n['session_start_time'].startswith('2026-04-15T15:35:56.081')
    assert 'Average' in n['filtering'] and 'trigger at t = 10.000 s' in n['comments'] and 'WaveForms header' in n['notes']
    assert [r['label'] for r in n['inputs']] == ['el1', 'el2'] and n['inputs'][1]['chip'] == 'A' and n['inputs'][1]['group_name'] == 'chipA_ch3'
    assert md['sample_mode'] == 'Average' and md['device'] == 'Discovery2'
    # loader-level helpers
    assert recording_channels(out) == ['el1', 'el2'] and input_columns(out) == ['1', '2']
    assert recording_datetime(out).year == 2026
    info = describe_recording(str(out))
    assert info['dual_tissue'] and info['tissue_tokens'] == ['chipC_ch3', 'chipA_ch3'] and info['experiment'] == 'Exp 7'
    # the inspector is happy (institution given, experimenter in DANDI form)
    issues = validate(out)
    if issues is not None:
        assert not [i for i in issues if i.startswith('CRITICAL') or 'VIOLATION' in i], issues
    # same analysis as the CSV, tissue identity from the file
    ra = analyze_single_file(str(two_tissue_csv), channel='el2', verbose=False)
    rb = analyze_single_file(str(out), channel='el2', verbose=False)
    assert np.array_equal(ra['beat_indices_raw'], rb['beat_indices_raw']) and len(rb['beat_indices_raw']) >= 12
    assert rb['summary']['beat_period_ms_mean'] == pytest.approx(ra['summary']['beat_period_ms_mean'], rel=1e-6)
    assert rb['metadata']['gain_applied'] == 1.0 and 'NWB' in rb['metadata']['processing']['testo']
    # batch planning: one unit per tissue, identities from the electrodes table
    units, plan = plan_recordings([out], tmp_path)
    assert [(u['electrode'], u['update']['chip'], u['update']['chamber'], u['update']['concentration']) for u in units] == \
        [('el1', 'C', 3, 'B'), ('el2', 'A', 3, 'B')]
    assert plan['two_tissue_files'] == 1
    with pytest.raises(FileExistsError):
        csv_to_nwb(two_tissue_csv, out)


def test_single_input_and_sample_sheet(tmp_path):
    x, raw, beats, reps = _trace(dur_s=20.0, amp=1.0, noise=0.02)
    exp = tmp_path / 'Exp 3'
    (exp / 'Day7' / 'baseline').mkdir(parents=True)
    p = exp / 'Day7' / 'baseline' / 'Exp3_chipB_ch2_baseline.csv'
    _write_csv(p, raw * 1e-4, two_inputs=False, t0=0.0)
    (exp / 'samples.csv').write_text('file;electrode;experiment;chip;chamber;item;dose;exclude;note\n'
                                     f'Day7/baseline/{p.name};auto;;B;2;;baseline;;\n')
    ins = inputs_of_csv(p)
    assert len(ins) == 1 and ins[0]['chip'] == 'B' and ins[0]['chamber'] == '2' and ins[0]['condition'] == 'baseline'
    out = csv_to_nwb(p, tmp_path / 'single.nwb')
    md, df = read_nwb(out)
    assert md['n_inputs'] == 1 and list(df.columns) == ['time', 'el1', 'el2'] and np.array_equal(df['el1'], df['el2'])
    assert md['nwb']['inputs'][0]['condition'] == 'baseline' and recording_channels(out) == ['el1']
    info = describe_recording(str(out))
    assert info['chip'] == 'B' and info['channel'] == 2 and info['is_baseline'] and info['tissue'] == 'exp3/day7/chipB_ch2'
    units, _ = plan_recordings([out], tmp_path)
    assert len(units) == 1 and units[0]['electrode'] == 'auto'
    r = analyze_single_file(str(out), verbose=False)
    assert r['file_info']['is_baseline'] and r['file_info']['chip'] == 'B' and len(r['beat_indices_raw']) >= 8
