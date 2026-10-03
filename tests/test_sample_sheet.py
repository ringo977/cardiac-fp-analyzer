"""Two tissues in one file, and the samples.csv sheet (Oct 2026).

In the GG layout each oscilloscope input records a different tissue and the
file name lists the tissues in input order. Before v3.9.0 the batch did not
pair those files at all (81 of 136 dose recordings on GG). These tests pin
the reading of such names, the sample sheet that overrides names, the draft
of that sheet, and the batch giving one recording per input. All signals
and names are synthetic.
"""

import numpy as np
import pytest

import cardiac_fp_analyzer.analyze as analyze_mod
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.loader import input_columns, parse_inputs, tissue_key
from cardiac_fp_analyzer.normalization import classify_drug, get_group_key, is_baseline, is_washout
from cardiac_fp_analyzer.sample_sheet import (
    DRAFT_NAME,
    draft_sample_sheet,
    plan_batch,
    read_sample_sheet,
)
from tests.golden_signals import generate_regular_fp


def _tissues(entries):
    return [None if e.get('empty') else (e['chip'], e['chamber'], e.get('item'), e.get('concentration'))
            for e in entries]


# ── 1. names ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('name, expected', [
    ('Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A', [('P', 1, 'K01', None), ('Q', 2, 'K02', 'A')]),
    ('Exp1_ChipP_na_ChipQ_ch2_K02_B', [None, ('Q', 2, 'K02', 'B')]),
    ('Exp1_na_ChipQ_ch3_K03_A_v2', [None, ('Q', 3, 'K03', 'A')]),
    ('Exp1_ChipP_ch1_K04_A_ChipQ_ch1_na', [('P', 1, 'K04', 'A'), None]),
    ('Exp1_chipP_ch2_K05_chipQ_ch2_K_06_C', [('P', 2, 'K05', None), ('Q', 2, 'K06', 'C')]),
    ('Exp1_chipP_ch3_K07post_rec_chipQ_ch1_vehicle_C', [('P', 3, 'K07', None), ('Q', 1, 'vehicle', 'C')]),
    ('chipA_ch1_terfe_300nM_chipB_ch2_dofe_1nM', [('A', 1, 'terfe', '300 nM'), ('B', 2, 'dofe', '1 nM')]),
    ('Exp1_ChipP_ch1_ChipQ_ch3_baseline', [('P', 1, None, None), ('Q', 3, None, None)]),
    # one tissue: 'na' after an item, or a word 'Na', is not an empty input
    ('Exp1_ChipP_ch3_K08_na_A', [('P', 3, 'K08', 'A')]),
    ('chipA_ch1_Na_blocker_1uM', [('A', 1, 'Na', '1 uM')]),
])
def test_tissues_are_read_in_input_order(name, expected):
    assert _tissues(parse_inputs(name)) == expected


def test_tissue_key_matches_the_folder_rule():
    assert tissue_key('Exp 5', 'Day7', 'b', 1) == 'exp5/day7/chipB_ch1'
    assert tissue_key(None, None, 'A', 2) == '-/-/chipA_ch2'


def _waveforms_csv(path, columns, fs=2000.0, when='2026-04-14 10:00:00.000'):
    """Digilent WaveForms CSV with one or two inputs (column labels kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(columns.values())))
    labels = ','.join(f'Channel {k} (V)' for k in columns)
    with open(path, 'w') as f:
        f.write('#Digilent WaveForms Oscilloscope Acquisition\n')
        f.write(f'#Date Time: {when}\n#Sample rate: {int(fs)}Hz\n#Samples: {n}\n\n')
        f.write(f'Time (s),{labels}\n')
        data = np.column_stack([np.arange(n) / fs] + list(columns.values()))
        np.savetxt(f, data, delimiter=',', fmt='%.6g')


def test_input_columns_reads_the_header(tmp_path):
    two, one = tmp_path / 'two.csv', tmp_path / 'one.csv'
    _waveforms_csv(two, {1: np.zeros(10), 2: np.zeros(10)})
    _waveforms_csv(one, {2: np.zeros(10)})
    assert input_columns(two) == ['1', '2']
    assert input_columns(one) == ['2']
    (tmp_path / 'empty.csv').write_text('')
    assert input_columns(tmp_path / 'empty.csv') == []


# ── 2. the sample sheet ────────────────────────────────────────────────


def test_sheet_accepts_semicolons_and_italian_headers(tmp_path):
    p = tmp_path / 'samples.csv'
    p.write_text('file;elettrodo;esperimento;chip;camera;farmaco;dose;escludi;note\n'
                 'A/f1.csv;el2;Exp4;chipS;ch2;K31;A;;\n'
                 'A/f1.csv;1;;;;;;no tissue;\n'
                 'A/f2.csv;input 3;;A;1;K01;A;;\n'
                 'A/f3.csv;;;;;K01;A;;\n', encoding='utf-8-sig')
    rows, problems = read_sample_sheet(p)
    assert [(r.file, r.electrode, r.experiment, r.chip, r.chamber, r.item, r.dose, r.exclude) for r in rows] == [
        ('A/f1.csv', 'el2', 'Exp4', 'S', 2, 'K31', 'A', ''), ('A/f1.csv', 'el1', '', '', None, '', '', 'no tissue')]
    assert len(problems) == 2 and 'electrode' in problems[0] and 'chip and chamber' in problems[1]


def _write_names(root, names):
    for rel, inputs in names.items():
        _waveforms_csv(root / rel, {k: np.zeros(20) for k in inputs})


def test_plan_gives_one_recording_per_input_of_a_two_tissue_file(tmp_path):
    _write_names(tmp_path, {
        'Exp1/Day7/A/Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv': (1, 2),
        'Exp1/Day7/A/Exp1_ChipP_na_ChipQ_ch3_K03_A.csv': (1, 2),
        'Exp1/Day7/A/Exp1_ChipR_ch1_K04_A.csv': (1, 2),
        'Exp1/Day7/baseline/Exp1_ChipP_ch1_ChipQ_ch2_baseline.csv': (1, 2),
    })
    files = sorted(tmp_path.rglob('*.csv'))
    units, report = plan_batch(files, tmp_path)
    got = [(u['file'].name, u['electrode'], (u['update'] or {}).get('tissue'), (u['update'] or {}).get('drug'),
            (u['update'] or {}).get('concentration'), u['multi']) for u in units]
    assert got == [
        ('Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv', 'el1', 'exp1/day7/chipP_ch1', 'K01', 'A', True),
        ('Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv', 'el2', 'exp1/day7/chipQ_ch2', 'K02', 'A', True),
        ('Exp1_ChipP_na_ChipQ_ch3_K03_A.csv', 'el2', 'exp1/day7/chipQ_ch3', 'K03', 'A', False),
        ('Exp1_ChipR_ch1_K04_A.csv', 'auto', None, None, None, False),       # one tissue: as before
        ('Exp1_ChipP_ch1_ChipQ_ch2_baseline.csv', 'el1', 'exp1/day7/chipP_ch1', 'baseline', '0', True),
        ('Exp1_ChipP_ch1_ChipQ_ch2_baseline.csv', 'el2', 'exp1/day7/chipQ_ch2', 'baseline', '0', True),
    ]
    assert report['two_tissue_files'] == 2
    assert report['excluded'] == [('Exp1/Day7/A/Exp1_ChipP_na_ChipQ_ch3_K03_A.csv', 'el1',
                                   'no tissue on this input (na)')]


def test_the_sheet_takes_precedence_over_names(tmp_path):
    _write_names(tmp_path, {
        'Day8/A/Exp2_ChipR_ch1_K21_ChipS_ch1_K22_A.csv': (1, 2),
        'Day8/A/Exp2_ChipR_ch2_K23_A_only.csv': (1, 2),
        'Day8/A/Exp2_ChipR_ch3_K24_A.csv': (1, 2),
    })
    (tmp_path / 'samples.csv').write_text(
        'file,electrode,experiment,chip,chamber,item,dose,exclude,note\n'
        'Day8/A/Exp2_ChipR_ch1_K21_ChipS_ch1_K22_A.csv,el1,,R,1,K21,A,,\n'
        'Day8/A/Exp2_ChipR_ch1_K21_ChipS_ch1_K22_A.csv,el2,Exp3,S,1,K22,A,,chip of Exp3\n'
        'Exp2_ChipR_ch2_K23_A_only.csv,el2,,R,2,K23,A,,\n'
        'Exp2_ChipR_ch2_K23_A_only.csv,el1,,,,,,input 1 empty,\n'
        'missing.csv,el1,,R,1,K21,A,,\n')
    files = sorted(p for p in tmp_path.rglob('*.csv') if p.name != 'samples.csv')
    units, report = plan_batch(files, tmp_path)
    got = [(u['file'].name, u['electrode'], u['update']['tissue'] if u['update'] else None, u['source'])
           for u in units]
    assert got == [
        ('Exp2_ChipR_ch1_K21_ChipS_ch1_K22_A.csv', 'el1', '-/day8/chipR_ch1', 'samples.csv'),
        ('Exp2_ChipR_ch1_K21_ChipS_ch1_K22_A.csv', 'el2', 'exp3/day8/chipS_ch1', 'samples.csv'),
        ('Exp2_ChipR_ch2_K23_A_only.csv', 'el2', '-/day8/chipR_ch2', 'samples.csv'),
        ('Exp2_ChipR_ch3_K24_A.csv', 'auto', None, ''),
    ]
    assert report['not_in_sheet'] == ['Day8/A/Exp2_ChipR_ch3_K24_A.csv']
    assert report['excluded'] == [('Day8/A/Exp2_ChipR_ch2_K23_A_only.csv', 'el1', 'input 1 empty')]
    assert any("'missing.csv' not found" in m for m in report['issues'])
    upd = units[1]['update']
    assert upd['experiment'] == 'Exp3' and upd['two_tissue_file'] is True and upd['role'] == 'dose'


def test_roles_from_the_sheet_decide_baseline_and_washout():
    base = {'metadata': {'filename': 'chipA_ch1_baseline_v2'}}
    assert is_baseline(base) is True
    assert is_baseline({**base, 'file_info': {'role': 'dose', 'drug': 'K01'}}) is False
    assert is_washout({'file_info': {'role': 'washout'}, 'metadata': {'filename': 'x'}}) is True
    assert is_washout({'file_info': {'role': 'dose'}, 'metadata': {'filename': 'x_wash'}}) is False


def test_group_key_ignores_the_input_when_the_batch_says_so():
    r = {'file_info': {'tissue': 'exp1/day7/chipP_ch1', 'analyzed_channel': 'el2'}, 'metadata': {}}
    assert get_group_key(r) == 'exp1/day7/chipP_ch1/el2'
    r['file_info']['group_electrode'] = ''
    assert get_group_key(r) == 'exp1/day7/chipP_ch1'


# ── 3. the draft ───────────────────────────────────────────────────────


def test_draft_lists_inputs_and_flags_what_to_check(tmp_path):
    _write_names(tmp_path, {
        'Day7/baseline/Exp1_ChipP_ch1_ChipQ_ch2_baseline.csv': (1, 2),
        'Day7/A/Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv': (1, 2),
        'Day7/B/Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_B.csv': (1, 2),
        'Day7/C/Exp1_ChipP_ch1_K01_ChipQ_ch2_K09_C.csv': (1, 2),        # item typed wrong
        'Day7/C/Exp1_ChipP_ch1_K01_C_v2.csv': (1,),                       # repeat
        'Day7/A/Exp1_ChipR_ch11_K05_A.csv': (1, 2),                       # chamber 11, two inputs
        'Day7/A/Exp1_ChipS_ch2_vehicle_ChipT_K06_ch3_A.csv': (1, 2),      # second tissue not read
    })
    rows, path = draft_sample_sheet(tmp_path)
    assert path == tmp_path / DRAFT_NAME and path.exists()
    note = {(r['file'].rsplit('/', 1)[-1], r['electrode']): r['note'] for r in rows}
    assert 'test item changes in this chamber' in note[('Exp1_ChipP_ch1_K01_ChipQ_ch2_K09_C.csv', 'el2')]
    assert '2 recordings of this chamber at C' in note[('Exp1_ChipP_ch1_K01_C_v2.csv', '')]
    assert 'chamber 11' in note[('Exp1_ChipR_ch11_K05_A.csv', '')]
    assert 'two inputs, one tissue named' in note[('Exp1_ChipR_ch11_K05_A.csv', '')]
    assert 'no reference for this chamber' in note[('Exp1_ChipR_ch11_K05_A.csv', '')]
    assert "2 'chip' but 1 tissue" in note[('Exp1_ChipS_ch2_vehicle_ChipT_K06_ch3_A.csv', '')]
    assert note[('Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv', 'el1')] == ''
    # the draft is read back as a sheet, and a sheet file is not a recording
    sheet_rows, problems = read_sample_sheet(path)
    assert problems == [] and len(sheet_rows) == len(rows)


# ── 4. the batch, on synthetic signals ─────────────────────────────────


def _two_tissue_series(root, fpd1, fpd2):
    """Chip P chamber 1 on input 1 (beat 800 ms), chip Q chamber 2 on
    input 2 (beat 1200 ms); a reference and doses A, B, C."""
    times = {'baseline': '10:00', 'A': '10:30', 'B': '11:00', 'C': '11:30'}
    for k, cond in enumerate(('baseline', 'A', 'B', 'C')):
        s1, _, _ = generate_regular_fp(duration_s=30.0, beat_period_ms=800.0, fpd_ms=fpd1[k], seed=k)
        s2, _, _ = generate_regular_fp(duration_s=30.0, beat_period_ms=1200.0, fpd_ms=fpd2[k], seed=10 + k)
        name = ('Exp1_ChipP_ch1_ChipQ_ch2_baseline' if cond == 'baseline'
                else f'Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_{cond}')
        _waveforms_csv(root / 'Exp1' / 'Day7' / cond / f'{name}.csv', {1: s1, 2: s2},
                       when=f'2026-04-14 {times[cond]}:00.000')


@pytest.fixture
def no_reports(monkeypatch):
    monkeypatch.setattr(analyze_mod, 'generate_excel_report', lambda *a, **k: None)
    monkeypatch.setattr(analyze_mod, 'generate_pdf_report', lambda *a, **k: None)


def _config():
    cfg = AnalysisConfig()
    cfg.inclusion.enabled_confidence = False    # noise-free synthetic beats score 0.5 on it
    return cfg


def test_batch_analyses_each_input_as_its_own_tissue(tmp_path, no_reports):
    _two_tissue_series(tmp_path, fpd1=(300, 300, 345, 390), fpd2=(400, 400, 400, 400))
    cfg = _config()
    results = analyze_mod.batch_analyze(tmp_path, channel='auto', output_dir=tmp_path / 'out',
                                        verbose=False, config=cfg)
    assert len(results) == 8                                # 4 files × 2 inputs
    by = {(r['file_info']['tissue'], r['file_info'].get('concentration')): r for r in results}
    for (tissue, conc), r in by.items():
        el = r['file_info']['analyzed_channel']
        assert el == ('el1' if tissue.endswith('chipP_ch1') else 'el2')
        assert r['metadata']['filename'].endswith(f'[{el}]')
        bp = r['summary']['beat_period_ms_median']
        assert abs(bp - (800 if el == 'el1' else 1200)) < 30, (tissue, conc, bp)
        if conc in ('A', 'B', 'C'):
            n = r['normalization']
            assert n['has_baseline'] and n['baseline_file'] == f"Exp1_ChipP_ch1_ChipQ_ch2_baseline [{el}]"
    p = [by[('exp1/day7/chipP_ch1', c)]['normalization']['pct_fpdc_change'] for c in 'ABC']
    q = [by[('exp1/day7/chipQ_ch2', c)]['normalization']['pct_fpdc_change'] for c in 'ABC']
    assert abs(p[0]) < 3 and 10 < p[1] < 20 and 25 < p[2] < 35, p
    assert all(abs(x) < 3 for x in q), q
    assert all(r['inclusion'].get('passed', True) for r in results)
    calls = classify_drug(results, cfg=cfg.normalization)
    assert set(calls) == {'k01', 'k02'}
    assert [row['n_tissues'] for row in calls['k01']['per_concentration']] == [1, 1, 1]
    assert results[0]['batch_plan']['two_tissue_files'] == 4


def test_batch_follows_the_sheet(tmp_path, no_reports):
    _two_tissue_series(tmp_path, fpd1=(300, 300, 300, 300), fpd2=(400, 400, 440, 480))
    # the name says chip Q chamber 2; the sheet says the tissue on input 2 is
    # chip Q chamber 3 of another experiment, and leaves input 1 of dose B out
    lines = ['file,electrode,experiment,chip,chamber,item,dose,exclude']
    for cond in ('baseline', 'A', 'B', 'C'):
        name = ('Exp1_ChipP_ch1_ChipQ_ch2_baseline' if cond == 'baseline'
                else f'Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_{cond}')
        rel = f'Exp1/Day7/{cond}/{name}.csv'
        lines.append(f"{rel},el1,,P,1,K01,{cond},{'operator note' if cond == 'B' else ''}")
        lines.append(f'{rel},el2,Exp2,Q,3,K02,{cond},')
    (tmp_path / 'samples.csv').write_text('\n'.join(lines) + '\n')
    results = analyze_mod.batch_analyze(tmp_path, channel='auto', output_dir=tmp_path / 'out',
                                        verbose=False, config=_config())
    assert len(results) == 7
    tissues = {r['file_info']['tissue'] for r in results}
    assert tissues == {'exp1/day7/chipP_ch1', 'exp2/day7/chipQ_ch3'}
    q = sorted((r['file_info']['concentration'], r['normalization']['pct_fpdc_change'])
               for r in results if r['file_info']['tissue'] == 'exp2/day7/chipQ_ch3' and not is_baseline(r))
    assert [c for c, _ in q] == ['A', 'B', 'C']
    assert abs(q[0][1]) < 3 and 5 < q[1][1] < 15 and 15 < q[2][1] < 25, q
    assert all(r['file_info']['source'] == 'samples.csv' for r in results)
    plan = results[0]['batch_plan']
    assert plan['sheets'] == ['samples.csv'] and len(plan['excluded']) == 1
