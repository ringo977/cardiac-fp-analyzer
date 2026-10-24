"""Results workbook export (cardiac_fp_analyzer.results_export): records, grouping, sheets, formulas."""
from __future__ import annotations

import numpy as np
import pytest

from cardiac_fp_analyzer.results_export import (
    BASELINE,
    _sheet_groups,
    _sheet_names,
    condition_label,
    condition_sort_key,
    record_from_state,
    records_from_folder,
    sheet_root,
    write_results_workbook,
)
from tests.test_review_model import FS, _state, _trace

openpyxl = pytest.importorskip('openpyxl')


def _record(chip='A', chamber=1, item='Ti03', condition='A', *, experiment='Exp 5', day='Day8', bp=2.0, fpd=0.9,
            amp=300.0, n=20, level='verde', corrected=False, name=None):
    beats = [{'n': k + 1, 'dep_s': 0.5 + k * bp, 'bp_s': bp if k + 1 < n else None, 'rep_s': 0.5 + k * bp + fpd,
              'fpd_s': fpd, 'v_up': 0.9, 'v_down': -2.1, 'amp_uV': amp, 'stato': 'auto', 'stato_rip': 'auto'}
             for k in range(n)]
    return {'file': f'/data/{name or f"chip{chip}_ch{chamber}_{condition}.csv"}',
            'name': name or f'chip{chip}_ch{chamber}_{condition}.csv', 'electrode': 'el1',
            'experiment': experiment, 'day': day, 'chip': chip, 'chamber': chamber,
            'item': None if condition == BASELINE else item, 'condition': condition, 'is_baseline': condition == BASELINE,
            'measures': {'n_battiti': n, 'n_intervalli': n - 1, 'intervalli_esclusi': 0, 'n_fpd': n, 'bp_s': bp,
                         'bp_sd_s': 0.05, 'bp_cv_pct': 2.5, 'fpd_s': fpd, 'amp_uV': amp, 'n_vetoed': 0,
                         'burst_frac': 0.0, 'fpdc_s': fpd / bp ** (1 / 3)},
            'beats': beats, 'changes': {'tolti': 1 if corrected else 0, 'aggiunti': 0, 'spostati': 0, 'ripristinati': 0,
                                        'ripol_corrette': 0, 'ripol_propagate': 0, 'polarita': None, 'corrected': corrected},
            'quality': {'livello': level, 'titolo': 'Buono', 'problemi': [] if level == 'verde' else ['rumore']},
            'not_analysable': False, 'processing': 'Filtraggio: test', 'version': 'x'}


def test_condition_label_and_order():
    assert condition_label({'is_baseline': True, 'concentration': '0'}) == BASELINE
    assert condition_label({'is_baseline': False, 'concentration': 'B', 'item': 'Ti03'}) == 'B'
    assert condition_label({'is_baseline': False, 'concentration': None, 'item': 'vehicle'}) == 'vehicle'
    labels = ['C', BASELINE, '0.1 µM', 'A', 'vehicle', '10 nM']
    assert sorted(labels, key=condition_sort_key) == [BASELINE, '10 nM', '0.1 µM', 'A', 'C', 'vehicle']


def test_sheet_root_walks_up_to_the_sample_sheet(tmp_path):
    exp = tmp_path / 'Exp 5'
    (exp / 'Day8' / 'B').mkdir(parents=True)
    assert sheet_root(exp / 'Day8' / 'B') == (exp / 'Day8' / 'B').resolve()
    (exp / 'samples.csv').write_text('file;electrode\n')
    assert sheet_root(exp / 'Day8' / 'B') == exp.resolve()
    assert sheet_root(exp) == exp.resolve()


def test_groups_keep_the_better_duplicate_and_name_sheets():
    recs = [_record(condition=BASELINE), _record(condition='A'), _record(condition='B', level='giallo'),
            _record(condition='B', level='verde', name='repeat_B.csv'),
            _record(chip='A', chamber=1, condition=BASELINE, experiment='Exp6', name='exp6_base.csv'),
            _record(chip='C', chamber=2, condition=BASELINE, name='c2.csv')]
    groups = _sheet_groups(recs)
    assert list(groups) == [('Exp 5', 'Day8', 'A'), ('Exp 5', 'Day8', 'C'), ('Exp6', 'Day8', 'A')]
    kept = groups[('Exp 5', 'Day8', 'A')][1]['B']
    assert kept['name'] == 'repeat_B.csv' and kept['alternates'] == ['chipA_ch1_B.csv']
    names = _sheet_names(groups)
    assert names[('Exp 5', 'Day8', 'A')] == 'Exp5 CHIP A' and names[('Exp6', 'Day8', 'A')] == 'Exp6 CHIP A'
    assert names[('Exp 5', 'Day8', 'C')] == 'Exp5 CHIP C'
    one_exp = _sheet_groups([r for r in recs if r['experiment'] == 'Exp 5'])
    assert set(_sheet_names(one_exp).values()) == {'CHIP A', 'CHIP C'}


def test_workbook_layout_and_formulas(tmp_path):
    recs = [_record(condition=BASELINE, bp=2.19, fpd=1.18, amp=367.0, n=28),
            _record(condition='A', bp=2.38, fpd=1.22, amp=299.0, n=25, corrected=True),
            _record(chamber=2, item='Ti04', condition=BASELINE, bp=1.8, fpd=1.06, n=30),
            _record(chamber=2, item='Ti04', condition='A', bp=2.5, fpd=1.3, n=24, level='rosso')]
    out = write_results_workbook(recs, tmp_path / 'Exp5_FP_analysis.xlsx')
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ['samples', 'CHIP A', 'CHIP A raw']
    ws = wb['CHIP A']
    assert ws['A1'].value == 'Exp5_FP_analysis' and ws['A3'].value == 'DAY 8' and ws['A4'].value == 'CHIP A'
    assert ws['A6'].value == 'Ch1 (Ti03)' and ws['B7'].value == 'Beating Period [s]'
    assert ws['A8'].value == BASELINE and ws['B8'].value == pytest.approx(2.19) and ws['G8'].value == pytest.approx(1.18)
    assert ws['D8'].value == '=C8/B8*100' and ws['J8'].value == '=G8/(B8^(1/3))' and ws['E8'].value == '=B8/$B$8*100'
    assert ws['A10'].value == 'A' and ws['E10'].value == '=B10/$B$8*100' and ws['F10'].value == '=E10-100'
    assert ws['K10'].value == '=J10/$J$8*100' and ws['N10'].value == '=M10/$M$8*100'
    assert ws['P10'].value.startswith('not yet approved — corrected by the operator: beats 1 removed')
    assert ws['P8'].value == 'not yet approved — automatic' and ws['Q8'].value == 'chipA_ch1_baseline.csv' and ws['R8'].value == 'el1'
    # two conditions per channel: the next block starts after three blank rows (the lab's spacing)
    assert ws['A14'].value == 'Ch2 (Ti04)' and ws['A16'].value == BASELINE and ws['A18'].value == 'A'
    assert 'rumore' in ws['P18'].value
    raw = wb['CHIP A raw']
    assert raw['B5'].value == 'Ch1 (Ti03)' and raw['K5'].value == 'Ch2 (Ti04)'
    assert raw['A7'].value == BASELINE and raw['B7'].value == 'Dep. time [s]' and raw['H7'].value == 'Dep. Voltage [µV]'
    assert raw['B8'].value == pytest.approx(0.5) and raw['C8'].value == '=B9-B8' and raw['E8'].value == '=D8-B8'
    assert raw['H8'].value == '=(F8-G8)*100' and raw['C35'].value is None          # last beat: no interval
    # the baseline block of ch1 has 28 beats, ch2 30: Mean row aligned on the longest (row 8 + 30 + 1)
    assert raw['A39'].value == 'Mean' and raw['J39'].value == 'Mean' and raw['A40'].value == 'St. dev'
    assert raw['C39'].value == '=IFERROR(AVERAGE(C8:C35),"")' and raw['L40'].value == '=IFERROR(STDEV(L8:L37),"")'
    assert raw['A44'].value == 'A' and raw['J44'].value == 'A'          # next condition after 3 blank rows
    samples = wb['samples']
    assert samples['A4'].value == '#exp' and samples.max_row == 4 + 4
    assert [samples.cell(5, j).value for j in (3, 4, 6, 13, 15, 18)] == ['A', 1, BASELINE, 'good', 'no', 'to review']


def test_record_from_state_uses_the_review_measures():
    st, beats, reps = _state(dur_s=30.0)
    result = {'file_info': {'chip': 'B', 'channel': 2, 'experiment': 'Exp 1', 'day': 'Day7', 'is_baseline': False,
                            'concentration': 'A', 'drug': 'Ti01', 'analyzed_channel': 'el1'},
              'metadata': {'sample_rate': FS, 'processing': {'testo': 'Filtraggio: x'}}, 'summary': {'not_analysable': False}}
    rec = record_from_state(st, result, file='/d/f.csv', electrode='el1', identity={'item': 'Ti01', 'chamber': 2},
                            quality={'livello': 'verde', 'titolo': 'Buono', 'problemi': []})
    assert rec['chip'] == 'B' and rec['chamber'] == 2 and rec['condition'] == 'A' and rec['item'] == 'Ti01'
    assert rec['measures']['n_battiti'] == len(beats) and rec['measures']['bp_s'] == pytest.approx(2.0, abs=1e-6)
    assert rec['measures']['fpd_s'] == pytest.approx(0.9, abs=1e-3) and len(rec['beats']) == len(beats)
    assert rec['changes']['corrected'] is False and rec['processing'] == 'Filtraggio: x'
    st.remove_beat(0)
    rec2 = record_from_state(st, result, file='/d/f.csv', electrode='el1', quality={'livello': 'verde', 'titolo': 'B', 'problemi': []})
    assert rec2['changes'] == {**rec2['changes'], 'tolti': 1, 'corrected': True}


def test_records_from_folder_with_sample_sheet(tmp_path):
    x, raw, beats, reps = _trace(dur_s=40.0, amp=1.0, noise=0.02, pos_wave=0.3)
    t = np.arange(len(x)) / FS
    exp = tmp_path / 'Exp 9'
    (exp / 'Day7' / 'baseline').mkdir(parents=True)
    p = exp / 'Day7' / 'baseline' / 'Exp9_chipA_ch1_chipB_ch1_baseline.csv'
    with open(p, 'w') as f:
        f.write('#Sample rate: 2000Hz\nTime (s),Channel 1 (V),Channel 2 (V)\n')
        for ti, xi in zip(t, raw * 1e-4):
            f.write(f'{ti:.6f},{xi:.9e},{0.5 * xi:.9e}\n')
    (exp / 'samples.csv').write_text('file;electrode;experiment;chip;chamber;item;dose;exclude;note\n'
                                     f'Day7/baseline/{p.name};el1;;A;1;;baseline;;\n'
                                     f'Day7/baseline/{p.name};el2;Exp10;B;1;Ti01;A;;\n')
    recs = records_from_folder(exp / 'Day7', progress=lambda i, n, f: None)
    assert len(recs) == 2
    a, b = sorted(recs, key=lambda r: r['electrode'])
    assert (a['chip'], a['chamber'], a['condition'], a['experiment']) == ('A', 1, BASELINE, 'Exp 9')
    assert (b['chip'], b['chamber'], b['condition'], b['item'], b['experiment']) == ('B', 1, 'A', 'Ti01', 'Exp10')
    assert a['measures']['n_battiti'] >= 15 and a['measures']['bp_s'] == pytest.approx(2.0, abs=0.05)
    out = write_results_workbook(recs, tmp_path / 'out.xlsx')
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ['samples', 'Exp9 CHIP A', 'Exp9 CHIP A raw', 'Exp10 CHIP B', 'Exp10 CHIP B raw']
