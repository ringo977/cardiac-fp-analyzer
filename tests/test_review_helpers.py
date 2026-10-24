"""Qt-free helpers of the Revisione screen (pyside_app.review_helpers)."""
import numpy as np
from pyside_app.review_helpers import (
    BEAT_COLUMNS,
    format_beat_rows,
    format_measures,
    inputs_of,
    operators_view,
    processing_text,
    queue_entry,
    queue_sort_key,
)

FS = 2000.0


def test_operators_view_keeps_the_1_40_band():
    t = np.arange(int(6 * FS)) / FS
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 300 * t)
    y = operators_view(x, FS)
    core = slice(int(FS), int(5 * FS))
    assert len(y) == len(x)
    assert np.std(y[core] - np.sin(2 * np.pi * 10 * t[core])) < 0.2
    assert len(operators_view(np.zeros(10), FS)) == 10


def test_inputs_of_csv_and_unknown(tmp_path):
    assert inputs_of(tmp_path / 'rec.csv') == [('el1', None), ('el2', None)]
    assert inputs_of(tmp_path / 'rec.txt') == [('auto', None)]


def test_queue_entry_and_sort_order(tmp_path):
    good = {'metadata': {'sample_rate': FS}, 'file_info': {'analyzed_channel': 'el1'}, 'detection_info': {},
            'arrhythmia_report': None,
            'summary': {'not_analysable': False, 'fpd_ms_median': 900.0, 'fpd_valid_ratio': 0.9, 'fpd_reliable': True,
                        'fpd_confidence': 0.9, 'beat_snr_median': 12.0, 'beat_period_ms_mean': 2000.0},
            'beat_indices_raw': np.arange(1000, 1000 + 30 * 4000, 4000)}
    bad = dict(good, summary=dict(good['summary'], not_analysable=True, not_analysable_reason='only 2 depolarisation(s) detected (< 3)'))
    e1 = queue_entry(tmp_path / 'a.csv', 'el1', None, good)
    e2 = queue_entry(tmp_path / 'a.csv', 'el2', None, bad)
    e3 = queue_entry(tmp_path / 'b.csv', 'el1', None, None, error='boom')
    assert e1['livello'] == 'verde' and e1['bp_s'] == 2.0 and e1['fpd_s'] == 0.9 and e1['n_battiti'] == 30
    assert e2['livello'] == 'rosso' and 'solo 2' in e2['riassunto']
    assert e3['livello'] == 'rosso' and e3['errore'] == 'boom'
    order = sorted([e1, e2, e3], key=queue_sort_key)
    assert [e['livello'] for e in order] == ['rosso', 'rosso', 'verde']
    e1['rivista'] = True
    assert queue_sort_key(e1) > queue_sort_key(dict(e1, rivista=False))


def test_format_measures_and_rows():
    m = {'bp_s': 2.1898, 'bp_sd_s': 0.0477, 'bp_cv_pct': 2.18, 'fpd_s': 1.1839, 'fpdc_s': 0.9117, 'amp_uV': 367.4,
         'n_battiti': 28, 'intervalli_esclusi': 0, 'n_fpd': 27, 'n_vetoed': 0, 'burst_frac': 0.41,
         'bp_s_pct': 108.5, 'bp_s_var': 8.5}
    rows = format_measures(m)
    assert rows[0] == ('Beating Period [s]', '2.190', '108.5', '+8.5')
    assert rows[5][1] == '367' and rows[-1][1] == '41 %'
    assert len(rows) == 13 and rows[6][0].startswith('Deriva frequenza')
    beats = format_beat_rows([{'n': 1, 'dep_s': 0.4278, 'bp_s': 2.273, 'rep_s': 1.6148, 'fpd_s': 1.187, 'amp_uV': 352.5,
                               'stato': 'auto', 'stato_rip': 'corretto'},
                              {'n': 2, 'dep_s': 2.7, 'bp_s': None, 'rep_s': None, 'fpd_s': None, 'amp_uV': 10.0,
                               'stato': 'aggiunto', 'stato_rip': 'assente'}])
    assert beats[0] == ('1', '0.428', '2.273', '1.615', '1.187', '352', 'auto', 'corretto')
    assert beats[1][2] == '—' and len(beats[0]) == len(BEAT_COLUMNS)
    assert processing_text({'metadata': {'processing': {'testo': 'Filtraggio: x'}}}) == 'Filtraggio: x'
    assert processing_text(None) == ''
