"""Interface language (v3.19): English source texts, Italian catalog, switch at run time.

* every ``tr()`` literal of ``cardiac_fp_analyzer`` and ``pyside_app`` has an Italian entry, and the
  format placeholders of each translation match the source;
* the settings SPEC labels/tooltips/pages and the review labels shown through ``tr(x)`` are covered;
* package texts (quality judgement, drift, workflow status, filtering description) follow the language;
* the GUI language setting: default English, saved choice honoured, Italian widgets when selected.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from cardiac_fp_analyzer import i18n
from cardiac_fp_analyzer.translations.it import CATALOG

ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDER = re.compile(r'\{[^{}]*\}')


@pytest.fixture
def italian():
    i18n.set_language('it')
    try:
        yield
    finally:
        i18n.set_language('en')


def _tr_literals():
    found = {}
    for pkg in ('cardiac_fp_analyzer', 'pyside_app'):
        for f in sorted((ROOT / pkg).glob('*.py')):
            tree = ast.parse(f.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                name = getattr(n.func, 'id', None) or getattr(n.func, 'attr', None)
                if name == 'tr' and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                    found.setdefault(n.args[0].value, []).append(f'{f.name}:{n.lineno}')
    return found


def test_every_tr_literal_has_an_italian_entry_with_the_same_placeholders():
    lits = _tr_literals()
    assert len(lits) > 400
    missing = {k: v for k, v in lits.items() if k not in CATALOG}
    assert not missing, f'missing Italian entries: {missing}'
    bad = [k for k, v in CATALOG.items() if set(_PLACEHOLDER.findall(k)) != set(_PLACEHOLDER.findall(v))]
    assert not bad, f'placeholders differ: {bad}'
    assert all(isinstance(k, str) and isinstance(v, str) and v for k, v in CATALOG.items())


def test_settings_and_review_tables_are_covered():
    from pyside_app.review_helpers import BEAT_COLUMNS, LEVEL_LABEL, SOURCE_WORDS
    from pyside_app.settings_dialog_helpers import SPEC, page_order
    texts = list(page_order()) + [fs.label for fs in SPEC] + [fs.tooltip for fs in SPEC if fs.tooltip]
    texts += list(BEAT_COLUMNS) + list(LEVEL_LABEL.values()) + list(SOURCE_WORDS.values())
    assert all(t in CATALOG for t in texts), [t for t in texts if t not in CATALOG]
    assert page_order()[0] == 'Signal' and CATALOG['Signal'] == 'Segnale'


def test_language_switch_and_normalisation():
    assert i18n.language() == 'en' and i18n.tr('Beats') == 'Beats'
    assert i18n.normalise('it_IT') == 'it' and i18n.normalise('Italian') == 'it'
    assert i18n.normalise('de') == 'en' and i18n.normalise(None) == 'en'
    try:
        assert i18n.set_language('it-IT') == 'it'
        assert i18n.tr('Beats') == 'Battiti' and i18n.tr('{0} beats included').format(3) == '3 battiti inclusi'
        assert i18n.tr('not in the catalog') == 'not in the catalog'
    finally:
        i18n.set_language('en')
    assert i18n.tr('Beats') == 'Beats'


def test_package_texts_follow_the_language(italian):
    from cardiac_fp_analyzer.drift import drift_text, rate_drift
    from cardiac_fp_analyzer.quality_summary import summarise_quality
    from cardiac_fp_analyzer.review import ReviewState
    q = summarise_quality({'summary': {'not_analysable': True, 'not_analysable_reason': 'only 2 depolarisations'},
                           'file_info': {}, 'metadata': {'sample_rate': 1000.0}, 'beat_indices_raw': [100, 900]})
    assert q['titolo'] == 'Problemi' and q['riassunto'].startswith('Problemi: non analizzabile: solo 2 depolarizzazioni')
    d = rate_drift([k * 0.5 for k in range(8)])
    assert 'intervalli' in d['motivo']
    txt = drift_text({'pendenza_bpm_min': 1.0, 'variazione_pct': 2.0, 'durata_s': 30.0, 'ci_bpm_min': (0.5, 1.5),
                      'affidabile': True, 'significativa': True, 'nota': ''})
    assert txt.startswith('frequenza +2.0 % in 30 s') and 'IC 95 %' in txt
    st = ReviewState.__new__(ReviewState)
    st.status, st.approved_by, st.approved_at, st.closed_at = 'da_rivedere', '', '', ''
    assert st.status_text() == 'da rivedere'
    st.status, st.approved_by, st.approved_at = 'approvato', 'marco', '2026-10-25T10:00:00'
    assert st.status_text() == 'approvato da marco il 2026-10-25 10:00'
    i18n.set_language('en')
    assert st.status_text() == 'approved by marco on 2026-10-25 10:00'


def test_review_helper_rows_follow_the_language(italian):
    from pyside_app.review_helpers import format_beat_rows, format_measures, source_word
    rows = format_measures({'bp_s': 1.0, 'n_battiti': 3})
    assert rows[6][0] == 'Deriva frequenza' and rows[8][0] == 'Battiti' and rows[0][0] == 'Beating Period [s]'
    assert source_word('aggiunto') == 'aggiunto' and source_word('assente') == 'assente'
    i18n.set_language('en')
    assert source_word('aggiunto') == 'added' and source_word('assente') == 'missing'
    out = format_beat_rows([{'n': 1, 'dep_s': 0.5, 'bp_s': None, 'rep_s': 0.8, 'fpd_s': 0.3, 'amp_uV': 100.0,
                             'stato': 'corretto', 'stato_rip': 'propagato'}])
    assert out[0][6:] == ('corrected', 'propagated')


# ── GUI ──────────────────────────────────────────────────────────────────────
pytest.importorskip('PySide6')


@pytest.fixture(scope='module')
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_language_setting_and_apply(app, tmp_path):
    from PySide6.QtCore import QSettings

    from pyside_app import i18n as gui_i18n
    settings = QSettings(str(tmp_path / 'ui.ini'), QSettings.Format.IniFormat)
    assert gui_i18n.configured_language(settings) == 'en'
    assert gui_i18n.save_language('it', settings) == 'it' and gui_i18n.configured_language(settings) == 'it'
    settings.setValue(gui_i18n.SETTINGS_KEY, 'xx')
    assert gui_i18n.configured_language(settings) == 'en'
    try:
        assert gui_i18n.apply_language(app, 'it') == 'it' and i18n.language() == 'it'
        assert gui_i18n.tr('Review queue') == 'Coda di revisione'
    finally:
        assert gui_i18n.apply_language(app, 'en') == 'en'
    assert gui_i18n.tr('Review queue') == 'Review queue' and not gui_i18n._translators


def test_review_panel_speaks_italian_when_selected(app, italian):
    from pyside_app.review_panel import ReviewPanel
    panel = ReviewPanel()
    assert panel._btn_save.text() == 'Salva correzioni' and panel._chk_approved.text() == 'Approvato'
    assert panel._btn_undo.text() == 'Annulla' and panel._btn_reanalyse.text() == "Rifai l'analisi"
    assert panel._table.horizontalHeaderItem(6).text() == 'Depolarizzazione'
    i18n.set_language('en')
    panel_en = ReviewPanel()
    assert panel_en._btn_save.text() == 'Save corrections' and panel_en._btn_undo.text() == 'Undo'


def test_settings_dialog_speaks_italian_when_selected(app, italian):
    from cardiac_fp_analyzer.config import AnalysisConfig
    from pyside_app.settings_dialog import SettingsDialog
    dlg = SettingsDialog(AnalysisConfig())
    assert dlg.windowTitle() == 'Impostazioni analisi'
    assert [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())] == ['Segnale', 'Detection', 'Ripolarizzazione', 'QC', 'Inclusione']
    assert dlg._widgets['_top.amplifier_gain'].toolTip().startswith('Fattore di guadagno hardware')
