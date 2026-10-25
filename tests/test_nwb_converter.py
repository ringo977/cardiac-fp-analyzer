"""NWB converter window (pyside_app.nwb_converter) and the shared ``convert_many``: files and folders, optional
metadata left empty, errors reported without stopping the batch, remembered fields, language."""
from __future__ import annotations

import pytest

pytest.importorskip('pynwb')
pytest.importorskip('PySide6')

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from cardiac_fp_analyzer.nwb_io import convert_many, find_csvs, read_header  # noqa: E402
from pyside_app import i18n  # noqa: E402
from pyside_app.nwb_converter import NwbConverterWindow, options_from_fields, split_names  # noqa: E402
from tests.test_nwb_io import _trace, _write_csv  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def folder(tmp_path):
    _, raw, _, _ = _trace(dur_s=12.0, amp=1.0, noise=0.02, pos_wave=0.3)
    d = tmp_path / 'Exp 7' / 'Day8' / 'B'
    d.mkdir(parents=True)
    _write_csv(d / 'Exp7_chipC_ch3_Ti05_chipA_ch3_Ti06_B.csv', raw * 1e-4)
    _write_csv(d / 'second.csv', raw * 1e-4, two_inputs=False)
    (d / 'samples.csv').write_text('file\n')
    (d / 'broken.csv').write_text('not a recording\n')
    return tmp_path


def test_helpers():
    assert split_names('Rossi, M.; Bianchi\n Verdi ;') == ['Rossi, M.', 'Bianchi', 'Verdi']
    assert split_names('') == []
    o = options_from_fields(gain=0, tz='', species='', subject='')
    assert o['gain'] == 10000.0 and o['tz'] == 'Europe/Rome' and o['species'] == 'Homo sapiens'
    assert o['lab'] == '' and o['experimenter'] == [] and o['overwrite'] is False
    with pytest.raises(ValueError):
        options_from_fields(gain=1, tz='Mars/Olympus')


def test_convert_many_folder_reports_each_file(folder, tmp_path):
    files = find_csvs([folder])
    assert sorted(f.name for f in files) == ['Exp7_chipC_ch3_Ti05_chipA_ch3_Ti06_B.csv', 'broken.csv', 'second.csv']
    assert all(f.name != 'samples.csv' for f in files)
    seen = []
    out = tmp_path / 'out'
    res = convert_many([folder], out_dir=out, progress=lambda *a: seen.append(a[0]), gain=10000.0)
    assert len(res) == 3 and seen == [1, 2, 3]
    ok = {pathlib_name(r['csv']): r for r in res}
    assert ok['broken.csv']['ok'] is False and ok['broken.csv']['message']
    assert ok['second.csv']['ok'] and (out / 'second.nwb').exists()
    # second run without overwrite fails per file, with overwrite succeeds
    again = convert_many([folder / 'Exp 7' / 'Day8' / 'B' / 'second.csv'], out_dir=out, gain=10000.0)
    assert not again[0]['ok']
    assert convert_many([folder / 'Exp 7' / 'Day8' / 'B' / 'second.csv'], out_dir=out, gain=10000.0, overwrite=True)[0]['ok']


def pathlib_name(p):
    from pathlib import Path
    return Path(p).name


def test_window_converts_a_single_file_with_empty_metadata(app, folder, tmp_path):
    settings = QSettings(str(tmp_path / 'conv.ini'), QSettings.Format.IniFormat)
    w = NwbConverterWindow(settings)
    csv = folder / 'Exp 7' / 'Day8' / 'B' / 'second.csv'
    w.set_paths([csv])
    w._out.setText(str(tmp_path / 'nwb_out'))
    w._lab.setText('Lab X')
    w.start()
    assert w._worker is not None
    w._worker.wait(60000)
    for _ in range(50):
        app.processEvents()
    assert w._worker is None and w._convert.isEnabled()
    assert '1 of 1' in w._status.text() and (tmp_path / 'nwb_out' / 'second.nwb').exists()
    assert '✓ second.csv' in w._log.toPlainText()
    # what was typed is remembered
    w.close()
    w2 = NwbConverterWindow(settings)
    assert w2._lab.text() == 'Lab X' and w2._out.text() == str(tmp_path / 'nwb_out') and w2._institution.text() == ''
    w2.close()


def test_window_without_files_does_not_start(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, 'information', staticmethod(lambda *a, **k: shown.append(a[2])))
    w = NwbConverterWindow(QSettings(str(tmp_path / 'c.ini'), QSettings.Format.IniFormat))
    w.start()
    assert w._worker is None and len(shown) == 1


def test_window_speaks_italian_when_selected(app, tmp_path):
    try:
        i18n.apply_language(app, 'it')
        w = NwbConverterWindow(QSettings(str(tmp_path / 'it.ini'), QSettings.Format.IniFormat))
        assert w.windowTitle() == 'Convertitore NWB' and w._convert.text() == 'Converti'
    finally:
        i18n.apply_language(app, 'en')


def test_header_still_readable(folder):
    fields = read_header(folder / 'Exp 7' / 'Day8' / 'B' / 'second.csv')
    assert fields
