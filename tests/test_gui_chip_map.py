"""GUI for multi-chamber chips (v3.13): chip map, chamber selector, electrode
list, analysis of a chamber from the in-memory recording.

Skipped where PySide6 widgets cannot load (sandbox without libEGL), as the
other Qt tests; runs in CI.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
pytest.importorskip("h5py")
try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:   # pragma: no cover — platform-dependent
    pytest.skip(f"PySide6 widget libs unavailable: {exc}", allow_module_level=True)

from pyside_app.chip_map import ChipMapWidget  # noqa: E402
from pyside_app.main import CHANNEL_AUTO, MainWindow, _SignalTab  # noqa: E402

from cardiac_fp_analyzer.chambers import UHEART_MVP_64  # noqa: E402
from tests.test_chambers import NAME, _chip_file  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def chip_path(tmp_path_factory):
    return _chip_file(tmp_path_factory.mktemp("gui") / ("2026-01-27T12-00-41" + NAME.format("baseline")),
                      seed=5, beat_ms=(800.0, 1000.0, 650.0, 900.0))


def test_chip_map_order_and_selection(app):
    w = ChipMapWidget()
    assert not w.isVisible()
    scores = {e: {'score': float(k), 'n_beats': 10, 'period_ms': 800.0, 'cv_pct': 3.0, 'snr': 10.0, 'repol_snr': 2.0}
              for k, e in enumerate(UHEART_MVP_64.channels)}
    w.set_chip(UHEART_MVP_64, scores, {'A': '800 ms'})
    assert not w.isHidden() and w.height() > 100
    order = w._order(UHEART_MVP_64['A'])
    assert order[:2] == ['E16', 'E17'] and order[-2:] == ['E29', 'E30'] and len(order) == 16
    assert order[2:-2] == list(UHEART_MVP_64['A'].row)
    w.set_selected('E24')
    w.resize(600, 140)
    w.grab()                                   # paints without error
    w.clear()
    assert w._layout is None


def test_signal_tab_chip_combos(app):
    tab = _SignalTab()
    assert tab.chamber_choice() is None and tab.channel_choice() == CHANNEL_AUTO
    scores = {e: {'score': 10.0, 'n_beats': 10, 'period_ms': 800.0, 'cv_pct': 3.0, 'snr': 10.0, 'repol_snr': 2.0}
              for e in UHEART_MVP_64.channels}
    tab.show()
    tab.set_chip(UHEART_MVP_64, 'B', scores, {'B': '1000 ms'})
    assert tab.chamber_choice() == 'B'
    assert tab._channel_values == (CHANNEL_AUTO,) + UHEART_MVP_64['B'].electrodes
    assert tab._combo_channel.count() == 17
    assert '(stim)' in tab._combo_channel.itemText(1)        # E31 is a stimulation electrode
    tab.set_channel('E40')
    assert tab.channel_choice() == 'E40'
    tab.set_channel('el1')                                    # not an electrode of the chamber -> Auto
    assert tab.channel_choice() == CHANNEL_AUTO
    tab.set_chip(None, None, {})
    assert tab.chamber_choice() is None and tab._channel_values == (CHANNEL_AUTO, 'el1', 'el2')


def test_main_window_opens_chip(app, chip_path):
    w = MainWindow()
    w._run_analysis(str(chip_path), channel=CHANNEL_AUTO)
    assert w._chip is not None and w._chip['layout'] is UHEART_MVP_64
    fi = w._current_result['file_info']
    assert fi['chamber'] in 'ABCD' and fi['analyzed_channel'] in UHEART_MVP_64[fi['chamber']].recording
    assert w._signal_tab.chamber_choice() == fi['chamber']
    assert 'camera' in w.windowTitle()
    # another chamber: electrode back to Auto, no reload
    chip_before = w._chip
    w._on_chamber_changed('C')
    assert w._chip is chip_before
    assert w._current_result['file_info']['chamber'] == 'C'
    assert w._current_result['summary']['beat_period_ms_median'] == pytest.approx(650, abs=5)
    # a click on an electrode of chamber A switches chamber and electrode
    w._on_electrode_clicked('E24')
    assert w._signal_tab.chamber_choice() == 'A' and w._current_result['file_info']['analyzed_channel'] == 'E24'
    assert w._current_result['summary']['beat_period_ms_median'] == pytest.approx(800, abs=5)
    assert np.isfinite(w._current_result['summary']['beat_period_ms_median'])
