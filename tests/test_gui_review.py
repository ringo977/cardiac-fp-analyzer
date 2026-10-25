"""Revisione screen (v3.17): review state in the viewer, measures dock, queue, corrections with undo.

Skipped where PySide6 widgets cannot load (sandbox without libEGL); runs in CI (offscreen).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except ImportError as exc:   # pragma: no cover — platform-dependent
    pytest.skip(f"PySide6 widget libs unavailable: {exc}", allow_module_level=True)

from cardiac_fp_analyzer.config import AnalysisConfig  # noqa: E402
from pyside_app.main import MainWindow  # noqa: E402
from pyside_app.review_panel import ReviewPanel  # noqa: E402
from pyside_app.review_queue import ReviewQueue  # noqa: E402
from tests.test_review_model import FS, _trace  # noqa: E402


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """A modal question would wait for a click that never comes in a test run: answer Yes unless a test
    installs its own answer."""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    x, raw, beats, reps = _trace(dur_s=60.0, amp=1.0, noise=0.02, pos_wave=0.3)
    t = np.arange(len(x)) / FS
    d = tmp_path_factory.mktemp("review")
    p = d / "Exp1_chipA_ch1_chipB_ch1_baseline.csv"
    with open(p, "w") as f:
        f.write("#Sample rate: 2000Hz\nTime (s),Channel 1 (V),Channel 2 (V)\n")
        for ti, xi in zip(t, raw * 1e-4):
            f.write(f"{ti:.6f},{xi:.9e},{0.3 * xi:.9e}\n")
    return p


def test_review_state_and_panel_after_open(app, csv_path):
    w = MainWindow()
    w._run_analysis(str(csv_path), "el1")
    st = w._review_state
    assert st is not None and st.n_beats() >= 25
    assert w.viewer.review_state() is st
    assert len(w.viewer.get_beat_indices()) == st.n_beats()
    panel: ReviewPanel = w._review_panel
    assert panel._table.rowCount() == st.n_beats()
    assert "Beating Period" in panel._measure_rows[0][0].text()
    assert panel._measure_rows[0][1].text().startswith("2.0")
    assert panel._badge.text()
    assert panel._processing.text().startswith("Filtering:")

    # a correction through the state, then undo from the window
    n = st.n_beats()
    st.remove_beat(0)
    w._review_after_edit()
    assert panel._table.rowCount() == n - 1
    w._review_undo()
    assert st.n_beats() == n and panel._table.rowCount() == n
    assert not st.can_undo and st.can_redo
    w._review_redo()
    assert st.n_beats() == n - 1
    w._review_undo()

    # selection and keyboard-like navigation
    w._review_step(1)
    assert w.viewer.selected_beat() == 0
    w._review_step(-1)
    assert w.viewer.selected_beat() == n - 1
    w._review_delete_selected()
    assert st.n_beats() == n - 1
    w._review_undo()

    # highlighted stretches and navigation: remove a beat → a red 'missing beat' stretch, Ctrl+G reaches it
    st.remove_beat(3)
    w._review_after_edit()
    probs = w.viewer.problems()
    assert any(p["tipo"] == "battito_mancante" for p in probs) and len(w.viewer._problem_regions) == len(probs)
    assert "missing beat" in panel._legend.text()
    w._review_next_problem(1)
    assert w.viewer.selected_beat() is not None
    w._review_next_problem(-1)
    w._review_undo()
    assert not any(p["tipo"] == "battito_mancante" for p in w.viewer.problems())

    # views
    for kind in ("grezzo", "operatori", "filtrato"):
        w.viewer.set_view(kind)
        assert w.viewer.view() == kind
        w.viewer.grab()

    # polarity propagation and save
    w._review_polarity(1)
    assert st.polarity == 1
    w._review_save()
    sidecar = csv_path.with_name(csv_path.name + ".review.json")
    assert sidecar.exists()

    # approval (tick) and closure (lock) through the panel, persisted, shown in the queue
    panel._chk_approved.setChecked(True)
    assert st.approved and st.approved_by and "approved" in panel._status_lbl.text()
    assert not w._signal_tab._btn_recompute.styleSheet() or "pending" not in w._signal_tab._btn_recompute.styleSheet().lower()
    assert w.windowTitle().endswith("✓")
    # a new correction withdraws the approval and the panel follows
    n0 = st.n_beats()
    st.remove_beat(0)
    w._review_after_edit()
    assert not st.approved and not panel._chk_approved.isChecked()
    w._review_undo()
    assert st.n_beats() == n0
    panel._chk_approved.setChecked(True)
    from cardiac_fp_analyzer.review import ReviewState
    assert ReviewState.saved_status(csv_path) == {"el1": "approvato"}
    panel._btn_close.click()
    assert st.locked and panel._btn_close.text().startswith("Unlock") and not panel._btn_pos.isEnabled()
    n_before = st.n_beats()
    w._review_delete_selected()                       # refused while closed
    w._review_polarity(-1)
    assert st.n_beats() == n_before and st.polarity == 1
    w.viewer._show_review_menu(1.0)                   # no menu on a closed recording, only a message
    assert ReviewState.saved_status(csv_path) == {"el1": "chiuso"}
    panel._btn_close.click()                          # reopen: still approved, editable again
    assert st.approved and not st.locked and panel._chk_approved.isEnabled()
    panel._chk_approved.setChecked(False)
    assert st.status == "da_rivedere" and ReviewState.saved_status(csv_path) == {"el1": "da_rivedere"}

    # reopening loads the sidecar
    w._run_analysis(str(csv_path), "el1")
    assert w._review_state.polarity == 1

    # clipboard copies: tab-separated with headers, as many lines as beats
    panel.copy_beats()
    text = QApplication.clipboard().text()
    assert text.splitlines()[0] == "\t".join(("#", "Dep. time [s]", "Beating Period [s]", "Rep. time [s]", "FPD [s]",
                                               "Dep. Voltage [µV]", "Depolarisation", "Repolarisation"))
    assert len(text.splitlines()) == w._review_state.n_beats() + 1
    panel.copy_measures()
    assert QApplication.clipboard().text().startswith("Measure\tValue\t%\tΔ baseline\nBeating Period [s]\t")
    # the measures table has the Results labels and the drift rows
    assert panel._measures.rowCount() == 13 and "drift" in panel._measure_rows[6][0].text()

    # 'Rifai l'analisi' drops the corrections in memory but leaves the sidecar alone
    w._review_state.remove_beat(0)
    w._review_state.save(str(csv_path))
    n_saved = w._review_state.n_beats()
    w._ignore_sidecar_once = True
    w._run_analysis(str(csv_path), "el1")
    assert w._review_state.n_beats() == n_saved + 1 and w._review_state.polarity is None
    assert ReviewState.saved_status(csv_path) and sidecar.exists()
    w._run_analysis(str(csv_path), "el1")             # a normal reopen loads the saved corrections again
    assert w._review_state.n_beats() == n_saved

    # recompute keeps the state
    w._on_recompute()
    assert w._review_state is not None and w._review_panel._table.rowCount() == w._review_state.n_beats()
    w.close()


def test_review_queue_lists_both_inputs(app, csv_path, tmp_path):
    q = ReviewQueue(lambda: AnalysisConfig())
    n = q.add_paths([csv_path.parent])
    assert n == 2
    q._pool.waitForDone(120000)
    for _ in range(50):
        app.processEvents()
        if all(e.get("livello") for e in q._entries.values()):
            break
    entries = q.entries()
    assert len(entries) == 2 and all(e["livello"] in ("verde", "giallo", "rosso") for e in entries)
    assert {e["ingresso"] for e in entries} == {"el1", "el2"}
    assert q._tree.topLevelItemCount() == 2 and q._tree.columnCount() == 10
    # every row carries its export record (identity from the two-tissue name) and the measures of the review model
    recs = q.records()
    assert len(recs) == 2 and {(r["chip"], r["chamber"]) for r in recs} == {("A", 1), ("B", 1)}
    assert all(r["condition"] == "baseline" and r["measures"]["n_battiti"] >= 25 for r in recs)
    el1 = next(e for e in entries if e["ingresso"] == "el1")
    assert el1["bp_s"] == pytest.approx(el1["record"]["measures"]["bp_s"])
    # saving corrections refreshes the record from the state
    from cardiac_fp_analyzer.analyze import analyze_single_file
    from cardiac_fp_analyzer.review import ReviewState
    result = analyze_single_file(str(csv_path), channel="el1", verbose=False)
    st = ReviewState.from_result(result)
    st.remove_beat(0)
    q.mark_reviewed(str(csv_path), "el1", state=st, result=result)
    el1 = next(e for e in q.entries() if e["ingresso"] == "el1")
    assert el1["rivista"] and el1["record"]["changes"]["corrected"] and el1["record"]["chip"] == "A"
    assert el1["record"]["measures"]["n_battiti"] == st.n_beats()
    # the workflow state travels with the record and is shown in the 'Stato' column, closed rows last
    st.close("Rossi, Mario")
    q.mark_reviewed(str(csv_path), "el1", state=st, result=result)
    rows = [(q._tree.topLevelItem(i).text(3), q._tree.topLevelItem(i).text(5)) for i in range(2)]
    assert ("EL1", "🔒") in rows and q.entries()[-1]["ingresso"] == "el1"
    assert q.records()[-1]["stato"] == "chiuso" and "Rossi" in q.records()[-1]["stato_testo"]
    # export of the queue
    out = tmp_path / "Results.xlsx"
    assert q.export_results(out) == 2 and out.exists() and out.stat().st_size > 5000
    q.clear()
    assert q._tree.topLevelItemCount() == 0 and not q._btn_export.isEnabled()


@pytest.fixture(scope="module")
def single_tissue_csv(tmp_path_factory):
    """One construct, two electrodes: input 2 is a weaker copy of input 1 (the software prefers el1)."""
    x, raw, beats, reps = _trace(dur_s=40.0, amp=1.0, noise=0.02, pos_wave=0.3)
    t = np.arange(len(x)) / FS
    d = tmp_path_factory.mktemp("single")
    p = d / "Exp2_chipA_ch2_baseline.csv"
    rng = np.random.default_rng(1)
    with open(p, "w") as f:
        f.write("#Sample rate: 2000Hz\nTime (s),Channel 1 (V),Channel 2 (V)\n")
        for ti, xi, ni in zip(t, raw * 1e-4, rng.normal(0, 0.3e-4, len(t))):
            f.write(f"{ti:.6f},{xi:.9e},{0.4 * xi + ni:.9e}\n")
    return p


def test_review_queue_groups_the_electrodes_of_one_tissue(app, single_tissue_csv, tmp_path):
    from cardiac_fp_analyzer.analyze import analyze_single_file
    from cardiac_fp_analyzer.review import ReviewState
    q = ReviewQueue(lambda: AnalysisConfig())
    assert q.add_paths([single_tissue_csv]) == 1
    q._pool.waitForDone(180000)
    for _ in range(50):
        app.processEvents()
        if all(e.get("livello") for e in q._entries.values()):
            break
    e = q.entries()[0]
    assert e["due_elettrodi"] and set(e["elettrodi"]) == {"el1", "el2"} and e["scelto"] == "el1"
    assert e["elettrodi"]["el1"]["scelta_software"] and not e["elettrodi"]["el2"]["scelta_software"]
    assert e["record"] is e["elettrodi"]["el1"]["record"] and e["record"]["electrode"] == "el1"
    top = q._tree.topLevelItem(0)
    assert top.childCount() == 2 and top.child(0).checkState(4) == Qt.Checked and top.child(1).checkState(4) == Qt.Unchecked
    assert "(used)" in top.text(3) and q.records()[0]["electrode"] == "el1"
    # the operator picks EL2: the tissue row follows, the choice is saved and 'auto' honours it
    assert q.choose_electrode(str(single_tissue_csv), "el2")
    e = q.entries()[0]
    assert e["scelto"] == "el2" and e["record"]["electrode"] == "el2" and q.records()[0]["electrode"] == "el2"
    assert ReviewState.chosen_electrode(single_tissue_csv) == "el2"
    r = analyze_single_file(str(single_tissue_csv), channel="auto", verbose=False)
    assert r["file_info"]["analyzed_channel"] == "el2" and r["file_info"]["electrode_choice"] == "operatore"
    # ticking in the tree does the same
    top = q._tree.topLevelItem(0)
    top.child(0).setCheckState(4, Qt.Checked)
    for _ in range(5):
        app.processEvents()
    assert q.entries()[0]["scelto"] == "el1" and ReviewState.chosen_electrode(single_tissue_csv) == "el1"
    assert q._tree.topLevelItem(0).child(0).checkState(4) == Qt.Checked
    # a re-run of the queue starts from the saved choice while still telling the software's own
    q.clear()
    q.add_paths([single_tissue_csv])
    q._pool.waitForDone(180000)
    for _ in range(50):
        app.processEvents()
        if all(e.get("livello") for e in q._entries.values()):
            break
    e = q.entries()[0]
    assert e["scelto"] == "el1" and e["elettrodi"]["el1"]["scelta_software"]
    ReviewState.save_choice(single_tissue_csv, None)
    assert ReviewState.chosen_electrode(single_tissue_csv) is None


def test_open_from_queue(app, csv_path):
    w = MainWindow()
    w._open_from_queue(str(csv_path), "el2", None)
    assert w._review_state is not None and w._review_state.channel.lower() == "el2"
    assert w._signal_tab.channel_choice() == "el2"       # the explicit request shows in the combo
    w.close()


def test_double_click_on_a_two_tissue_row_opens_that_input(app, csv_path):
    """Each row of a two-tissue file (one tissue per input) opens its own electrode: the lab's channel
    number in the identity is not an MCS chamber (it made both rows analyse 'auto' → the same electrode,
    and the two tissues shared one review state)."""
    w = MainWindow()
    q = w._review_queue
    q.add_paths([csv_path])
    q._pool.waitForDone(120000)
    for _ in range(50):
        app.processEvents()
        if all(e.get("livello") for e in q._entries.values()):
            break
    seen = []
    for i in range(q._tree.topLevelItemCount()):
        it = q._tree.topLevelItem(i)
        assert it.data(0, Qt.UserRole)[2] is None            # no MCS chamber for a CSV input
        q._on_double_click(it, 0)
        for _ in range(20):
            app.processEvents()
        seen.append((w._review_state.channel, w._signal_tab.channel_choice()))
    assert sorted(seen) == [("el1", "el1"), ("el2", "el2")]
    w.close()


def test_worker_pools_have_large_stacks(app):
    """OpenBLAS's parallel LU overflows macOS's 512 KB thread stacks: every GUI pool asks for 16 MB."""
    from pyside_app.workers import WORKER_STACK_BYTES, analysis_pool
    q = ReviewQueue(lambda: AnalysisConfig())
    assert q._pool.stackSize() == WORKER_STACK_BYTES == 16 * 1024 * 1024
    assert analysis_pool(max_threads=2).maxThreadCount() == 2


def test_message_boxes_are_qt_drawn(app):
    """The native macOS alert aborted the process (24 Oct 2026): the static helpers are replaced by Qt boxes."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox

    from pyside_app import dialogs
    assert QMessageBox.question is not dialogs._NATIVE.get("question") and dialogs._NATIVE
    answers = []

    def _click():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QMessageBox) and w.isVisible():
                assert w.testOption(QMessageBox.Option.DontUseNativeDialog)
                w.button(QMessageBox.Yes).click()
    QTimer.singleShot(50, _click)
    answers.append(QMessageBox.question(None, "t", "q", QMessageBox.Yes | QMessageBox.No, QMessageBox.No))
    assert answers == [QMessageBox.Yes]


def test_approving_or_closing_with_pending_edits_asks_to_recompute(app, csv_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w = MainWindow()
    w._run_analysis(str(csv_path), "el1")
    st, panel = w._review_state, w._review_panel
    calls, asked = [], []
    real = w._on_recompute
    w._on_recompute = lambda: (calls.append(1), real())[1]
    answer = {"v": QMessageBox.No}

    def _question(parent, title, text, *a, **k):
        asked.append(text)
        return answer["v"]
    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))

    st.remove_beat(0)
    w._review_after_edit()
    assert w._signal_tab.recompute_pending()
    panel._chk_approved.setChecked(True)               # declined: nothing is approved, the tick goes back
    assert len(asked) == 1 and "approve" in asked[0] and not calls
    assert not st.approved and not panel._chk_approved.isChecked() and w._signal_tab.recompute_pending()
    answer["v"] = QMessageBox.Yes
    panel._chk_approved.setChecked(True)               # accepted: recompute, then approve
    assert calls == [1] and st.approved and panel._chk_approved.isChecked()
    assert not w._signal_tab.recompute_pending()
    panel._chk_approved.setChecked(False)
    st.remove_beat(0)
    w._review_after_edit()
    answer["v"] = QMessageBox.No
    panel._btn_close.click()                           # same for closing
    assert len(asked) == 3 and "close" in asked[2] and not st.locked
    answer["v"] = QMessageBox.Yes
    panel._btn_close.click()
    assert len(calls) == 2 and st.locked and not w._signal_tab.recompute_pending()
    panel._btn_close.click()
    panel._chk_approved.setChecked(False)
    panel._chk_approved.setChecked(True)               # nothing pending: no question
    assert len(asked) == 4
