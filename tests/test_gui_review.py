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


class _FakeDrag:
    """The parts of pyqtgraph's MouseDragEvent the viewer reads (scene positions, start/finish, button, keys)."""

    def __init__(self, down, pos, *, start=False, finish=False, shift=False, button=None):
        self._down, self._pos, self._start, self._finish = down, pos, start, finish
        self._mods = Qt.ShiftModifier if shift else Qt.NoModifier
        self._button = Qt.LeftButton if button is None else button
        self.accepted = False

    def isStart(self):
        return self._start

    def isFinish(self):
        return self._finish

    def button(self):
        return self._button

    def buttonDownScenePos(self):
        return self._down

    def scenePos(self):
        return self._pos

    def modifiers(self):
        return self._mods

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def _drag(viewer, down, to, shift=False):
    viewer._on_view_drag(_FakeDrag(down, down, start=True, shift=shift))
    viewer._on_view_drag(_FakeDrag(down, to, shift=shift))
    viewer._on_view_drag(_FakeDrag(down, to, finish=True, shift=shift))


def test_markers_can_be_dragged_in_edit_mode(app, csv_path, monkeypatch):
    """Cate's feedback (26 Oct 2026): drag a marker to move it. Edit mode, left button on a marker: on release
    the depolarisation snaps to the spike, the repolarisation to the nearest extremum; Shift keeps the exact
    drop point. Undoable, refused on a closed recording, measures marked out of date; a drag that does not
    start on a marker (or in View mode) pans / zooms as before."""
    import pyqtgraph as pg
    from PySide6.QtCore import QPointF
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w._run_analysis(str(csv_path), "el1")
    app.processEvents()
    st, v = w._review_state, w.viewer
    reports = []
    v.review_report.connect(reports.append)
    w._signal_tab._rb_edit.setChecked(True)
    k = 3
    d0, r0 = st.beats[k].dep, st.beats[k].repol
    assert r0 is not None
    t = v.get_time_vector()
    v.setXRange(float(t[d0]) - 0.3, float(t[r0]) + 0.3, padding=0)
    app.processEvents()
    vb = v.getPlotItem().vb

    def scene(i, dt=0.0):
        trace = v._displayed_trace()
        return vb.mapViewToScene(QPointF(float(t[i] + dt), float(trace[i] + v._marker_offset)))

    assert v.marker_at(scene(d0)) == ("dep", k) and v.marker_at(scene(r0)) == ("rep", k)
    assert v.marker_at(scene((d0 + r0) // 2)) is None

    # Shift: the depolarisation stays exactly where it is dropped
    _drag(v, scene(d0), scene(d0, 0.030), shift=True)
    assert abs(st.beats[k].dep - (d0 + int(0.030 * FS))) <= 2 and st.beats[k].source == "corretto"
    assert w._signal_tab.recompute_pending()
    # without Shift it lands on the spike, like the click commands
    d1 = st.beats[k].dep
    expected = st.snapped_dep(d0 + int(0.010 * FS))
    assert abs(expected - d0) <= 3
    _drag(v, scene(d1), scene(d0, 0.010))
    assert st.beats[k].dep == expected
    # the repolarisation, exactly where dropped (Shift) and snapped to the nearest extremum (no Shift)
    _drag(v, scene(r0), scene(r0, 0.040), shift=True)
    assert abs(st.beats[k].repol - (r0 + int(0.040 * FS))) <= 2 and st.beats[k].repol_source == "corretto"
    r1 = st.beats[k].repol
    _drag(v, scene(r1), scene(r0, 0.005))
    assert st.beats[k].repol == st.snapped_repol(r0 + int(0.005 * FS), trace=v._displayed_trace())
    # a depolarisation cannot be dropped on another one
    d_next = st.beats[k + 1].dep
    _drag(v, scene(st.beats[k].dep), scene(d_next), shift=True)
    assert st.beats[k].dep != d_next and reports and "already" in reports[-1]
    # all undoable
    for _ in range(4):
        st.undo()
    assert st.beats[k].dep == d0 and st.beats[k].repol == r0

    # a drag that does not start on a marker, or any drag in View mode, is the ViewBox's (pan / zoom)
    passed = []
    monkeypatch.setattr(pg.ViewBox, "mouseDragEvent", lambda self, ev, axis=None: passed.append(ev))
    _drag(v, scene((d0 + r0) // 2), scene((d0 + r0) // 2, 0.05))
    assert len(passed) == 3 and st.beats[k].dep == d0
    w._signal_tab._rb_view.setChecked(True)
    _drag(v, scene(d0), scene(d0, 0.03), shift=True)
    assert len(passed) == 6 and st.beats[k].dep == d0
    # a closed recording refuses the drag
    w._signal_tab._rb_edit.setChecked(True)
    st.close()
    n_rep = len(reports)
    _drag(v, scene(d0), scene(d0, 0.03), shift=True)
    assert st.beats[k].dep == d0 and len(reports) > n_rep
    st.reopen()
    hints = []
    v.hover_info.connect(hints.append)
    v._on_mouse_moved((scene(d0),))                    # hovering a marker in Edit mode says it can be dragged
    assert hints and "drag" in hints[-1]
    w.close()


def test_dragging_a_marker_with_real_mouse_events(app, csv_path, monkeypatch):
    """The same drag through Qt mouse events (press, moves 30 ms apart — pyqtgraph drops moves closer than its
    rate limit — release): the ViewBox hands it to the viewer, which moves the marker; nothing opens a menu."""
    import time

    from PySide6.QtCore import QEvent, QPoint, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QMenu
    menus = []
    monkeypatch.setattr(QMenu, "exec", lambda *a, **k: menus.append(1))
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w._run_analysis(str(csv_path), "el1")
    app.processEvents()
    st, v = w._review_state, w.viewer
    w._signal_tab._rb_edit.setChecked(True)
    k = 3
    d0 = st.beats[k].dep
    t = v.get_time_vector()
    v.setXRange(float(t[d0]) - 0.3, float(t[d0]) + 1.2, padding=0)
    app.processEvents()
    vb, vp = v.getPlotItem().vb, v.viewport()
    trace = v._displayed_trace()

    def at(i, dt=0.0):
        return v.mapFromScene(vb.mapViewToScene(QPointF(float(t[i] + dt), float(trace[i] + v._marker_offset))))

    def send(kind, pos, button, buttons):
        ev = QMouseEvent(kind, QPointF(pos), QPointF(vp.mapToGlobal(pos)), button, buttons, Qt.ShiftModifier)
        QApplication.sendEvent(vp, ev)
        app.processEvents()

    a, b = at(d0), at(d0, 0.030)
    send(QEvent.MouseButtonPress, a, Qt.LeftButton, Qt.LeftButton)
    for f in np.linspace(0, 1, 6)[1:]:
        time.sleep(0.03)
        send(QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * f), a.y()), Qt.NoButton, Qt.LeftButton)
    send(QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
    assert not menus
    assert abs(st.beats[k].dep - (d0 + int(0.030 * FS))) <= 3 and st.beats[k].source == "corretto"
    w.close()


def test_the_whole_quality_judgement_can_be_unfolded(app, csv_path):
    """Cate's feedback: the judgement stopped at three problems with "… and N more (hover here)" and hovering
    did not open it. Now a link unfolds the whole list and folds it back; the text is escaped (a '<' in a
    problem must not eat the rest); the tooltip keeps the whole list too."""
    w = MainWindow()
    w._run_analysis(str(csv_path), "el1")
    panel = w._review_panel
    probs = [{"testo": f"problem {i}"} for i in range(5)] + [{"testo": "FPD < 50 % of the beats"}]
    panel._quality = {"livello": "rosso", "titolo": "Problems", "problemi": probs}
    panel.refresh()
    text = panel._problems.text()
    assert "problem 2" in text and "problem 3" not in text and 'href="more"' in text and "3 more" in text
    panel._problems.linkActivated.emit("more")
    text = panel._problems.text()
    assert "problem 4" in text and "FPD &lt; 50 % of the beats" in text and 'href="less"' in text
    assert panel._problems.toolTip().count("•") == 6
    panel._problems.linkActivated.emit("less")
    assert "problem 4" not in panel._problems.text()
    assert ReviewPanel.problems_html(["• a", "• b"], False) == "• a<br>• b"     # nothing to fold
    w.close()


def test_depolarisation_markers_are_drawn_on_the_visible_crest_and_can_be_nudged(app, csv_path):
    """Operators' screenshot (26 Oct): after 'align' the red triangles sat below the crests — the time comes from
    the raw peak, a sample or two from the filtered crest, and on a narrow spike that sample is much lower.
    The triangle is now drawn on the displayed crest within ±2 ms (the time does not change), and Alt+←/→
    moves the selected depolarisation one sample (Alt+Shift: the repolarisation)."""
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w._run_analysis(str(csv_path), "el1")
    app.processEvents()
    st, v = w._review_state, w.viewer
    k = 2
    trace = v._displayed_trace()
    side = st._spike_sign()
    d0 = st.beats[k].dep
    a, b = d0 - int(0.004 * FS), d0 + int(0.004 * FS)
    crest = a + int(np.argmax(side * trace[a:b]))
    st.beats[k].dep = crest + 2                                  # two samples off the crest
    v.refresh_from_state()
    xs, ys = v._beats_scatter.getData()
    j = int(np.argmin(np.abs(xs - v.get_time_vector()[crest])))
    assert abs(xs[j] - v.get_time_vector()[crest]) < 1e-9 and abs(ys[j] - (trace[crest] + v._marker_offset)) < 1e-12
    assert st.beats[k].dep == crest + 2                          # drawn on the crest, time unchanged
    # Alt+→ / Alt+← on the selected beat
    w._review_nudge("dep", 1)                                    # nothing selected yet: refused with a message
    assert st.beats[k].dep == crest + 2
    v.select_beat(k, center=False)
    w._review_nudge("dep", 1)
    assert st.beats[k].dep == crest + 3 and v.selected_beat() == k
    r0 = st.beats[k].repol
    w._review_nudge("rep", -1)
    assert st.beats[k].repol == r0 - 1
    assert w._signal_tab.recompute_pending()
    from PySide6.QtGui import QAction
    acts = [a for a in w.findChildren(QAction) if a.shortcut().toString() == "Alt+Right"]
    assert acts, "Alt+Right action missing"
    acts[0].trigger()
    assert st.beats[k].dep == crest + 4
    st.close()
    w._review_nudge("dep", 1)                                    # closed: refused
    assert st.beats[k].dep == crest + 4
    st.reopen()
    w.close()


def test_operator_chooses_the_side_of_the_spikes(app, csv_path):
    """Operators: 'P / N correct all (and only) the repolarisations' — the depolarisations get the same:
    Shift+P / Shift+N and the Dep +/− buttons put every depolarisation on that side of the spike."""
    from PySide6.QtGui import QAction
    w = MainWindow()
    w._run_analysis(str(csv_path), "el1")
    st, panel = w._review_state, w._review_panel
    side = st._spike_sign()
    other = -side
    panel._btn_dep_pos.click() if other > 0 else panel._btn_dep_neg.click()
    assert st.dep_polarity == other and st._spike_sign() == other
    assert w._signal_tab.recompute_pending()
    seq = "Shift+P" if side > 0 else "Shift+N"
    acts = [a for a in w.findChildren(QAction) if a.shortcut().toString() == seq]
    assert acts
    acts[0].trigger()
    assert st.dep_polarity == side
    st.close()
    panel.refresh()
    assert not panel._btn_dep_pos.isEnabled() and not panel._btn_dep_neg.isEnabled()
    st.reopen()
    w.close()


def test_lab_convention_in_the_review_window(app, tmp_path):
    """Settings ▸ Detection ▸ Depolarisation time = steepest downstroke (lab MATLAB): markers drawn at the
    downstroke, the complex's lowest/highest points shown, Dep +/− not needed; a sidecar saved with the
    other convention is announced on opening."""
    import shutil

    from cardiac_fp_analyzer.config import AnalysisConfig as AC
    from pyside_app.settings_dialog import SettingsDialog
    from tests.test_lab_convention import FS as LFS
    from tests.test_lab_convention import _complex_train
    x, beats = _complex_train(n_s=30.0, period_s=1.5, seed=4)
    t = np.arange(len(x)) / LFS
    p = tmp_path / "Exp1_chipA_ch1_baseline.csv"
    with open(p, "w") as f:
        f.write("#Sample rate: 2000Hz\nTime (s),Channel 1 (V)\n")
        for ti, xi in zip(t, x * 1e-4):
            f.write(f"{ti:.6f},{xi:.9e}\n")
    cfg = AC()
    dlg = SettingsDialog(cfg)
    combo = dlg._widgets["beat_detection.dep_time_rule"]
    assert [combo.itemData(i) for i in range(combo.count())] == ["peak", "steepest_fall"]
    assert combo.itemText(1).startswith("Steepest downstroke")
    w = MainWindow()
    w._config.beat_detection.dep_time_rule = "steepest_fall"
    w._run_analysis(str(p), "el1")
    st, v, panel = w._review_state, w.viewer, w._review_panel
    assert st.lab_rule and not panel._btn_dep_pos.isEnabled() and not panel._btn_dep_neg.isEnabled()
    xs, _ = v._beats_scatter.getData()
    tv = v.get_time_vector()
    assert np.allclose(sorted(xs), sorted(tv[st.deps()]))            # drawn where the time is
    ex, _ = v._extrema_scatter.getData()
    assert len(ex) == 2 * st.n_beats()
    msgs = []
    w._review_dep_polarity(1)                                         # refused: not needed
    assert st.dep_polarity is None
    st.save(str(p))
    shutil.copy(str(p) + ".review.json", str(p) + ".review.json.bak")
    w._config.beat_detection.dep_time_rule = "peak"
    w.statusBar().messageChanged.connect(msgs.append)
    w._run_analysis(str(p), "el1")
    assert not w._review_state.lab_rule and w._review_state.saved_dep_rule == "steepest_fall"
    assert any("other depolarisation convention" in m for m in msgs)
    ex, _ = v._extrema_scatter.getData()
    assert len(ex) == 0
    w.close()
