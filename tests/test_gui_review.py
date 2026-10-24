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

from pyside_app.main import MainWindow  # noqa: E402
from pyside_app.review_panel import ReviewPanel  # noqa: E402
from pyside_app.review_queue import ReviewQueue  # noqa: E402

from cardiac_fp_analyzer.config import AnalysisConfig  # noqa: E402
from tests.test_review_model import FS, _trace  # noqa: E402


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
    assert panel._processing.text().startswith("Filtraggio:")

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
    assert "battito mancante" in panel._legend.text()
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
    assert st.approved and st.approved_by and "approvato" in panel._status_lbl.text()
    from cardiac_fp_analyzer.review import ReviewState
    assert ReviewState.saved_status(csv_path) == {"el1": "approvato"}
    panel._btn_close.click()
    assert st.locked and panel._btn_close.text().startswith("Sblocca") and not panel._btn_pos.isEnabled()
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
    assert "(usato)" in top.text(3) and q.records()[0]["electrode"] == "el1"
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
    w.close()


def test_worker_pools_have_large_stacks(app):
    """OpenBLAS's parallel LU overflows macOS's 512 KB thread stacks: every GUI pool asks for 16 MB."""
    from pyside_app.workers import WORKER_STACK_BYTES, analysis_pool
    q = ReviewQueue(lambda: AnalysisConfig())
    assert q._pool.stackSize() == WORKER_STACK_BYTES == 16 * 1024 * 1024
    assert analysis_pool(max_threads=2).maxThreadCount() == 2
