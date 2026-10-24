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
    assert q._tree.topLevelItemCount() == 2 and q._tree.columnCount() == 8
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
    # export of the queue
    out = tmp_path / "Results.xlsx"
    assert q.export_results(out) == 2 and out.exists() and out.stat().st_size > 5000
    q.clear()
    assert q._tree.topLevelItemCount() == 0 and not q._btn_export.isEnabled()


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
