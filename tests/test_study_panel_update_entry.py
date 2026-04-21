"""Regression test for ``StudyPanel.update_entry`` (issue #6, Fase 3).

Fase 3 of the Group-config-as-source-of-truth work (GH #6) introduces
a public cache write-back method on :class:`StudyPanel`: after a
Ricalcola, :class:`MainWindow` calls
``update_entry(group_name, csv_relpath, result_dict, config)`` to
replace the stale :class:`FileResult` in ``_results`` with one built
from the recomputed pipeline output.  This is the step that makes
the Studi tree and the dose-response dialog reflect the user's
manual beat edits *immediately* — without it, a Ricalcola of a file
in a Group produces a fresh in-memory ``_current_result`` but the
cache (and therefore every aggregate downstream) stays frozen on the
value the batch wrote.

Channel canonicalisation is a critical part of this contract: the
batch (:class:`_BatchWorker`) stamps fingerprints using the
:class:`FileEntry` channel (typically ``'auto'``), while the
pipeline output records the *resolved* channel in
``file_info.analyzed_channel`` (typically ``'EL1'``).  If
``update_entry`` used the pipeline's resolved channel for the
fingerprint, it would diverge from what :func:`_is_stale` expects
on its next pass — which is exactly the bug Marco hit during
Fase 3 manual testing: the ● badge persisted after Ricalcola even
though the numbers had been updated.  The regression test
:meth:`test_fingerprint_uses_fileentry_channel_not_analyzed`
pins this behaviour.

The tests here cover:

1. **Happy path** — a plausible pipeline output dict goes through
   ``update_entry`` and the cached :class:`FileResult` picks up the
   new FPD / FPDc / BPM / STV scalars from the supplied summary.
   Fingerprint matches what the batch would stamp, so the staleness
   gate stops flagging the row as ●.

2. **Channel canonicalisation** — FileEntry.channel='auto' must
   produce a fingerprint matching ``_result_fingerprint(cfg, 'auto')``
   even when the pipeline resolved to ``EL1``.  This is the
   staleness-gate-after-Ricalcola regression.

3. **Different config fingerprint** — sanity check that the
   fingerprint really reflects the config passed in.

4. **Unknown group name** — silently ignored, no exception, cache
   untouched.  Covers the rare race where the Group was removed
   between the Ricalcola emit and the slot dispatch.

5. **Unknown csv_relpath** — same as above at the file level.

6. **Closed study** — ``_study is None`` → no-op, no exception.

Skipped automatically on environments without PySide6 widget support
(sandbox has no libEGL/libGL — same pattern as every other Qt test
in this suite).
"""
from __future__ import annotations

import os
import sys

import pytest

# Offscreen backend lets QApplication start without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:   # pragma: no cover — platform-dependent
    pytest.skip(
        f"PySide6 widget libs unavailable: {exc}",
        allow_module_level=True,
    )

from pyside_app.study_panel import (
    FileResult,
    StudyPanel,
    _result_fingerprint,
)

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.study import (
    FileEntry,
    Group,
    Study,
    save_study,
)


@pytest.fixture(scope="module")
def qapp():
    """Shared ``QApplication`` for all tests in this module."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("CardiacFP-Test")
    app.setApplicationName("study-panel-update-entry")
    yield app


def _make_panel_with_study(tmp_path) -> tuple[StudyPanel, Study, Group]:
    """Build a ``StudyPanel`` loaded with a 1-group / 1-file Study.

    The CSV has a minimal header so ``is_file()`` passes; content is
    irrelevant because we never actually run analysis — we only call
    :meth:`update_entry` with a hand-crafted result dict.
    """
    csv = tmp_path / "a.csv"
    csv.write_text("Time (s),EL1 (V)\n0.000,0.0\n0.001,0.0\n")
    group = Group(
        name="DoseX",
        files=[FileEntry(csv_relpath="a.csv")],
    )
    study = Study(name="TestStudy", folder=str(tmp_path), groups=[group])
    save_study(study)
    panel = StudyPanel()
    panel.set_study(study)
    return panel, study, group


def _fake_pipeline_result(
    *,
    analyzed_channel: str,
    n_beats: int,
    n_fpd: int,
    fpd_ms: float,
    fpdc_ms: float,
    bpm: float,
    stv_ms: float,
) -> dict:
    """Craft the minimal result dict shape that ``update_entry``
    consumes.  Must mirror what ``analyze_single_file`` produces for
    the fields the helper extracts (``file_info.analyzed_channel``,
    ``beat_indices``, ``beat_indices_fpd``, ``summary.*``).
    """
    return {
        'file_info': {'analyzed_channel': analyzed_channel},
        'beat_indices': list(range(n_beats)),
        'beat_indices_fpd': list(range(n_fpd)),
        'summary': {
            'fpd_ms_mean': fpd_ms,
            'fpdc_ms_mean': fpdc_ms,
            'bpm_mean': bpm,
            'stv_ms': stv_ms,
        },
    }


class TestUpdateEntry:
    """Cache write-back contract for ``StudyPanel.update_entry``."""

    def test_updates_cached_file_result_with_matching_fingerprint(
        self, qapp, tmp_path,
    ):
        """Happy path: update_entry rewrites ``_results`` from the
        recomputed pipeline output.

        Pre-seeds the cache with an "old" FileResult carrying batch
        numbers, then calls ``update_entry`` with new summary values
        (simulating a Ricalcola that changed the FPD).  The cached
        entry must now hold the new scalars and a fingerprint
        computed against the group's current config — exactly what
        ``_BatchWorker`` would have stamped.
        """
        panel, _study, group = _make_panel_with_study(tmp_path)
        key = (group.name, "a.csv")

        # Pre-seed a "batch-done" FileResult with the SAME canonical
        # channel the batch would use for this FileEntry ('auto' is
        # the default in Group-level Studies).  update_entry should
        # replace this row, not merge with it.
        old_fp = _result_fingerprint(group.config, "auto")
        panel._results[key] = FileResult(
            status='ok',
            channel_analyzed="EL1",   # resolved channel, display-only
            n_included=50,
            n_total=50,
            fingerprint=old_fp,
            fpd_ms=300.0,
            fpdc_ms=310.0,
            bpm=60.0,
            stv_ms=2.5,
        )

        # Ricalcola produces these new numbers.  A real pipeline run
        # after a manual beat edit would produce similar deltas.
        new_result = _fake_pipeline_result(
            analyzed_channel="EL1",
            n_beats=45,
            n_fpd=42,
            fpd_ms=325.0,
            fpdc_ms=330.0,
            bpm=58.0,
            stv_ms=3.1,
        )

        panel.update_entry(
            group_name=group.name,
            csv_relpath="a.csv",
            result_dict=new_result,
            config=group.config,
        )

        fr = panel._results[key]
        assert fr.status == 'ok'
        assert fr.channel_analyzed == "EL1"
        # n_included comes from beat_indices_fpd (preferred when >0).
        assert fr.n_included == 42
        assert fr.n_total == 45
        # Scalars picked from summary.
        assert fr.fpd_ms == pytest.approx(325.0)
        assert fr.fpdc_ms == pytest.approx(330.0)
        assert fr.bpm == pytest.approx(58.0)
        assert fr.stv_ms == pytest.approx(3.1)
        # Fingerprint matches what a batch rerun with the same
        # (group.config, fe.channel) would compute — the whole point
        # of GH #6 Fase 2 + 3.  FileEntry.channel defaults to 'auto'
        # and that's what the fingerprint is keyed on, NOT the
        # resolved analyzed_channel (which is display-only).
        expected_fp = _result_fingerprint(group.config, "auto")
        assert fr.fingerprint == expected_fp

    def test_fingerprint_uses_fileentry_channel_not_analyzed(
        self, qapp, tmp_path,
    ):
        """Regression: fingerprint keyed on FileEntry.channel, not on
        the pipeline's resolved ``analyzed_channel``.

        This is the exact bug Marco reported during Fase 3 manual
        validation: after Ricalcola the ● badge persisted because the
        cached fingerprint diverged from what :func:`_is_stale`
        recomputed on the next pass.  Root cause:

        - ``_BatchWorker`` canonicalises ``fe.channel`` to one of
          ``'auto' / 'EL1' / 'EL2'`` (falling back to ``'auto'`` for
          anything else) before hashing — see ``study_panel.py``
          line ~680.
        - The pipeline then resolves ``'auto'`` → ``'EL1'`` (say) and
          writes that to ``file_info.analyzed_channel``.
        - Pre-fix, ``update_entry`` happily hashed the *resolved*
          channel ('EL1'), so the next batch pass compared its
          ``'auto'``-keyed recompute against the cache's
          ``'EL1'``-keyed entry and always flagged ● stale.

        Post-fix, ``update_entry`` looks up the FileEntry internally
        and uses ITS channel ('auto' by default) — matching the
        batch.  This test pins that invariant.
        """
        panel, _study, group = _make_panel_with_study(tmp_path)
        # Confirm the precondition: the FileEntry channel is 'auto'.
        # If the Study model ever changes its default, this test
        # should fail loudly and get updated, not silently pass.
        assert group.files[0].channel == 'auto'

        # Pipeline reports 'EL1' as the actually-analysed channel —
        # this is the exact shape analyze_single_file emits after
        # resolving 'auto'.
        result = _fake_pipeline_result(
            analyzed_channel="EL1",
            n_beats=20, n_fpd=20,
            fpd_ms=300.0, fpdc_ms=310.0, bpm=60.0, stv_ms=2.0,
        )
        panel.update_entry(
            group_name=group.name,
            csv_relpath="a.csv",
            result_dict=result,
            config=group.config,
        )

        fr = panel._results[(group.name, "a.csv")]
        # channel_analyzed is display-only and DOES carry the resolved
        # value — no regression on that front.
        assert fr.channel_analyzed == "EL1"
        # The load-bearing assertion: fingerprint is keyed on 'auto'
        # (the FileEntry channel), NOT 'EL1' (the analyzed channel).
        assert fr.fingerprint == _result_fingerprint(group.config, "auto")
        assert fr.fingerprint != _result_fingerprint(group.config, "EL1"), (
            "fingerprint must not leak the resolved channel — "
            "otherwise the staleness gate flags ● after every "
            "Ricalcola, which is the GH #6 Fase 3 bug."
        )

    def test_different_config_produces_different_fingerprint(
        self, qapp, tmp_path,
    ):
        """Sanity: calling update_entry with a config *different* from
        ``group.config`` stamps a fingerprint that does NOT match the
        group's current config.

        This is the observable side-effect of the bug #5 / #6 wiring:
        pre-Fase-2, Ricalcola always passed the global config, so the
        cached row (if this write-back had existed then) would carry
        the wrong fingerprint and the staleness gate would flag it ●.
        Post-Fase-2, MainWindow passes ``group.config`` and the
        fingerprint matches — this test pins that causality.
        """
        panel, _study, group = _make_panel_with_study(tmp_path)

        # Build a config that definitely differs from the group's
        # default.  Bump the beat-detection threshold factor by 1.0 —
        # a safe numeric knob whose change is enough to perturb the
        # fingerprint hash (``_result_fingerprint`` serialises the
        # whole config dict, so any field change flips the SHA-1).
        rogue = AnalysisConfig()
        rogue.beat_detection.threshold_factor = (
            group.config.beat_detection.threshold_factor + 1.0
        )

        result = _fake_pipeline_result(
            analyzed_channel="EL1",
            n_beats=10,
            n_fpd=10,
            fpd_ms=320.0,
            fpdc_ms=325.0,
            bpm=60.0,
            stv_ms=2.0,
        )

        panel.update_entry(
            group_name=group.name,
            csv_relpath="a.csv",
            result_dict=result,
            config=rogue,
        )

        fr = panel._results[(group.name, "a.csv")]
        # FileEntry.channel defaults to 'auto' — that's what the
        # batch uses, so that's what the canonical fingerprint is
        # keyed on.  update_entry stamps _result_fingerprint(rogue,
        # 'auto'); the group's own config would hash to something
        # different, proving the config param is load-bearing.
        group_fp = _result_fingerprint(group.config, "auto")
        assert fr.fingerprint != group_fp, (
            "fingerprint must reflect the config actually used — "
            "otherwise the staleness gate can't detect divergence."
        )

    def test_unknown_group_is_noop(self, qapp, tmp_path):
        """Passing an unknown ``group_name`` is silently ignored.

        Rare race: the user fires Ricalcola just as another action
        removes the Group from the study.  Losing the write-back is
        acceptable (the user will re-batch anyway); raising would
        surface as a dialog and look like a real error.
        """
        panel, _study, _group = _make_panel_with_study(tmp_path)

        result = _fake_pipeline_result(
            analyzed_channel="EL1", n_beats=10, n_fpd=10,
            fpd_ms=320.0, fpdc_ms=325.0, bpm=60.0, stv_ms=2.0,
        )

        panel.update_entry(
            group_name="GhostGroup",
            csv_relpath="a.csv",
            result_dict=result,
            config=_study_config_or_default(panel),
        )

        # Cache left empty — update_entry refused the unknown key.
        assert panel._results == {}

    def test_unknown_csv_relpath_is_noop(self, qapp, tmp_path):
        """Passing a ``csv_relpath`` that doesn't live in the group
        is silently ignored (file was removed between emit and slot).
        """
        panel, _study, group = _make_panel_with_study(tmp_path)

        result = _fake_pipeline_result(
            analyzed_channel="EL1", n_beats=10, n_fpd=10,
            fpd_ms=320.0, fpdc_ms=325.0, bpm=60.0, stv_ms=2.0,
        )

        panel.update_entry(
            group_name=group.name,
            csv_relpath="ghost.csv",
            result_dict=result,
            config=group.config,
        )

        assert panel._results == {}

    def test_no_study_loaded_is_noop(self, qapp):
        """With no study loaded, ``update_entry`` is a silent no-op."""
        panel = StudyPanel()   # no set_study() call
        # Minimal AnalysisConfig — never actually used past the
        # early ``_study is None`` guard, but required by the API.
        cfg = AnalysisConfig()
        result = _fake_pipeline_result(
            analyzed_channel="EL1", n_beats=1, n_fpd=1,
            fpd_ms=300.0, fpdc_ms=310.0, bpm=60.0, stv_ms=2.0,
        )

        panel.update_entry(
            group_name="AnyGroup",
            csv_relpath="whatever.csv",
            result_dict=result,
            config=cfg,
        )

        assert panel._results == {}


def _study_config_or_default(panel: StudyPanel) -> AnalysisConfig:
    """Return the first group's config if loaded, else a fresh default.

    Helper for tests that need *some* AnalysisConfig to pass to
    ``update_entry`` but don't care about which one.
    """
    study = panel.current_study()
    if study is not None and study.groups:
        return study.groups[0].config
    return AnalysisConfig()
