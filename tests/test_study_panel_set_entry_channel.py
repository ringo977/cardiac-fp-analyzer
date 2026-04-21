"""Regression tests for ``StudyPanel.set_entry_channel`` (#82 follow-up, GH #7).

When the user changes the channel in the Signal-tab combo for a file
that was opened from a Study, :class:`MainWindow` persists the choice
into :attr:`FileEntry.channel` via ``StudyPanel.set_entry_channel``.
This method also re-stamps the cached :class:`FileResult` fingerprint
so :func:`_is_stale` now flags the row with ``●`` — the visual cue
that the group aggregate is out of sync with the viewer-level choice.

The tests below pin that contract:

1. **Happy path** — auto→EL1 with an existing cached result: returns
   ``True``, ``fe.channel`` is updated, the cached FileResult body is
   preserved (FPD / amp stay), and its fingerprint re-points to the
   *old* channel so ``_is_stale`` flips to True.
2. **No-op on same channel** — auto→auto returns ``False``, study is
   not saved, cache untouched.
3. **Canonicalisation** — unknown value ('xyz') collapses to 'auto'
   (same rule as ``_BatchWorker``); auto→'xyz' is therefore a no-op.
4. **Valid EL2 path** — sanity: EL1→EL2 works symmetrically.
5. **Unknown group / csv_relpath / closed study** — returns ``False``,
   nothing mutates, no exception.
6. **Missing cached result** — just writes ``fe.channel``; no cache
   entry to re-stamp is fine.

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
    _is_stale,
    _result_fingerprint,
)

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.study import (
    FileEntry,
    Group,
    Study,
    load_study,
    save_study,
)


@pytest.fixture(scope="module")
def qapp():
    """Shared ``QApplication`` for all tests in this module."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("CardiacFP-Test")
    app.setApplicationName("study-panel-set-entry-channel")
    yield app


def _make_panel_with_study(
    tmp_path,
    *,
    fe_channel: str = 'auto',
    with_cached_result: bool = True,
    cached_channel_analyzed: str = 'EL2',
) -> tuple[StudyPanel, Study, Group, FileEntry]:
    """Build a ``StudyPanel`` loaded with a 1-group / 1-file Study.

    Optionally seeds the cache with a FileResult whose fingerprint
    matches the current (config, channel) so ``_is_stale`` starts at
    False — the test then asserts the method flips it to True.

    ``cached_channel_analyzed`` defaults to ``'EL2'`` so that the common
    ``auto→EL1`` test vector is GENUINELY stale (auto had picked EL2,
    the user now explicitly overrides to EL1 — different signal, must
    re-run).  Tests that need the semantic-equal shortcut from Fix #1
    override this to match the explicit target (see
    ``test_auto_to_analyzed_channel_semantic_equal_no_stale``).
    """
    csv = tmp_path / "a.csv"
    csv.write_text("Time (s),EL1 (V)\n0.000,0.0\n0.001,0.0\n")
    group = Group(
        name="DoseX",
        files=[FileEntry(csv_relpath="a.csv", channel=fe_channel)],
    )
    study = Study(name="TestStudy", folder=str(tmp_path), groups=[group])
    save_study(study)

    panel = StudyPanel()
    panel.set_study(study)

    if with_cached_result:
        # Stamp a fingerprint that matches the current (config, channel)
        # so _is_stale returns False before the test mutates channel.
        fp = _result_fingerprint(group.config, fe_channel)
        panel._results[(group.name, "a.csv")] = FileResult(
            status='ok',
            channel_analyzed=cached_channel_analyzed,
            n_included=5,
            n_total=6,
            fingerprint=fp,
            fpd_ms=320.0,
            fpdc_ms=330.0,
            bpm=72.0,
            stv_ms=4.5,
            amp_uv=450.0,
        )

    return panel, study, group, group.files[0]


class TestSetEntryChannel:
    """Contract for ``StudyPanel.set_entry_channel`` — #82 follow-up."""

    def test_happy_path_auto_to_el1_marks_stale_and_persists(
        self, qapp, tmp_path,
    ):
        panel, study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
        )
        # Sanity: fresh cache, not stale.
        cached_before = panel._results[(group.name, "a.csv")]
        assert not _is_stale(cached_before, group.config, fe.channel)

        changed = panel.set_entry_channel(group.name, "a.csv", 'EL1')

        assert changed is True
        assert fe.channel == 'EL1'

        # Cached FileResult body is preserved — the user's prior
        # aggregate is still readable while they decide to Ricalcola.
        cached_after = panel._results[(group.name, "a.csv")]
        assert cached_after.fpd_ms == pytest.approx(320.0)
        assert cached_after.amp_uv == pytest.approx(450.0)
        assert cached_after.n_included == 5
        assert cached_after.status == 'ok'

        # But its fingerprint now anchors to the OLD channel, so the
        # current (config, new-channel) hash differs → the row will
        # render with ● on the next repaint.
        assert _is_stale(cached_after, group.config, fe.channel)
        assert cached_after.fingerprint == _result_fingerprint(
            group.config, 'auto',
        )

        # Persistence: re-load the study from disk and verify the new
        # channel survived the save.  This is the whole point of the
        # save-on-channel-change rule.
        reloaded = load_study(str(tmp_path))
        assert reloaded is not None
        assert reloaded.groups[0].files[0].channel == 'EL1'

    def test_noop_on_same_channel_returns_false(self, qapp, tmp_path):
        panel, study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
        )
        before_fp = panel._results[(group.name, "a.csv")].fingerprint

        changed = panel.set_entry_channel(group.name, "a.csv", 'auto')

        assert changed is False
        assert fe.channel == 'auto'
        # Cache untouched — same fingerprint, same body.
        after_fp = panel._results[(group.name, "a.csv")].fingerprint
        assert after_fp == before_fp

    def test_unknown_channel_canonicalises_to_auto(self, qapp, tmp_path):
        # fe starts on 'auto', user somehow sends 'xyz' → must
        # collapse to 'auto' (same rule as _BatchWorker) → no change.
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'xyz')
        assert changed is False
        assert fe.channel == 'auto'

    def test_lowercase_combo_value_el2_is_stored_as_uppercase(
        self, qapp, tmp_path,
    ):
        # The Signal-tab combo emits lowercase 'el1'/'el2' (see
        # _CHANNEL_VALUES in main.py).  _BatchWorker and _is_stale
        # fingerprint on uppercase 'EL1'/'EL2'.  set_entry_channel
        # must bridge: accept lowercase, store uppercase — otherwise
        # the ● stale badge never fires on real user input.  This is
        # the regression pinned by Marco's smoke test where the
        # combo change silently produced a no-op.
        #
        # We pin ``cached_channel_analyzed='EL1'`` so the auto→EL2
        # transition is genuinely stale (auto had picked EL1, user
        # overrides to EL2 — different signal).  The semantic-equal
        # shortcut (Fix #1) is covered by
        # ``test_auto_to_analyzed_channel_semantic_equal_no_stale``.
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
            cached_channel_analyzed='EL1',
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'el2')

        assert changed is True
        assert fe.channel == 'EL2'
        cached = panel._results[(group.name, "a.csv")]
        assert _is_stale(cached, group.config, fe.channel)

    def test_lowercase_el1_is_stored_as_uppercase(
        self, qapp, tmp_path,
    ):
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'el1')

        assert changed is True
        assert fe.channel == 'EL1'

    def test_unknown_channel_canonicalises_when_fe_is_el1(
        self, qapp, tmp_path,
    ):
        # Starting from 'EL1', 'xyz' canonicalises to 'auto' → DOES
        # change the entry.  Pins the canonicalisation rule.
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='EL1', with_cached_result=True,
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'xyz')
        assert changed is True
        assert fe.channel == 'auto'

    def test_el1_to_el2_path_marks_stale(self, qapp, tmp_path):
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='EL1', with_cached_result=True,
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'EL2')

        assert changed is True
        assert fe.channel == 'EL2'
        cached = panel._results[(group.name, "a.csv")]
        assert _is_stale(cached, group.config, fe.channel)
        assert cached.fingerprint == _result_fingerprint(
            group.config, 'EL1',
        )

    def test_missing_cached_result_still_persists_channel(
        self, qapp, tmp_path,
    ):
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=False,
        )
        assert (group.name, "a.csv") not in panel._results

        changed = panel.set_entry_channel(group.name, "a.csv", 'EL2')

        assert changed is True
        assert fe.channel == 'EL2'
        # No cache entry was created — this method doesn't invent
        # FileResults, it only flags existing ones stale.
        assert (group.name, "a.csv") not in panel._results

    def test_unknown_group_returns_false(self, qapp, tmp_path):
        panel, _study, group, fe = _make_panel_with_study(tmp_path)
        changed = panel.set_entry_channel('NOPE', "a.csv", 'EL1')
        assert changed is False
        assert fe.channel == 'auto'  # unchanged

    def test_unknown_csv_relpath_returns_false(self, qapp, tmp_path):
        panel, _study, group, fe = _make_panel_with_study(tmp_path)
        changed = panel.set_entry_channel(group.name, "ghost.csv", 'EL1')
        assert changed is False
        assert fe.channel == 'auto'  # unchanged

    def test_closed_study_returns_false(self, qapp, tmp_path):
        panel, _study, _group, _fe = _make_panel_with_study(tmp_path)
        panel.set_study(None)
        changed = panel.set_entry_channel('DoseX', "a.csv", 'EL1')
        assert changed is False

    # ─── Fix #1 (GH #7 followup) — semantic-equal shortcut ────────────
    # When ``auto`` has already picked ``EL1`` and the user now
    # explicitly selects ``EL1`` from the Signal-tab combo, the cached
    # numbers are literally the ones we'd get re-running on ``EL1`` —
    # same config, same samples, same beats.  Stamping the row as stale
    # (●) would be a false positive: Ricalcola would just reprocess
    # identical input.  The contract below pins this: the channel is
    # persisted (important for auditing / export — the explicit label
    # is meaningful) but the fingerprint is re-anchored to the NEW
    # ``(config, canon)`` so ``_is_stale`` stays False.
    # ─────────────────────────────────────────────────────────────────

    def test_auto_to_analyzed_channel_semantic_equal_no_stale(
        self, qapp, tmp_path,
    ):
        # auto had picked EL1 → user now explicitly selects EL1 → no ●.
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
            cached_channel_analyzed='EL1',
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'EL1')

        # Choice is persisted — important for export/audit.
        assert changed is True
        assert fe.channel == 'EL1'

        # Cache body preserved (the numbers are correct for EL1 anyway).
        cached_after = panel._results[(group.name, "a.csv")]
        assert cached_after.fpd_ms == pytest.approx(320.0)
        assert cached_after.amp_uv == pytest.approx(450.0)
        assert cached_after.channel_analyzed == 'EL1'

        # Fingerprint re-anchored to (config, 'EL1') — NOT stale.  The
        # row should render with ✓ on the next repaint, not ●.
        assert not _is_stale(cached_after, group.config, fe.channel)
        assert cached_after.fingerprint == _result_fingerprint(
            group.config, 'EL1',
        )

        # And of course the channel choice survived to disk.
        reloaded = load_study(str(tmp_path))
        assert reloaded is not None
        assert reloaded.groups[0].files[0].channel == 'EL1'

    def test_auto_to_non_analyzed_channel_is_genuinely_stale(
        self, qapp, tmp_path,
    ):
        # auto had picked EL1 but user now overrides to EL2 → ● (stale).
        # This is the complementary case: semantic-equal must NOT fire
        # when the explicit target differs from what auto had chosen.
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
            cached_channel_analyzed='EL1',
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'EL2')

        assert changed is True
        assert fe.channel == 'EL2'

        cached_after = panel._results[(group.name, "a.csv")]
        assert _is_stale(cached_after, group.config, fe.channel)
        assert cached_after.fingerprint == _result_fingerprint(
            group.config, 'auto',
        )

    def test_semantic_equal_case_insensitive_on_cached_channel(
        self, qapp, tmp_path,
    ):
        # ``cached.channel_analyzed`` COULD be lowercase (legacy cache
        # payloads, future pipeline refactors, etc.).  The semantic-
        # equal comparison must match case-insensitively so the
        # shortcut fires even when the cache hasn't been re-canonicalised.
        panel, _study, group, fe = _make_panel_with_study(
            tmp_path, fe_channel='auto', with_cached_result=True,
            cached_channel_analyzed='el2',   # lowercase, unusual
        )
        changed = panel.set_entry_channel(group.name, "a.csv", 'EL2')

        assert changed is True
        assert fe.channel == 'EL2'

        cached_after = panel._results[(group.name, "a.csv")]
        assert not _is_stale(cached_after, group.config, fe.channel)
