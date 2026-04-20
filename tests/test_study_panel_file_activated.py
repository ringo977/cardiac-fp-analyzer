"""Regression test for ``StudyPanel.file_activated`` signal payload (issue #6).

Fase 1 of the Group-config-as-source-of-truth work (GH #6) changes
``file_activated`` from a single-argument ``Signal(str)`` to a
two-argument ``Signal(str, str)`` carrying ``(abspath, group_name)``.
The group name lets ``MainWindow`` track which Group the active file
belongs to, so Ricalcola can later read ``group.config`` instead of
the global app config and keep the fingerprint consistent with the
batch cache entry.

The test:
  1. Builds a real ``Study`` on disk (tmp_path) with one Group and one
     CSV, so ``resolve_file_path`` + ``is_file`` guards in
     ``_on_item_double_clicked`` both pass.
  2. Loads it into a ``StudyPanel``.
  3. Invokes ``_on_item_double_clicked`` directly with a crafted
     ``QTreeWidgetItem`` (simpler than driving real mouse events and
     sufficient because we're covering the signal contract, not Qt
     event dispatch).
  4. Asserts the captured payload matches ``(resolved_abspath,
     group_name)`` — both as strings.

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
    from PySide6.QtWidgets import QApplication, QTreeWidgetItem
except ImportError as exc:   # pragma: no cover — platform-dependent
    pytest.skip(
        f"PySide6 widget libs unavailable: {exc}",
        allow_module_level=True,
    )

from pyside_app.study_panel import (
    _FILE_INDEX_ROLE,
    _GROUP_NAME_ROLE,
    _KIND_ROLE,
    StudyPanel,
)

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
    app.setApplicationName("study-panel-file-activated")
    yield app


def _make_study_with_one_file(tmp_path) -> tuple[Study, str]:
    """Build a Study with one Group and one on-disk CSV.

    Returns ``(study, group_name)``.  The CSV is a minimal two-column
    header so ``is_file()`` returns True; content doesn't matter for
    the signal contract test.
    """
    csv = tmp_path / "a.csv"
    csv.write_text("Time (s),EL1 (V)\n0.000,0.0\n0.001,0.0\n")
    group = Group(
        name="DoseX",
        files=[FileEntry(csv_relpath="a.csv")],
    )
    study = Study(name="TestStudy", folder=tmp_path, groups=[group])
    save_study(study)
    return study, "DoseX"


class TestFileActivatedSignal:
    """Signal contract for ``StudyPanel.file_activated`` post-#6 Fase 1."""

    def test_double_click_emits_path_and_group_name(self, qapp, tmp_path):
        """Double-click on a file row emits ``(abspath, group_name)``.

        Regression target: the signal was single-arg ``Signal(str)``
        before #6 Fase 1.  Now it must carry the group name so
        MainWindow can stash the Group reference for Ricalcola.
        """
        study, group_name = _make_study_with_one_file(tmp_path)
        panel = StudyPanel()
        panel.set_study(study)

        captured: list[tuple[str, str, str]] = []
        panel.file_activated.connect(
            lambda path, gn, rel: captured.append((path, gn, rel)),
        )

        # Craft an item that satisfies the guards in
        # ``_on_item_double_clicked`` without re-driving the tree:
        # KIND must be "file", GROUP_NAME and FILE_INDEX must resolve
        # to a real FileEntry on disk.
        item = QTreeWidgetItem()
        item.setData(0, _KIND_ROLE, "file")
        item.setData(0, _GROUP_NAME_ROLE, group_name)
        item.setData(0, _FILE_INDEX_ROLE, 0)

        panel._on_item_double_clicked(item, 0)

        assert len(captured) == 1, (
            f"expected exactly one emission, got {len(captured)}"
        )
        path_emitted, gn_emitted, rel_emitted = captured[0]
        # ``resolve_file_path`` returns an absolute resolved Path; the
        # slot emits ``str(abspath)``.  We compare string-wise because
        # tmp_path may itself be symlinked on macOS (``/private/var``).
        expected_abs = str((tmp_path / "a.csv").resolve())
        assert path_emitted == expected_abs
        assert gn_emitted == group_name
        # GH #6 Fase 3: the third payload element is the POSIX-style
        # csv_relpath as stored in the Group's FileEntry — used by
        # MainWindow to write the recomputed result back into the
        # Studi batch cache via ``StudyPanel.update_entry``.
        assert rel_emitted == "a.csv"

    def test_double_click_on_non_file_kind_does_not_emit(
        self, qapp, tmp_path,
    ):
        """Guard: double-clicking a Group or Study row must NOT emit.

        Regression coverage for the ``if item.data(0, _KIND_ROLE) !=
        'file': return`` early-out — we only fire the signal for file
        rows, never group or study headers.
        """
        study, _ = _make_study_with_one_file(tmp_path)
        panel = StudyPanel()
        panel.set_study(study)

        captured: list[tuple[str, str, str]] = []
        panel.file_activated.connect(
            lambda path, gn, rel: captured.append((path, gn, rel)),
        )

        # A "group" kind item — no FILE_INDEX set, would crash in the
        # lookup path if the guard were broken.
        group_item = QTreeWidgetItem()
        group_item.setData(0, _KIND_ROLE, "group")
        group_item.setData(0, _GROUP_NAME_ROLE, "DoseX")
        panel._on_item_double_clicked(group_item, 0)

        assert captured == [], (
            "file_activated must stay silent for non-file rows"
        )

    def test_emits_even_when_no_receivers_connected(
        self, qapp, tmp_path,
    ):
        """Sanity: the slot must not raise if nothing is listening.

        Qt allows emitting a signal with zero receivers; we're just
        verifying the internal call ``emit(str, str)`` matches the
        new two-argument declaration (mismatches would raise a
        ``TypeError`` at emit time under PySide6).
        """
        study, group_name = _make_study_with_one_file(tmp_path)
        panel = StudyPanel()
        panel.set_study(study)

        item = QTreeWidgetItem()
        item.setData(0, _KIND_ROLE, "file")
        item.setData(0, _GROUP_NAME_ROLE, group_name)
        item.setData(0, _FILE_INDEX_ROLE, 0)

        # No connect() call — just verify emit doesn't raise.
        panel._on_item_double_clicked(item, 0)
