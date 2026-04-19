"""Unit tests for cardiac_fp_analyzer.project (Project / Group / File model).

Covers:

1. Dataclass round-trip — ``to_dict`` / ``from_dict`` preserve every
   field including the embedded :class:`AnalysisConfig`.
2. Save / load round-trip to disk — the on-disk JSON is stable and
   portable (relative paths, no absolute folder).
3. Missing / malformed sidecar behaviour — callers get ``None`` rather
   than an exception.
4. Atomic write — writing to an existing project replaces it without
   leaving a half-written sidecar.
5. Portability — moving / renaming the project folder and pointing
   :func:`load_project` at the new location still resolves file paths
   correctly.
6. :func:`make_file_entry` — accepts absolute + relative CSV paths,
   rejects paths outside the project folder.
7. Group management — duplicate-name guard, ``remove_group``,
   ``add_file_to_group`` error on unknown group.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.project import (
    PROJECT_FILENAME,
    SCHEMA_VERSION,
    FileEntry,
    Group,
    Project,
    add_file_to_group,
    add_group,
    find_group,
    load_project,
    make_file_entry,
    remove_group,
    resolve_file_path,
    save_project,
)

# ─────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def project_folder(tmp_path):
    """An empty directory we'll treat as a project root."""
    folder = tmp_path / "myproject"
    folder.mkdir()
    return folder


@pytest.fixture
def project_with_files(project_folder):
    """Project with two groups, each with one CSV stub on disk.

    The CSVs are empty byte-strings — we just need the paths to exist
    so ``make_file_entry`` / ``resolve_file_path`` can round-trip.
    """
    (project_folder / "data").mkdir()
    (project_folder / "data" / "baseline.csv").write_text("x")
    (project_folder / "data" / "dose1.csv").write_text("y")

    p = Project(name="Exp6-DofRR", folder=str(project_folder))
    cfg_baseline = AnalysisConfig()
    cfg_baseline.inclusion.max_cv_bp = 25.0  # sentinel we can read back
    add_group(p, Group(
        name="baseline", drug="Control", dose_uM=None,
        condition="pre-drug", analysis_date="2026-04-19",
        notes="stable rhythm", config=cfg_baseline,
    ))
    add_group(p, Group(
        name="dof-10nM", drug="Dofetilide", dose_uM=0.010,
        condition="dose1", analysis_date="2026-04-19",
        notes="", config=AnalysisConfig(),
    ))
    add_file_to_group(p, "baseline", project_folder / "data" / "baseline.csv",
                     channel="EL1", note="clean")
    add_file_to_group(p, "dof-10nM", project_folder / "data" / "dose1.csv",
                     channel="EL1")
    return p


# ─────────────────────────────────────────────────────────────────────────
#  Dataclass round-trip (in-memory)
# ─────────────────────────────────────────────────────────────────────────

class TestDictRoundTrip:
    def test_file_entry_round_trip(self):
        f = FileEntry(csv_relpath="data/a.csv", channel="EL2", note="ok")
        assert FileEntry.from_dict(f.to_dict()) == f

    def test_file_entry_from_dict_tolerates_missing_fields(self):
        f = FileEntry.from_dict({"csv_relpath": "x.csv"})
        assert f.csv_relpath == "x.csv"
        assert f.channel == "auto"
        assert f.note == ""

    def test_group_round_trip_preserves_config(self):
        cfg = AnalysisConfig()
        cfg.inclusion.max_cv_bp = 42.0
        g = Group(
            name="g1", drug="Sotalol", dose_uM=0.5, condition="wash",
            analysis_date="2026-04-19", notes="n", config=cfg,
            files=[FileEntry(csv_relpath="a.csv")],
        )
        g2 = Group.from_dict(g.to_dict())
        assert g2.name == "g1"
        assert g2.drug == "Sotalol"
        assert g2.dose_uM == 0.5
        assert g2.condition == "wash"
        assert g2.notes == "n"
        assert len(g2.files) == 1
        assert g2.files[0].csv_relpath == "a.csv"
        # The config must survive the round-trip — pick a sentinel field.
        assert g2.config.inclusion.max_cv_bp == 42.0

    def test_group_none_dose_stays_none(self):
        g = Group(name="baseline", drug="Control", dose_uM=None)
        g2 = Group.from_dict(g.to_dict())
        assert g2.dose_uM is None

    def test_group_ignores_non_numeric_dose(self):
        # Tolerant loading: a corrupt field becomes None rather than
        # blowing up the whole project.
        g = Group.from_dict({"name": "x", "dose_uM": "not a number"})
        assert g.dose_uM is None

    def test_project_from_dict_does_not_serialise_folder(self, tmp_path):
        p = Project(name="p", folder=str(tmp_path))
        d = p.to_dict()
        assert "folder" not in d
        # Re-read gets folder from the caller, not the JSON
        tmp_path2 = tmp_path / "elsewhere"
        tmp_path2.mkdir()
        p2 = Project.from_dict(d, folder=tmp_path2)
        assert Path(p2.folder) == tmp_path2.resolve()


# ─────────────────────────────────────────────────────────────────────────
#  Save / load round-trip
# ─────────────────────────────────────────────────────────────────────────

class TestDiskRoundTrip:
    def test_save_then_load(self, project_with_files):
        p = project_with_files
        sidecar = save_project(p)
        assert sidecar.is_file()
        assert sidecar.name == PROJECT_FILENAME

        p2 = load_project(p.folder)
        assert p2 is not None
        assert p2.name == p.name
        assert p2.version == SCHEMA_VERSION
        assert len(p2.groups) == 2
        assert {g.name for g in p2.groups} == {"baseline", "dof-10nM"}

    def test_load_preserves_relative_paths(self, project_with_files):
        save_project(project_with_files)
        p2 = load_project(project_with_files.folder)
        baseline = find_group(p2, "baseline")
        assert baseline is not None
        assert baseline.files[0].csv_relpath == "data/baseline.csv"

    def test_load_preserves_config_per_group(self, project_with_files):
        save_project(project_with_files)
        p2 = load_project(project_with_files.folder)
        baseline = find_group(p2, "baseline")
        assert baseline is not None
        assert baseline.config.inclusion.max_cv_bp == 25.0

    def test_json_has_no_absolute_paths(self, project_with_files):
        sidecar = save_project(project_with_files)
        text = sidecar.read_text(encoding="utf-8")
        assert str(project_with_files.folder) not in text
        # Individual file paths are stored POSIX-relative.
        data = json.loads(text)
        for group in data["groups"]:
            for f in group["files"]:
                assert not Path(f["csv_relpath"]).is_absolute()
                # On disk we always use forward slashes.
                assert "\\" not in f["csv_relpath"]


# ─────────────────────────────────────────────────────────────────────────
#  Missing / malformed sidecar
# ─────────────────────────────────────────────────────────────────────────

class TestTolerantLoading:
    def test_missing_sidecar_returns_none(self, project_folder):
        assert load_project(project_folder) is None

    def test_malformed_json_returns_none(self, project_folder):
        (project_folder / PROJECT_FILENAME).write_text("this is not json")
        assert load_project(project_folder) is None

    def test_non_object_json_returns_none(self, project_folder):
        (project_folder / PROJECT_FILENAME).write_text("[1, 2, 3]")
        assert load_project(project_folder) is None

    def test_unknown_fields_are_ignored(self, project_folder):
        (project_folder / PROJECT_FILENAME).write_text(json.dumps({
            "version": "99",  # forward-compat: newer schema
            "name": "future",
            "groups": [],
            "mystery_field": "not a crash",
        }))
        p = load_project(project_folder)
        assert p is not None
        assert p.name == "future"


# ─────────────────────────────────────────────────────────────────────────
#  Atomic write
# ─────────────────────────────────────────────────────────────────────────

class TestAtomicWrite:
    def test_overwrite_replaces_cleanly(self, project_with_files):
        save_project(project_with_files)
        # Mutate + re-save
        project_with_files.groups[0].notes = "updated"
        save_project(project_with_files)

        p2 = load_project(project_with_files.folder)
        assert p2 is not None
        baseline = find_group(p2, "baseline")
        assert baseline is not None
        assert baseline.notes == "updated"

    def test_no_leftover_tmp_files(self, project_with_files):
        save_project(project_with_files)
        save_project(project_with_files)
        folder = Path(project_with_files.folder)
        leftovers = [
            p for p in folder.iterdir()
            if p.name.startswith(PROJECT_FILENAME + ".")
        ]
        assert leftovers == []

    def test_save_refuses_missing_folder(self, tmp_path):
        p = Project(name="x", folder=str(tmp_path / "does-not-exist"))
        with pytest.raises(FileNotFoundError):
            save_project(p)


# ─────────────────────────────────────────────────────────────────────────
#  Portability — move the project folder
# ─────────────────────────────────────────────────────────────────────────

class TestPortability:
    def test_moved_project_still_resolves_files(
        self, project_with_files, tmp_path,
    ):
        save_project(project_with_files)
        src = Path(project_with_files.folder)
        dst = tmp_path / "moved-project"
        shutil.move(str(src), str(dst))

        p2 = load_project(dst)
        assert p2 is not None
        baseline = find_group(p2, "baseline")
        assert baseline is not None
        resolved = resolve_file_path(p2, baseline.files[0])
        assert resolved == (dst / "data" / "baseline.csv").resolve()
        assert resolved.is_file()


# ─────────────────────────────────────────────────────────────────────────
#  make_file_entry
# ─────────────────────────────────────────────────────────────────────────

class TestMakeFileEntry:
    def test_absolute_path_inside_project(self, project_folder):
        (project_folder / "a.csv").write_text("x")
        p = Project(name="x", folder=str(project_folder))
        entry = make_file_entry(p, project_folder / "a.csv", channel="EL1")
        assert entry.csv_relpath == "a.csv"
        assert entry.channel == "EL1"

    def test_relative_path_is_resolved_from_project_root(self, project_folder):
        (project_folder / "sub").mkdir()
        (project_folder / "sub" / "a.csv").write_text("x")
        p = Project(name="x", folder=str(project_folder))
        entry = make_file_entry(p, "sub/a.csv")
        assert entry.csv_relpath == "sub/a.csv"

    def test_path_outside_project_raises(self, tmp_path, project_folder):
        outside = tmp_path / "other.csv"
        outside.write_text("x")
        p = Project(name="x", folder=str(project_folder))
        with pytest.raises(ValueError, match="outside the project folder"):
            make_file_entry(p, outside)


# ─────────────────────────────────────────────────────────────────────────
#  Group mutation helpers
# ─────────────────────────────────────────────────────────────────────────

class TestGroupMutation:
    def test_add_group_rejects_duplicate_name(self, project_folder):
        p = Project(name="x", folder=str(project_folder))
        add_group(p, Group(name="baseline"))
        with pytest.raises(ValueError, match="duplicate"):
            add_group(p, Group(name="baseline"))

    def test_add_group_rejects_empty_name(self, project_folder):
        p = Project(name="x", folder=str(project_folder))
        with pytest.raises(ValueError, match="non-empty"):
            add_group(p, Group(name=""))

    def test_remove_group(self, project_folder):
        p = Project(name="x", folder=str(project_folder))
        add_group(p, Group(name="a"))
        add_group(p, Group(name="b"))
        assert remove_group(p, "a") is True
        assert [g.name for g in p.groups] == ["b"]
        assert remove_group(p, "nope") is False

    def test_add_file_to_unknown_group_raises(self, project_folder):
        (project_folder / "a.csv").write_text("x")
        p = Project(name="x", folder=str(project_folder))
        with pytest.raises(KeyError):
            add_file_to_group(p, "ghost", project_folder / "a.csv")

    def test_find_group(self, project_folder):
        p = Project(name="x", folder=str(project_folder))
        g = Group(name="a")
        add_group(p, g)
        assert find_group(p, "a") is g
        assert find_group(p, "nope") is None
