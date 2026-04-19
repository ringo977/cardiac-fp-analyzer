"""Unit tests for cardiac_fp_analyzer.study (Study / Group / File model).

Covers:

1. Dataclass round-trip — ``to_dict`` / ``from_dict`` preserve every
   field including the embedded :class:`AnalysisConfig`.
2. Save / load round-trip to disk — the on-disk JSON is stable and
   portable (relative paths, no absolute folder).
3. Missing / malformed sidecar behaviour — callers get ``None`` rather
   than an exception.
4. Atomic write — writing to an existing study replaces it without
   leaving a half-written sidecar.
5. Portability — moving / renaming the study folder and pointing
   :func:`load_study` at the new location still resolves file paths
   correctly.
6. :func:`make_file_entry` — accepts absolute + relative CSV paths,
   rejects paths outside the study folder.
7. Group management — duplicate-name guard, ``remove_group``,
   ``add_file_to_group`` error on unknown group.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.study import (
    SCHEMA_VERSION,
    STUDY_FILENAME,
    FileEntry,
    Group,
    Study,
    add_file_to_group,
    add_group,
    find_group,
    load_study,
    make_file_entry,
    remove_group,
    resolve_file_path,
    save_study,
)

# ─────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def study_folder(tmp_path):
    """An empty directory we'll treat as a study root."""
    folder = tmp_path / "mystudy"
    folder.mkdir()
    return folder


@pytest.fixture
def study_with_files(study_folder):
    """Study with two groups, each with one CSV stub on disk.

    The CSVs are empty byte-strings — we just need the paths to exist
    so ``make_file_entry`` / ``resolve_file_path`` can round-trip.
    """
    (study_folder / "data").mkdir()
    (study_folder / "data" / "baseline.csv").write_text("x")
    (study_folder / "data" / "dose1.csv").write_text("y")

    s = Study(name="Exp6-DofRR", folder=str(study_folder))
    cfg_baseline = AnalysisConfig()
    cfg_baseline.inclusion.max_cv_bp = 25.0  # sentinel we can read back
    add_group(s, Group(
        name="baseline", drug="Control", dose_uM=None,
        condition="pre-drug", analysis_date="2026-04-19",
        notes="stable rhythm", config=cfg_baseline,
    ))
    add_group(s, Group(
        name="dof-10nM", drug="Dofetilide", dose_uM=0.010,
        condition="dose1", analysis_date="2026-04-19",
        notes="", config=AnalysisConfig(),
    ))
    add_file_to_group(s, "baseline", study_folder / "data" / "baseline.csv",
                     channel="EL1", note="clean")
    add_file_to_group(s, "dof-10nM", study_folder / "data" / "dose1.csv",
                     channel="EL1")
    return s


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
        # blowing up the whole study.
        g = Group.from_dict({"name": "x", "dose_uM": "not a number"})
        assert g.dose_uM is None

    def test_study_from_dict_does_not_serialise_folder(self, tmp_path):
        s = Study(name="s", folder=str(tmp_path))
        d = s.to_dict()
        assert "folder" not in d
        # Re-read gets folder from the caller, not the JSON
        tmp_path2 = tmp_path / "elsewhere"
        tmp_path2.mkdir()
        s2 = Study.from_dict(d, folder=tmp_path2)
        assert Path(s2.folder) == tmp_path2.resolve()


# ─────────────────────────────────────────────────────────────────────────
#  Save / load round-trip
# ─────────────────────────────────────────────────────────────────────────

class TestDiskRoundTrip:
    def test_save_then_load(self, study_with_files):
        s = study_with_files
        sidecar = save_study(s)
        assert sidecar.is_file()
        assert sidecar.name == STUDY_FILENAME

        s2 = load_study(s.folder)
        assert s2 is not None
        assert s2.name == s.name
        assert s2.version == SCHEMA_VERSION
        assert len(s2.groups) == 2
        assert {g.name for g in s2.groups} == {"baseline", "dof-10nM"}

    def test_load_preserves_relative_paths(self, study_with_files):
        save_study(study_with_files)
        s2 = load_study(study_with_files.folder)
        baseline = find_group(s2, "baseline")
        assert baseline is not None
        assert baseline.files[0].csv_relpath == "data/baseline.csv"

    def test_load_preserves_config_per_group(self, study_with_files):
        save_study(study_with_files)
        s2 = load_study(study_with_files.folder)
        baseline = find_group(s2, "baseline")
        assert baseline is not None
        assert baseline.config.inclusion.max_cv_bp == 25.0

    def test_json_has_no_absolute_paths(self, study_with_files):
        sidecar = save_study(study_with_files)
        text = sidecar.read_text(encoding="utf-8")
        assert str(study_with_files.folder) not in text
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
    def test_missing_sidecar_returns_none(self, study_folder):
        assert load_study(study_folder) is None

    def test_malformed_json_returns_none(self, study_folder):
        (study_folder / STUDY_FILENAME).write_text("this is not json")
        assert load_study(study_folder) is None

    def test_non_object_json_returns_none(self, study_folder):
        (study_folder / STUDY_FILENAME).write_text("[1, 2, 3]")
        assert load_study(study_folder) is None

    def test_unknown_fields_are_ignored(self, study_folder):
        (study_folder / STUDY_FILENAME).write_text(json.dumps({
            "version": "99",  # forward-compat: newer schema
            "name": "future",
            "groups": [],
            "mystery_field": "not a crash",
        }))
        s = load_study(study_folder)
        assert s is not None
        assert s.name == "future"


# ─────────────────────────────────────────────────────────────────────────
#  Atomic write
# ─────────────────────────────────────────────────────────────────────────

class TestAtomicWrite:
    def test_overwrite_replaces_cleanly(self, study_with_files):
        save_study(study_with_files)
        # Mutate + re-save
        study_with_files.groups[0].notes = "updated"
        save_study(study_with_files)

        s2 = load_study(study_with_files.folder)
        assert s2 is not None
        baseline = find_group(s2, "baseline")
        assert baseline is not None
        assert baseline.notes == "updated"

    def test_no_leftover_tmp_files(self, study_with_files):
        save_study(study_with_files)
        save_study(study_with_files)
        folder = Path(study_with_files.folder)
        leftovers = [
            p for p in folder.iterdir()
            if p.name.startswith(STUDY_FILENAME + ".")
        ]
        assert leftovers == []

    def test_save_refuses_missing_folder(self, tmp_path):
        s = Study(name="x", folder=str(tmp_path / "does-not-exist"))
        with pytest.raises(FileNotFoundError):
            save_study(s)


# ─────────────────────────────────────────────────────────────────────────
#  Portability — move the study folder
# ─────────────────────────────────────────────────────────────────────────

class TestPortability:
    def test_moved_study_still_resolves_files(
        self, study_with_files, tmp_path,
    ):
        save_study(study_with_files)
        src = Path(study_with_files.folder)
        dst = tmp_path / "moved-study"
        shutil.move(str(src), str(dst))

        s2 = load_study(dst)
        assert s2 is not None
        baseline = find_group(s2, "baseline")
        assert baseline is not None
        resolved = resolve_file_path(s2, baseline.files[0])
        assert resolved == (dst / "data" / "baseline.csv").resolve()
        assert resolved.is_file()


# ─────────────────────────────────────────────────────────────────────────
#  make_file_entry
# ─────────────────────────────────────────────────────────────────────────

class TestMakeFileEntry:
    def test_absolute_path_inside_study(self, study_folder):
        (study_folder / "a.csv").write_text("x")
        s = Study(name="x", folder=str(study_folder))
        entry = make_file_entry(s, study_folder / "a.csv", channel="EL1")
        assert entry.csv_relpath == "a.csv"
        assert entry.channel == "EL1"

    def test_relative_path_is_resolved_from_study_root(self, study_folder):
        (study_folder / "sub").mkdir()
        (study_folder / "sub" / "a.csv").write_text("x")
        s = Study(name="x", folder=str(study_folder))
        entry = make_file_entry(s, "sub/a.csv")
        assert entry.csv_relpath == "sub/a.csv"

    def test_path_outside_study_raises(self, tmp_path, study_folder):
        outside = tmp_path / "other.csv"
        outside.write_text("x")
        s = Study(name="x", folder=str(study_folder))
        with pytest.raises(ValueError, match="outside the study folder"):
            make_file_entry(s, outside)


# ─────────────────────────────────────────────────────────────────────────
#  Group mutation helpers
# ─────────────────────────────────────────────────────────────────────────

class TestGroupMutation:
    def test_add_group_rejects_duplicate_name(self, study_folder):
        s = Study(name="x", folder=str(study_folder))
        add_group(s, Group(name="baseline"))
        with pytest.raises(ValueError, match="duplicate"):
            add_group(s, Group(name="baseline"))

    def test_add_group_rejects_empty_name(self, study_folder):
        s = Study(name="x", folder=str(study_folder))
        with pytest.raises(ValueError, match="non-empty"):
            add_group(s, Group(name=""))

    def test_remove_group(self, study_folder):
        s = Study(name="x", folder=str(study_folder))
        add_group(s, Group(name="a"))
        add_group(s, Group(name="b"))
        assert remove_group(s, "a") is True
        assert [g.name for g in s.groups] == ["b"]
        assert remove_group(s, "nope") is False

    def test_add_file_to_unknown_group_raises(self, study_folder):
        (study_folder / "a.csv").write_text("x")
        s = Study(name="x", folder=str(study_folder))
        with pytest.raises(KeyError):
            add_file_to_group(s, "ghost", study_folder / "a.csv")

    def test_find_group(self, study_folder):
        s = Study(name="x", folder=str(study_folder))
        g = Group(name="a")
        add_group(s, g)
        assert find_group(s, "a") is g
        assert find_group(s, "nope") is None
