"""Unit tests for cardiac_fp_analyzer.overrides (manual beat picking).

Covers:

1. Path resolution (``overrides_path_for``) — suffix ``.overrides.json``
   is appended, not substituted, so adjacent files aren't clobbered.
2. Round-trip serialisation — ``save_overrides`` / ``load_overrides``
   on a populated record returns an equivalent ``BeatOverrides``.
3. Missing / malformed sidecars never raise — they return ``None``.
4. ``is_empty()`` deletes the sidecar on save instead of leaving a stub.
5. ``apply_overrides`` semantics:
   a. removed timestamps kick the nearest auto beat within ``tol_s``;
   b. unmatched removals are reported but do not crash;
   c. added timestamps that collide with an existing beat within
      ``tol_s`` are silently deduped (no double-count on re-clicks);
   d. the resulting beat set is sorted and unique;
   e. ``fs <= 0`` raises.
6. ``diff_to_overrides`` infers the minimal record from (auto, final).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardiac_fp_analyzer.overrides import (
    DEFAULT_TOL_S,
    OVERRIDES_SUFFIX,
    SCHEMA_VERSION,
    BeatOverrides,
    apply_overrides,
    diff_to_overrides,
    load_overrides,
    overrides_path_for,
    save_overrides,
)

# ─────────────────────────────────────────────────────────────────────────
#  Path handling
# ─────────────────────────────────────────────────────────────────────────

class TestPathHandling:
    def test_suffix_is_appended_not_substituted(self, tmp_path):
        csv = tmp_path / "foo.csv"
        assert overrides_path_for(csv) == tmp_path / ("foo.csv" + OVERRIDES_SUFFIX)

    def test_works_with_path_object_and_string(self, tmp_path):
        csv = tmp_path / "foo.csv"
        p1 = overrides_path_for(csv)
        p2 = overrides_path_for(str(csv))
        assert p1 == p2


# ─────────────────────────────────────────────────────────────────────────
#  Round-trip
# ─────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_save_then_load_returns_equivalent(self, tmp_path):
        csv = tmp_path / "sig.csv"
        # CSV does not need to exist; save_overrides only needs the path
        # to derive the sidecar location.
        ov = BeatOverrides(
            added_s=[1.234, 5.678],
            removed_s=[3.141],
            tol_s=0.040,
        )
        save_overrides(csv, ov)
        loaded = load_overrides(csv)
        assert loaded is not None
        assert loaded.added_s == ov.added_s
        assert loaded.removed_s == ov.removed_s
        assert loaded.tol_s == ov.tol_s
        assert loaded.version == SCHEMA_VERSION

    def test_missing_sidecar_returns_none(self, tmp_path):
        csv = tmp_path / "nope.csv"
        assert load_overrides(csv) is None

    def test_malformed_json_returns_none_does_not_raise(self, tmp_path):
        csv = tmp_path / "broken.csv"
        sidecar = overrides_path_for(csv)
        sidecar.write_text("{ not valid json ", encoding='utf-8')
        assert load_overrides(csv) is None

    def test_json_non_object_returns_none(self, tmp_path):
        csv = tmp_path / "list_instead.csv"
        sidecar = overrides_path_for(csv)
        sidecar.write_text("[1, 2, 3]", encoding='utf-8')
        assert load_overrides(csv) is None

    def test_unknown_fields_are_ignored(self, tmp_path):
        csv = tmp_path / "extra.csv"
        sidecar = overrides_path_for(csv)
        sidecar.write_text(json.dumps({
            'added_s': [1.0],
            'removed_s': [],
            'tol_s': 0.05,
            'version': '1',
            'future_field': 42,  # should be ignored, not crash
        }), encoding='utf-8')
        loaded = load_overrides(csv)
        assert loaded is not None
        assert loaded.added_s == [1.0]

    def test_missing_optional_fields_use_defaults(self, tmp_path):
        csv = tmp_path / "minimal.csv"
        sidecar = overrides_path_for(csv)
        sidecar.write_text(json.dumps({}), encoding='utf-8')
        loaded = load_overrides(csv)
        assert loaded is not None
        assert loaded.added_s == []
        assert loaded.removed_s == []
        assert loaded.tol_s == DEFAULT_TOL_S


# ─────────────────────────────────────────────────────────────────────────
#  Empty → delete
# ─────────────────────────────────────────────────────────────────────────

class TestSaveEmpty:
    def test_saving_empty_deletes_existing_sidecar(self, tmp_path):
        csv = tmp_path / "sig.csv"
        save_overrides(csv, BeatOverrides(added_s=[1.0]))
        assert overrides_path_for(csv).exists()
        save_overrides(csv, BeatOverrides())  # empty
        assert not overrides_path_for(csv).exists()

    def test_saving_empty_is_noop_if_no_sidecar_exists(self, tmp_path):
        csv = tmp_path / "sig.csv"
        # Should not raise
        save_overrides(csv, BeatOverrides())
        assert not overrides_path_for(csv).exists()


# ─────────────────────────────────────────────────────────────────────────
#  apply_overrides semantics
# ─────────────────────────────────────────────────────────────────────────

class TestApplyRemove:
    def test_removes_nearest_within_tol(self):
        fs = 1000.0
        bi_auto = np.array([1000, 2000, 3000, 4000], dtype=int)  # 1..4 s
        ov = BeatOverrides(removed_s=[2.010], tol_s=0.050)  # 10 ms off
        bi_manual, info = apply_overrides(bi_auto, ov, fs)
        assert 2000 not in bi_manual.tolist()
        assert info['n_removed'] == 1
        assert info['unmatched_removals'] == []

    def test_unmatched_remove_is_reported_not_applied(self):
        fs = 1000.0
        bi_auto = np.array([1000, 2000, 3000], dtype=int)
        # Click landed at 2.5 s — 500 ms away from nearest beat
        ov = BeatOverrides(removed_s=[2.500], tol_s=0.050)
        bi_manual, info = apply_overrides(bi_auto, ov, fs)
        assert bi_manual.tolist() == [1000, 2000, 3000]
        assert info['n_removed'] == 0
        assert info['unmatched_removals'] == [2.500]

    def test_multiple_removals_do_not_collide(self):
        """Two removals pointing near the same auto beat should each
        remove a distinct auto beat when possible, and otherwise report
        the second as unmatched (no crash, no spurious re-adds)."""
        fs = 1000.0
        bi_auto = np.array([1000, 2000], dtype=int)
        ov = BeatOverrides(removed_s=[1.005, 1.010], tol_s=0.050)
        bi_manual, info = apply_overrides(bi_auto, ov, fs)
        # First removal kicks beat 1000. Second has no near-enough beat
        # left (2000 is 1 s away) → unmatched.
        assert 1000 not in bi_manual.tolist()
        assert 2000 in bi_manual.tolist()
        assert info['n_removed'] == 1
        assert len(info['unmatched_removals']) == 1


class TestApplyAdd:
    def test_add_inserts_index(self):
        fs = 1000.0
        bi_auto = np.array([1000, 3000], dtype=int)
        ov = BeatOverrides(added_s=[2.0])
        bi_manual, info = apply_overrides(bi_auto, ov, fs)
        assert bi_manual.tolist() == [1000, 2000, 3000]
        assert info['n_added'] == 1

    def test_add_is_deduped_against_existing_beats(self):
        """Click near an already-detected beat should NOT double it."""
        fs = 1000.0
        bi_auto = np.array([1000, 2000, 3000], dtype=int)
        ov = BeatOverrides(added_s=[2.010], tol_s=0.050)  # 10 ms off 2000
        bi_manual, info = apply_overrides(bi_auto, ov, fs)
        assert bi_manual.tolist() == [1000, 2000, 3000]
        assert info['n_added'] == 0

    def test_add_dedupes_against_each_other(self):
        """Two near-identical add-clicks produce a single beat."""
        fs = 1000.0
        bi_auto = np.array([1000, 3000], dtype=int)
        ov = BeatOverrides(added_s=[2.000, 2.005], tol_s=0.050)
        bi_manual, info = apply_overrides(bi_auto, ov, fs)
        assert bi_manual.tolist() == [1000, 2000, 3000]
        assert info['n_added'] == 1


class TestApplyInvariants:
    def test_result_is_sorted_and_unique(self):
        fs = 1000.0
        bi_auto = np.array([3000, 1000, 2000, 2000], dtype=int)
        ov = BeatOverrides()
        bi_manual, _ = apply_overrides(bi_auto, ov, fs)
        assert bi_manual.tolist() == sorted(set(bi_manual.tolist()))

    def test_empty_auto_with_adds_still_works(self):
        fs = 1000.0
        ov = BeatOverrides(added_s=[1.0, 2.0])
        bi_manual, info = apply_overrides([], ov, fs)
        assert bi_manual.tolist() == [1000, 2000]
        assert info['n_added'] == 2

    def test_empty_auto_with_removes_reports_unmatched(self):
        fs = 1000.0
        ov = BeatOverrides(removed_s=[1.0])
        bi_manual, info = apply_overrides([], ov, fs)
        assert bi_manual.size == 0
        assert info['unmatched_removals'] == [1.0]

    def test_negative_fs_raises(self):
        with pytest.raises(ValueError):
            apply_overrides([1, 2], BeatOverrides(), fs=-1.0)

    def test_zero_fs_raises(self):
        with pytest.raises(ValueError):
            apply_overrides([1, 2], BeatOverrides(), fs=0.0)


# ─────────────────────────────────────────────────────────────────────────
#  diff_to_overrides
# ─────────────────────────────────────────────────────────────────────────

class TestDiff:
    def test_pure_remove(self):
        ov = diff_to_overrides(
            bi_auto=[1000, 2000, 3000],
            bi_final=[1000, 3000],
            fs=1000.0,
        )
        assert ov.removed_s == [2.0]
        assert ov.added_s == []

    def test_pure_add(self):
        ov = diff_to_overrides(
            bi_auto=[1000, 3000],
            bi_final=[1000, 2000, 3000],
            fs=1000.0,
        )
        assert ov.added_s == [2.0]
        assert ov.removed_s == []

    def test_identity_yields_empty(self):
        ov = diff_to_overrides(
            bi_auto=[1000, 2000],
            bi_final=[1000, 2000],
            fs=1000.0,
        )
        assert ov.is_empty()


# ─────────────────────────────────────────────────────────────────────────
#  End-to-end: diff → save → load → apply reproduces final beat set
# ─────────────────────────────────────────────────────────────────────────

class TestE2E:
    def test_diff_save_load_apply_round_trip(self, tmp_path: Path):
        fs = 1000.0
        bi_auto = [1000, 2000, 3000, 4000, 5000]
        bi_final = [1000, 2000, 2500, 4000, 5000]  # dropped 3000, added 2500
        ov = diff_to_overrides(bi_auto, bi_final, fs)
        csv = tmp_path / "signal.csv"
        save_overrides(csv, ov)
        reloaded = load_overrides(csv)
        assert reloaded is not None
        bi_manual, info = apply_overrides(bi_auto, reloaded, fs)
        assert bi_manual.tolist() == bi_final
        assert info['n_added'] == 1
        assert info['n_removed'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
