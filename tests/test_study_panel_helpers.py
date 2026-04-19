"""Unit tests for the pure (non-Qt) helpers in ``pyside_app.study_panel``.

Covers the step-3 batch-summary helpers so they can be validated in CI
without importing PySide6 / spinning up a headless Qt app:

* :class:`FileResult` — dataclass defaults.
* :func:`_badge_for` — tree glyph mapping.
* :func:`_tooltip_for` — human-readable hover text.
* :func:`_aggregate_status` — group-row summary of mixed results.
* :func:`_len_or_zero` — defensive len() for pipeline result dicts.

These feed the visual cues in the Studi panel (task #84 step 3), so
regressions here would silently mis-label batch results — hence the
coverage despite their small size.
"""
from __future__ import annotations

import importlib

import pytest

# We import lazily via importlib so a missing PySide6 in the test env
# only skips these tests instead of erroring the whole module. We need
# to probe QtGui specifically because that's where the native (libEGL /
# libGL) dependency actually loads — the bare ``PySide6`` package alone
# imports cleanly even on headless CI boxes without GL.
pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
try:
    sp = importlib.import_module("pyside_app.study_panel")
except ImportError as exc:  # pragma: no cover — CI-only skip path
    pytest.skip(
        f"pyside_app.study_panel not importable in this env: {exc}",
        allow_module_level=True,
    )


FileResult = sp.FileResult
_badge_for = sp._badge_for
_tooltip_for = sp._tooltip_for
_aggregate_status = sp._aggregate_status
_len_or_zero = sp._len_or_zero


# ─────────────────────────────────────────────────────────────────────────
#  FileResult
# ─────────────────────────────────────────────────────────────────────────

class TestFileResult:
    def test_defaults_for_error_leaves_zero_counts(self):
        r = FileResult(status='error', error='boom')
        assert r.status == 'error'
        assert r.error == 'boom'
        assert r.n_included == 0
        assert r.n_total == 0
        assert r.channel_analyzed == ''

    def test_defaults_for_ok(self):
        r = FileResult(
            status='ok', channel_analyzed='EL1',
            n_included=12, n_total=14,
        )
        assert r.error == ''


# ─────────────────────────────────────────────────────────────────────────
#  _badge_for
# ─────────────────────────────────────────────────────────────────────────

class TestBadgeFor:
    def test_none_has_no_badge(self):
        assert _badge_for(None) == ""

    def test_ok_is_checkmark(self):
        assert _badge_for(FileResult(status='ok')) == "✓"

    def test_error_is_cross(self):
        assert _badge_for(FileResult(status='error', error='x')) == "✗"

    def test_unknown_status_is_no_badge(self):
        # Defensive: future statuses (e.g. 'stale') shouldn't crash
        # the renderer — they just don't get a badge yet.
        assert _badge_for(FileResult(status='stale')) == ""


# ─────────────────────────────────────────────────────────────────────────
#  _tooltip_for
# ─────────────────────────────────────────────────────────────────────────

class TestTooltipFor:
    def test_none_has_no_tooltip(self):
        assert _tooltip_for(None) == ""

    def test_error_shows_message(self):
        r = FileResult(status='error', error='no signal')
        assert "no signal" in _tooltip_for(r)
        assert "Errore" in _tooltip_for(r)

    def test_error_without_message_still_labels_itself(self):
        r = FileResult(status='error')
        assert _tooltip_for(r) == "Errore"

    def test_ok_shows_channel_and_counts_when_equal(self):
        r = FileResult(
            status='ok', channel_analyzed='EL1',
            n_included=14, n_total=14,
        )
        t = _tooltip_for(r)
        assert "EL1" in t
        assert "14 battiti" in t
        # No "inclusi" suffix when all beats are included.
        assert "/" not in t

    def test_ok_shows_fraction_when_some_excluded(self):
        r = FileResult(
            status='ok', channel_analyzed='EL2',
            n_included=10, n_total=14,
        )
        t = _tooltip_for(r)
        assert "EL2" in t
        assert "10/14" in t
        assert "inclusi" in t

    def test_ok_without_counts_still_shows_channel(self):
        r = FileResult(
            status='ok', channel_analyzed='EL1',
            n_included=0, n_total=0,
        )
        assert _tooltip_for(r) == "Canale: EL1"


# ─────────────────────────────────────────────────────────────────────────
#  _aggregate_status
# ─────────────────────────────────────────────────────────────────────────

class TestAggregateStatus:
    def test_empty_list(self):
        assert _aggregate_status([]) == ""

    def test_all_none(self):
        assert _aggregate_status([None, None, None]) == ""

    def test_all_ok_uses_compact_form(self):
        results = [FileResult(status='ok') for _ in range(3)]
        assert _aggregate_status(results) == "✓ 3/3"

    def test_mixed_ok_and_error(self):
        results = [
            FileResult(status='ok'),
            FileResult(status='ok'),
            FileResult(status='error', error='x'),
        ]
        out = _aggregate_status(results)
        assert "2✓" in out
        assert "1✗" in out

    def test_error_only_shows_cross_count(self):
        results = [
            FileResult(status='error', error='x'),
            FileResult(status='error', error='y'),
        ]
        out = _aggregate_status(results)
        assert "2✗" in out
        assert "✓" not in out

    def test_ok_plus_unanalysed_still_reports_ok(self):
        # When some files are pending, we still want to see "1 is ok"
        # rather than hiding progress.
        results = [FileResult(status='ok'), None, None]
        out = _aggregate_status(results)
        assert "1✓" in out


# ─────────────────────────────────────────────────────────────────────────
#  _len_or_zero
# ─────────────────────────────────────────────────────────────────────────

class TestLenOrZero:
    def test_list(self):
        assert _len_or_zero([1, 2, 3]) == 3

    def test_none(self):
        assert _len_or_zero(None) == 0

    def test_empty(self):
        assert _len_or_zero([]) == 0

    def test_unsized_falls_back_to_zero(self):
        class NoLen:
            pass
        assert _len_or_zero(NoLen()) == 0
