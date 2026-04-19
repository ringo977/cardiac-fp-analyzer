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
_BatchWorker = sp._BatchWorker
_BatchWorkerSignals = sp._BatchWorkerSignals
_BATCH_MAX_THREADS = sp._BATCH_MAX_THREADS


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


# ─────────────────────────────────────────────────────────────────────────
#  _BatchWorker / _BatchWorkerSignals (step 3b — background batch pool)
#
#  These tests exercise the worker in isolation: we monkeypatch
#  ``cardiac_fp_analyzer.analyze.analyze_single_file`` so the worker
#  never touches disk / numpy, and we invoke ``run()`` directly on the
#  test thread.  That's enough to verify that both success and failure
#  paths emit *exactly one* signal with the right ``FileResult`` shape
#  — which is the full contract the UI relies on.
#
#  We deliberately don't spin up a ``QThreadPool`` here: that would
#  require a ``QApplication`` + event loop, and the _on_analyze_group
#  orchestration (pool, progress dialog, cancel) is covered at a higher
#  level.  The worker-level contract is what regressions will hit first.
# ─────────────────────────────────────────────────────────────────────────

class TestBatchMaxThreads:
    def test_is_a_positive_int(self):
        # Pure sanity: must be ≥ 1 so ``min(idealThreadCount(), cap)``
        # never produces 0 threads on a weird host.
        assert isinstance(_BATCH_MAX_THREADS, int)
        assert _BATCH_MAX_THREADS >= 1


class _SignalCapture:
    """Tiny helper: connect to ``signals.done`` and record the args.

    Avoids pulling in pytest-qt (which would require a running Qt app
    loop) — the worker emits via a plain ``Signal`` and we connect a
    Python callable, which Qt dispatches synchronously for direct
    connections by default.
    """
    def __init__(self, signals):
        self.events: list[tuple[int, object]] = []
        signals.done.connect(self._on_done)

    def _on_done(self, idx, result):
        self.events.append((idx, result))


class TestBatchWorkerRun:
    def test_success_emits_ok_result(self, monkeypatch):
        # Fake pipeline result — matches the shape the worker reads.
        fake_result = {
            'file_info': {'analyzed_channel': 'EL2'},
            'beat_indices': list(range(14)),
            'beat_indices_fpd': list(range(10)),
        }
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: fake_result,
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        worker = _BatchWorker(
            file_index=3,
            abspath='/fake/chip.csv',
            channel='auto',
            config=None,
            signals=signals,
        )
        worker.run()

        assert len(cap.events) == 1
        idx, fr = cap.events[0]
        assert idx == 3
        assert fr.status == 'ok'
        assert fr.channel_analyzed == 'EL2'
        assert fr.n_total == 14
        assert fr.n_included == 10
        assert fr.error == ''

    def test_pipeline_returning_none_becomes_error(self, monkeypatch):
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: None,
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=0, abspath='/fake.csv',
            channel='auto', config=None, signals=signals,
        ).run()

        idx, fr = cap.events[0]
        assert idx == 0
        assert fr.status == 'error'
        assert 'vuota' in fr.error.lower()

    def test_exception_becomes_error_with_exc_type(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("bad signal")
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file', boom,
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=7, abspath='/fake.csv',
            channel='EL1', config=None, signals=signals,
        ).run()

        idx, fr = cap.events[0]
        assert idx == 7
        assert fr.status == 'error'
        # Error string should carry the exception class name and msg
        # so the tooltip gives the user something actionable.
        assert 'RuntimeError' in fr.error
        assert 'bad signal' in fr.error

    def test_fpd_fallback_when_only_beat_indices_present(self, monkeypatch):
        # Older schemas (pre-#73) don't populate beat_indices_fpd —
        # the worker must fall back to the bare beat_indices count
        # rather than report 0 included beats.
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: {
                'file_info': {'analyzed_channel': 'EL1'},
                'beat_indices': list(range(8)),
                # no beat_indices_fpd
            },
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=0, abspath='/fake.csv',
            channel='auto', config=None, signals=signals,
        ).run()

        _, fr = cap.events[0]
        assert fr.n_total == 8
        assert fr.n_included == 8   # fallback to total

    def test_emits_exactly_once(self, monkeypatch):
        # Regression guard: the UI counts completions — if a worker
        # ever emits 0 or 2+ signals for one file, the progress bar
        # desyncs and either hangs or finishes early.
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: {'file_info': {'analyzed_channel': 'EL1'},
                              'beat_indices': [1], 'beat_indices_fpd': [1]},
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=0, abspath='/fake.csv',
            channel='auto', config=None, signals=signals,
        ).run()

        assert len(cap.events) == 1
