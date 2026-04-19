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
from dataclasses import dataclass as _dc

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
_enumerate_csvs_under = sp._enumerate_csvs_under
_result_fingerprint = sp._result_fingerprint
_is_stale = sp._is_stale


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
            fingerprint='deadbeef0001',
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
        # Fingerprint must be stamped into the success result so
        # _rebuild_tree can detect staleness later.
        assert fr.fingerprint == 'deadbeef0001'

    def test_pipeline_returning_none_becomes_error(self, monkeypatch):
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: None,
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=0, abspath='/fake.csv',
            channel='auto', config=None,
            fingerprint='abc123', signals=signals,
        ).run()

        idx, fr = cap.events[0]
        assert idx == 0
        assert fr.status == 'error'
        assert 'vuota' in fr.error.lower()
        # Even the "None-pipeline → error" branch must carry the
        # fingerprint — otherwise re-analysing wouldn't clear the
        # stale ● badge on the next run.
        assert fr.fingerprint == 'abc123'

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
            channel='EL1', config=None,
            fingerprint='exc-fp-77', signals=signals,
        ).run()

        idx, fr = cap.events[0]
        assert idx == 7
        assert fr.status == 'error'
        # Error string should carry the exception class name and msg
        # so the tooltip gives the user something actionable.
        assert 'RuntimeError' in fr.error
        assert 'bad signal' in fr.error
        # Exception branch also stamps the fingerprint.
        assert fr.fingerprint == 'exc-fp-77'

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
            channel='auto', config=None,
            fingerprint='fp-fallback', signals=signals,
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
            channel='auto', config=None,
            fingerprint='once-fp', signals=signals,
        ).run()

        assert len(cap.events) == 1


# ─────────────────────────────────────────────────────────────────────────
#  _result_fingerprint — hash of (config, channel) that seeds staleness.
#
#  We build a small dataclass that *looks* like an AnalysisConfig (only
#  what matters is that ``asdict`` / ``__dict__`` works) so we can avoid
#  pulling the scientific core into the test.  Determinism is the whole
#  contract — we assert equal inputs hash equal, and a changed value
#  hashes differently.
# ─────────────────────────────────────────────────────────────────────────


@_dc
class _FakeConfig:
    threshold: float = 1.5
    window_ms: int = 80
    mode: str = "strict"


class TestResultFingerprint:
    def test_returns_stable_string(self):
        fp = _result_fingerprint(_FakeConfig(), "auto")
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_same_inputs_same_fingerprint(self):
        a = _result_fingerprint(_FakeConfig(), "auto")
        b = _result_fingerprint(_FakeConfig(), "auto")
        assert a == b

    def test_different_channel_differs(self):
        a = _result_fingerprint(_FakeConfig(), "auto")
        b = _result_fingerprint(_FakeConfig(), "EL1")
        assert a != b

    def test_different_config_value_differs(self):
        a = _result_fingerprint(_FakeConfig(threshold=1.5), "auto")
        b = _result_fingerprint(_FakeConfig(threshold=1.6), "auto")
        assert a != b

    def test_field_order_does_not_matter(self):
        # Rebuilding the same config via different keyword order must
        # still produce the same fingerprint — otherwise random dict
        # iteration order would spuriously mark every row stale.
        a = _result_fingerprint(
            _FakeConfig(threshold=1.5, window_ms=80, mode="strict"), "auto",
        )
        b = _result_fingerprint(
            _FakeConfig(mode="strict", window_ms=80, threshold=1.5), "auto",
        )
        assert a == b

    def test_handles_non_dataclass_object(self):
        # Defensive: some call sites may hand us a plain namespace
        # rather than an AnalysisConfig dataclass — we fall back to
        # ``__dict__`` and must still produce something stable.
        class Bag:
            def __init__(self):
                self.x = 1
                self.y = "two"
        a = _result_fingerprint(Bag(), "auto")
        b = _result_fingerprint(Bag(), "auto")
        assert a == b

    def test_handles_empty_channel(self):
        # Empty channel is a legitimate input (file-entry default).
        fp = _result_fingerprint(_FakeConfig(), "")
        assert isinstance(fp, str)
        assert len(fp) > 0


# ─────────────────────────────────────────────────────────────────────────
#  _is_stale — tolerance rules for the ● badge
# ─────────────────────────────────────────────────────────────────────────

class TestIsStale:
    def test_none_is_not_stale(self):
        # No result at all → no badge of any kind.  "Stale" is only
        # meaningful when there's a cached result to compare against.
        assert _is_stale(None, _FakeConfig(), "auto") is False

    def test_empty_fingerprint_is_not_stale(self):
        # Legacy / pre-3c cached entries have fingerprint=''.  We
        # tolerate them as "fresh" so loading a pre-3c study doesn't
        # instantly flag every row ●.
        r = FileResult(status='ok', fingerprint='')
        assert _is_stale(r, _FakeConfig(), "auto") is False

    def test_matching_fingerprint_is_not_stale(self):
        cfg = _FakeConfig()
        fp = _result_fingerprint(cfg, "auto")
        r = FileResult(status='ok', fingerprint=fp)
        assert _is_stale(r, cfg, "auto") is False

    def test_mismatched_fingerprint_is_stale(self):
        r = FileResult(status='ok', fingerprint='deadbeef0000')
        # The stored fingerprint is synthetic; any real re-compute will
        # differ → stale.
        assert _is_stale(r, _FakeConfig(), "auto") is True

    def test_config_change_flips_to_stale(self):
        cfg_old = _FakeConfig(threshold=1.5)
        fp_old = _result_fingerprint(cfg_old, "auto")
        r = FileResult(status='ok', fingerprint=fp_old)

        # Simulate the user tweaking the threshold in the settings
        # dialog — the previously-cached result should now flag as
        # stale against the *new* config.
        cfg_new = _FakeConfig(threshold=1.9)
        assert _is_stale(r, cfg_new, "auto") is True

    def test_channel_change_flips_to_stale(self):
        cfg = _FakeConfig()
        fp_auto = _result_fingerprint(cfg, "auto")
        r = FileResult(status='ok', fingerprint=fp_auto)
        # Per-file channel override changed → different analysed signal,
        # so the old result is now stale even though the group config
        # hasn't moved.
        assert _is_stale(r, cfg, "EL1") is True

    def test_error_result_also_stale_aware(self):
        # A stale error is still a stale result: the user edited the
        # config *and* we haven't re-run, so "it used to fail under the
        # old config" is not a statement about the new one.
        r = FileResult(status='error', error='x', fingerprint='old-fp')
        assert _is_stale(r, _FakeConfig(), "auto") is True


# ─────────────────────────────────────────────────────────────────────────
#  _badge_for / _tooltip_for / _aggregate_status — stale variants
# ─────────────────────────────────────────────────────────────────────────

class TestBadgeForStale:
    def test_stale_ok_is_dot(self):
        r = FileResult(status='ok', fingerprint='old')
        assert _badge_for(r, stale=True) == "●"

    def test_stale_error_is_also_dot(self):
        # Stale takes precedence over error — the user should first
        # know "inputs changed" before reading the old error message.
        r = FileResult(status='error', error='x', fingerprint='old')
        assert _badge_for(r, stale=True) == "●"

    def test_stale_on_none_still_empty(self):
        # Defensive: if caller passes stale=True for a row with no
        # cached result, we still show nothing (the badge has no
        # semantic meaning without a backing result).
        assert _badge_for(None, stale=True) == ""

    def test_fresh_ok_unchanged(self):
        # Back-compat with pre-3c callers (default stale=False).
        assert _badge_for(FileResult(status='ok')) == "✓"


class TestTooltipForStale:
    def test_stale_ok_prefixed_with_reason(self):
        r = FileResult(
            status='ok', channel_analyzed='EL1',
            n_included=14, n_total=14, fingerprint='old',
        )
        t = _tooltip_for(r, stale=True)
        assert "non aggiornato" in t.lower() or "config cambiata" in t
        # Underlying ok detail is still there.
        assert "EL1" in t
        assert "14 battiti" in t

    def test_stale_error_prefixed_with_reason(self):
        r = FileResult(status='error', error='boom', fingerprint='old')
        t = _tooltip_for(r, stale=True)
        assert "non aggiornato" in t.lower() or "config cambiata" in t
        assert "boom" in t

    def test_fresh_unchanged(self):
        r = FileResult(status='ok', channel_analyzed='EL1', n_included=5, n_total=5)
        t = _tooltip_for(r)
        assert "non aggiornato" not in t.lower()


class TestAggregateStatusStale:
    def test_all_fresh_ok_still_compact(self):
        results = [FileResult(status='ok') for _ in range(3)]
        stale = [False, False, False]
        assert _aggregate_status(results, stale) == "✓ 3/3"

    def test_single_stale_breaks_compact_form(self):
        # Even if everything succeeded, one stale result should force
        # the mixed form so the user sees the ● in the summary.
        results = [FileResult(status='ok') for _ in range(3)]
        stale = [False, True, False]
        out = _aggregate_status(results, stale)
        assert "2✓" in out
        assert "1●" in out
        assert "/" not in out   # not the compact "✓ N/N" form

    def test_mixed_ok_stale_error(self):
        results = [
            FileResult(status='ok'),
            FileResult(status='ok', fingerprint='old'),
            FileResult(status='error', error='x'),
        ]
        stale = [False, True, False]
        out = _aggregate_status(results, stale)
        assert "1✓" in out
        assert "1●" in out
        assert "1✗" in out

    def test_stale_error_counts_as_error_not_dot(self):
        # Rationale: errors are more actionable than staleness — the
        # user needs to fix the error regardless of whether inputs
        # changed.  Double-counting would inflate the summary.
        results = [FileResult(status='error', error='x')]
        stale = [True]
        out = _aggregate_status(results, stale)
        assert "1✗" in out
        assert "●" not in out

    def test_stale_argument_is_optional(self):
        # Pre-3c callers pass just ``results`` — behaviour must match
        # the old two-symbol summary.
        results = [FileResult(status='ok'), FileResult(status='error')]
        out_legacy = _aggregate_status(results)
        out_no_stale = _aggregate_status(results, [False, False])
        assert out_legacy == out_no_stale


# ─────────────────────────────────────────────────────────────────────────
#  _enumerate_csvs_under — recursive CSV scanner for "Aggiungi cartella"
#
#  Uses ``tmp_path`` so the filesystem layout is deterministic per test.
#  The action backing this helper is UX-critical (it replaces the
#  multi-select dialog workflow for cross-subfolder selection), so any
#  regression in "hidden file skip" or "ordering" would surface as
#  confusing tree order or .DS_Store pollution.
# ─────────────────────────────────────────────────────────────────────────

class TestEnumerateCsvsUnder:
    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert _enumerate_csvs_under(tmp_path) == []

    def test_finds_top_level_csvs(self, tmp_path):
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "b.csv").write_text("x")
        result = _enumerate_csvs_under(tmp_path)
        assert len(result) == 2
        assert {p.name for p in result} == {"a.csv", "b.csv"}

    def test_finds_nested_csvs_recursively(self, tmp_path):
        # Real-world dose-response layout: baseline/ dose1/ dose2/
        for sub in ("baseline", "dose1", "dose2"):
            d = tmp_path / sub
            d.mkdir()
            (d / f"chip_{sub}.csv").write_text("x")
        result = _enumerate_csvs_under(tmp_path)
        assert len(result) == 3
        # Sorted → baseline/ comes before dose1/ comes before dose2/
        assert [p.parent.name for p in result] == ["baseline", "dose1", "dose2"]

    def test_ignores_non_csv_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "archive.zip").write_text("x")
        result = _enumerate_csvs_under(tmp_path)
        assert len(result) == 1
        assert result[0].name == "data.csv"

    def test_skips_hidden_files_at_top_level(self, tmp_path):
        (tmp_path / "visible.csv").write_text("x")
        (tmp_path / ".hidden.csv").write_text("x")
        result = _enumerate_csvs_under(tmp_path)
        assert [p.name for p in result] == ["visible.csv"]

    def test_skips_csvs_inside_hidden_directory(self, tmp_path):
        # ``.cache/baseline.csv`` is a very plausible accident on macOS.
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        (cache_dir / "baseline.csv").write_text("x")
        (tmp_path / "keeper.csv").write_text("x")
        result = _enumerate_csvs_under(tmp_path)
        assert [p.name for p in result] == ["keeper.csv"]

    def test_is_sorted_for_deterministic_tree_order(self, tmp_path):
        # Insert in reverse order to prove we sort (not filesystem mtime)
        for name in ("zebra.csv", "alpha.csv", "middle.csv"):
            (tmp_path / name).write_text("x")
        result = _enumerate_csvs_under(tmp_path)
        assert [p.name for p in result] == [
            "alpha.csv", "middle.csv", "zebra.csv",
        ]

    def test_missing_folder_returns_empty_not_raise(self, tmp_path):
        # Defensive: UI shouldn't crash if the folder vanished between
        # dialog pick and scan.
        result = _enumerate_csvs_under(tmp_path / "does-not-exist")
        assert result == []
