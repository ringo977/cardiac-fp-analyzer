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
import math
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
# Step 4 — scalar metrics + per-group aggregate.
_to_float_or_nan = sp._to_float_or_nan
_extract_summary_metrics = sp._extract_summary_metrics
_mean_sd_n = sp._mean_sd_n
_aggregate_group_metrics = sp._aggregate_group_metrics
_fmt_mean_sd = sp._fmt_mean_sd
_format_group_metrics_line = sp._format_group_metrics_line
_format_file_metrics = sp._format_file_metrics
# Step 5 — dose-response helpers.
DoseResponsePoint = sp.DoseResponsePoint
_DOSE_RESPONSE_METRICS = sp._DOSE_RESPONSE_METRICS
_metric_meta = sp._metric_meta
_collect_dose_response_points = sp._collect_dose_response_points
_assemble_dose_response_series = sp._assemble_dose_response_series
_can_plot_log_x = sp._can_plot_log_x
_drug_colour = sp._drug_colour
# Step 6 — CDISC export helpers.
_format_concentration_from_dose_uM = sp._format_concentration_from_dose_uM
_enrich_file_info_for_export = sp._enrich_file_info_for_export
_collect_export_inputs = sp._collect_export_inputs
_suggest_study_id = sp._suggest_study_id
# Task #91 — study-wide batch helper.
_collect_batch_inputs = sp._collect_batch_inputs


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


# ═════════════════════════════════════════════════════════════════════════
#  Step 4 — per-group scalar metrics (FPD / FPDc / BPM / STV)
#
#  Tests the metric-extraction → aggregation → formatting chain that
#  turns pipeline ``summary`` dicts into the "FPDc 385±12 ms · n=4/5"
#  string the user reads on the group row.  Pure functions, no Qt.
# ═════════════════════════════════════════════════════════════════════════

def _nan():
    return float('nan')


class TestFileResultMetrics:
    def test_new_metric_fields_default_to_nan(self):
        # Back-compat: constructing a FileResult the old way (no
        # metric kwargs) must still work and leave metrics as NaN so
        # aggregate/format code can skip them uniformly.
        r = FileResult(status='ok')
        assert math.isnan(r.fpd_ms)
        assert math.isnan(r.fpdc_ms)
        assert math.isnan(r.bpm)
        assert math.isnan(r.stv_ms)

    def test_metrics_are_stored(self):
        r = FileResult(
            status='ok',
            fpd_ms=385.0, fpdc_ms=410.0, bpm=58.0, stv_ms=2.1,
        )
        assert r.fpd_ms == 385.0
        assert r.fpdc_ms == 410.0
        assert r.bpm == 58.0
        assert r.stv_ms == 2.1


class TestToFloatOrNan:
    def test_plain_float_round_trips(self):
        assert _to_float_or_nan(1.5) == 1.5

    def test_int_becomes_float(self):
        out = _to_float_or_nan(3)
        assert isinstance(out, float)
        assert out == 3.0

    def test_none_is_nan(self):
        assert math.isnan(_to_float_or_nan(None))

    def test_unparseable_string_is_nan(self):
        assert math.isnan(_to_float_or_nan("not a number"))

    def test_numeric_string_is_parsed(self):
        assert _to_float_or_nan("385.5") == 385.5

    def test_nan_stays_nan(self):
        assert math.isnan(_to_float_or_nan(float('nan')))


class TestExtractSummaryMetrics:
    def test_full_summary_preferred_median(self):
        summary = {
            'fpd_ms_median': 385.0, 'fpd_ms_mean': 400.0,
            'fpdc_ms_median': 410.0, 'fpdc_ms_mean': 420.0,
            'bpm_mean': 58.0,
            'stv_ms': 2.1,
        }
        fpd, fpdc, bpm, stv = _extract_summary_metrics(summary)
        # Median preferred — matches scientific convention for
        # heavy-tailed FPD distributions.
        assert fpd == 385.0
        assert fpdc == 410.0
        assert bpm == 58.0
        assert stv == 2.1

    def test_mean_fallback_when_median_missing(self):
        # Older result schemas may have only ``*_mean`` — we should
        # still produce a number rather than NaN.
        summary = {'fpd_ms_mean': 400.0, 'fpdc_ms_mean': 420.0,
                   'bpm_mean': 60.0, 'stv_ms': 1.9}
        fpd, fpdc, bpm, stv = _extract_summary_metrics(summary)
        assert fpd == 400.0
        assert fpdc == 420.0

    def test_empty_summary_is_all_nan(self):
        fpd, fpdc, bpm, stv = _extract_summary_metrics({})
        assert all(math.isnan(v) for v in (fpd, fpdc, bpm, stv))

    def test_none_summary_is_all_nan(self):
        # Defensive — the pipeline returning ``result['summary'] = None``
        # must not crash the worker.
        fpd, fpdc, bpm, stv = _extract_summary_metrics(None)
        assert all(math.isnan(v) for v in (fpd, fpdc, bpm, stv))

    def test_partial_summary_only_fills_known_fields(self):
        # Degenerate signal: beats detected but no valid FPD → rate
        # is measurable, FPD is not.  Aggregate must keep the rate.
        summary = {'bpm_mean': 72.0, 'stv_ms': 1.5}
        fpd, fpdc, bpm, stv = _extract_summary_metrics(summary)
        assert math.isnan(fpd)
        assert math.isnan(fpdc)
        assert bpm == 72.0
        assert stv == 1.5


class TestMeanSdN:
    def test_empty_list_is_nan(self):
        mean, sd, n = _mean_sd_n([])
        assert math.isnan(mean)
        assert math.isnan(sd)
        assert n == 0

    def test_all_nan_list_is_nan(self):
        mean, sd, n = _mean_sd_n([_nan(), _nan(), _nan()])
        assert math.isnan(mean)
        assert math.isnan(sd)
        assert n == 0

    def test_single_value_gives_zero_sd(self):
        # Can't compute SD from 1 sample; "0 observed spread" is the
        # safe reading that doesn't mis-sell precision.
        mean, sd, n = _mean_sd_n([42.0])
        assert mean == 42.0
        assert sd == 0.0
        assert n == 1

    def test_two_values_uses_sample_sd(self):
        # ddof=1 → sample SD: for [10, 20], mean=15, sd = sqrt(50)
        # ≈ 7.07.  That's what scientists see as "mean ± SD".
        mean, sd, n = _mean_sd_n([10.0, 20.0])
        assert mean == 15.0
        assert sd == pytest.approx(math.sqrt(50), rel=1e-9)
        assert n == 2

    def test_nan_entries_are_dropped(self):
        # Files with NaN for a given metric simply don't contribute,
        # but n reflects the finite count actually used — not the
        # input length.  Prevents NaN-pollution of the aggregate.
        mean, sd, n = _mean_sd_n([10.0, _nan(), 20.0, _nan()])
        assert mean == 15.0
        assert n == 2


class TestAggregateGroupMetrics:
    def _fresh_ok(self, **kw):
        # Convenience factory — always status='ok', fingerprint is
        # irrelevant to the aggregator (it uses the parallel stale[]
        # flags instead).
        return FileResult(status='ok', **kw)

    def test_empty_group_returns_zero_n(self):
        agg = _aggregate_group_metrics([])
        assert agg['n'] == 0
        assert agg['n_total'] == 0

    def test_all_none_results_returns_zero_n(self):
        agg = _aggregate_group_metrics([None, None, None])
        assert agg['n'] == 0
        assert agg['n_total'] == 3

    def test_fresh_ok_feeds_aggregate(self):
        results = [
            self._fresh_ok(fpdc_ms=400.0, bpm=60.0, stv_ms=2.0),
            self._fresh_ok(fpdc_ms=420.0, bpm=62.0, stv_ms=2.2),
            self._fresh_ok(fpdc_ms=410.0, bpm=61.0, stv_ms=2.1),
        ]
        agg = _aggregate_group_metrics(results, [False, False, False])
        assert agg['n'] == 3
        assert agg['n_total'] == 3
        fpdc_mean, fpdc_sd, fpdc_n = agg['fpdc_ms']
        assert fpdc_n == 3
        assert fpdc_mean == pytest.approx(410.0)
        assert fpdc_sd > 0

    def test_stale_results_are_excluded(self):
        # Rationale: showing a mean over stale numbers would be
        # misleading after the user edited the config.  Once they
        # re-run, the aggregate repopulates.
        results = [
            self._fresh_ok(fpdc_ms=400.0),
            self._fresh_ok(fpdc_ms=420.0),
            self._fresh_ok(fpdc_ms=9999.0),   # stale → excluded
        ]
        agg = _aggregate_group_metrics(results, [False, False, True])
        assert agg['n'] == 2
        fpdc_mean, _, fpdc_n = agg['fpdc_ms']
        assert fpdc_n == 2
        assert fpdc_mean == pytest.approx(410.0)

    def test_error_results_are_excluded(self):
        results = [
            self._fresh_ok(fpdc_ms=400.0),
            FileResult(status='error', error='boom'),
            self._fresh_ok(fpdc_ms=420.0),
        ]
        agg = _aggregate_group_metrics(results, [False, False, False])
        assert agg['n'] == 2

    def test_stale_absent_defaults_to_all_fresh(self):
        # Back-compat: pre-3c callers pass just results; behaviour
        # must treat every row as fresh.
        results = [self._fresh_ok(fpdc_ms=400.0) for _ in range(2)]
        agg = _aggregate_group_metrics(results)
        assert agg['n'] == 2

    def test_per_metric_n_finite_reflects_nan_drops(self):
        # File 2 is fresh-ok but has NaN FPDc (no valid repol on that
        # signal) — it contributes to BPM but not to FPDc.
        results = [
            self._fresh_ok(fpdc_ms=400.0, bpm=60.0),
            self._fresh_ok(fpdc_ms=_nan(), bpm=72.0),
            self._fresh_ok(fpdc_ms=420.0, bpm=61.0),
        ]
        agg = _aggregate_group_metrics(results, [False, False, False])
        _, _, fpdc_n = agg['fpdc_ms']
        _, _, bpm_n = agg['bpm']
        assert fpdc_n == 2
        assert bpm_n == 3


class TestFmtMeanSd:
    def test_n_zero_returns_empty(self):
        assert _fmt_mean_sd((_nan(), _nan(), 0), unit='ms') == ""

    def test_nan_mean_returns_empty(self):
        assert _fmt_mean_sd((_nan(), 0.0, 1), unit='ms') == ""

    def test_n_one_shows_count(self):
        # n=1 is flagged explicitly so the user doesn't read "±0"
        # as "the group is perfectly consistent".
        out = _fmt_mean_sd((385.0, 0.0, 1), unit='ms', decimals=0)
        assert "385" in out
        assert "ms" in out
        assert "n=1" in out
        assert "±" not in out

    def test_n_two_uses_mean_pm_sd(self):
        out = _fmt_mean_sd((385.0, 12.0, 2), unit='ms', decimals=0)
        assert out == "385 ± 12 ms"

    def test_decimals_are_respected(self):
        out = _fmt_mean_sd((2.15, 0.33, 4), unit='ms', decimals=1)
        assert out == "2.1 ± 0.3 ms"

    def test_unit_can_be_omitted(self):
        out = _fmt_mean_sd((58.0, 3.0, 4), decimals=0)
        assert out == "58 ± 3"


class TestFormatGroupMetricsLine:
    def _agg_with(self, **metrics):
        # Build a plausible agg dict for the formatter; defaults are
        # all-NaN so a test can supply only what it needs.
        base = {
            'n': 0, 'n_total': 0,
            'fpd_ms': (_nan(), _nan(), 0),
            'fpdc_ms': (_nan(), _nan(), 0),
            'bpm': (_nan(), _nan(), 0),
            'stv_ms': (_nan(), _nan(), 0),
        }
        base.update(metrics)
        return base

    def test_empty_aggregate_returns_empty(self):
        assert _format_group_metrics_line(self._agg_with()) == ""

    def test_n_zero_returns_empty_even_if_metrics_present(self):
        # Defensive: if somehow n=0 but metric tuples leak a value,
        # we still return empty so the group row stays clean.
        agg = self._agg_with(n=0, n_total=3, fpdc_ms=(400.0, 10.0, 3))
        assert _format_group_metrics_line(agg) == ""

    def test_full_metrics_produces_readable_line(self):
        agg = self._agg_with(
            n=4, n_total=5,
            fpdc_ms=(410.0, 12.0, 4),
            bpm=(58.0, 3.0, 4),
            stv_ms=(2.1, 0.3, 4),
        )
        out = _format_group_metrics_line(agg)
        assert "FPDc 410 ± 12 ms" in out
        assert "58 ± 3 BPM" in out
        assert "STV 2.1 ± 0.3 ms" in out
        assert "n=4/5" in out

    def test_fpd_falls_back_when_fpdc_missing(self):
        # Tail case: no rate correction available (e.g. QC rejected
        # RR outliers too aggressively) — show raw FPD rather than
        # nothing.
        agg = self._agg_with(
            n=3, n_total=3,
            fpdc_ms=(_nan(), _nan(), 0),
            fpd_ms=(400.0, 10.0, 3),
        )
        out = _format_group_metrics_line(agg)
        assert "FPD 400 ± 10 ms" in out
        assert "FPDc" not in out

    def test_fpd_suppressed_when_fpdc_present(self):
        # Anti-clutter: when FPDc is available, showing raw FPD next
        # to it is just two near-duplicate numbers.
        agg = self._agg_with(
            n=3, n_total=3,
            fpdc_ms=(410.0, 8.0, 3),
            fpd_ms=(400.0, 10.0, 3),
        )
        out = _format_group_metrics_line(agg)
        assert "FPDc 410" in out
        assert "FPD 400" not in out


class TestFormatFileMetrics:
    def test_none_is_empty(self):
        assert _format_file_metrics(None) == ""

    def test_error_is_empty(self):
        # Error rows show the error message, not metrics.
        assert _format_file_metrics(FileResult(status='error', error='x')) == ""

    def test_all_nan_is_empty(self):
        assert _format_file_metrics(FileResult(status='ok')) == ""

    def test_full_metrics(self):
        r = FileResult(
            status='ok', fpdc_ms=410.0, bpm=58.0, stv_ms=2.1,
        )
        out = _format_file_metrics(r)
        assert "FPDc: 410 ms" in out
        assert "58 BPM" in out
        assert "STV: 2.1 ms" in out

    def test_fpd_fallback_single_file(self):
        r = FileResult(status='ok', fpd_ms=400.0, bpm=60.0)
        out = _format_file_metrics(r)
        assert "FPD: 400 ms" in out
        assert "FPDc" not in out


class TestTooltipForWithMetrics:
    def test_ok_tooltip_includes_metrics(self):
        # Verify step 4 actually made it into the tooltip chain.
        r = FileResult(
            status='ok', channel_analyzed='EL1',
            n_included=14, n_total=14,
            fpdc_ms=410.0, bpm=58.0,
        )
        tip = _tooltip_for(r)
        assert "EL1" in tip
        assert "14 battiti" in tip
        assert "FPDc: 410 ms" in tip
        assert "58 BPM" in tip

    def test_ok_tooltip_without_metrics_still_works(self):
        # Back-compat: a legacy cached result with NaN metrics
        # renders exactly like pre-step-4.
        r = FileResult(
            status='ok', channel_analyzed='EL1',
            n_included=14, n_total=14,
        )
        tip = _tooltip_for(r)
        assert "EL1" in tip
        assert "14 battiti" in tip
        assert "FPDc" not in tip


class TestBatchWorkerExtractsMetrics:
    """Round-trip: _BatchWorker.run → FileResult.fpdc_ms etc."""

    def test_worker_populates_metrics_from_summary(self, monkeypatch):
        fake_result = {
            'file_info': {'analyzed_channel': 'EL1'},
            'beat_indices': list(range(14)),
            'beat_indices_fpd': list(range(12)),
            'summary': {
                'fpd_ms_median': 385.0,
                'fpdc_ms_median': 410.0,
                'bpm_mean': 58.0,
                'stv_ms': 2.1,
            },
        }
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: fake_result,
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=0, abspath='/fake.csv',
            channel='auto', config=None,
            fingerprint='metrics-fp', signals=signals,
        ).run()

        _, fr = cap.events[0]
        assert fr.fpd_ms == 385.0
        assert fr.fpdc_ms == 410.0
        assert fr.bpm == 58.0
        assert fr.stv_ms == 2.1

    def test_worker_tolerates_missing_summary(self, monkeypatch):
        # Pipeline that returns beats but no summary dict (degenerate
        # / early-exit branch) must still produce an ok result with
        # NaN metrics — not crash.
        monkeypatch.setattr(
            'cardiac_fp_analyzer.analyze.analyze_single_file',
            lambda *a, **kw: {
                'file_info': {'analyzed_channel': 'EL1'},
                'beat_indices': [1, 2, 3],
                'beat_indices_fpd': [1, 2, 3],
                # no 'summary' key
            },
        )
        signals = _BatchWorkerSignals()
        cap = _SignalCapture(signals)

        _BatchWorker(
            file_index=0, abspath='/fake.csv',
            channel='auto', config=None,
            fingerprint='no-summary-fp', signals=signals,
        ).run()

        _, fr = cap.events[0]
        assert fr.status == 'ok'
        assert math.isnan(fr.fpd_ms)
        assert math.isnan(fr.fpdc_ms)
        assert math.isnan(fr.bpm)
        assert math.isnan(fr.stv_ms)


# ═════════════════════════════════════════════════════════════════════════
#  Step 5 — dose-response helpers.
#
#  The plot itself requires a QApplication; these tests only exercise the
#  pure data-preparation pipeline (collection + series assembly +
#  log-scale guard + colour palette).  The dialog render is validated
#  on Marco's laptop because pyqtgraph offscreen still needs GL libs
#  that aren't guaranteed in CI.
# ═════════════════════════════════════════════════════════════════════════

# Minimal fake Group/Study — we don't import the real dataclasses here
# to keep these tests independent of the cardiac_fp_analyzer.study
# module schema (and faster: no AnalysisConfig instantiation).


@_dc
class _FakeFile:
    csv_relpath: str
    channel: str = 'auto'


@_dc
class _FakeGroup:
    name: str
    drug: str = ''
    dose_uM: float | None = None
    files: list = None   # type: ignore[assignment]
    config: object = None

    def __post_init__(self):
        if self.files is None:
            self.files = []


@_dc
class _FakeStudy:
    name: str = 'test-study'
    groups: list = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.groups is None:
            self.groups = []


def _ok(**metrics) -> FileResult:
    """Shortcut: a fresh-ok FileResult with the requested metrics.

    Empty fingerprint = "legacy fresh" per :func:`_is_stale` contract,
    so tests can use a plain ``object()`` as the group's config
    without computing the matching fingerprint.  Tests that want to
    exercise staleness pass ``fingerprint=...`` explicitly.
    """
    defaults = {'status': 'ok', 'n_included': 10, 'n_total': 10,
                'fingerprint': '', 'channel_analyzed': 'EL1'}
    defaults.update(metrics)
    return FileResult(**defaults)


def _err(msg: str = 'boom') -> FileResult:
    return FileResult(status='error', error=msg)


class TestDoseResponsePoint:
    def test_dataclass_holds_all_fields(self):
        p = DoseResponsePoint(
            group_name='baseline', drug='dof',
            dose_uM=0.01, mean=385.0, sd=5.5, n=3,
        )
        assert p.group_name == 'baseline'
        assert p.drug == 'dof'
        assert p.dose_uM == 0.01
        assert p.mean == 385.0
        assert p.sd == 5.5
        assert p.n == 3

    def test_dose_can_be_none_for_baseline(self):
        p = DoseResponsePoint(
            group_name='baseline', drug='',
            dose_uM=None, mean=400.0, sd=0.0, n=1,
        )
        assert p.dose_uM is None


class TestMetricMeta:
    def test_known_keys_resolve(self):
        # Exercise every entry of the whitelist so a future
        # reorder/rename of _DOSE_RESPONSE_METRICS doesn't silently
        # break the dialog.
        for key, label, unit, decimals in _DOSE_RESPONSE_METRICS:
            lbl, un, dec = _metric_meta(key)
            assert lbl == label
            assert un == unit
            assert dec == decimals

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            _metric_meta('not-a-metric')

    def test_whitelist_starts_with_fpdc(self):
        # FPDc is the clinically-primary endpoint — it must be the
        # default (first) entry so the dialog opens on the right
        # metric without extra clicks.
        assert _DOSE_RESPONSE_METRICS[0][0] == 'fpdc_ms'


class TestCollectDoseResponsePoints:
    def test_empty_study_returns_empty_list(self):
        study = _FakeStudy(groups=[])
        assert _collect_dose_response_points(study, {}) == []

    def test_unknown_metric_raises_keyerror(self):
        study = _FakeStudy(groups=[])
        with pytest.raises(KeyError):
            _collect_dose_response_points(study, {}, metric='not-a-metric')

    def test_single_group_single_file_makes_one_point(self):
        g = _FakeGroup(
            name='dose-0.01', drug='dof', dose_uM=0.01,
            files=[_FakeFile(csv_relpath='f1.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {('dose-0.01', 'f1.csv'): _ok(fpdc_ms=410.0)}
        pts = _collect_dose_response_points(study, cache)
        assert len(pts) == 1
        assert pts[0].drug == 'dof'
        assert pts[0].dose_uM == 0.01
        assert pts[0].mean == 410.0
        # Single-file group → sd is 0.0 (by the mean_sd_n contract).
        assert pts[0].sd == 0.0
        assert pts[0].n == 1

    def test_multi_file_group_averages(self):
        g = _FakeGroup(
            name='dose-0.1', drug='dof', dose_uM=0.1,
            files=[_FakeFile('a.csv'), _FakeFile('b.csv'),
                   _FakeFile('c.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {
            ('dose-0.1', 'a.csv'): _ok(fpdc_ms=410.0),
            ('dose-0.1', 'b.csv'): _ok(fpdc_ms=420.0),
            ('dose-0.1', 'c.csv'): _ok(fpdc_ms=430.0),
        }
        pts = _collect_dose_response_points(study, cache)
        assert len(pts) == 1
        assert pts[0].mean == 420.0   # (410+420+430)/3
        assert pts[0].n == 3

    def test_group_with_no_fresh_ok_is_dropped(self):
        # A group where every file errored should not produce a point —
        # the plot should simply skip it rather than showing a gap.
        g = _FakeGroup(
            name='dose-0.1', drug='dof', dose_uM=0.1,
            files=[_FakeFile('a.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {('dose-0.1', 'a.csv'): _err("pipeline crash")}
        assert _collect_dose_response_points(study, cache) == []

    def test_group_with_no_cached_results_is_dropped(self):
        g = _FakeGroup(
            name='dose-0.1', drug='dof', dose_uM=0.1,
            files=[_FakeFile('a.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        assert _collect_dose_response_points(study, {}) == []

    def test_baseline_group_is_included(self):
        g = _FakeGroup(
            name='baseline', drug='', dose_uM=None,
            files=[_FakeFile('ctrl.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {('baseline', 'ctrl.csv'): _ok(fpdc_ms=400.0)}
        pts = _collect_dose_response_points(study, cache)
        assert len(pts) == 1
        assert pts[0].dose_uM is None

    def test_different_metric_can_be_selected(self):
        g = _FakeGroup(
            name='dose-0.1', drug='dof', dose_uM=0.1,
            files=[_FakeFile('a.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {('dose-0.1', 'a.csv'): _ok(fpdc_ms=410.0, bpm=58.0,
                                              stv_ms=2.1)}
        pts_bpm = _collect_dose_response_points(
            study, cache, metric='bpm',
        )
        assert pts_bpm[0].mean == 58.0
        pts_stv = _collect_dose_response_points(
            study, cache, metric='stv_ms',
        )
        assert pts_stv[0].mean == 2.1

    def test_error_file_doesnt_poison_aggregate(self):
        # One OK + one error file → aggregate should be the OK value
        # alone, not NaN, not averaged with NaN.
        g = _FakeGroup(
            name='dose-0.1', drug='dof', dose_uM=0.1,
            files=[_FakeFile('a.csv'), _FakeFile('b.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {
            ('dose-0.1', 'a.csv'): _ok(fpdc_ms=410.0),
            ('dose-0.1', 'b.csv'): _err(),
        }
        pts = _collect_dose_response_points(study, cache)
        assert len(pts) == 1
        assert pts[0].mean == 410.0
        assert pts[0].n == 1

    def test_empty_group_produces_no_point(self):
        # Group defined but zero files — don't even attempt to plot.
        g = _FakeGroup(name='empty', drug='dof', dose_uM=1.0,
                      files=[], config=object())
        study = _FakeStudy(groups=[g])
        assert _collect_dose_response_points(study, {}) == []

    def test_stale_is_not_part_of_point(self):
        # A result with a fingerprint that doesn't match the current
        # (config, channel) pair is stale and must be excluded — the
        # aggregate in the plot has to match the tree-row aggregate.
        @_dc
        class _Cfg:
            v: int = 1

        g = _FakeGroup(
            name='dose-0.1', drug='dof', dose_uM=0.1,
            files=[_FakeFile('a.csv')],
            config=_Cfg(v=2),   # different from the stamped fingerprint
        )
        study = _FakeStudy(groups=[g])
        # Stamp a fingerprint matching a DIFFERENT config, so the
        # cached result is stale w.r.t. the group's current config.
        old_fp = _result_fingerprint(_Cfg(v=1), 'auto')
        cache = {('dose-0.1', 'a.csv'): _ok(
            fpdc_ms=410.0, fingerprint=old_fp,
        )}
        pts = _collect_dose_response_points(study, cache)
        # Stale result excluded → no fresh-ok → no point.
        assert pts == []


class TestAssembleDoseResponseSeries:
    def test_empty_input_returns_empty_structures(self):
        out = _assemble_dose_response_series([])
        assert out['baselines'] == []
        assert out['doses'] == {}

    def test_single_dose_point_single_drug(self):
        p = DoseResponsePoint(
            group_name='g1', drug='dof', dose_uM=0.01,
            mean=410.0, sd=1.0, n=1,
        )
        out = _assemble_dose_response_series([p])
        assert out['baselines'] == []
        assert list(out['doses'].keys()) == ['dof']
        assert out['doses']['dof'] == [p]

    def test_doses_sorted_ascending_per_drug(self):
        # Input given in a jumbled order; output must be sorted.
        points = [
            DoseResponsePoint('g3', 'dof', 1.0, 450.0, 0.0, 1),
            DoseResponsePoint('g1', 'dof', 0.01, 400.0, 0.0, 1),
            DoseResponsePoint('g2', 'dof', 0.1, 420.0, 0.0, 1),
        ]
        out = _assemble_dose_response_series(points)
        dof = out['doses']['dof']
        assert [p.dose_uM for p in dof] == [0.01, 0.1, 1.0]

    def test_baselines_and_doses_split(self):
        # One baseline + two dose points → splits cleanly, baseline
        # doesn't end up in the dose curve.
        points = [
            DoseResponsePoint('bl', '', None, 400.0, 5.0, 3),
            DoseResponsePoint('d1', 'dof', 0.01, 410.0, 0.0, 1),
            DoseResponsePoint('d2', 'dof', 0.1, 425.0, 0.0, 1),
        ]
        out = _assemble_dose_response_series(points)
        assert len(out['baselines']) == 1
        assert out['baselines'][0].group_name == 'bl'
        assert [p.dose_uM for p in out['doses']['dof']] == [0.01, 0.1]

    def test_multi_drug_gets_separate_curves(self):
        points = [
            DoseResponsePoint('a', 'dof', 0.01, 410.0, 0.0, 1),
            DoseResponsePoint('b', 'mox', 0.1, 380.0, 0.0, 1),
            DoseResponsePoint('c', 'dof', 0.1, 430.0, 0.0, 1),
            DoseResponsePoint('d', 'mox', 1.0, 360.0, 0.0, 1),
        ]
        out = _assemble_dose_response_series(points)
        assert set(out['doses'].keys()) == {'dof', 'mox'}
        assert [p.dose_uM for p in out['doses']['dof']] == [0.01, 0.1]
        assert [p.dose_uM for p in out['doses']['mox']] == [0.1, 1.0]

    def test_same_dose_tie_break_by_group_name(self):
        # Two replicate groups at the same dose → deterministic sort
        # (alpha by group_name) so the plot doesn't render a different
        # polyline each redraw.
        points = [
            DoseResponsePoint('zz', 'dof', 0.1, 410.0, 0.0, 1),
            DoseResponsePoint('aa', 'dof', 0.1, 420.0, 0.0, 1),
        ]
        out = _assemble_dose_response_series(points)
        dof = out['doses']['dof']
        assert [p.group_name for p in dof] == ['aa', 'zz']

    def test_baseline_insertion_order_preserved(self):
        # When there are multiple baselines (multi-drug study with
        # per-drug vehicle controls), legend order follows the
        # study's groups[] order — not alpha.  Verified by inserting
        # them in a non-alpha order and checking the output matches.
        points = [
            DoseResponsePoint('bl-mox', 'mox', None, 390.0, 0.0, 1),
            DoseResponsePoint('bl-dof', 'dof', None, 395.0, 0.0, 1),
        ]
        out = _assemble_dose_response_series(points)
        assert [p.group_name for p in out['baselines']] == [
            'bl-mox', 'bl-dof',
        ]

    def test_drug_without_name_keeps_empty_string_key(self):
        # Empty drug string is a legal key — e.g. a single-drug study
        # where the user didn't bother filling in the name.  The
        # legend will show "(senza nome)", handled in the dialog.
        points = [
            DoseResponsePoint('g1', '', 0.1, 410.0, 0.0, 1),
            DoseResponsePoint('g2', '', 1.0, 420.0, 0.0, 1),
        ]
        out = _assemble_dose_response_series(points)
        assert list(out['doses'].keys()) == ['']
        assert len(out['doses']['']) == 2


class TestCanPlotLogX:
    def test_empty_input_is_false(self):
        assert _can_plot_log_x([]) is False

    def test_only_baselines_is_false(self):
        # No dose points → log-x is vacuously unavailable (nothing to
        # plot on x).  The dialog's checkbox should grey out.
        pts = [DoseResponsePoint('bl', '', None, 400.0, 0.0, 1)]
        assert _can_plot_log_x(pts) is False

    def test_all_positive_doses_is_true(self):
        pts = [
            DoseResponsePoint('a', 'dof', 0.01, 1.0, 0.0, 1),
            DoseResponsePoint('b', 'dof', 0.1, 1.0, 0.0, 1),
            DoseResponsePoint('c', 'dof', 1.0, 1.0, 0.0, 1),
        ]
        assert _can_plot_log_x(pts) is True

    def test_zero_dose_disables_log(self):
        # A literal "0 µM" group forces linear scale — log can't
        # render x=0.  In practice this shouldn't happen (baselines
        # should use dose_uM=None), but we're defensive.
        pts = [
            DoseResponsePoint('a', 'dof', 0.0, 1.0, 0.0, 1),
            DoseResponsePoint('b', 'dof', 1.0, 1.0, 0.0, 1),
        ]
        assert _can_plot_log_x(pts) is False

    def test_negative_dose_disables_log(self):
        # Pathological but covered for defensiveness.
        pts = [
            DoseResponsePoint('a', 'dof', -1.0, 1.0, 0.0, 1),
        ]
        assert _can_plot_log_x(pts) is False

    def test_baseline_mixed_with_doses_ignores_baseline(self):
        # Baselines (dose_uM=None) don't participate in the check —
        # they're drawn separately as horizontal lines.
        pts = [
            DoseResponsePoint('bl', '', None, 400.0, 0.0, 1),
            DoseResponsePoint('a', 'dof', 0.01, 1.0, 0.0, 1),
        ]
        assert _can_plot_log_x(pts) is True


class TestDrugColour:
    def test_zero_index_is_first_colour(self):
        assert _drug_colour(0) == sp._DRUG_PALETTE[0]

    def test_wraps_around_palette(self):
        n = len(sp._DRUG_PALETTE)
        assert _drug_colour(n) == _drug_colour(0)
        assert _drug_colour(n + 1) == _drug_colour(1)

    def test_returns_hex_string(self):
        # Always a 7-char hex colour so callers can feed it to
        # pyqtgraph / QColor without reformatting.
        for i in range(len(sp._DRUG_PALETTE)):
            c = _drug_colour(i)
            assert isinstance(c, str)
            assert c.startswith('#')
            assert len(c) == 7


# ═════════════════════════════════════════════════════════════════════════
#  Step 6 — CDISC SEND export helpers (task #89).
#
#  These exercise the pure Python glue that marshals a study + results
#  cache into the list[dict] shape :func:`cardiac_fp_analyzer.cdisc_export
#  .export_send_package` expects.  The worker itself needs Qt and the
#  scientific core on real signals — that's validated on Marco's
#  laptop; CI only covers the deterministic transform here.
# ═════════════════════════════════════════════════════════════════════════


class TestFormatConcentrationFromDoseUM:
    def test_none_returns_empty_string(self):
        # Baseline groups have dose_uM=None — the CDISC exporter
        # routes them to CONTROL based on is_baseline / drug heuristics,
        # so we emit '' rather than an ambiguous '0 uM'.
        assert _format_concentration_from_dose_uM(None) == ''

    def test_small_dose_uses_g_format(self):
        # :g strips trailing zeros — 0.010 µM should read "0.01 uM"
        # (i.e. 10 nM), which is what a scientist expects.
        assert _format_concentration_from_dose_uM(0.010) == '0.01 uM'

    def test_whole_number_dose(self):
        assert _format_concentration_from_dose_uM(100) == '100 uM'

    def test_fractional_dose(self):
        assert _format_concentration_from_dose_uM(0.5) == '0.5 uM'

    def test_drug_argument_is_accepted_but_does_not_change_unit(self):
        # Signature allows a drug hint for future specialisation; it
        # must not change the rendered unit today.
        assert _format_concentration_from_dose_uM(
            1.0, drug='Dofetilide',
        ) == '1 uM'

    def test_roundtrips_through_exporter_parser(self):
        # Contract: whatever we emit must be parseable by the exporter's
        # _parse_concentration helper — otherwise EXDOSE would silently
        # fall back to 0.0 and EXDOSU to the default nmol/L.
        from cardiac_fp_analyzer.cdisc_export import _parse_concentration
        for dose in (0.01, 0.1, 1.0, 10.0, 100.0):
            s = _format_concentration_from_dose_uM(dose)
            val, unit = _parse_concentration(s)
            assert val == dose, f"roundtrip lost value for {dose}"
            assert unit == 'umol/L', f"roundtrip lost unit for {dose}"


class TestEnrichFileInfoForExport:
    def test_adds_drug_and_concentration_from_group(self):
        g = _FakeGroup(
            name='dose-0.01', drug='Dofetilide', dose_uM=0.01,
            files=[], config=object(),
        )
        f = _FakeFile(csv_relpath='a.csv')
        raw = {
            'file_info': {'chip': 'A', 'analyzed_channel': 'EL1'},
            'summary': {'fpd_ms_median': 400.0},
        }
        out = _enrich_file_info_for_export(raw, g, f)
        assert out['file_info']['drug'] == 'Dofetilide'
        assert out['file_info']['concentration'] == '0.01 uM'
        assert out['file_info']['is_baseline'] is False
        # Pipeline-owned keys (chip, analyzed_channel) survive untouched.
        assert out['file_info']['chip'] == 'A'
        assert out['file_info']['analyzed_channel'] == 'EL1'

    def test_does_not_mutate_original_result(self):
        g = _FakeGroup(
            name='dose-0.01', drug='Dofetilide', dose_uM=0.01,
            files=[], config=object(),
        )
        f = _FakeFile(csv_relpath='a.csv')
        raw = {
            'file_info': {'chip': 'A', 'drug': '', 'concentration': ''},
            'summary': {},
        }
        out = _enrich_file_info_for_export(raw, g, f)
        # Original dict and its nested file_info must be unchanged.
        assert raw['file_info']['drug'] == ''
        assert raw['file_info']['concentration'] == ''
        assert out['file_info'] is not raw['file_info']

    def test_baseline_group_marks_is_baseline(self):
        g = _FakeGroup(
            name='baseline', drug='', dose_uM=None,
            files=[], config=object(),
        )
        f = _FakeFile(csv_relpath='ctrl.csv')
        raw = {'file_info': {}, 'summary': {}}
        out = _enrich_file_info_for_export(raw, g, f)
        assert out['file_info']['drug'] == ''
        assert out['file_info']['concentration'] == ''
        assert out['file_info']['is_baseline'] is True

    def test_summary_and_other_top_level_keys_passthrough(self):
        # The exporter reads file_info + summary + metadata; anything
        # else in the result dict must round-trip so future fields don't
        # need changes here.
        g = _FakeGroup(
            name='g', drug='X', dose_uM=1.0, files=[], config=object(),
        )
        f = _FakeFile(csv_relpath='x.csv')
        raw = {
            'file_info': {'chip': 'A'},
            'summary': {'bpm_mean': 58.0},
            'metadata': {'filename': 'x.csv'},
            'beat_indices': [1, 2, 3],
        }
        out = _enrich_file_info_for_export(raw, g, f)
        # Summary/metadata/beat_indices must be the same objects —
        # we only shallow-copy ``file_info``.
        assert out['summary'] is raw['summary']
        assert out['metadata'] is raw['metadata']
        assert out['beat_indices'] is raw['beat_indices']


class TestCollectExportInputs:
    def test_empty_study_returns_empty(self):
        assert _collect_export_inputs(_FakeStudy(groups=[]), {}) == []

    def test_fresh_ok_file_included(self):
        g = _FakeGroup(
            name='dose', drug='dof', dose_uM=0.01,
            files=[_FakeFile('a.csv')], config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {('dose', 'a.csv'): _ok(fpdc_ms=410.0)}
        out = _collect_export_inputs(study, cache)
        assert len(out) == 1
        assert out[0][0] is g
        assert out[0][1].csv_relpath == 'a.csv'

    def test_error_file_excluded(self):
        g = _FakeGroup(
            name='dose', drug='dof', dose_uM=0.01,
            files=[_FakeFile('a.csv'), _FakeFile('b.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {
            ('dose', 'a.csv'): _ok(),
            ('dose', 'b.csv'): _err('pipeline crash'),
        }
        out = _collect_export_inputs(study, cache)
        assert len(out) == 1
        assert out[0][1].csv_relpath == 'a.csv'

    def test_missing_cache_entry_excluded(self):
        # Never-analysed file → dropped.  Exporter only sees files the
        # user has actually batch-analysed.
        g = _FakeGroup(
            name='dose', drug='dof', dose_uM=0.01,
            files=[_FakeFile('a.csv'), _FakeFile('b.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g])
        cache = {('dose', 'a.csv'): _ok()}
        out = _collect_export_inputs(study, cache)
        assert len(out) == 1
        assert out[0][1].csv_relpath == 'a.csv'

    def test_stale_file_excluded(self):
        # Stale = fingerprint mismatch between cache and current
        # (config, channel).  The cache's fingerprint is a bogus
        # non-empty hex string; the live fingerprint is computed by
        # _result_fingerprint on the current config, so they won't match.
        cfg = object()
        g = _FakeGroup(
            name='dose', drug='dof', dose_uM=0.01,
            files=[_FakeFile('a.csv', channel='EL1')],
            config=cfg,
        )
        study = _FakeStudy(groups=[g])
        stale_fr = FileResult(
            status='ok', n_included=10, n_total=10,
            fingerprint='deadbeefcafe',   # non-empty → triggers compare
            channel_analyzed='EL1',
        )
        cache = {('dose', 'a.csv'): stale_fr}
        out = _collect_export_inputs(study, cache)
        assert out == []

    def test_preserves_study_group_file_order(self):
        g1 = _FakeGroup(
            name='baseline', drug='', dose_uM=None,
            files=[_FakeFile('ctrl.csv')],
            config=object(),
        )
        g2 = _FakeGroup(
            name='dose-0.01', drug='dof', dose_uM=0.01,
            files=[_FakeFile('a.csv'), _FakeFile('b.csv')],
            config=object(),
        )
        study = _FakeStudy(groups=[g1, g2])
        cache = {
            ('baseline', 'ctrl.csv'): _ok(),
            ('dose-0.01', 'a.csv'): _ok(),
            ('dose-0.01', 'b.csv'): _ok(),
        }
        out = _collect_export_inputs(study, cache)
        rels = [f.csv_relpath for _, f in out]
        assert rels == ['ctrl.csv', 'a.csv', 'b.csv']


class TestSuggestStudyId:
    def test_basic_slug(self):
        assert _suggest_study_id('Exp6 dof-rr') == 'EXP6-DOF-RR'

    def test_strips_punctuation(self):
        # Middle dot, non-ASCII punctuation and mixed whitespace must
        # all collapse to single dashes.
        assert _suggest_study_id('Exp6 · dof-rr') == 'EXP6-DOF-RR'

    def test_strips_leading_and_trailing_dashes(self):
        assert _suggest_study_id('!!Exp6!!') == 'EXP6'

    def test_empty_input_returns_default(self):
        assert _suggest_study_id('') == 'STUDY'

    def test_all_punctuation_returns_default(self):
        assert _suggest_study_id('!!!') == 'STUDY'
        assert _suggest_study_id('...---...') == 'STUDY'

    def test_cap_at_32_characters(self):
        long = 'a' * 50
        out = _suggest_study_id(long)
        assert len(out) == 32
        assert out == 'A' * 32

    def test_is_idempotent(self):
        # Running the slugifier twice is a no-op — the dialog calls
        # it on both the suggested default AND on the user-edited
        # value before passing to export_send_package, so idempotence
        # is load-bearing.
        for s in ['Exp6 dof-rr', 'CIPA001', 'study-2025-04']:
            a = _suggest_study_id(s)
            b = _suggest_study_id(a)
            assert a == b

    def test_preserves_digits(self):
        assert _suggest_study_id('study2026') == 'STUDY2026'


# ─────────────────────────────────────────────────────────────────────────
#  _collect_batch_inputs — study-wide + per-group batch collector (#91).
#
#  The collector is the shared foundation of both "Analizza gruppo" and
#  "Analizza studio": it resolves CSV paths via the study folder,
#  clamps each file's ``channel`` to the known triple, and splits the
#  (group, file) pairs into submittable (ready) vs. pre-flagged
#  (missing).  These tests cover the split logic end-to-end without
#  Qt — the UI-side threadpool / progress-dialog plumbing is exercised
#  at a higher level, but a regression in collection would quietly
#  mis-attribute or drop files, which is the kind of bug the user
#  won't notice until results are wrong.
# ─────────────────────────────────────────────────────────────────────────

from cardiac_fp_analyzer.study import (  # noqa: E402 — after sp import
    FileEntry,
    Group,
    Study,
)


def _make_study(tmp_path, layout):
    """Build a :class:`Study` on disk with the requested ``layout``.

    ``layout`` is a list of ``(group_name, [(csv_relpath, exists, channel)])``
    — a tiny DSL so tests read top-down.  Missing files are represented
    by ``exists=False`` and no file is actually created on disk; the
    study folder itself always exists.
    """
    folder = tmp_path / "study"
    folder.mkdir()
    groups = []
    for gname, files in layout:
        fes = []
        for rel, exists, ch in files:
            if exists:
                p = folder / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("time,el1\n0,0\n", encoding="utf-8")
            fes.append(FileEntry(csv_relpath=rel, channel=ch))
        groups.append(Group(name=gname, files=fes))
    return Study(name="t", folder=str(folder), groups=groups)


class TestCollectBatchInputs:
    def test_empty_groups_list_yields_empty_result(self, tmp_path):
        study = _make_study(tmp_path, [])
        ready, missing = _collect_batch_inputs(study, [])
        assert ready == []
        assert missing == {}

    def test_single_group_all_ready(self, tmp_path):
        study = _make_study(
            tmp_path,
            [("g1", [("a.csv", True, "auto"), ("b.csv", True, "EL1")])],
        )
        ready, missing = _collect_batch_inputs(study, study.groups)
        assert missing == {}
        assert len(ready) == 2
        # Channels preserved; order follows group.files.
        assert [r[1].csv_relpath for r in ready] == ["a.csv", "b.csv"]
        assert [r[3] for r in ready] == ["auto", "EL1"]
        # Each ready entry references the same Group instance — the
        # run-batch layer relies on this to read per-group config.
        assert all(r[0] is study.groups[0] for r in ready)

    def test_missing_file_populates_missing_dict(self, tmp_path):
        study = _make_study(
            tmp_path,
            [("g1", [("there.csv", True, "auto"),
                     ("gone.csv", False, "auto")])],
        )
        ready, missing = _collect_batch_inputs(study, study.groups)
        assert len(ready) == 1
        assert ready[0][1].csv_relpath == "there.csv"
        # Missing file lands in the cache-ready dict keyed by
        # (group_name, csv_relpath) with an Italian error message —
        # the UI merges this straight into _results before the pool
        # starts, so ✗ badges appear without phantom "pending" rows.
        assert ("g1", "gone.csv") in missing
        fr = missing[("g1", "gone.csv")]
        assert fr.status == "error"
        assert "non trovato" in fr.error.lower()
        assert fr.error.endswith("gone.csv")

    def test_cross_group_study_wide(self, tmp_path):
        # Two groups × two files each — mirrors the real dose-response
        # shape (baseline + dose).  Study-wide batch must see every
        # (group, file) pair and keep the per-group Group reference so
        # each worker can honour that group's AnalysisConfig.
        study = _make_study(
            tmp_path,
            [
                ("baseline", [("b1.csv", True, "auto"),
                              ("b2.csv", True, "auto")]),
                ("dose-10nM", [("d1.csv", True, "EL2"),
                               ("d2.csv", False, "auto")]),
            ],
        )
        ready, missing = _collect_batch_inputs(study, study.groups)
        # One missing (d2), three ready.
        assert len(ready) == 3
        assert set(missing.keys()) == {("dose-10nM", "d2.csv")}
        # Group references survive the collection — the run-batch
        # layer compares ``r[0] is group`` when looking up per-group
        # config.
        group_by_name = {g.name: g for g in study.groups}
        for group, fe, _abs, _ch in ready:
            assert group is group_by_name[group.name]

    def test_unknown_channel_clamps_to_auto(self, tmp_path):
        # Legacy / corrupted sidecar might carry 'EL3' or a typo like
        # 'el1'; the collector must fall back to 'auto' rather than
        # propagate a bogus label into ``analyze_single_file``.
        study = _make_study(
            tmp_path,
            [("g1", [
                ("a.csv", True, "EL3"),     # unknown
                ("b.csv", True, "el1"),      # wrong case
                ("c.csv", True, ""),         # empty
                ("d.csv", True, "EL1"),      # valid → preserved
                ("e.csv", True, "auto"),     # valid → preserved
                ("f.csv", True, "EL2"),      # valid → preserved
            ])],
        )
        ready, _ = _collect_batch_inputs(study, study.groups)
        channels = [r[3] for r in ready]
        # First three clamped, last three preserved exactly.
        assert channels == ["auto", "auto", "auto", "EL1", "auto", "EL2"]

    def test_empty_group_is_skipped_silently(self, tmp_path):
        # A group with no files contributes nothing — not an error,
        # not a missing row.  The "Analizza studio" caller already
        # filters these out for cleaner logs, but the collector must
        # tolerate them anyway (defence in depth + future callers).
        study = _make_study(
            tmp_path,
            [
                ("empty", []),
                ("populated", [("a.csv", True, "auto")]),
            ],
        )
        ready, missing = _collect_batch_inputs(study, study.groups)
        assert len(ready) == 1
        assert ready[0][0].name == "populated"
        assert missing == {}

    def test_subset_of_groups_honoured(self, tmp_path):
        # The per-group flow passes ``[selected]``; the collector must
        # see only the requested group(s), never sibling groups — a
        # regression here would leak analysis work across groups.
        study = _make_study(
            tmp_path,
            [
                ("g1", [("a.csv", True, "auto")]),
                ("g2", [("b.csv", True, "auto")]),
            ],
        )
        ready, missing = _collect_batch_inputs(study, [study.groups[0]])
        assert [r[1].csv_relpath for r in ready] == ["a.csv"]
        assert missing == {}
        assert _suggest_study_id('2026-q2') == '2026-Q2'
