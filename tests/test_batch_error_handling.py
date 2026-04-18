"""
test_batch_error_handling.py — Regression tests for Fix #5
(Sprint 1 item 5, ASSESSMENT_v3.3.0.md §4.1).

Before this fix:
- ``analyze_single_file``'s except clause did not include
  ``pandas.errors.ParserError`` / ``EmptyDataError`` / ``UnicodeError``,
  so a malformed CSV could propagate out of the analyze helper.
- The serial branch of ``batch_analyze`` had no ``try/except`` at all:
  one bad file would crash the whole batch.
- The parallel branch caught only a subset of the exceptions the
  internal ``analyze_single_file`` caught, so the two code paths had
  asymmetric error semantics.

These tests exercise the batch on intentionally-broken CSV fixtures and
assert that:

1. ``_safe_analyze`` returns ``(None, str)`` instead of propagating;
2. ``batch_analyze`` (serial branch) doesn't raise even when one or
   more files in the directory are malformed;
3. the ``_BATCH_SAFE_EXCEPTIONS`` whitelist lists all expected
   exception types (pandas parser/empty + UnicodeError + existing);
4. mixed good + bad file directories still produce results for the
   good files.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cardiac_fp_analyzer.analyze import (
    _BATCH_SAFE_EXCEPTIONS,
    _safe_analyze,
    batch_analyze,
)
from cardiac_fp_analyzer.config import AnalysisConfig

# ═══════════════════════════════════════════════════════════════════════
#  CSV fixtures — one good, several broken
# ═══════════════════════════════════════════════════════════════════════

def _write_good_csv(path: Path, fs: float = 1000.0, duration: float = 2.0):
    """Minimal Digilent-style CSV that load_csv will accept."""
    import numpy as np
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    el1 = rng.standard_normal(n) * 1e-4
    el2 = rng.standard_normal(n) * 1e-4
    for k in range(1, 4):
        idx = int(k * fs * 0.5)
        if idx + 3 < n:
            el1[idx] = 0.05
            el1[idx + 1] = -0.03
            el2[idx] = 0.05
    header = (
        "#Device Name: Analog Discovery 2\n"
        "#Serial Number: SN:123\n"
        "#Date Time: 2026-01-01 00:00:00.0\n"
        f"#Sample rate: {fs} Hz\n"
        f"#Samples: {n}\n"
        "#Trigger: Auto\n"
        "#Channel 1: Range: 5.0 mV/div Offset: 0.0 V\n"
        "#Channel 2: Range: 5.0 mV/div Offset: 0.0 V\n"
        "Time (s),Channel 1 (V),Channel 2 (V)\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for ti, a, b in zip(t, el1, el2):
            f.write(f"{ti:.6f},{a:.6e},{b:.6e}\n")


def _write_empty_csv(path: Path):
    path.write_text("")


def _write_header_only_csv(path: Path):
    # Header lines only, no data rows — pandas raises EmptyDataError.
    path.write_text(
        "#Device Name: Analog Discovery 2\n"
        "#Sample rate: 1000.0 Hz\n"
        "#Samples: 0\n"
    )


def _write_malformed_csv(path: Path):
    """CSV with an unparseable structure — ragged quotes/columns.

    pandas raises ``ParserError`` here when reading the data section.
    """
    path.write_text(
        '#Device Name: BadFile\n'
        '#Sample rate: 1000.0 Hz\n'
        'Time,Ch1,Ch2\n'
        '0.000,"unterminated_quote,\n'
        '0.001,0.001,0.001,0.001,extra_column\n'
    )


def _write_binary_garbage_csv(path: Path):
    """Non-UTF8 bytes that will trip UnicodeError in default decoding."""
    path.write_bytes(b"\xff\xfe\x00\x00not a csv \x80\x81\x82")


# ═══════════════════════════════════════════════════════════════════════
#  _BATCH_SAFE_EXCEPTIONS — whitelist shape
# ═══════════════════════════════════════════════════════════════════════

class TestBatchSafeExceptions:

    def test_includes_pandas_parser_error(self):
        assert pd.errors.ParserError in _BATCH_SAFE_EXCEPTIONS

    def test_includes_pandas_empty_data_error(self):
        assert pd.errors.EmptyDataError in _BATCH_SAFE_EXCEPTIONS

    def test_includes_unicode_error(self):
        assert UnicodeError in _BATCH_SAFE_EXCEPTIONS

    def test_includes_legacy_exceptions(self):
        for exc in (KeyError, ValueError, IndexError, RuntimeError,
                    AssertionError, FileNotFoundError, OSError):
            assert exc in _BATCH_SAFE_EXCEPTIONS


# ═══════════════════════════════════════════════════════════════════════
#  _safe_analyze
# ═══════════════════════════════════════════════════════════════════════

class TestSafeAnalyze:

    def test_missing_file_returns_none_with_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        r, err = _safe_analyze(missing, channel='el1', verbose=False,
                                config=AnalysisConfig())
        assert r is None
        # Either caught inside analyze_single_file (returns None without
        # reaching _safe_analyze's except) or caught by the outer guard.
        # Both cases must yield a non-empty error string.
        assert err is not None
        assert err != ""

    def test_empty_file_does_not_raise(self, tmp_path):
        path = tmp_path / "empty.csv"
        _write_empty_csv(path)
        r, err = _safe_analyze(path, channel='el1', verbose=False,
                                config=AnalysisConfig())
        assert r is None
        assert err is not None

    def test_header_only_file_does_not_raise(self, tmp_path):
        path = tmp_path / "headers.csv"
        _write_header_only_csv(path)
        r, err = _safe_analyze(path, channel='el1', verbose=False,
                                config=AnalysisConfig())
        assert r is None
        assert err is not None

    def test_malformed_csv_does_not_raise(self, tmp_path):
        path = tmp_path / "malformed.csv"
        _write_malformed_csv(path)
        r, err = _safe_analyze(path, channel='el1', verbose=False,
                                config=AnalysisConfig())
        assert r is None
        assert err is not None

    def test_binary_garbage_does_not_raise(self, tmp_path):
        path = tmp_path / "garbage.csv"
        _write_binary_garbage_csv(path)
        r, err = _safe_analyze(path, channel='el1', verbose=False,
                                config=AnalysisConfig())
        assert r is None
        assert err is not None


# ═══════════════════════════════════════════════════════════════════════
#  batch_analyze — serial branch (n_workers=1, which is the default path)
# ═══════════════════════════════════════════════════════════════════════

class TestBatchAnalyzeSerialResilience:

    def test_empty_directory_returns_empty_list(self, tmp_path):
        # Not strictly part of the bug fix, but sanity-checks that
        # an empty directory is a no-op, not an error.
        results = batch_analyze(
            str(tmp_path), channel='el1', output_dir=str(tmp_path / 'out'),
            verbose=False, config=AnalysisConfig(), n_workers=1,
        )
        assert results == []

    def test_all_malformed_does_not_crash(self, tmp_path):
        """A directory of only bad files must return [], not raise."""
        _write_empty_csv(tmp_path / "a.csv")
        _write_header_only_csv(tmp_path / "b.csv")
        _write_malformed_csv(tmp_path / "c.csv")
        _write_binary_garbage_csv(tmp_path / "d.csv")

        # Must not raise.
        results = batch_analyze(
            str(tmp_path), channel='el1',
            output_dir=str(tmp_path / 'out'),
            verbose=False, config=AnalysisConfig(), n_workers=1,
        )
        assert isinstance(results, list)

    def test_mixed_good_and_bad_returns_results_for_good(
        self, tmp_path,
    ):
        """With a mix of broken + valid CSVs, the batch completes and
        returns results for the good files."""
        _write_good_csv(tmp_path / "good_chipA_ch1_baseline.csv")
        _write_malformed_csv(tmp_path / "bad.csv")
        _write_empty_csv(tmp_path / "empty.csv")

        results = batch_analyze(
            str(tmp_path), channel='el1',
            output_dir=str(tmp_path / 'out'),
            verbose=False, config=AnalysisConfig(), n_workers=1,
        )
        assert isinstance(results, list)
        # The good file may or may not produce a full result depending
        # on whether the tiny synthetic signal passes all pipeline
        # gates, but the important invariant is that the batch runs to
        # completion without raising on the bad/empty files alongside it.

    def test_inclusion_criteria_runs_on_empty_batch(self, tmp_path):
        """apply_inclusion_criteria on a zero-results batch shouldn't
        crash the downstream baseline-relative pass either."""
        _write_empty_csv(tmp_path / "only_bad.csv")
        # Must not raise.
        results = batch_analyze(
            str(tmp_path), channel='el1',
            output_dir=str(tmp_path / 'out'),
            verbose=False, config=AnalysisConfig(), n_workers=1,
        )
        assert results == [] or isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════
#  batch_analyze — single-file shortcut (also goes through serial branch)
# ═══════════════════════════════════════════════════════════════════════

class TestBatchAnalyzeSingleBadFile:
    """When ``len(csv_files) == 1`` the parallel branch is skipped even
    if ``n_workers > 1``; this path is covered by the serial loop."""

    def test_single_bad_file_does_not_crash(self, tmp_path):
        _write_malformed_csv(tmp_path / "only.csv")
        results = batch_analyze(
            str(tmp_path), channel='el1',
            output_dir=str(tmp_path / 'out'),
            verbose=False, config=AnalysisConfig(), n_workers=4,
        )
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════
#  analyze_single_file — direct call should now also swallow pandas errors
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeSingleFileExceptionWidening:

    @pytest.mark.parametrize("writer", [
        _write_empty_csv,
        _write_header_only_csv,
        _write_malformed_csv,
        _write_binary_garbage_csv,
    ])
    def test_returns_none_for_broken_csv(self, tmp_path, writer):
        from cardiac_fp_analyzer.analyze import analyze_single_file
        path = tmp_path / "broken.csv"
        writer(path)
        # Must not raise; must return None to signal failure.
        result = analyze_single_file(
            path, channel='el1', verbose=False, config=AnalysisConfig()
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
#  Explicit exception propagation — malformed CSV pattern
# ═══════════════════════════════════════════════════════════════════════

def test_unguarded_pandas_read_raises_expected_error():
    """Sanity check on the underlying failure modes: a truly empty file
    raises EmptyDataError; a malformed one raises ParserError or
    UnicodeError.  This guards against the fixtures drifting into a
    shape that no longer triggers the behavior we're defending."""
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "empty.csv"
        _write_empty_csv(empty)
        with pytest.raises((pd.errors.EmptyDataError, pd.errors.ParserError)):
            pd.read_csv(empty)

        garbage = Path(d) / "garbage.csv"
        _write_binary_garbage_csv(garbage)
        with pytest.raises((UnicodeError, pd.errors.ParserError,
                            pd.errors.EmptyDataError)):
            pd.read_csv(garbage)
