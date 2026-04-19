"""Regression tests for the post-detection refactor.

``analyze_single_file`` was refactored in Fase 2.F to delegate everything
downstream of beat detection to a new ``_analyze_from_beats`` helper,
which is also the entry point for the UI "Ricalcola" button via
``recompute_from_beats``.

The refactor is a pure code move: for any fixed ``bi`` the two paths
(full pipeline / detect + recompute) must produce identical results.
These tests pin that invariant.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cardiac_fp_analyzer.analyze import (
    analyze_single_file,
    recompute_from_beats,
)
from cardiac_fp_analyzer.config import AnalysisConfig

# ═══════════════════════════════════════════════════════════════════════
#  Synthetic CSV fixture — the same pattern the batch-error tests use
# ═══════════════════════════════════════════════════════════════════════

def _write_synthetic_csv(path: Path, fs: float = 1000.0,
                          duration: float = 8.0) -> None:
    """Write a minimal Digilent-style CSV with periodic beat-like spikes.

    Enough beats (≈ 1 Hz for 8 s → 8 beats) for the pipeline to build
    a template, run QC, and populate the arrhythmia report.  The
    exact numeric values don't matter for the non-regression test —
    we only compare ``full`` vs ``recompute`` on the same signal.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    el1 = rng.standard_normal(n) * 1e-4
    el2 = rng.standard_normal(n) * 1e-4

    # Sharp spike every ~1 s followed by a small positive bump 300 ms
    # later (a crude T-wave analogue).  Eight full cycles fit in 8 s.
    for k in range(1, 9):
        spike_idx = int(k * fs * 1.0)
        if spike_idx + int(0.35 * fs) >= n:
            continue
        el1[spike_idx] = 2.0e-3
        el1[spike_idx + 1] = -1.0e-3
        el2[spike_idx] = 2.0e-3
        el2[spike_idx + 1] = -1.0e-3
        twave = spike_idx + int(0.30 * fs)
        el1[twave] = 3.5e-4
        el2[twave] = 3.5e-4

    header = (
        "#Device Name: Analog Discovery 2\n"
        "#Serial Number: SN:TEST\n"
        "#Date Time: 2026-04-19 00:00:00.0\n"
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


def _assert_deep_equal(a, b, path: str = "") -> None:
    """Recursively compare two Python objects, with numpy-array support.

    Used for summary dicts and other result-dict fields that mix
    scalars, arrays, dicts, and lists.  The test is strict: numeric
    values must match exactly (no tolerance) — the refactor is a pure
    code move, so any drift is a bug.  NaN compares equal to NaN
    (same-slot semantics), matching numpy's ``array_equal(equal_nan=True)``.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), (
            f"{path}: key sets differ ({set(a)} vs {set(b)})"
        )
        for k in a:
            _assert_deep_equal(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_deep_equal(x, y, f"{path}[{i}]")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{path}: array drifted",
        )
        return
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return
        assert a == b, f"{path}: {a} vs {b}"
        return
    assert a == b, f"{path}: {a!r} vs {b!r}"


@pytest.fixture(scope="module")
def baseline_result():
    """Run the full pipeline once and share the result across tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "synthetic.csv"
        _write_synthetic_csv(csv_path)
        cfg = AnalysisConfig()
        result = analyze_single_file(
            str(csv_path), channel="el1", verbose=False, config=cfg,
        )
    assert result is not None, "pipeline returned None on synthetic fixture"
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Non-regression: full pipeline == detect + recompute on same bi
# ═══════════════════════════════════════════════════════════════════════

def test_recompute_with_same_bi_preserves_all_params(baseline_result):
    """recompute_from_beats(result, result['beat_indices_raw']) must
    reproduce the original full-pipeline result.  The signal + time
    vectors, detection_info, and config are the same, so the post-
    detection pipeline is called with the same inputs — the output
    must be bit-identical.
    """
    bi_raw = baseline_result["beat_indices_raw"]
    recomputed = recompute_from_beats(
        baseline_result, bi_raw, verbose=False,
    )

    # --- beat indices after QC / filters must be identical ----------
    np.testing.assert_array_equal(
        recomputed["beat_indices"], baseline_result["beat_indices"],
        err_msg="beat_indices drifted between full pipeline and recompute",
    )
    np.testing.assert_array_equal(
        recomputed["beat_indices_fpd"], baseline_result["beat_indices_fpd"],
        err_msg="beat_indices_fpd drifted between full pipeline and recompute",
    )
    np.testing.assert_array_equal(
        recomputed["beat_indices_raw"], baseline_result["beat_indices_raw"],
        err_msg="beat_indices_raw drifted between full pipeline and recompute",
    )

    # --- per-beat parameters ---------------------------------------
    assert len(recomputed["all_params"]) == len(baseline_result["all_params"])
    for a, b in zip(recomputed["all_params"], baseline_result["all_params"]):
        # Compare numeric fields with tight tolerance; string fields exactly.
        for key in a:
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if np.isnan(va) and np.isnan(vb):
                    continue
                assert va == pytest.approx(vb, rel=0, abs=0), (
                    f"all_params[{key}] drifted: {va} vs {vb}"
                )
            else:
                assert va == vb, f"all_params[{key}] drifted: {va!r} vs {vb!r}"


def test_recompute_preserves_summary_and_arrhythmia(baseline_result):
    """Summary dict and arrhythmia report must also match."""
    bi_raw = baseline_result["beat_indices_raw"]
    recomputed = recompute_from_beats(
        baseline_result, bi_raw, verbose=False,
    )

    # --- summary dict ----------------------------------------------
    # Recursively compare — the summary may contain scalars, arrays,
    # dicts (rr_outlier_filter, resegmentation_info, …) and lists.
    a_summary = recomputed["summary"]
    b_summary = baseline_result["summary"]
    _assert_deep_equal(a_summary, b_summary, path="summary")

    # --- arrhythmia report -----------------------------------------
    a_ar = recomputed["arrhythmia_report"]
    b_ar = baseline_result["arrhythmia_report"]
    assert a_ar.classification == b_ar.classification
    assert a_ar.risk_score == b_ar.risk_score
    assert a_ar.flags == b_ar.flags
    assert a_ar.events == b_ar.events


def test_recompute_with_fewer_beats_reduces_param_rows(baseline_result):
    """Dropping the last beat before recompute must shrink all_params.

    This is the user-level contract the UI depends on: if you remove a
    beat in the viewer and click Ricalcola, the Parametri table should
    have one fewer row.  We do not assert exact numeric changes — QC
    + rhythm filters may cascade — only that the pipeline responds.
    """
    bi_raw = np.asarray(baseline_result["beat_indices_raw"])
    if len(bi_raw) < 3:
        pytest.skip("synthetic fixture produced too few beats")

    recomputed = recompute_from_beats(
        baseline_result, bi_raw[:-1], verbose=False,
    )

    # bi_raw is the user-supplied input → must match what we passed in.
    np.testing.assert_array_equal(
        recomputed["beat_indices_raw"], bi_raw[:-1],
    )
    # Downstream sets are at most as large as the input; they may be
    # smaller if QC / filters reject additional beats.
    assert len(recomputed["beat_indices"]) <= len(bi_raw) - 1
    assert len(recomputed["all_params"]) <= len(bi_raw) - 1


def test_recompute_shares_signal_arrays(baseline_result):
    """The signal / time arrays in the new result must be the same
    objects as in the input — no hidden copies slow down the UI path.
    """
    recomputed = recompute_from_beats(
        baseline_result, baseline_result["beat_indices_raw"],
        verbose=False,
    )
    assert recomputed["filtered_signal"] is baseline_result["filtered_signal"]
    assert recomputed["raw_signal"] is baseline_result["raw_signal"]
    assert recomputed["time_vector"] is baseline_result["time_vector"]


def test_recompute_default_config_does_not_mutate_input(baseline_result):
    """Calling recompute without a config must not leave side effects
    in the input result.  The UI re-uses the same result repeatedly —
    accidental mutation would show up as cross-run bleed.
    """
    # Snapshot a few keys that would be most obviously damaged.
    before_n = len(baseline_result["all_params"])
    before_class = baseline_result["arrhythmia_report"].classification

    _ = recompute_from_beats(
        baseline_result,
        baseline_result["beat_indices_raw"][:-1],
        verbose=False,
    )

    assert len(baseline_result["all_params"]) == before_n
    assert baseline_result["arrhythmia_report"].classification == before_class
