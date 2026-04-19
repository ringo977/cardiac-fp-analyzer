"""Regression tests for ``analyze.py`` sidecar-override integration (task #64).

When a user corrects the automatic beat detection in the PySide6 viewer
the correction is persisted as a ``.overrides.json`` sidecar next to the
CSV (see ``cardiac_fp_analyzer.overrides``).  ``analyze_single_file``
must pick up that sidecar automatically so the correction survives
re-opens and batch re-runs.

These tests pin the user-visible contract:

* No sidecar → pipeline behaves exactly as before.
* Sidecar with ``removed_s`` → the matched automatic beats disappear
  from ``beat_indices_raw`` and the downstream parameter table shrinks.
* Sidecar with ``added_s`` → a new beat enters ``beat_indices_raw`` and
  reaches the downstream pipeline.
* ``config.use_overrides = False`` → sidecar is ignored even when
  present (escape hatch for pristine re-runs).
* ``det['overrides_applied']`` is attached when a sidecar was applied,
  absent otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cardiac_fp_analyzer.analyze import analyze_single_file
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.overrides import (
    BeatOverrides,
    overrides_path_for,
    save_overrides,
)

# ═══════════════════════════════════════════════════════════════════════
#  Synthetic CSV fixture — mirrors tests/test_recompute_from_beats.py
# ═══════════════════════════════════════════════════════════════════════


def _write_synthetic_csv(path: Path, fs: float = 1000.0,
                          duration: float = 8.0) -> None:
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    el1 = rng.standard_normal(n) * 1e-4
    el2 = rng.standard_normal(n) * 1e-4

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


@pytest.fixture
def csv_with_baseline(tmp_path):
    """Write a synthetic CSV and return (csv_path, baseline_result).

    ``baseline_result`` is the output of ``analyze_single_file`` with no
    sidecar present.  Subsequent calls in each test can compare against
    it to verify that overrides make a difference.
    """
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)
    cfg = AnalysisConfig()
    result = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert result is not None, "baseline pipeline returned None"
    # Guard: the sidecar must NOT have been created by the baseline run.
    assert not overrides_path_for(csv_path).exists()
    return csv_path, result


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════


def test_no_sidecar_is_transparent(csv_with_baseline):
    """Without a sidecar the pipeline behaves exactly as before.

    We rerun ``analyze_single_file`` on the same CSV and compare
    ``beat_indices_raw`` and the number of parameter rows — both must
    be identical to the baseline run.  ``det['overrides_applied']``
    must be absent (no sidecar touched the beat set).
    """
    csv_path, baseline = csv_with_baseline

    cfg = AnalysisConfig()
    again = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert again is not None

    np.testing.assert_array_equal(
        again["beat_indices_raw"], baseline["beat_indices_raw"],
    )
    assert len(again["all_params"]) == len(baseline["all_params"])
    assert "overrides_applied" not in again.get("detection_info", {})


def test_sidecar_removes_beat(csv_with_baseline):
    """A sidecar that removes an automatic beat must shrink the set.

    We pick the middle beat of the baseline detection (so it is safely
    inside the analysis window, not at an edge where segmentation
    would drop it anyway) and write a sidecar removing it.  The new
    run must produce one fewer beat in ``beat_indices_raw`` and its
    ``detection_info['overrides_applied']`` must report ``n_removed=1``.
    """
    csv_path, baseline = csv_with_baseline
    bi_raw = np.asarray(baseline["beat_indices_raw"])
    if len(bi_raw) < 3:
        pytest.skip("synthetic fixture produced too few beats")

    fs = float(baseline["metadata"]["sample_rate"])
    mid_idx = bi_raw[len(bi_raw) // 2]
    ov = BeatOverrides(removed_s=[float(mid_idx) / fs])
    save_overrides(str(csv_path), ov)

    cfg = AnalysisConfig()
    edited = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert edited is not None

    assert len(edited["beat_indices_raw"]) == len(bi_raw) - 1
    assert mid_idx not in set(edited["beat_indices_raw"].tolist())

    info = edited["detection_info"]["overrides_applied"]
    assert info["n_removed"] == 1
    assert info["n_added"] == 0
    assert info["unmatched_removals"] == []


def test_sidecar_adds_beat(csv_with_baseline):
    """A sidecar that adds a beat must grow the raw beat set.

    We insert a beat at a timestamp that is not within the 50 ms
    tolerance of any automatic beat — the synthetic fixture puts
    spikes at 1, 2, …, 8 s, so 4.5 s is safe.  The sidecar should add
    exactly one beat to ``beat_indices_raw``.
    """
    csv_path, baseline = csv_with_baseline
    bi_raw_before = np.asarray(baseline["beat_indices_raw"])
    fs = float(baseline["metadata"]["sample_rate"])

    # Guard the chosen timestamp: any existing beat within tol_s blocks
    # the insert (see apply_overrides de-duplication).  4.5 s is half
    # way between the 4th and 5th spike of the synthetic fixture.
    t_add = 4.5
    target_idx = int(round(t_add * fs))
    tol_samples = int(round(0.050 * fs))
    assert np.min(np.abs(bi_raw_before - target_idx)) > tol_samples, (
        "chosen t_add is too close to an existing automatic beat"
    )

    ov = BeatOverrides(added_s=[t_add])
    save_overrides(str(csv_path), ov)

    cfg = AnalysisConfig()
    edited = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert edited is not None

    bi_raw_after = np.asarray(edited["beat_indices_raw"])
    assert len(bi_raw_after) == len(bi_raw_before) + 1
    assert target_idx in set(bi_raw_after.tolist())

    info = edited["detection_info"]["overrides_applied"]
    assert info["n_added"] == 1
    assert info["n_removed"] == 0


def test_use_overrides_false_ignores_sidecar(csv_with_baseline):
    """``config.use_overrides=False`` must ignore a present sidecar.

    The sidecar stays on disk (we do not touch the user's file) but
    the pipeline must behave exactly like the no-sidecar baseline.
    """
    csv_path, baseline = csv_with_baseline
    bi_raw = np.asarray(baseline["beat_indices_raw"])
    fs = float(baseline["metadata"]["sample_rate"])

    mid_idx = bi_raw[len(bi_raw) // 2]
    ov = BeatOverrides(removed_s=[float(mid_idx) / fs])
    save_overrides(str(csv_path), ov)

    cfg = AnalysisConfig()
    cfg.use_overrides = False
    ignored = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert ignored is not None

    np.testing.assert_array_equal(
        ignored["beat_indices_raw"], baseline["beat_indices_raw"],
    )
    assert "overrides_applied" not in ignored["detection_info"]


def test_empty_sidecar_is_noop(csv_with_baseline, tmp_path):
    """A sidecar with empty add/remove lists must not affect the pipeline.

    This exercises the ``is_empty()`` early-return in ``apply_overrides``
    reached via the pipeline: the sidecar file is on disk but contains
    no actual corrections.  ``save_overrides`` normally deletes an empty
    sidecar, so we write one manually instead.
    """
    csv_path, baseline = csv_with_baseline
    sidecar = overrides_path_for(str(csv_path))
    sidecar.write_text(
        '{"version": "1", "added_s": [], "removed_s": [], "tol_s": 0.05}\n',
        encoding="utf-8",
    )

    cfg = AnalysisConfig()
    result = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert result is not None
    np.testing.assert_array_equal(
        result["beat_indices_raw"], baseline["beat_indices_raw"],
    )
    # No override info attached because is_empty() short-circuits.
    assert "overrides_applied" not in result["detection_info"]


def test_sidecar_unmatched_removal_is_reported(csv_with_baseline):
    """A removal far from any automatic beat must be reported unmatched.

    ``apply_overrides`` keeps unmatched removals in the info dict so the
    UI can surface them to the user and the sidecar remains useful after
    a future re-detection exposes the beat.  We pick a timestamp at
    least 200 ms from every automatic beat (≫ 50 ms default tol_s).
    """
    csv_path, baseline = csv_with_baseline
    bi_raw = np.asarray(baseline["beat_indices_raw"])
    fs = float(baseline["metadata"]["sample_rate"])
    auto_times = bi_raw / fs

    # 10 ms grid → pick a timestamp that sits > 200 ms from every
    # detected beat.  Robust to the fixture picking up minor transients
    # at the record edges (the 8 s duration × 8 spikes means the
    # detector may land slightly off the nominal 1/2/…/7 s positions).
    grid = np.arange(0.0, 8.0, 0.01)
    gaps = np.min(np.abs(grid[:, None] - auto_times[None, :]), axis=1)
    safe = grid[gaps > 0.200]
    assert safe.size > 0, "could not find a beat-free timestamp in fixture"
    t_rm = float(safe[safe.size // 2])

    ov = BeatOverrides(removed_s=[t_rm])
    save_overrides(str(csv_path), ov)

    cfg = AnalysisConfig()
    result = analyze_single_file(
        str(csv_path), channel="el1", verbose=False, config=cfg,
    )
    assert result is not None

    info = result["detection_info"]["overrides_applied"]
    assert info["n_removed"] == 0
    assert info["unmatched_removals"] == [pytest.approx(t_rm)]
    # Beat set is unchanged because no automatic beat matched.
    np.testing.assert_array_equal(
        result["beat_indices_raw"], baseline["beat_indices_raw"],
    )
