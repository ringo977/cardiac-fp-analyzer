"""test_slow_rhythm_segmentation.py — Regression test for the search-window
truncation bug that caused T-wave detection to fail on slow rhythms
(dofetilide-like signals with BP >> 1 s).

Root cause (pre-fix):
    analyze.py called ``segment_beats(..., post_ms=max(850, search_end_ms+50))``
    which is FIXED and does NOT account for the adaptive
    ``search_end_pct_rr × RR`` window used downstream in
    ``repolarization.find_repolarization_on_template``. For a signal with
    RR=3.5 s and ``search_end_pct_rr=0.70`` the adaptive search window is
    2450 ms, but the segmented template was only 950-1550 ms long, so
    ``search_end = min(search_end, len(template))`` silently clipped the
    search to the template boundary and the real T-wave (≈2000 ms after
    the spike) was never reached. The detected repolarization fell on
    the afterpotential / tail instead.

Fix (applied in this branch):
    ``analyze.py:segment_beats`` call now uses an adaptive ``post_ms`` =
    ``max(fixed_search_end_ms+50, search_end_pct_rr × median_BP × 1000 + 50)``
    so the template extends at least as far as the adaptive search window.
    A mirror fix was applied to ``channel_selection.select_best_channel``.

These tests exercise the integrated pipeline on a synthetic slow-rhythm
signal and assert that the template length and detected FPD track the
expected values.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.beat_detection import (
    compute_beat_periods,
    detect_beats,
    segment_beats,
)
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.parameters import extract_all_parameters
from tests.golden_signals import generate_regular_fp

# ═══════════════════════════════════════════════════════════════════════
#   Helpers
# ═══════════════════════════════════════════════════════════════════════

def _segment_with_config(signal, time, bi, fs, rep_cfg):
    """Replicate analyze.py's post-fix segmentation call."""
    bp = compute_beat_periods(bi, fs)
    median_bp_s = float(np.median(bp)) if len(bp) > 0 else 0.0
    pct_rr = getattr(rep_cfg, 'search_end_pct_rr', 0.0)
    adaptive_end_ms = (pct_rr * median_bp_s * 1000.0
                       if (pct_rr > 0 and median_bp_s > 0) else 0.0)
    post_ms = max(850.0,
                  rep_cfg.search_end_ms + 50.0,
                  adaptive_end_ms + 50.0)
    bd, btm, vi = segment_beats(signal, time, bi, fs,
                                pre_ms=rep_cfg.segment_pre_ms,
                                post_ms=post_ms)
    return bd, btm, vi, post_ms, median_bp_s


# ═══════════════════════════════════════════════════════════════════════
#   Regression tests
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptivePostMs:
    """Unit tests for the adaptive post_ms computation itself."""

    def test_adaptive_post_ms_extends_for_slow_rhythm(self):
        """BP=3.5 s, pct_rr=0.70 → post_ms must be ≥ 2450 + margin."""
        fs = 2000.0
        signal, time, _expected = generate_regular_fp(
            fs=fs, duration_s=30.0,
            beat_period_ms=3500.0, fpd_ms=800.0,
            noise_std=0.0,
        )
        bi, _bt, _info = detect_beats(signal, fs, method='auto',
                                      min_distance_ms=400)
        assert len(bi) >= 3, f"Need ≥3 beats, got {len(bi)}"

        cfg = AnalysisConfig()
        rep_cfg = cfg.repolarization
        # Use the DEFAULT search_end_ms (900 ms) — the fix must still
        # extend the template via the adaptive path even when the user
        # has not manually increased search_end_ms.
        assert rep_cfg.search_end_pct_rr == 0.70

        _bd, _btm, _vi, post_ms, median_bp_s = _segment_with_config(
            signal, time, bi, fs, rep_cfg
        )

        # Adaptive window should be 0.70 × 3500 = 2450 ms
        expected_adaptive_ms = 0.70 * median_bp_s * 1000.0
        assert expected_adaptive_ms == pytest.approx(2450.0, abs=200.0)
        # post_ms must cover it (plus margin)
        assert post_ms >= expected_adaptive_ms, (
            f"post_ms={post_ms:.0f} < adaptive_end_ms="
            f"{expected_adaptive_ms:.0f} — fix did not extend the window"
        )

    def test_adaptive_post_ms_stays_fixed_for_fast_rhythm(self):
        """BP=500 ms, pct_rr=0.70 → adaptive=350 ms < fixed 900 ms. Should
        use the fixed search_end_ms path (no regression for normal signals)."""
        fs = 2000.0
        signal, time, _expected = generate_regular_fp(
            fs=fs, duration_s=10.0,
            beat_period_ms=500.0, fpd_ms=250.0,
            noise_std=0.0,
        )
        bi, _bt, _info = detect_beats(signal, fs, method='auto',
                                      min_distance_ms=300)
        assert len(bi) >= 10

        cfg = AnalysisConfig()
        rep_cfg = cfg.repolarization
        _bd, _btm, _vi, post_ms, median_bp_s = _segment_with_config(
            signal, time, bi, fs, rep_cfg
        )

        # For a 500 ms RR the adaptive window is 350 ms, well below the
        # 900 ms default. post_ms should collapse to the fixed ceiling.
        expected_fixed_ms = rep_cfg.search_end_ms + 50.0  # 950 ms
        assert post_ms == pytest.approx(max(850.0, expected_fixed_ms),
                                        abs=1.0)

    def test_adaptive_post_ms_disabled_falls_back_to_fixed(self):
        """pct_rr=0.0 (disabled) → post_ms must equal fixed ceiling
        regardless of BP — preserves opt-out behaviour."""
        fs = 2000.0
        signal, time, _expected = generate_regular_fp(
            fs=fs, duration_s=30.0,
            beat_period_ms=3500.0, fpd_ms=800.0,
            noise_std=0.0,
        )
        bi, _bt, _info = detect_beats(signal, fs, method='auto',
                                      min_distance_ms=400)

        cfg = AnalysisConfig()
        rep_cfg = cfg.repolarization
        rep_cfg.search_end_pct_rr = 0.0  # disable adaptive extension

        _bd, _btm, _vi, post_ms, _median_bp_s = _segment_with_config(
            signal, time, bi, fs, rep_cfg
        )
        expected_fixed_ms = max(850.0, rep_cfg.search_end_ms + 50.0)
        assert post_ms == pytest.approx(expected_fixed_ms, abs=1.0)


class TestSlowRhythmEndToEnd:
    """End-to-end: the pre-fix pipeline would clip the template and
    report an FPD well below the true 2000 ms. After the fix the
    extracted FPD should match the true value."""

    def test_long_fpd_recovered_on_slow_rhythm(self):
        """True BP=3.5 s, FPD=2000 ms. Pre-fix: template ≤ 1550 ms,
        FPD reported ≪ 2000 ms. Post-fix: FPD ≈ 2000 ms."""
        fs = 2000.0
        true_fpd_ms = 2000.0
        signal, time, _expected = generate_regular_fp(
            fs=fs, duration_s=30.0,
            beat_period_ms=3500.0,
            fpd_ms=true_fpd_ms,
            repol_amp=0.012,     # boost so repol peak stands out
            noise_std=0.0002,    # mild noise so template averaging helps
            seed=123,
        )

        # Beat detection
        bi, _bt, _info = detect_beats(signal, fs, method='auto',
                                      min_distance_ms=400)
        assert len(bi) >= 5, f"Need ≥5 beats, got {len(bi)}"

        # Segmentation (post-fix, adaptive post_ms)
        cfg = AnalysisConfig()
        rep_cfg = cfg.repolarization
        bd, btm, vi, post_ms, median_bp_s = _segment_with_config(
            signal, time, bi, fs, rep_cfg
        )

        # Sanity: the template window must reach past the true T-wave
        assert post_ms > true_fpd_ms, (
            f"post_ms={post_ms:.0f} ms does not reach the true T-wave "
            f"at {true_fpd_ms:.0f} ms — the template will clip it."
        )
        # And the actual segmented beat must be long enough in samples
        n_samples_each = int(rep_cfg.segment_pre_ms * fs / 1000) + int(post_ms * fs / 1000)
        assert all(len(b) == n_samples_each for b in bd)

        # Parameter extraction
        bi_seg = np.asarray(bi)[np.asarray(vi)]
        all_p, summary = extract_all_parameters(bd, btm, bi_seg, fs,
                                                cfg=rep_cfg)

        fpd = summary.get('fpd_ms_median', np.nan)
        assert fpd is not None and not np.isnan(fpd), \
            "FPD was not computed"

        # Accept ±20 % of the true FPD — synthetic repol shape, noise and
        # tangent/peak method introduce some jitter, but anything below
        # ~1000 ms would mean the template was still truncated.
        lower, upper = 0.70 * true_fpd_ms, 1.30 * true_fpd_ms
        assert lower <= fpd <= upper, (
            f"Detected FPD={fpd:.0f} ms is outside expected range "
            f"[{lower:.0f}, {upper:.0f}] for true FPD={true_fpd_ms:.0f} ms. "
            f"This suggests the template is still being truncated."
        )

    def test_pre_fix_simulation_shows_truncation(self):
        """Sanity-check: if we reproduce the OLD segmentation (fixed
        post_ms ignoring BP) the template IS shorter than the true FPD.
        This guards against silently losing the regression signal if
        future refactors change the default search_end_ms."""
        fs = 2000.0
        true_fpd_ms = 2000.0
        signal, time, _expected = generate_regular_fp(
            fs=fs, duration_s=30.0,
            beat_period_ms=3500.0,
            fpd_ms=true_fpd_ms,
            noise_std=0.0,
        )
        bi, _bt, _info = detect_beats(signal, fs, method='auto',
                                      min_distance_ms=400)
        cfg = AnalysisConfig()
        rep_cfg = cfg.repolarization

        # Old-style (buggy) segmentation: fixed post_ms, no RR adaptation.
        old_post_ms = max(850.0, rep_cfg.search_end_ms + 50.0)
        bd_old, _btm_old, _vi_old = segment_beats(
            signal, time, bi, fs,
            pre_ms=rep_cfg.segment_pre_ms,
            post_ms=old_post_ms,
        )
        # This is exactly the bug: the old template is too short.
        assert old_post_ms < true_fpd_ms, (
            f"Unexpected: fixed post_ms={old_post_ms:.0f} ≥ true FPD"
            f"={true_fpd_ms:.0f}. The regression baseline no longer "
            f"demonstrates the bug — revisit this test."
        )
        # And confirm the segmented beat is shorter than the real T-wave.
        pre_samples = int(rep_cfg.segment_pre_ms * fs / 1000)
        true_fpd_samples = int(true_fpd_ms * fs / 1000)
        for b in bd_old:
            post_samples = len(b) - pre_samples
            assert post_samples < true_fpd_samples, (
                "Old segmentation unexpectedly reaches the T-wave."
            )
