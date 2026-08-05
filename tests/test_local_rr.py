"""Tests for true-local-RR computation and its effect on FPDc / CV.

Background
----------
Rate correction is a per-beat physiological relationship: FPD depends on
the cycle that preceded it. The correct RR for beat N is the time since
the depolarization immediately before it.

QC rejects beats on spike amplitude and template correlation — on
*waveform shape*. A rejected beat still depolarized the tissue and still
set the diastolic interval for the next beat. The old code derived RR
from the surviving beats only (``beat_periods[i-1]`` over the accepted
list), so intervals spanned the QC gaps.

Measured on 67 real recordings: RR inflated up to 3.72x (>10% on 35 of
them), FPDc deflated by up to 55%, CV(BP) inflated by up to 250
percentage points — and 21 of 67 pushed over the 25% inclusion
threshold by the artefact alone, including the dofetilide baseline
(true CV 23.7%, reported 31.8%).

``analyze.py`` already carried the right idea in a comment ("Beat period
from ALL detected beats ... avoids artificial gaps from QC rejection")
but applied it to ``result['beat_periods']`` only.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.parameters import compute_local_rr

FS = 1000.0


# ── Core behaviour ──────────────────────────────────────────────────────

def test_no_rejections_matches_consecutive_differences():
    """With nothing rejected the result is the plain diff, in seconds."""
    beats = [0, 1000, 2000, 3000]
    rr = compute_local_rr(beats, beats, FS)
    assert rr[0] is None                      # first beat has no predecessor
    assert rr[1:] == pytest.approx([1.0, 1.0, 1.0])


def test_first_beat_has_no_rr():
    """FPDc of the first beat is undefined by construction."""
    rr = compute_local_rr([0, 1000], [0, 1000], FS)
    assert rr[0] is None


def test_rejected_beat_still_provides_the_interval():
    """The core fix.

    Beats at 0, 1000, 2000, 3000; the one at 2000 is QC-rejected. The
    beat at 3000 must still be corrected against 2000 (1.0 s), not
    against 1000 (2.0 s).
    """
    allb = [0, 1000, 2000, 3000]
    accepted = [0, 1000, 3000]

    rr = compute_local_rr(accepted, allb, FS)

    assert rr[0] is None
    assert rr[1] == pytest.approx(1.0)
    assert rr[2] == pytest.approx(1.0), (
        "the rejected beat at 2000 still set the cycle length for 3000"
    )


def test_legacy_behaviour_would_have_doubled_it():
    """Pins the magnitude of the bug being fixed.

    The old computation was ``np.diff(accepted)`` — shown here explicitly
    so the contrast is in the test rather than only in the commit message.
    """
    allb = [0, 1000, 2000, 3000]
    accepted = [0, 1000, 3000]

    legacy = np.diff(accepted) / FS
    fixed = compute_local_rr(accepted, allb, FS)

    assert legacy[-1] == pytest.approx(2.0)    # spans the gap
    assert fixed[-1] == pytest.approx(1.0)     # true cycle


def test_every_other_beat_rejected_halves_the_rr():
    """The regime that produced the 2x inflation on real data."""
    allb = list(range(0, 10000, 500))          # 20 beats, 0.5 s apart
    accepted = allb[::2]                        # keep every other one

    rr = compute_local_rr(accepted, allb, FS)

    assert all(v == pytest.approx(0.5) for v in rr[1:]), (
        "each accepted beat's predecessor is the rejected beat 0.5 s before"
    )
    legacy = np.diff(accepted) / FS
    assert all(v == pytest.approx(1.0) for v in legacy)


# ── The indexing trap ───────────────────────────────────────────────────

def test_alignment_goes_through_sample_positions_not_list_positions():
    """Each accepted beat must get *its own* RR.

    Indexing a raw period array by the accepted-list position silently
    hands each beat somebody else's RR as soon as anything is rejected.
    Irregular spacing makes the mix-up detectable.
    """
    allb = [0, 100, 500, 1200, 1800, 3000]
    accepted = [500, 1800, 3000]

    rr = compute_local_rr(accepted, allb, FS)

    assert rr == pytest.approx([0.400, 0.600, 1.200])

    # What indexing the raw diff array by list position would have given:
    raw_diffs = np.diff(allb) / FS            # [.1, .4, .7, .6, 1.2]
    wrong = [raw_diffs[i - 1] if i > 0 else None
             for i in range(len(accepted))]
    assert wrong[1] != pytest.approx(rr[1])   # 0.1 vs 0.6
    assert wrong[2] != pytest.approx(rr[2])   # 0.4 vs 1.2


# ── Degenerate inputs ───────────────────────────────────────────────────

def test_empty_accepted_returns_empty():
    assert compute_local_rr([], [0, 1000], FS) == []


def test_empty_raw_train_yields_all_none():
    """No raw train (legacy caller) must not crash or invent values."""
    assert compute_local_rr([0, 1000], [], FS) == [None, None]


def test_single_beat():
    assert compute_local_rr([500], [500], FS) == [None]


def test_resegmented_beat_is_not_its_own_predecessor():
    """Re-segmentation can shift a beat's index by a few samples.

    The shifted beat must not be rate-corrected against its own former
    position: that would give an RR of a few milliseconds and an absurd
    FPDc. The predecessor is the raw beat one cycle back.
    """
    allb = [0, 1000, 2000]
    accepted = [1005]                          # the beat at 1000, shifted

    rr = compute_local_rr(accepted, allb, FS)
    assert rr[0] == pytest.approx(1.005), (
        "expected the cycle back to 0, not the 5 ms to its own old index"
    )


def test_beats_just_outside_the_tolerance_are_real_predecessors():
    """The guard must not swallow genuinely close beats.

    60 ms apart is beyond the 50 ms same-beat tolerance, so it counts.
    """
    rr = compute_local_rr([1060], [0, 1000, 2000], FS)
    assert rr[0] == pytest.approx(0.060)


def test_same_beat_tolerance_is_configurable():
    allb = [0, 1000, 2000]
    assert compute_local_rr([1005], allb, FS,
                            same_beat_tol_s=0.0)[0] == pytest.approx(0.005)
    assert compute_local_rr([1005], allb, FS,
                            same_beat_tol_s=0.05)[0] == pytest.approx(1.005)


def test_accepted_beat_before_every_raw_beat():
    rr = compute_local_rr([50], [100, 200], FS)
    assert rr == [None]


def test_result_length_always_matches_accepted():
    for accepted, allb in [([], [1, 2]), ([1], [1]), ([1, 5, 9], [1, 3, 5, 7, 9])]:
        assert len(compute_local_rr(accepted, allb, FS)) == len(accepted)


# ── Downstream consequence ──────────────────────────────────────────────

def test_fpdc_direction_of_the_error():
    """An inflated RR deflates FPDc, since FPDc = FPD / RR**(1/3).

    Documents why the reported historical FPDc values are too low.
    """
    fpd_s = 0.400
    rr_true, rr_inflated = 0.5, 1.0

    fpdc_true = fpd_s / rr_true ** (1 / 3)
    fpdc_wrong = fpd_s / rr_inflated ** (1 / 3)

    assert fpdc_wrong < fpdc_true
    # 2x RR inflation → the correct value is ~26% higher than reported.
    assert fpdc_true / fpdc_wrong == pytest.approx(2 ** (1 / 3), rel=1e-9)
    assert fpdc_true / fpdc_wrong == pytest.approx(1.26, abs=0.01)


def test_cv_of_true_rhythm_is_lower_than_of_the_gapped_one():
    """Why 21 of 67 recordings crossed the inclusion threshold.

    A perfectly regular train with alternating rejections has CV 0 in
    truth; measured over the survivors it stays 0 only if the rejection
    pattern is itself regular. Make it irregular and the artefact appears.
    """
    allb = [0, 500, 1000, 1500, 2000, 2500, 3000]      # perfectly regular
    accepted = [0, 500, 1500, 3000]                     # irregular rejects

    true_rr = np.array([v for v in compute_local_rr(accepted, allb, FS)
                        if v is not None])
    gapped = np.diff(accepted) / FS

    true_cv = true_rr.std() / true_rr.mean() * 100
    gapped_cv = gapped.std() / gapped.mean() * 100

    assert true_cv == pytest.approx(0.0, abs=1e-9)
    assert gapped_cv > 40.0
