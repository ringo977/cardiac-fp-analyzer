"""
test_beat_length_invariant.py — Regression tests for Fix #6
(Sprint 1 item 6, ASSESSMENT_v3.3.0.md §3.2 "Nota su un falso positivo").

Before this fix the three template-building helpers:

  - ``parameters.build_beat_template``
  - ``quality_control.compute_beat_template``
  - ``residual_analysis.compute_template``

each contained a defensive per-beat slice of the form::

    min_len = min(len(b) for b in beats)
    mat = np.array([b[:min_len] for b in beats])

which was *inoperative* because ``segment_beats`` produces beats of
uniform length by construction (``data[idx-pre : idx+post]`` with
fixed ``pre`` / ``post`` samples), and ``_align_beats_xcorr`` further
truncates + pads every aligned beat back to ``min_len``.  The
assessment in rev.1 flagged this as visual noise that could mislead a
future reader into thinking the code handled heterogeneous-length
input.

Fix #6 replaces each slice with an ``assert`` that explicitly
documents and enforces the invariant, while preserving the two
functional sanity checks that rode alongside the slice
(``beat_len < 10`` in ``quality_control`` and ``beat_len < 20`` in
``residual_analysis``).

These tests verify:

1. the underlying invariant — ``segment_beats`` really does return
   equal-length beats — holds across a range of pre/post/fs settings;
2. the alignment helper (``_align_beats_xcorr``) preserves the
   uniform-length invariant even when given more than three beats;
3. all three template helpers accept uniform-length input and return
   a template whose length matches the input beat length;
4. each of the three helpers *raises* ``AssertionError`` if fed a
   hand-constructed heterogeneous-length list (guard against silent
   regressions under a future refactor that loses the invariant).
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_beats(n_beats: int = 8, beat_len: int = 300) -> list[np.ndarray]:
    """Return ``n_beats`` float arrays of identical length ``beat_len``."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal(beat_len).astype(float)
            for _ in range(n_beats)]


def _make_uneven_beats() -> list[np.ndarray]:
    """Hand-rolled heterogeneous-length beats — used to confirm the
    invariant is actually enforced."""
    rng = np.random.default_rng(1)
    return [
        rng.standard_normal(300).astype(float),
        rng.standard_normal(290).astype(float),  # one short
        rng.standard_normal(300).astype(float),
        rng.standard_normal(300).astype(float),
        rng.standard_normal(300).astype(float),
    ]


# ═══════════════════════════════════════════════════════════════════════
#  1. segment_beats produces uniform-length output
# ═══════════════════════════════════════════════════════════════════════

class TestSegmentBeatsUniformLength:
    """The invariant all three helpers depend on."""

    @pytest.mark.parametrize("pre_ms,post_ms,fs", [
        (50, 500, 1000.0),
        (100, 800, 1000.0),
        (50, 1550, 1000.0),  # slow-rhythm window from Fix #2 / §3.2
        (30, 300, 2000.0),
    ])
    def test_all_beats_share_length(self, pre_ms, post_ms, fs):
        from cardiac_fp_analyzer.beat_detection import segment_beats

        n = int(5.0 * fs)
        # Synthetic signal: baseline + small sharp deflections every
        # 500 ms (so we get ~10 detectable peaks in 5 s).
        t = np.arange(n) / fs
        sig = np.zeros(n)
        spike_indices = np.arange(int(0.25 * fs), n - int(post_ms/1000*fs),
                                  int(0.5 * fs))
        for idx in spike_indices:
            sig[idx] = 0.05
            sig[idx + 1] = -0.03

        beats_data, _beats_time, _valid = segment_beats(
            sig, t, spike_indices, fs, pre_ms=pre_ms, post_ms=post_ms,
        )
        if len(beats_data) == 0:
            pytest.skip("no beats segmented for this parameter combination")

        expected_len = int(pre_ms / 1000 * fs) + int(post_ms / 1000 * fs)
        assert all(len(b) == expected_len for b in beats_data), (
            f"segment_beats returned heterogeneous lengths: "
            f"{set(len(b) for b in beats_data)} (expected {expected_len})"
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. _align_beats_xcorr preserves uniform length
# ═══════════════════════════════════════════════════════════════════════

class TestAlignBeatsXcorrPreservesLength:

    def test_aligned_beats_share_input_length(self):
        from cardiac_fp_analyzer.parameters import _align_beats_xcorr

        beats = _make_beats(n_beats=10, beat_len=400)
        aligned = _align_beats_xcorr(beats, fs=1000.0)
        assert len(aligned) == len(beats)
        # Every aligned beat must match the (uniform) input length.
        input_len = len(beats[0])
        assert all(len(b) == input_len for b in aligned)

    def test_short_input_takes_early_return(self):
        """With fewer than 3 beats the function returns the input
        unmodified — callers of build_beat_template guard against this
        with their own ``len < 5`` check."""
        from cardiac_fp_analyzer.parameters import _align_beats_xcorr

        beats = _make_beats(n_beats=2, beat_len=400)
        aligned = _align_beats_xcorr(beats, fs=1000.0)
        assert aligned is beats  # early return is by reference


# ═══════════════════════════════════════════════════════════════════════
#  3. Template helpers accept uniform input and produce correct shape
# ═══════════════════════════════════════════════════════════════════════

class TestBuildBeatTemplateHappyPath:

    def test_parameters_build_beat_template_uniform_input(self):
        from cardiac_fp_analyzer.parameters import build_beat_template

        beats = _make_beats(n_beats=8, beat_len=300)
        tmpl = build_beat_template(beats, fs=1000.0)
        assert tmpl is not None
        assert tmpl.shape == (300,)

    def test_quality_control_compute_beat_template_uniform_input(self):
        from cardiac_fp_analyzer.quality_control import compute_beat_template

        beats = _make_beats(n_beats=8, beat_len=300)
        tmpl = compute_beat_template(beats)
        assert tmpl is not None
        assert tmpl.shape == (300,)

    def test_residual_analysis_compute_template_uniform_input(self):
        from cardiac_fp_analyzer.residual_analysis import compute_template

        beats = _make_beats(n_beats=8, beat_len=300)
        tmpl = compute_template(beats)
        assert tmpl is not None
        assert tmpl.shape == (300,)


# ═══════════════════════════════════════════════════════════════════════
#  4. Functional sanity checks still fire on too-short beats
# ═══════════════════════════════════════════════════════════════════════

class TestShortBeatsReturnNone:
    """Guards that rode alongside the old min_len slice."""

    def test_quality_control_returns_none_below_10(self):
        from cardiac_fp_analyzer.quality_control import compute_beat_template

        beats = _make_beats(n_beats=8, beat_len=9)
        assert compute_beat_template(beats) is None

    def test_quality_control_accepts_10_sample_beats(self):
        from cardiac_fp_analyzer.quality_control import compute_beat_template

        beats = _make_beats(n_beats=8, beat_len=10)
        tmpl = compute_beat_template(beats)
        assert tmpl is not None
        assert tmpl.shape == (10,)

    def test_residual_analysis_returns_none_below_20(self):
        from cardiac_fp_analyzer.residual_analysis import compute_template

        beats = _make_beats(n_beats=8, beat_len=19)
        assert compute_template(beats) is None

    def test_residual_analysis_accepts_20_sample_beats(self):
        from cardiac_fp_analyzer.residual_analysis import compute_template

        beats = _make_beats(n_beats=8, beat_len=20)
        tmpl = compute_template(beats)
        assert tmpl is not None
        assert tmpl.shape == (20,)


# ═══════════════════════════════════════════════════════════════════════
#  5. Invariant is actively enforced — heterogeneous input raises
# ═══════════════════════════════════════════════════════════════════════

class TestInvariantEnforcement:
    """If the invariant is ever broken by a future refactor, every
    template helper should raise loudly (``AssertionError``) rather
    than silently producing an object array or a surprisingly shaped
    template."""

    def test_parameters_align_normalises_before_assert(self):
        """``build_beat_template``'s new assertion only fires if
        ``_align_beats_xcorr`` itself loses the invariant — in
        practice alignment truncates to ``min_len`` internally, so
        downstream receives uniform-length beats even if the raw input
        is heterogeneous.  This test pins that contract so the
        invariant assertion upstream is trustworthy."""
        from cardiac_fp_analyzer.parameters import _align_beats_xcorr

        aligned = _align_beats_xcorr(_make_uneven_beats(), fs=1000.0)
        lengths = {len(b) for b in aligned}
        assert len(lengths) == 1, (
            "_align_beats_xcorr must normalise lengths before the "
            "invariant assert in build_beat_template can be trusted"
        )

    def test_quality_control_compute_beat_template_rejects_uneven(self):
        from cardiac_fp_analyzer.quality_control import compute_beat_template

        with pytest.raises(AssertionError, match="same length"):
            compute_beat_template(_make_uneven_beats())

    def test_residual_analysis_compute_template_rejects_uneven(self):
        from cardiac_fp_analyzer.residual_analysis import compute_template

        with pytest.raises(AssertionError, match="same length"):
            compute_template(_make_uneven_beats())
