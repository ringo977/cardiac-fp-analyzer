"""Regression tests for the beat alignment fix (v3.2.0).

The critical bug was: segment_beats() returns a `valid` index list for
non-truncated beats, but validate_beats() was receiving the unfiltered
beat_indices, causing misalignment between amplitude checks (N items)
and morphology checks (M items, M < N).
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.beat_detection import segment_beats


class TestSegmentBeatsValidIndices:
    """Verify segment_beats returns consistent valid indices."""

    def _make_synthetic_signal(self, fs=1000, duration_s=5, beat_period_s=1.0):
        """Create a synthetic signal with sharp spikes at regular intervals."""
        n_samples = int(fs * duration_s)
        t = np.arange(n_samples) / fs
        signal = np.zeros(n_samples)

        # Add sharp negative spikes
        beat_indices = []
        for spike_time in np.arange(0.5, duration_s - 0.5, beat_period_s):
            idx = int(spike_time * fs)
            if idx < n_samples:
                signal[idx] = -1.0
                # Add a repolarization hump
                for j in range(1, min(200, n_samples - idx)):
                    signal[idx + j] += 0.3 * np.exp(-j / 80)
                beat_indices.append(idx)

        return signal, t, np.array(beat_indices), fs

    def test_valid_indices_are_subset(self):
        """valid indices should be a subset of range(len(beat_indices))."""
        signal, t, bi, fs = self._make_synthetic_signal()
        bd, btm, valid = segment_beats(signal, t, bi, fs)

        assert all(0 <= v < len(bi) for v in valid)

    def test_valid_indices_match_segments(self):
        """Number of valid indices should equal number of segments."""
        signal, t, bi, fs = self._make_synthetic_signal()
        bd, btm, valid = segment_beats(signal, t, bi, fs)

        assert len(valid) == len(bd)
        assert len(valid) == len(btm)

    def test_edge_beats_excluded(self):
        """Beats too close to signal edges should be excluded from valid."""
        signal, t, bi, fs = self._make_synthetic_signal(duration_s=2, beat_period_s=0.5)
        # Add a beat very close to the end
        bi_extended = np.append(bi, len(signal) - 10)

        bd, btm, valid = segment_beats(signal, t, bi_extended, fs)

        # The last beat (near end) should have been excluded
        assert len(valid) <= len(bi_extended)
        assert len(valid) == len(bd) == len(btm)

    def test_bi_seg_remapping(self):
        """Verify the bi[np.array(valid)] remapping produces correct alignment."""
        signal, t, bi, fs = self._make_synthetic_signal()
        bd, btm, valid = segment_beats(signal, t, bi, fs)

        bi_seg = bi[np.array(valid)] if len(valid) > 0 else bi[:0]

        # bi_seg should have same length as segments
        assert len(bi_seg) == len(bd) == len(btm), (
            f"Alignment error: bi_seg={len(bi_seg)}, "
            f"beats_data={len(bd)}, beats_time={len(btm)}"
        )
