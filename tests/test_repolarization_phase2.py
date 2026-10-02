"""Phase-2 repolarisation changes (Oct 2026), pinned on synthetic signals.

(The template alignment and inversion fixes of the same release are in
tests/test_template_alignment.py.)

1. Candidate wave selection: ``prefer_positive`` picks the positive lobe of
   a biphasic repolarisation (the manual reference's convention).
2. The template search window uses the rhythm of the beats forming the
   template when the full detected train is shorter (over-detection), and
   the extension never crosses a depolarisation-like event.

The calibration of these rules on real data (manual gold standard) is
documented in docs/FPD_vs_gold_standard_2026-10.md; the data themselves are
not redistributable and are not part of the test corpus.
"""

import numpy as np
import pytest

from cardiac_fp_analyzer.config import RepolarizationConfig
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.repolarization import (
    _select_candidate,
    find_repolarization_on_template,
    next_spike_cut_index,
)

FS = 2000.0
PRE_MS = 50.0


def _gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def _beat(post_ms=1450.0, t_pos=None, a_pos=0.0, t_neg=None, a_neg=0.0, spike=1.0,
          noise=0.0, rng=None, invert=False):
    """Synthetic FP beat: biphasic spike at PRE_MS + optional ±T lobes."""
    n = int((PRE_MS + post_ms) / 1000 * FS)
    t = np.arange(n) / FS * 1000 - PRE_MS            # ms from spike
    x = spike * (_gauss(t, 0, 2.0) - 0.7 * _gauss(t, 6, 3.0))
    if t_pos is not None:
        x = x + a_pos * _gauss(t, t_pos, 40.0)
    if t_neg is not None:
        x = x - a_neg * _gauss(t, t_neg, 40.0)
    if noise and rng is not None:
        x = x + rng.normal(0, noise, n)
    return -x if invert else x


def _beat_time(n):
    return np.arange(n) / FS - PRE_MS / 1000


# ── 1. candidate selection ──────────────────────────────────────────────

class TestCandidateRule:
    rc = RepolarizationConfig()

    def test_prefer_positive_lobe_of_biphasic_wave(self):
        # (peak_idx, prominence, sign): negative lobe larger, positive 0.6×
        cands = [(1000, 1.0, -1), (760, 0.6, +1)]
        assert _select_candidate(cands, self.rc, FS)[2] == +1

    def test_small_positive_bump_does_not_win(self):
        cands = [(1000, 1.0, -1), (900, 0.3, +1)]            # below 0.5×
        assert _select_candidate(cands, self.rc, FS)[2] == -1

    def test_distant_positive_peak_is_another_wave(self):
        cands = [(1000, 1.0, -1), (1000 + int(0.6 * FS), 0.9, +1)]   # 600 ms away
        assert _select_candidate(cands, self.rc, FS)[2] == -1

    def test_legacy_rule_takes_max_prominence(self):
        from dataclasses import replace
        rc = replace(self.rc, repol_candidate_rule='max_prominence')
        cands = [(1000, 1.0, -1), (760, 0.6, +1)]
        assert _select_candidate(cands, rc, FS)[2] == -1

    def test_on_a_biphasic_template(self):
        """Positive lobe at 600 ms, larger negative lobe at 720 ms: default
        reports the positive peak; the legacy rule the negative one."""
        from dataclasses import replace
        tpl = _beat(t_pos=600, a_pos=0.12, t_neg=720, a_neg=0.2)
        fpd, sign, *_ = find_repolarization_on_template(tpl, FS, pre_ms=PRE_MS, median_bp_s=1.5)
        assert sign == +1 and abs(fpd / FS * 1000 - 600) < 15
        legacy = replace(RepolarizationConfig(), repol_candidate_rule='max_prominence')
        fpd_l, sign_l, *_ = find_repolarization_on_template(tpl, FS, pre_ms=PRE_MS, cfg=legacy,
                                                              median_bp_s=1.5)
        assert sign_l == -1 and abs(fpd_l / FS * 1000 - 720) < 15

    def test_monophasic_negative_wave_is_kept(self):
        tpl = _beat(t_neg=650, a_neg=0.2)
        fpd, sign, *_ = find_repolarization_on_template(tpl, FS, pre_ms=PRE_MS, median_bp_s=1.5)
        assert sign == -1 and abs(fpd / FS * 1000 - 650) < 15

    def test_default_endpoint_is_the_peak(self):
        assert RepolarizationConfig().fpd_method == 'peak'


# ── 2. window rhythm and next-beat guard ────────────────────────────────

class TestTemplateWindowRhythm:
    def _recording(self, rr_s=2.0, t_wave_ms=1100.0, n_beats=12, seed=4):
        """Kept beats every rr_s with a T-wave at 0.55×RR; the full detected
        train also contains a spurious detection halfway between beats."""
        rng = np.random.default_rng(seed)
        post_ms = 0.7 * rr_s * 1000 + 50
        beats = [_beat(post_ms=post_ms, t_pos=t_wave_ms, a_pos=0.2, noise=0.005, rng=rng)
                 for _ in range(n_beats)]
        times = [_beat_time(len(b)) for b in beats]
        step = int(rr_s * FS)
        kept = np.arange(n_beats) * step + int(1.0 * FS)
        raw = np.sort(np.concatenate([kept, kept[:-1] + step // 2]))
        return beats, times, kept, raw

    def test_over_detected_train_no_longer_hides_the_t_wave(self):
        beats, times, kept, raw = self._recording()
        _, s = extract_all_parameters(beats, times, kept, FS, cfg=RepolarizationConfig(),
                                      all_beat_indices=raw)
        assert s.get('repol_window_extended') is True
        assert abs(s['template_fpd_ms'] - 1100) < 20

    def test_legacy_window_misses_it(self):
        from dataclasses import replace
        beats, times, kept, raw = self._recording()
        rc = replace(RepolarizationConfig(), window_rr_from_template_beats=False)
        _, s = extract_all_parameters(beats, times, kept, FS, cfg=rc, all_beat_indices=raw)
        tf = s.get('template_fpd_ms')
        assert tf is None or abs(tf - 1100) > 100

    def test_correct_train_is_unchanged(self):
        beats, times, kept, _ = self._recording()
        _, s = extract_all_parameters(beats, times, kept, FS, cfg=RepolarizationConfig(),
                                      all_beat_indices=kept)
        assert s.get('repol_window_extended') is False
        assert abs(s['template_fpd_ms'] - 1100) < 20


class TestNextSpikeGuard:
    def test_finds_a_spike_shaped_event(self):
        tpl = _beat(post_ms=2000)
        nxt = int((PRE_MS + 1500) / 1000 * FS)
        tpl = tpl + np.roll(_beat(post_ms=2000), nxt - int(PRE_MS / 1000 * FS))
        pre = int(PRE_MS / 1000 * FS)
        cut = next_spike_cut_index(tpl, FS, pre, pre + int(0.9 * FS), len(tpl))
        assert cut is not None and abs((cut - pre) / FS * 1000 - 1500) < 15

    def test_ignores_noise(self):
        rng = np.random.default_rng(5)
        tpl = _beat(post_ms=2000) + rng.normal(0, 0.05, int((PRE_MS + 2000) / 1000 * FS))
        pre = int(PRE_MS / 1000 * FS)
        assert next_spike_cut_index(tpl, FS, pre, pre + int(0.9 * FS), len(tpl)) is None


@pytest.mark.parametrize('rule', ['max_prominence', 'prefer_positive'])
def test_unknown_rule_is_rejected_but_known_ones_work(rule):
    from dataclasses import replace
    rc = replace(RepolarizationConfig(), repol_candidate_rule=rule)
    assert _select_candidate([(10, 1.0, 1)], rc, FS) is not None
    with pytest.raises(ValueError):
        _select_candidate([(10, 1.0, 1)], replace(rc, repol_candidate_rule='nope'), FS)
