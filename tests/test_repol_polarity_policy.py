"""Per-beat repolarisation polarity policy (v3.23).

Operators' feedback (Exp1, 26 Oct 2026): "sometimes repolarisation markers are put in opposite positions
throughout the signal" — on a biphasic wave some beats took the positive lobe, others the negative one.
``per_beat_polarity``: 'free' (both signs compete, up to v3.22), 'template' (template's sign first),
'majority' (default): a free pass, then every beat searched with the recording's sign first — the template's,
unless two thirds of the beats chose the other one. Evaluated on the GG corpus: recordings with mixed
polarity 71 → 9 (DEV) and 28 → 4 (TEST), repolarisations within 20 ms of the operator's 57 → 62 % (DEV) and
64 → 68 % (TEST), recording FPD within 5 % unchanged on DEV (55 %) and 63 → 66 % on TEST.
"""
from __future__ import annotations

import numpy as np
import pytest

import cardiac_fp_analyzer.parameters as P
from cardiac_fp_analyzer.config import AnalysisConfig

FS = 2000.0
RR = 1.5


def _beats(n=12, seed=0):
    """Beat segments like the pipeline's (spike 50 ms into the segment, to 85 % of the RR): a sharp spike
    and a biphasic repolarisation (positive lobe at 400 ms, negative at 550 ms)."""
    rng = np.random.default_rng(seed)
    data, times, idx = [], [], []
    for k in range(n):
        m = int((0.05 + 0.85 * RR) * FS)
        t = np.arange(m) / FS - 0.05
        x = 2e-6 * rng.standard_normal(m)
        x += -300e-6 * np.exp(-(t / 0.003) ** 2 / 2)
        x += 80e-6 * np.exp(-((t - 0.40) / 0.04) ** 2 / 2) - 80e-6 * np.exp(-((t - 0.55) / 0.04) ** 2 / 2)
        data.append(x)
        times.append(t)
        idx.append(int((1.0 + k * RR) * FS))
    return data, times, np.array(idx)


def _fingerprint(x):
    return round(float(np.asarray(x)[10]) * 1e12)


def _fake_finder(opposite, beats):
    """Stand-in for find_repolarization_per_beat: with the template's sign preferred it returns that sign;
    searching freely, the beats in ``opposite`` take the other lobe (beats recognised by their samples)."""
    number = {_fingerprint(x): k for k, x in enumerate(beats)}

    def fake(data, t, spike_idx, fs, template_repol_sign=1, cfg=None, info=None, **kw):
        k = number[_fingerprint(data)]
        sign = template_repol_sign
        if not cfg.per_beat_prefer_template_sign and k in opposite:
            sign = -template_repol_sign
        if info is not None:
            info['sign'] = sign
        lat = 0.40 if sign > 0 else 0.55
        return lat, sign * 80e-6, spike_idx + int(lat * fs), spike_idx + int(lat * fs)
    return fake


def _run(monkeypatch, mode, opposite, threshold=None):
    data, times, idx = _beats()
    monkeypatch.setattr(P, '_find_repolarization_per_beat', _fake_finder(set(opposite), data))
    rc = AnalysisConfig().repolarization
    rc.per_beat_polarity = mode
    if threshold is not None:
        rc.per_beat_polarity_majority = threshold
    params, summary = P.extract_all_parameters(data, times, idx, FS, cfg=rc)
    assert 'template_repol_sign' in summary                     # the template found the wave
    return [p['repol_sign'] for p in params], summary


def test_default_is_majority_of_two_thirds():
    rc = AnalysisConfig().repolarization
    assert rc.per_beat_polarity == 'majority' and rc.per_beat_polarity_majority == pytest.approx(2 / 3, abs=0.01)


def test_free_mode_keeps_the_mixture(monkeypatch):
    signs, s = _run(monkeypatch, 'free', opposite={1, 4, 7})
    ts = s['template_repol_sign']
    assert sum(1 for x in signs if x == -ts) == 3 and s['repol_polarity_mode'] == 'free'


def test_majority_follows_the_template_when_most_beats_agree(monkeypatch):
    signs, s = _run(monkeypatch, 'majority', opposite={1, 4, 7, 10})            # 4 / 12 chose the other lobe
    ts = s['template_repol_sign']
    assert set(signs) == {ts} and s['repol_polarity_ref'] == ts
    assert s['repol_polarity_opposite_frac'] == pytest.approx(4 / 12)


def test_majority_follows_the_beats_when_two_thirds_disagree_with_the_template(monkeypatch):
    signs, s = _run(monkeypatch, 'majority', opposite=set(range(9)))           # 9 / 12 chose the other lobe
    ts = s['template_repol_sign']
    assert set(signs) == {-ts} and s['repol_polarity_ref'] == -ts
    signs, s = _run(monkeypatch, 'majority', opposite=set(range(9)), threshold=0.8)   # 0.75 < 0.8: template
    assert set(signs) == {ts}


def test_template_mode_and_the_legacy_flag(monkeypatch):
    signs, s = _run(monkeypatch, 'template', opposite=set(range(9)))
    assert set(signs) == {s['template_repol_sign']} and s['repol_polarity_mode'] == 'template'
    data, times, idx = _beats()
    monkeypatch.setattr(P, '_find_repolarization_per_beat', _fake_finder(set(range(9)), data))
    rc = AnalysisConfig().repolarization
    rc.per_beat_polarity, rc.per_beat_prefer_template_sign = 'free', True     # the v3.22 switch still works
    params, s = P.extract_all_parameters(data, times, idx, FS, cfg=rc)
    assert {p['repol_sign'] for p in params} == {s['template_repol_sign']} and s['repol_polarity_mode'] == 'template'
