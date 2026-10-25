"""Review model (review.py) and quality judgement (quality_summary.py), v3.17."""
import dataclasses
import json

import numpy as np
import pytest

from cardiac_fp_analyzer.config import AnalysisConfig, BeatDetectionConfig
from cardiac_fp_analyzer.detection_vetoes import enforce_min_rr
from cardiac_fp_analyzer.filtering import describe_filtering
from cardiac_fp_analyzer.quality_summary import summarise_quality
from cardiac_fp_analyzer.review import REVIEW_SUFFIX, BeatMark, ReviewState

FS = 2000.0


def _trace(dur_s=40.0, period_s=2.0, fpd_s=0.9, amp=1.0, noise=0.01, seed=0, pos_wave=0.3, neg_wave=0.0):
    """Beats with a sharp negative spike and a repolarisation wave at fpd_s (positive, optionally also a negative
    one 150 ms later). Returns filtered-like trace, raw trace (x10), beats, true repol samples."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    x = noise * rng.standard_normal(n)
    beats = np.arange(int(0.5 * FS), n - int(1.5 * FS), int(period_s * FS))
    reps = []
    for b in beats:
        tb = t - t[b]
        x += -amp * np.exp(-(tb / 0.003) ** 2 / 2) + 0.3 * amp * np.exp(-((tb - 0.008) / 0.004) ** 2 / 2)
        x += pos_wave * np.exp(-((tb - fpd_s) / 0.04) ** 2 / 2)
        if neg_wave:
            x += -neg_wave * np.exp(-((tb - fpd_s - 0.15) / 0.04) ** 2 / 2)
        reps.append(b + int(fpd_s * FS))
    return x, 10 * x, beats, np.array(reps)


def _state(**kw):
    x, raw, beats, reps = _trace(**kw)
    st = ReviewState(FS, x, raw, channel='el1', gain=1e4)
    st.beats = [BeatMark(int(b), int(r), 'auto', 'auto') for b, r in zip(beats, reps)]
    return st, beats, reps


def test_measures_follow_the_workbook_conventions():
    st, beats, reps = _state()
    m = st.measures()
    assert m['n_battiti'] == len(beats) and m['n_intervalli'] == len(beats) - 1
    assert m['bp_s'] == pytest.approx(2.0, abs=1e-6)
    assert m['bp_sd_s'] == pytest.approx(0.0, abs=1e-6)
    assert m['fpd_s'] == pytest.approx(0.9, abs=1e-3)
    assert m['fpdc_s'] == pytest.approx(0.9 / 2.0 ** (1 / 3), rel=1e-3)
    rows = st.per_beat_rows()
    assert rows[-1]['bp_s'] is None and rows[0]['bp_s'] == pytest.approx(2.0, abs=1e-6)
    # amplitude: peak-to-peak of the raw trace within ±10 ms, in µV; up/down back in oscilloscope volts
    assert rows[0]['amp_uV'] == pytest.approx((rows[0]['v_up'] - rows[0]['v_down']) / 1e4 * 1e6, rel=1e-9)
    assert m['amp_uV'] > 0


def test_measures_exclude_gaps_and_reference_variation():
    st, beats, reps = _state()
    del st.beats[5]                                   # a missing beat → one 4 s interval
    m = st.measures()
    assert m['intervalli_esclusi'] == 1 and m['bp_s'] == pytest.approx(2.0, abs=1e-6)
    m0 = st.measures(exclude_gaps=False)
    assert m0['bp_s'] > 2.0
    ref = {'bp_s': 1.6, 'fpd_s': 0.8, 'fpdc_s': 0.7, 'amp_uV': m['amp_uV'] / 2}
    mr = st.measures(reference=ref)
    assert mr['bp_s_pct'] == pytest.approx(125.0, rel=1e-6) and mr['bp_s_var'] == pytest.approx(25.0, rel=1e-6)
    assert mr['amp_uV_var'] == pytest.approx(100.0, rel=1e-6)


def test_add_remove_move_with_undo_redo():
    st, beats, reps = _state()
    n = st.n_beats()
    k = st.add_beat(int(beats[3] + 1.0 * FS))          # between two beats, on noise: snapped, repol proposed
    assert k is not None and st.n_beats() == n + 1 and st.beats[k].source == 'aggiunto'
    assert st.add_beat(int(beats[3] + 1.0 * FS)) is None      # too close to the one just added
    st.remove_beat(k)
    assert st.n_beats() == n
    assert st.undo() and st.n_beats() == n + 1
    assert st.undo() and st.n_beats() == n and not st.can_undo
    assert st.redo() and st.n_beats() == n + 1 and st.redo() and st.n_beats() == n
    assert not st.can_redo
    st.move_beat(0, int(beats[0]) + 12)                # snaps back onto the spike (within ±10 ms)
    assert abs(st.beats[0].dep - beats[0]) <= 2 and st.beats[0].source == 'corretto'


def test_set_repol_and_propagate_to_negative_wave():
    st, beats, reps = _state(neg_wave=0.3)
    # the operator moves one marker from the positive wave to the negative one 150 ms later and propagates
    rep = st.set_repol(4, int(reps[4] + 0.15 * FS), propagate=True)
    assert rep['criterio']['polarita'] == -1
    assert rep['spostati'] == len(beats) - 1 and rep['ambigui'] == 0
    lat = np.array([(b.repol - b.dep) / FS for b in st.beats])
    assert np.allclose(lat, 1.05, atol=0.02)
    assert {b.repol_source for b in st.beats} == {'corretto', 'propagato'}
    assert st.undo()
    assert all(b.repol_source == 'auto' for b in st.beats)


def test_propagation_flags_beats_without_a_wave():
    st, beats, reps = _state(neg_wave=0.0)              # no negative wave anywhere
    st.beats[2].repol = int(reps[2] + 0.15 * FS)        # pretend the operator put one on noise
    rep = st.propagate_from(2)
    assert rep['spostati'] == 0 and rep['criterio'] is None and 'motivo' in rep
    assert all(b.repol_source == 'auto' for i, b in enumerate(st.beats) if i != 2)


def test_set_polarity_and_vetoed_restore():
    st, beats, reps = _state(neg_wave=0.3)
    st.vetoed = [int(beats[2] + int(0.7 * FS))]
    rep = st.set_polarity(-1)
    assert rep['spostati'] == len(beats)
    k = st.restore_vetoed(st.vetoed[0] + 10)
    assert k is not None and st.beats[k].source == 'ripristinato' and not st.vetoed
    st.undo()
    assert st.vetoed and st.n_beats() == len(beats)


def test_sidecar_round_trip(tmp_path):
    st, beats, reps = _state()
    st.set_repol(1, int(reps[1]) + 3)
    st.notes = 'prova'
    rec = tmp_path / 'rec.csv'
    rec.write_text('x')
    p = st.save(rec)
    assert p.name == 'rec.csv' + REVIEW_SUFFIX
    other = ReviewState(FS, st.filtered, st.raw, channel='el2')
    other.beats = [BeatMark(10, None)]
    other.save(rec)                                  # second input, same file
    data = json.loads(p.read_text())
    assert set(data['ingressi']) == {'el1', 'el2'}
    fresh = ReviewState(FS, st.filtered, st.raw, channel='el1')
    assert ReviewState.load_into(fresh, rec)
    assert fresh.n_beats() == st.n_beats() and fresh.beats[1].repol_source == 'corretto' and fresh.notes == 'prova'
    assert fresh.beats[1].repol == st.beats[1].repol


def test_from_result_and_recompute(tmp_path):
    from cardiac_fp_analyzer.analyze import analyze_single_file
    x, raw, beats, reps = _trace(dur_s=60.0, amp=1.0, noise=0.02, pos_wave=0.3)
    t = np.arange(len(x)) / FS
    p = tmp_path / 'synthetic.csv'
    with open(p, 'w') as f:
        f.write('#Sample rate: 2000Hz\nTime (s),Channel 1 (V),Channel 2 (V)\n')
        for ti, xi in zip(t, raw * 1e-4):
            f.write(f'{ti:.6f},{xi:.9e},0\n')
    cfg = AnalysisConfig()
    r = analyze_single_file(str(p), channel='el1', verbose=False, config=cfg)
    st = ReviewState.from_result(r, cfg)
    assert st.n_beats() >= len(beats) - 2
    assert sum(b.repol is not None for b in st.beats) >= st.n_beats() // 2
    m = st.measures()
    assert m['bp_s'] == pytest.approx(2.0, rel=0.02)
    assert r['metadata']['processing']['testo'].startswith('Filtering:')
    st.remove_beat(0)
    new = st.recompute(r, cfg)
    assert len(new['beat_indices_raw']) == st.n_beats()
    q = summarise_quality(new)
    assert q['livello'] in ('verde', 'giallo', 'rosso') and q['titolo']


def test_enforce_min_rr_keeps_the_larger_spike():
    x = np.zeros(int(10 * FS))
    bi = np.array([2000, 2300, 6000, 6200, 6400])
    x[2000] = 1.0; x[2300] = 0.4; x[6000] = 0.5; x[6200] = 0.9; x[6400] = 0.3
    kept, info = enforce_min_rr(x, FS, bi, 250.0)
    assert list(kept) == [2000, 6200]
    assert info['n_removed'] == 3
    kept0, _ = enforce_min_rr(x, FS, bi, 0.0)
    assert np.array_equal(kept0, bi)


def test_describe_filtering_text():
    cfg = AnalysisConfig()
    d = describe_filtering(cfg, 2000.0, {'sample_mode': 'Average', 'decimation_factor': 5, 'original_sample_rate': 10000.0})
    assert 'notch 50/100/150 Hz' in d['testo'] and 'decimation ×5' in d['testo'] and 'Average' in d['testo']
    assert d['software'][1]['passo'] == 'passa-banda'


def test_quality_levels_on_synthetic_cases():
    base = {'metadata': {'sample_rate': FS}, 'file_info': {}, 'detection_info': {}, 'arrhythmia_report': None}
    good = dict(base, summary={'not_analysable': False, 'fpd_ms_median': 900.0, 'fpd_valid_ratio': 0.9, 'fpd_reliable': True,
                               'fpd_confidence': 0.9, 'beat_snr_median': 12.0},
                beat_indices_raw=np.arange(1000, 1000 + 30 * 4000, 4000))
    assert summarise_quality(good)['livello'] == 'verde'
    bad = dict(good, summary=dict(good['summary'], not_analysable=True,
                                  not_analysable_reason='median beat amplitude 1.20× the noise floor (< 1.6): no depolarisation pattern'))
    q = summarise_quality(bad)
    assert q['livello'] == 'rosso' and 'weak signal' in q['problemi'][0]['testo']
    gaps = dict(good, beat_indices_raw=np.array([1000, 5000, 9000, 17000, 21000, 29000, 33000, 37000, 45000, 49000, 53000, 57000]))
    q = summarise_quality(gaps)
    assert q['livello'] == 'giallo' and any(p['codice'] == 'battiti_mancanti' for p in q['problemi'])
    na_input = dict(good, file_info={'two_tissue_file': True, 'chamber': None})
    assert summarise_quality(na_input)['livello'] == 'rosso'


def test_review_default_config_flags_unchanged():
    assert BeatDetectionConfig().min_rr_physiological_ms == 0.0
    assert BeatDetectionConfig().min_distance_floor_ms == 100.0
    assert dataclasses.replace(BeatDetectionConfig(), min_rr_physiological_ms=250.0).min_rr_physiological_ms == 250.0


def test_removed_beats_are_tracked_and_saved(tmp_path):
    st, beats, reps = _state()
    n = st.n_beats()
    st.remove_beat(2)
    st.add_beat(int(beats[3]) + int(1.0 * FS))
    k = st.beat_near(int(beats[4]))
    st.move_beat(k, int(beats[4]) + 4)
    c = st.changes()
    assert c['tolti'] == 1 and c['aggiunti'] == 1 and c['spostati'] == 1 and c['corrected']
    st.undo(); st.undo(); st.undo()
    assert st.changes()['corrected'] is False and st.removed == [] and st.n_beats() == n
    st.redo()
    assert st.removed == [int(beats[2])]
    st.source_file = str(tmp_path / 'rec.csv')
    st.save()
    other, _, _ = _state()
    other.source_file = st.source_file
    assert ReviewState.load_into(other) and other.removed == [int(beats[2])] and other.n_beats() == n - 1


def test_problems_flag_gaps_close_beats_and_missing_repol():
    st, beats, reps = _state()
    assert st.problems() == []
    st.remove_beat(3)                                   # a gap of two periods
    st.clear_repol(5)                                   # a beat without repolarisation
    st.add_beat(int(beats[8]) + int(0.3 * FS))          # a detection 0.3 s after a beat
    probs = st.problems()
    kinds = [p['tipo'] for p in probs]
    assert kinds.count('battito_mancante') == 1 and kinds.count('ripolarizzazione_assente') >= 1 and kinds.count('battiti_vicini') == 1
    gap = next(p for p in probs if p['tipo'] == 'battito_mancante')
    assert gap['start'] == int(beats[2]) and gap['end'] == int(beats[4]) and gap['k'] == 2
    close = next(p for p in probs if p['tipo'] == 'battiti_vicini')
    assert abs(close['end'] - close['start'] - int(0.3 * FS)) <= int(0.025 * FS)      # snapped within ±25 ms
    assert probs == sorted(probs, key=lambda p: (p['start'], p['tipo']))


def test_approval_and_closure_are_saved_and_lock_the_corrections(tmp_path):
    from cardiac_fp_analyzer.review import RecordingClosedError
    st, beats, reps = _state()
    assert st.status == 'da_rivedere' and not st.approved and not st.locked and st.status_text() == 'to review'
    st.approve('Rossi, Mario')
    assert st.approved and st.approved_by == 'Rossi, Mario' and st.approved_at and 'Rossi' in st.status_text()
    st.remove_beat(0)                                   # a correction withdraws the approval
    assert st.status == 'da_rivedere'
    st.close('Rossi, Mario')                            # closing approves first
    assert st.locked and st.closed_at and st.status_text().startswith('closed')
    with pytest.raises(RecordingClosedError):
        st.remove_beat(0)
    with pytest.raises(RecordingClosedError):
        st.set_polarity(1)
    st.source_file = str(tmp_path / 'rec.csv')
    st.save()
    assert ReviewState.saved_status(st.source_file) == {'el1': 'chiuso'}
    other, _, _ = _state()
    other.source_file = st.source_file
    assert ReviewState.load_into(other) and other.locked and other.approved_by == 'Rossi, Mario' and other.n_beats() == len(beats) - 1
    other.reopen()
    assert other.approved and not other.locked and other.closed_at == ''
    other.remove_beat(0)
    other.unapprove()
    assert other.status == 'da_rivedere' and other.approved_by == ''
    other.close('Bianchi, Anna')                        # closing approves first
    assert other.approved_by == 'Bianchi, Anna' and other.locked


def test_a_correction_withdraws_the_approval():
    st, beats, reps = _state()
    st.approve('Rossi, Mario')
    st.remove_beat(0)
    assert st.status == 'da_rivedere' and st.approved_by == ''
    st.approve('Rossi, Mario')
    st.undo()                                           # undo is a correction too: approve again when done
    assert st.status == 'da_rivedere'


def test_click_snaps_to_the_highest_peak_of_the_displayed_trace():
    """A click up to ±25 ms off lands on the spike; with another trace (the operators' 1–40 Hz view) the marker
    goes on the peak of that trace, which is what the operator sees."""
    import numpy as np

    from pyside_app.review_helpers import operators_view
    st, beats, reps = _state(dur_s=20.0, amp=1.0, noise=0.02, pos_wave=0.3)
    raw = st.raw
    spike = int(beats[2])
    st.remove_beat(2)
    k = st.add_beat(spike + int(0.020 * FS))             # 20 ms late: still the spike
    assert abs(st.beats[k].dep - spike) <= 2
    st.remove_beat(k)
    ops = operators_view(raw, FS)
    k = st.add_beat(spike - int(0.015 * FS), trace=ops)   # on the 1–40 Hz view
    lo, hi = spike - int(0.025 * FS), spike + int(0.025 * FS)
    expected = lo + int(np.argmax(np.abs(ops[lo:hi + 1] - np.median(ops[lo:hi + 1]))))
    assert st.beats[k].dep == expected
    # a wrong-length trace is ignored, not an error
    st.move_beat(k, spike + 5, trace=ops[:-10])
    assert abs(st.beats[k].dep - spike) <= 2
