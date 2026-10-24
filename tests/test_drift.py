"""Rate / FPD drift within a recording (cardiac_fp_analyzer.drift) and its integration."""
import numpy as np
import pytest

from cardiac_fp_analyzer.drift import drift_text, rate_drift
from tests.test_review_model import FS, _state


def _times(rate0, rate1, dur):
    t = [0.0]
    while t[-1] < dur:
        r = rate0 + (rate1 - rate0) * t[-1] / dur
        t.append(t[-1] + 60.0 / r)
    return np.array(t)


def test_rising_rate_and_falling_fpd():
    t = _times(30.0, 33.0, 60.0)
    d = rate_drift(t, fpd_s=1.0 - 0.002 * t)
    assert d['affidabile'] and d['significativa'] and d['n_intervalli'] == len(t) - 1
    assert d['bpm_inizio'] == pytest.approx(30.0, abs=0.1) and d['bpm_fine'] == pytest.approx(33.0, abs=0.2)
    assert d['pendenza_bpm_min'] == pytest.approx(3.0, abs=0.1) and d['variazione_pct'] == pytest.approx(10.0, abs=0.5)
    assert d['variazione_pct_min'] == pytest.approx(10.0, abs=0.5) and d['nota'] == ''
    f = d['fpd']
    assert f['significativa'] and f['pendenza_ms_min'] == pytest.approx(-120.0, abs=1) and f['variazione_pct'] == pytest.approx(-12.2, abs=0.3)
    txt = drift_text(d)
    assert txt.startswith('frequenza +9.9 % in 61 s') and 'FPD -12.2 %' in txt


def test_short_constant_and_outliers():
    t = np.arange(0, 20, 2.0)                          # 10 beats, 18 s, constant
    d = rate_drift(t)
    assert not d['affidabile'] and 'intervalli' in d['motivo'] and not d['significativa']
    assert d['pendenza_bpm_min'] == pytest.approx(0.0, abs=1e-9) and 'estrapolata' in d['nota']
    t = _times(30.0, 30.0, 70.0)
    t = np.delete(t, 10)                                # a missing beat
    t = np.sort(np.append(t, t[20] + 0.3))              # a detection in excess
    d = rate_drift(t)
    assert d['affidabile'] and d['n_intervalli'] == len(t) - 1 - 2 and abs(d['pendenza_bpm_min']) < 0.05
    assert rate_drift([0.0, 1.0])['motivo'] == 'troppo pochi battiti'


def test_review_state_and_measures_carry_the_drift():
    st, beats, reps = _state(dur_s=60.0)
    d = st.drift()
    assert d['affidabile'] and abs(d['pendenza_bpm_min']) < 0.1
    m = st.measures()
    assert m['drift_ok'] and m['drift_pct'] == pytest.approx(0.0, abs=0.5) and m['fpd_drift_pct'] == pytest.approx(0.0, abs=0.5)


def test_quality_summary_flags_a_drifting_rate():
    from cardiac_fp_analyzer.quality_summary import summarise_quality
    t = _times(30.0, 36.0, 60.0)
    drift = rate_drift(t)
    res = {'metadata': {'sample_rate': FS}, 'file_info': {'analyzed_channel': 'el1'}, 'detection_info': {},
           'arrhythmia_report': None,
           'summary': {'not_analysable': False, 'fpd_ms_median': 900.0, 'fpd_valid_ratio': 0.9, 'fpd_reliable': True,
                       'fpd_confidence': 0.9, 'beat_snr_median': 12.0, 'beat_period_ms_mean': 1900.0, 'drift': drift},
           'beat_indices_raw': (t * FS).astype(int)}
    q = summarise_quality(res)
    codes = {p['codice']: p for p in q['problemi']}
    assert 'deriva' in codes and codes['deriva']['gravita'] == 'medio' and 'frequenza in deriva: +19 %' in codes['deriva']['testo']
    assert q['livello'] == 'giallo'
