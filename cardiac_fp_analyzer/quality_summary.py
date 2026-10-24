"""One-glance quality judgement of an analysed recording (v3.17, Oct 2026) — no Qt.

``summarise_quality(result)`` turns the pieces the pipeline already produces
(analysability verdict and its reason, QC grade, FPD reliability, rhythm,
noise bursts, detections removed by the vetoes, missing beats, the input's
tissue, the chamber status of multi-electrode chips) into:

* ``livello``: ``'verde'`` (analysable, nothing to check), ``'giallo'``
  (analysable, something to check by eye), ``'rosso'`` (not analysable, or a
  problem that makes the measures unreliable);
* ``titolo``: a short Italian label;
* ``problemi``: a list of ``{'codice', 'gravita', 'testo', 'valore'}`` sorted
  by severity (``alto`` > ``medio`` > ``basso``);
* ``riassunto``: one line for tables and tooltips.

The review queue of the GUI sorts recordings by this judgement.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ['summarise_quality', 'LEVEL_COLOURS']

LEVEL_COLOURS = {'verde': '#1a7f37', 'giallo': '#d19a00', 'rosso': '#c0392b'}
_SEV = {'alto': 3, 'medio': 2, 'basso': 1}


def _reason_it(reason):
    """The pipeline's English verdict reasons in Italian, numbers kept."""
    r = reason or ''
    m = re.search(r'median beat amplitude ([\d.]+)× the noise floor', r)
    if m:
        return f'segnale debole: battiti {m.group(1)}× il rumore di fondo'
    m = re.search(r'only (\d+) depolarisations with RR CV (\d+) %', r)
    if m:
        return f'solo {m.group(1)} battiti con intervalli irregolari (CV {m.group(2)} %)'
    m = re.search(r'only (\d+) depolarisation', r)
    if m:
        return f'solo {m.group(1)} depolarizzazioni trovate'
    if 'irregular' in r:
        return 'ritmo irregolare nella camera'
    if 'conduction' in r:
        return 'conduzione persa tra gli elettrodi della camera'
    if 'silent' in r:
        return 'tessuto fermo'
    return r


def summarise_quality(result, *, burst_warn=0.10, cv_warn=25.0, min_beats=10, drift_warn=5.0):
    """See the module docstring. ``result`` is what ``analyze_single_file`` returns."""
    s = result.get('summary') or {}
    fi = result.get('file_info') or {}
    det = result.get('detection_info') or {}
    md = result.get('metadata') or {}
    fs = float(md.get('sample_rate') or 0) or None
    qc = result.get('qc_report')
    grade = getattr(qc, 'grade', None)
    bi = np.asarray(result.get('beat_indices_raw', []), dtype=int)
    n = int(len(bi))
    problems = []

    def add(code, sev, text, value=None):
        problems.append({'codice': code, 'gravita': sev, 'testo': text, 'valore': value})

    # input without tissue (two-tissue WaveForms file, 'na' in the name / samples sheet)
    if fi.get('two_tissue_file') and fi.get('chamber') in (None, '', 'na'):
        add('ingresso_senza_tessuto', 'alto', 'ingresso senza tessuto (na): da non analizzare')

    # analysability
    if s.get('not_analysable'):
        add('non_analizzabile', 'alto', 'non analizzabile: ' + _reason_it(s.get('not_analysable_reason')))
    # chamber status (multi-electrode chips)
    ch = result.get('chamber') or {}
    if ch.get('rhythm_status') in ('irregular', 'conduction_lost', 'silent'):
        add('camera', 'alto', 'camera: ' + _reason_it(ch['rhythm_status']), ch['rhythm_status'])

    # beats
    if 0 < n < min_beats:
        add('pochi_battiti', 'medio', f'pochi battiti ({n})', n)
    elif n == 0:
        add('nessun_battito', 'alto', 'nessun battito trovato')
    if fs and n >= 3:
        rr = np.diff(bi) / fs
        cv = float(np.std(rr) / np.mean(rr) * 100)
        med = float(np.median(rr))
        gaps = int((rr > 1.5 * med).sum())
        if gaps:
            add('battiti_mancanti', 'medio' if gaps >= 3 else 'basso',
                f'{gaps} intervall{"o" if gaps == 1 else "i"} con battito mancante', gaps)
        if cv > cv_warn and not s.get('not_analysable'):
            from .quality_control import common_period
            P, _, info = common_period(rr)
            if P is not None and info.get('cv_folded', 1e9) <= cv_warn:
                add('ritmo_con_buchi', 'basso', f'ritmo regolare ({P * 1000:.0f} ms) con {info["missing"]} battiti mancanti', P)
            else:
                add('ritmo_irregolare', 'medio', f'ritmo irregolare (CV {cv:.0f} %)', cv)
    # noise bursts
    vet = det.get('post_detection_vetoes') or {}
    windows = (vet.get('burst_veto') or {}).get('windows')
    if windows is None and fs and 'filtered_signal' in result:
        from .config import BeatDetectionConfig
        from .detection_vetoes import burst_windows
        windows = burst_windows(np.asarray(result['filtered_signal'], dtype=float), fs, BeatDetectionConfig())
    if windows:
        frac = sum(e - a for a, e in windows) / max(1, len(result.get('filtered_signal', [])))
        if frac >= burst_warn:
            add('burst', 'medio' if frac >= 0.3 else 'basso',
                f'burst di rumore (incubatore) nel {frac * 100:.0f} % del tempo', frac)
    # detections removed by the vetoes
    removed = sum(int((vet.get(k) or {}).get('n_removed') or 0) for k in ('weak_population_split', 'burst_veto', 'min_rr'))
    if removed:
        add('rilevamenti_tolti', 'medio' if removed >= max(5, 0.3 * (n + removed)) else 'basso',
            f'{removed} rilevamenti deboli tolti (ripristinabili)', removed)
    # weak signal (even when analysable)
    snr = s.get('beat_snr_median')
    if snr is None and fs and n and 'filtered_signal' in result:
        from .beat_detection import estimate_noise_floor
        x = np.asarray(result['filtered_signal'], dtype=float)
        nf = estimate_noise_floor(x, fs)
        half = max(1, int(0.02 * fs))
        if nf > 0:
            snr = float(np.median([np.ptp(x[max(0, i - half):i + half]) for i in bi]) / nf)
    if snr is not None and not s.get('not_analysable') and snr < 3.0:
        add('segnale_debole', 'medio', f'segnale debole: battiti {snr:.1f}× il rumore di fondo', snr)
    # FPD
    if not s.get('not_analysable') and n:
        fpd = s.get('fpd_ms_median')
        valid = s.get('fpd_valid_ratio')
        if fpd is None or not np.isfinite(fpd):
            add('fpd_non_misurabile', 'alto', 'onda di ripolarizzazione non trovata: FPD non misurabile')
        else:
            conf = s.get('fpd_confidence')
            if s.get('fpd_reliable') is False or (valid is not None and valid < 0.5):
                add('fpd_incerto', 'medio', f'FPD misurato solo sul {(valid or 0) * 100:.0f} % dei battiti', valid)
            elif conf is not None and np.isfinite(conf) and conf < 0.66:
                add('fpd_incerto', 'basso', f'FPD con confidenza bassa ({conf:.2f})', conf)
    # QC grade
    if grade in ('C', 'D') and not s.get('not_analysable'):
        add('qualita', 'basso' if grade == 'C' else 'medio', f'qualità del segnale {grade}', grade)
    # arrhythmia class
    ar = result.get('arrhythmia_report')
    cls = getattr(ar, 'classification', None)
    if cls and not s.get('not_analysable') and not str(cls).startswith(('Normal', 'Not analysable', 'Bradycardia', 'Borderline')):
        add('aritmia', 'medio' if 'Irregular' in str(cls) else 'basso', f'classificazione del ritmo: {cls}', cls)
    # very slow rhythm: a beat missed every other cycle leaves no visible gap
    if fs and n >= 3 and not s.get('not_analysable'):
        bp_s = float(np.median(np.diff(bi)) / fs)
        if bp_s > 3.0:
            add('ritmo_lento', 'basso', f'ritmo molto lento ({bp_s:.1f} s): controllare che non manchino battiti', bp_s)

    # drift of the rate within the recording (temperature, run-down): ≥ 10 % → to verify, 5–10 % → note
    d = s.get('drift') or {}
    if d.get('affidabile') and d.get('significativa') and d.get('variazione_pct') is not None and not s.get('not_analysable'):
        pct = float(d['variazione_pct'])
        if abs(pct) >= drift_warn:
            txt = (f"frequenza in deriva: {pct:+.0f} % in {d['durata_s']:.0f} s ({d['pendenza_bpm_min']:+.1f} bpm/min)"
                   + ('; ' + d['nota'] if d.get('nota') else ''))
            add('deriva', 'medio' if abs(pct) >= 2 * drift_warn else 'basso', txt, pct)
        f = d.get('fpd') or {}
        if f.get('affidabile') and f.get('significativa') and f.get('variazione_pct') is not None and abs(f['variazione_pct']) >= drift_warn:
            add('deriva_fpd', 'basso', f"FPD in deriva: {f['variazione_pct']:+.0f} % ({f['pendenza_ms_min']:+.0f} ms/min)",
                float(f['variazione_pct']))

    problems.sort(key=lambda p: -_SEV[p['gravita']])
    worst = max((_SEV[p['gravita']] for p in problems), default=0)
    level = 'rosso' if worst == 3 else 'giallo' if worst == 2 else 'verde'
    title = {'rosso': 'Problemi', 'giallo': 'Da verificare', 'verde': 'Buono'}[level]
    if level == 'verde' and any(p['gravita'] == 'basso' for p in problems):
        title = 'Buono (note)'
    summary = title + (': ' + '; '.join(p['testo'] for p in problems[:3]) if problems else '')
    return {'livello': level, 'colore': LEVEL_COLOURS[level], 'titolo': title, 'problemi': problems,
            'riassunto': summary, 'grado_qc': grade, 'n_battiti': n}
