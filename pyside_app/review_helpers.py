"""Qt-free helpers of the Revisione screen (testable without PySide6).

* the operators' view of the trace (notch 50 Hz + Butterworth 1–40 Hz, order 1, zero phase);
* the recordings a file holds for the review queue (two inputs of a WaveForms CSV, the chambers of a chip);
* queue entries from an analysis result (quality judgement, measures) and their sort order;
* text formatting of the measures and of the per-beat rows with the labels of the lab's Results sheets.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import signal as sg

from cardiac_fp_analyzer.quality_summary import summarise_quality

LEVEL_ORDER = {'rosso': 0, 'giallo': 1, 'verde': 2}
LEVEL_LABEL = {'rosso': 'Problemi', 'giallo': 'Da verificare', 'verde': 'Buono'}

MEASURE_LABELS = (
    ('bp_s', 'Beating Period [s]', '{:.3f}'),
    ('bp_sd_s', 'St. dev. BP [s]', '{:.3f}'),
    ('bp_cv_pct', 'Coefficient of Variation - BP [%]', '{:.1f}'),
    ('fpd_s', 'Field Potential Duration [s]', '{:.3f}'),
    ('fpdc_s', "FPD Fridericia's Method [s]", '{:.3f}'),
    ('amp_uV', 'Dep. Amplitude [µV]', '{:.0f}'),
)

BEAT_COLUMNS = ('#', 'Dep. time [s]', 'Beating Period [s]', 'Rep. time [s]', 'FPD [s]', 'Dep. Voltage [µV]',
                'Depolarizzazione', 'Ripolarizzazione')


def operators_view(raw, fs):
    """The trace as the operators look at it: notch 50 Hz (Q 30) then Butterworth band-pass 1–40 Hz of order 1,
    both zero-phase. Returns an array of the same length."""
    x = np.asarray(raw, dtype=float)
    if len(x) < 32 or fs <= 100:
        return x.copy()
    b, a = sg.iirnotch(50.0, 30.0, fs=fs)
    y = sg.filtfilt(b, a, x)
    b, a = sg.butter(1, [1.0, 40.0], btype='band', fs=fs)
    return sg.filtfilt(b, a, y)


def inputs_of(path):
    """Recordings to review in one file: ``[('el1', None), ('el2', None)]`` for a WaveForms CSV (two inputs),
    ``[('auto', chamber), ...]`` for a multi-chamber chip (one per chamber, electrode chosen automatically),
    ``[('auto', None)]`` otherwise."""
    p = Path(path)
    if p.suffix.lower() == '.csv':
        return [('el1', None), ('el2', None)]
    try:
        from cardiac_fp_analyzer.chambers import layout_for
        from cardiac_fp_analyzer.loader import electrode_columns, is_recording_file, load_recording
        if not is_recording_file(str(p)):
            return [('auto', None)]
        metadata, df = load_recording(str(p))
        layout = layout_for(electrode_columns(df), 'auto')
        if layout is None:
            return [('auto', None)]
        return [('auto', c.name) for c in layout.chambers]
    except Exception:   # noqa: BLE001 — the queue must not die on one file
        return [('auto', None)]


def queue_entry(path, channel, chamber, result, error=None, record=None):
    """One row of the review queue from an analysis result (or an error). With ``record`` (the export
    record of the recording, ``results_export.record_from_state``) the measures shown are the review
    model's — the same the operator sees in the panel and that the Results export writes."""
    e = {'file': str(path), 'nome': Path(path).name, 'ingresso': channel, 'camera': chamber, 'errore': error,
         'livello': 'rosso', 'titolo': 'Errore', 'riassunto': error or '', 'problemi': [], 'bp_s': None, 'fpd_s': None,
         'n_battiti': 0, 'analizzato': None, 'rivista': False, 'record': record, 'condizione': ''}
    if result is None:
        return e
    q = summarise_quality(result)
    s = result.get('summary') or {}
    fi = result.get('file_info') or {}
    bp = s.get('beat_period_ms_mean'); fpd = s.get('fpd_ms_median')
    e.update(livello=q['livello'], titolo=q['titolo'], riassunto=q['riassunto'], problemi=q['problemi'],
             bp_s=bp / 1000 if bp is not None and np.isfinite(bp) else None,
             fpd_s=fpd / 1000 if fpd is not None and np.isfinite(fpd) else None,
             n_battiti=int(len(result.get('beat_indices_raw', []))), analizzato=fi.get('analyzed_channel'),
             chip=fi.get('chip'), camera=chamber or fi.get('chamber'))
    if record:
        m = record.get('measures') or {}
        e.update(bp_s=m.get('bp_s'), fpd_s=m.get('fpd_s'), n_battiti=m.get('n_battiti', e['n_battiti']),
                 chip=record.get('chip', e['chip']), camera=record.get('chamber', e['camera']),
                 item=record.get('item'), condizione=record.get('condition', ''))
    return e


def queue_sort_key(entry):
    """Problems first, then to-verify, then good; within a level by file name and input."""
    return (LEVEL_ORDER.get(entry.get('livello'), 0), entry.get('rivista', False), entry.get('nome', ''), str(entry.get('ingresso')))


def format_measures(m, reference=None):
    """Rows ``(label, value, pct, variation)`` with the Results labels; percentages only with a reference."""
    rows = []
    for key, label, fmt in MEASURE_LABELS:
        v = m.get(key)
        txt = fmt.format(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
        pct = m.get(key + '_pct'); var = m.get(key + '_var')
        rows.append((label, txt, f'{pct:.1f}' if pct is not None else '', f'{var:+.1f}' if var is not None else ''))
    extra = [('Battiti', f"{m.get('n_battiti', 0)}", '', ''),
             ('Intervalli esclusi dal BP', f"{m.get('intervalli_esclusi', 0)}", '', ''),
             ('Battiti con FPD', f"{m.get('n_fpd', 0)}", '', ''),
             ('Rilevamenti tolti dai veti', f"{m.get('n_vetoed', 0)}", '', ''),
             ('Burst di rumore (tempo)', f"{100 * m.get('burst_frac', 0):.0f} %", '', '')]
    return rows + extra


def format_beat_rows(rows):
    """Strings for the per-beat table (``BEAT_COLUMNS``)."""
    def f(v, fmt):
        return fmt.format(v) if v is not None else '—'
    out = []
    for r in rows:
        out.append((str(r['n']), f(r['dep_s'], '{:.3f}'), f(r['bp_s'], '{:.3f}'), f(r['rep_s'], '{:.3f}'),
                    f(r['fpd_s'], '{:.3f}'), f(r['amp_uV'], '{:.0f}'), r['stato'], r['stato_rip']))
    return out


def processing_text(result):
    md = (result or {}).get('metadata') or {}
    p = md.get('processing') or {}
    return p.get('testo', '')


__all__ = ['operators_view', 'inputs_of', 'queue_entry', 'queue_sort_key', 'format_measures', 'format_beat_rows',
           'processing_text', 'MEASURE_LABELS', 'BEAT_COLUMNS', 'LEVEL_ORDER', 'LEVEL_LABEL']
