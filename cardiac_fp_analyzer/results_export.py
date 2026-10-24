"""Export in the lab's Results workbook format: one sheet ``CHIP X`` per chip (the measures per channel and
condition, with the lab's column labels and formulas) and one sheet ``CHIP X raw`` (the per-beat rows), plus
a ``samples`` sheet listing every recording with its judgement and corrections.

A *record* is one recording of one tissue (one input of a two-tissue file, one chamber of a chip):
identity (experiment, day, chip, chamber, item, condition), the measures and per-beat rows of its
``ReviewState`` (software proposal plus the operator's corrections), the quality judgement, what was
corrected, and the processing line. Records come from the review queue of the GUI or from
``records_from_folder`` (every recording under a folder, analysed with the sidecar corrections applied).

Layout of ``CHIP X`` (as the lab writes it): A1 title, A3 ``DAY n``, A4 ``CHIP X``; per channel a block
``Chn (item)``, a header row, the baseline row and one row per dose two rows apart; % and variation
columns are Excel formulas against the baseline row, FPDc the Fridericia formula; column P a note
(corrections, problems), Q the file, R the input. ``CHIP X raw``: one group of seven columns per channel
(label column, Dep. time, Beating Period, Rep. time, FPD, Dep. Voltage up/down in volts at the 10k
amplifier, Dep. Voltage in µV), conditions one below the other, Beating Period / FPD / µV as formulas,
``Mean`` and ``St. dev`` rows per condition, the spare column after each group for per-beat notes.
"""
from __future__ import annotations

import logging
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from . import __version__
from .normalization import concentration_value
from .quality_summary import summarise_quality

logger = logging.getLogger(__name__)

BASELINE = 'baseline'

SUMMARY_HEADERS = (
    'Beating Period [s]', 'St. dev. BP [s]', 'Coefficient of Variation - BP [%]', 'Beating Period [%]',
    'Beating Period [%] - variation to baseline', 'Field Potential Duration [s]', 'Field Potential Duration [%]',
    'FPD [%]  - variation to baseline', "FPD Fridericia's Method [s]", "FPD Fridericia's Method [%]",
    "FPD Fridericia's Method [%]  - variation to baseline", 'Dep. Amplitude [µV]', 'Dep. Amplitude [%]',
    'Dep. Amplitude [%]  - variation to baseline', 'Note', 'File', 'Input')
RAW_HEADERS = ('Dep. time [s]', 'Beating Period [s]', 'Rep. time [s]', 'Field Potential Duration [s]',
               'Dep. Voltage up (10k amp) [V]', 'Dep. Voltage down (10k amp) [V]', 'Dep. Voltage [µV]')
RAW_GROUP_WIDTH = 9                     # label column + 7 data columns + 1 note column
_BEAT_NOTE = {'aggiunto': 'added by the operator', 'corretto': 'moved by the operator',
              'ripristinato': 'restored (removed by a veto)'}
_REPOL_NOTE = {'corretto': 'repolarisation set by the operator', 'propagato': 'repolarisation propagated',
               'ambiguo': 'repolarisation ambiguous', 'assente': 'no repolarisation'}


# ──────────────────────────────────────────────────────────────────────
#   Records
# ──────────────────────────────────────────────────────────────────────

def condition_label(identity):
    """'baseline', else the concentration / dose written in the name ('B', '0.01 µM'), else the item."""
    if identity.get('is_baseline') or str(identity.get('role') or '') == 'baseline':
        return BASELINE
    conc = identity.get('concentration')
    if conc not in (None, '', '0'):
        return str(conc)
    return str(identity.get('item') or identity.get('drug') or '?')


def condition_sort_key(label):
    if label == BASELINE:
        return (0, 0.0, '')
    v = concentration_value(label)
    return (1, v, '') if v is not None else (2, 0.0, str(label))


def _chamber_key(ch):
    try:
        return (0, int(ch), '')
    except (TypeError, ValueError):
        return (1, 0, str(ch))


def record_from_state(state, result, *, file, electrode, identity=None, quality=None):
    """One export record from a ``ReviewState`` (with the operator's corrections) and the analysis result."""
    fi = dict(result.get('file_info') or {})
    fi.update(identity or {})
    md = result.get('metadata') or {}
    q = quality if quality is not None else summarise_quality(result)
    ch = fi.get('chamber') if fi.get('chamber') not in (None, '') else fi.get('channel')
    rec = {'file': str(file), 'name': Path(file).name, 'electrode': str(electrode or fi.get('analyzed_channel') or ''),
           'experiment': fi.get('experiment'), 'day': fi.get('day'), 'chip': fi.get('chip'), 'chamber': ch,
           'item': fi.get('item') or (None if fi.get('is_baseline') else fi.get('drug')),
           'condition': condition_label(fi), 'is_baseline': bool(fi.get('is_baseline')),
           'measures': state.measures(), 'beats': state.per_beat_rows(), 'changes': state.changes(),
           'quality': {'livello': q.get('livello'), 'titolo': q.get('titolo'),
                       'problemi': [p.get('testo', '') for p in q.get('problemi', [])]},
           'not_analysable': bool((result.get('summary') or {}).get('not_analysable')),
           'processing': ((md.get('processing') or {}).get('testo') or ''), 'version': __version__}
    return rec


def sheet_root(folder, levels=3):
    """The folder whose ``samples.csv`` governs ``folder``: the highest ancestor (up to ``levels`` above)
    holding a sample sheet at its top level, else ``folder`` itself. The sheet of an experiment usually
    sits in its root while the operator opens a day or a dose folder."""
    from .sample_sheet import SHEET_NAME
    folder = Path(folder).resolve()
    root = folder
    cand = folder
    for _ in range(levels + 1):
        if any(p.name.lower() == SHEET_NAME for p in cand.iterdir() if p.is_file()):
            root = cand
        if cand.parent == cand:
            break
        cand = cand.parent
    return root


def plan_recordings(files, data_dir, config=None):
    """``sample_sheet.plan_batch`` for the given recordings with the sheet under ``data_dir`` honoured:
    one unit per tissue input (file, electrode, identity update)."""
    from .analyze import describe_recording
    from .sample_sheet import plan_batch
    # Path objects, absolute on both sides: plan_batch keys its tables by the objects it is given and
    # matches the sheet rows by position under the sheet's folder.
    files = [Path(f).resolve() for f in files if not str(f).endswith('.review.json')]
    infos = {f: describe_recording(str(f)) for f in files}
    units, plan = plan_batch(files, Path(data_dir).resolve(), infos, layout=getattr(config, 'chamber_layout', 'auto'))
    for u in units:
        u['file'] = str(u['file'])
    return units, plan


def records_from_folder(folder, config=None, *, progress=None, files=None):
    """Every recording under ``folder`` (or the given ``files``) analysed with ``config``, the saved
    corrections (``.review.json``) applied, one record per tissue input; the ``samples.csv`` of the
    experiment (``sheet_root``) decides the identities. ``progress(i, n, path)`` is called before each
    analysis. Recordings that fail are logged and skipped."""
    from .analyze import analyze_single_file, find_recordings
    from .config import AnalysisConfig
    from .review import ReviewState
    from .sample_sheet import is_sheet_file
    cfg = config or AnalysisConfig()
    folder = Path(folder)
    files = [Path(f) for f in files] if files is not None else find_recordings(folder, is_sheet_file)
    units, _plan = plan_recordings(files, sheet_root(folder), cfg)
    out = []
    for i, u in enumerate(units):
        if progress:
            progress(i, len(units), u['file'])
        try:
            result = analyze_single_file(u['file'], channel=u['electrode'], verbose=False, config=cfg,
                                         file_info_update=u['update'])
            st = ReviewState.from_result(result, cfg)
            ReviewState.load_into(st, u['file'])
            # 'auto' resolves to the electrode analysed (the operator's saved choice when there is one)
            actual = (result.get('file_info') or {}).get('analyzed_channel') or u['electrode']
            out.append(record_from_state(st, result, file=u['file'], electrode=actual, identity=u['update']))
        except Exception as exc:     # noqa: BLE001 — one bad file must not stop the export
            logger.exception('results export: %s (%s): %s', u['file'], u['electrode'], exc)
    return out


# ──────────────────────────────────────────────────────────────────────
#   Grouping
# ──────────────────────────────────────────────────────────────────────

_LEVEL_RANK = {'verde': 0, 'giallo': 1, 'rosso': 2}


def _preference(r):
    """Which of two recordings of the same tissue and condition goes into the chip sheet: one the operator
    corrected first, then the better judgement, then more beats, then the file name."""
    q = r.get('quality') or {}
    return (0 if (r.get('changes') or {}).get('corrected') else 1, _LEVEL_RANK.get(q.get('livello'), 3),
            -int((r.get('measures') or {}).get('n_battiti') or 0), r.get('name') or '')


def _sheet_groups(records):
    """Records grouped per sheet: key (experiment, day, chip) → {chamber → {condition → record}}, keys
    sorted. A second recording of the same tissue and condition (a repeat) stays out of the chip sheet
    (``_preference`` decides) and is named in the note of the one kept (``alternates``)."""
    groups = {}
    for r in records:
        key = (r.get('experiment'), r.get('day'), r.get('chip'))
        conds = groups.setdefault(key, {}).setdefault(r.get('chamber'), {})
        label = r['condition']
        if label in conds:
            keep, other = sorted((conds[label], r), key=_preference)
            keep.setdefault('alternates', []).extend(other.pop('alternates', []) + [other['name']])
            logger.info('results export: %s and %s are both %s of chip %s ch %s; %s kept',
                        conds[label]['name'], r['name'], label, r.get('chip'), r.get('chamber'), keep['name'])
            conds[label] = keep
        else:
            conds[label] = r
    return OrderedDict(sorted(groups.items(), key=lambda kv: tuple(str(x or '') for x in kv[0])))


def _sheet_names(groups):
    """'CHIP A' for a workbook of one experiment; with several, every sheet carries its experiment
    ('Exp5 CHIP A'), and the day when the same chip appears on more days ('Exp5 Day8 CHIP A'); ≤ 25 chars."""
    exps = {key[0] for key in groups}
    days_of = defaultdict(set)
    for exp, day, chip in groups:
        days_of[(exp, chip)].add(day)
    names = {}
    for key in groups:
        exp, day, chip = key
        base = f'CHIP {chip}' if chip not in (None, '') else 'CHIP ?'
        pre = []
        if len(exps) > 1 and exp:
            pre.append(str(exp).replace(' ', ''))
        if len(days_of[(exp, chip)]) > 1 and day:
            pre.append(str(day).replace(' ', ''))
        base = ' '.join(pre + [base])
        base = re.sub(r'[\[\]:*?/\\]', '-', base)[:25]
        n, cand = 1, base
        while cand in names.values():
            n += 1
            cand = f'{base} {n}'
        names[key] = cand
    return names


def _channel_title(chamber, conds):
    items = []
    for lab in sorted(conds, key=condition_sort_key):
        it = conds[lab].get('item')
        if it and str(it) not in items:
            items.append(str(it))
    ch = f'Ch{chamber}' if _chamber_key(chamber)[0] == 0 else f'Chamber {chamber}'
    return f'{ch} ({"/".join(items)})' if items else ch


def _note(rec):
    parts = []
    c = rec.get('changes') or {}
    if c.get('corrected'):
        bits = [f"{c[k]} {w}" for k, w in (('tolti', 'removed'), ('aggiunti', 'added'), ('spostati', 'moved'),
                                            ('ripristinati', 'restored')) if c.get(k)]
        rp = [f"{c[k]} {w}" for k, w in (('ripol_corrette', 'set'), ('ripol_propagate', 'propagated')) if c.get(k)]
        txt = 'corrected by the operator'
        if bits:
            txt += ': beats ' + ', '.join(bits)
        if rp:
            txt += ('; ' if bits else ': ') + 'repolarisations ' + ', '.join(rp)
        if c.get('polarita'):
            txt += f"; polarity {'+' if c['polarita'] > 0 else '−'}"
        parts.append(txt)
    else:
        parts.append('automatic')
    if rec.get('not_analysable'):
        parts.append('NOT ANALYSABLE')
    q = rec.get('quality') or {}
    if q.get('livello') in ('giallo', 'rosso') and q.get('problemi'):
        parts.append('; '.join(q['problemi'][:3]))
    m = rec.get('measures') or {}
    if m.get('intervalli_esclusi'):
        parts.append(f"{m['intervalli_esclusi']} interval(s) with a missing beat excluded from the beating period")
    if m.get('n_vetoed'):
        parts.append(f"{m['n_vetoed']} weak/burst detection(s) removed by the vetoes")
    if rec.get('alternates'):
        parts.append('other recording(s) of this tissue and condition not used: ' + ', '.join(rec['alternates']))
    return ' — '.join(parts)


# ──────────────────────────────────────────────────────────────────────
#   Workbook
# ──────────────────────────────────────────────────────────────────────

def _col(c):
    """0-based column index → Excel letters."""
    s = ''
    c += 1
    while c:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


def _num(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def write_results_workbook(records, path, title=None):
    """Write the workbook; returns the path. Sheets: ``samples``, then ``CHIP X`` and ``CHIP X raw`` per chip."""
    import xlsxwriter
    path = Path(path)
    title = title or path.stem
    groups = _sheet_groups(records)
    names = _sheet_names(groups)
    wb = xlsxwriter.Workbook(str(path), {'nan_inf_to_errors': True})
    F = {
        'title': wb.add_format({'bold': True, 'font_size': 12}),
        'bold': wb.add_format({'bold': True}),
        'hdr': wb.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1, 'bg_color': '#DDEBF7'}),
        'lab': wb.add_format({'bold': True, 'border': 1}),
        's3': wb.add_format({'num_format': '0.000', 'border': 1}),
        'p1': wb.add_format({'num_format': '0.0', 'border': 1}),
        'uv': wb.add_format({'num_format': '0', 'border': 1}),
        'v6': wb.add_format({'num_format': '0.000000', 'border': 1}),
        'txt': wb.add_format({'border': 1, 'text_wrap': True, 'valign': 'top'}),
        'txtw': wb.add_format({'text_wrap': True, 'valign': 'top'}),
        'grey': wb.add_format({'font_color': '#777777'}),
        'cell': wb.add_format({'border': 1}),
    }
    _write_samples_sheet(wb, F, records, title)
    for key, chambers in groups.items():
        sheet = names[key]
        _write_summary_sheet(wb, F, sheet, key, chambers, title)
        _write_raw_sheet(wb, F, sheet + ' raw', key, chambers, title)
    wb.close()
    return path


def _header_lines(ws, F, title, key, records):
    exp, day, chip = key
    ws.write(0, 0, title, F['title'])
    proc = next((r.get('processing') for r in records if r.get('processing')), '')
    ws.write(1, 0, f"Cardiac FP Analyzer {__version__}" + (f" · {proc}" if proc else ''), F['grey'])
    d = re.sub(r'(?i)^day\s*', '', str(day)) if day else None
    ws.write(2, 0, f'DAY {d}' if d else '', F['bold'])
    ws.write(3, 0, f'CHIP {chip}' if chip not in (None, '') else 'CHIP ?', F['bold'])


def _write_samples_sheet(wb, F, records, title):
    ws = wb.add_worksheet('samples')
    ws.write(0, 0, title, F['title'])
    ws.write(1, 0, f'Cardiac FP Analyzer {__version__} — one row per recording (tissue input)', F['grey'])
    heads = ('#exp', 'Day', 'CHIP', 'Ch', 'Item', 'Condition', 'Beats', 'Beating Period [s]', 'CV BP [%]',
             'FPD [s]', "FPDc Fridericia [s]", 'Dep. Amplitude [µV]', 'Judgement', 'Problems', 'Corrections',
             'File', 'Input')
    for j, h in enumerate(heads):
        ws.write(3, j, h, F['hdr'])
    rows = sorted(records, key=lambda r: (str(r.get('experiment') or ''), str(r.get('day') or ''),
                                          str(r.get('chip') or ''), _chamber_key(r.get('chamber')),
                                          condition_sort_key(r['condition'])))
    for i, r in enumerate(rows, start=4):
        m = r.get('measures') or {}
        q = r.get('quality') or {}
        c = r.get('changes') or {}
        vals = (r.get('experiment'), r.get('day'), r.get('chip'), r.get('chamber'), r.get('item'), r['condition'],
                m.get('n_battiti'), _num(m.get('bp_s')), _num(m.get('bp_cv_pct')), _num(m.get('fpd_s')),
                _num(m.get('fpdc_s')), _num(m.get('amp_uV')),
                {'verde': 'good', 'giallo': 'to verify', 'rosso': 'problems'}.get(q.get('livello'), q.get('livello')),
                '; '.join(q.get('problemi') or []),
                'yes' if c.get('corrected') else 'no', r.get('name'), r.get('electrode'))
        fmts = (None, None, None, None, None, None, None, F['s3'], F['p1'], F['s3'], F['s3'], F['uv'],
                None, F['txtw'], None, None, None)
        for j, (v, f) in enumerate(zip(vals, fmts)):
            if v is None:
                continue
            if f is None:
                ws.write(i, j, v)
            else:
                ws.write(i, j, v, f)
    ws.set_column(0, 5, 10)
    ws.set_column(6, 11, 14)
    ws.set_column(12, 12, 11)
    ws.set_column(13, 13, 60)
    ws.set_column(14, 14, 11)
    ws.set_column(15, 15, 48)
    ws.freeze_panes(4, 0)


def _write_summary_sheet(wb, F, name, key, chambers, title):
    ws = wb.add_worksheet(name)
    allrec = [r for conds in chambers.values() for r in conds.values()]
    _header_lines(ws, F, title, key, allrec)
    ws.set_column(0, 0, 14)
    ws.set_column(1, 14, 16)
    ws.set_column(15, 15, 60)
    ws.set_column(16, 16, 44)
    ws.set_column(17, 17, 8)
    row = 5
    for chamber in sorted(chambers, key=_chamber_key):
        conds = chambers[chamber]
        ws.write(row, 0, _channel_title(chamber, conds), F['bold'])
        ws.set_row(row + 1, 45)
        for j, h in enumerate(SUMMARY_HEADERS):
            ws.write(row + 1, 1 + j, h, F['hdr'])
        labels = sorted(conds, key=condition_sort_key)
        base_row = row + 2 if labels and labels[0] == BASELINE else None
        r = row + 2
        for lab in labels:
            rec = conds[lab]
            m = rec.get('measures') or {}
            ws.write(r, 0, lab, F['lab'])
            bp, sd, fpd, amp = _num(m.get('bp_s')), _num(m.get('bp_sd_s')), _num(m.get('fpd_s')), _num(m.get('amp_uV'))
            x = r + 1                                                   # Excel row number
            B, C, G, J, M = f'B{x}', f'C{x}', f'G{x}', f'J{x}', f'M{x}'
            if bp is not None:
                ws.write(r, 1, bp, F['s3'])
            if sd is not None:
                ws.write(r, 2, sd, F['s3'])
            if bp is not None and sd is not None:
                ws.write_formula(r, 3, f'={C}/{B}*100', F['p1'], _num(m.get('bp_cv_pct')))
            if fpd is not None:
                ws.write(r, 6, fpd, F['s3'])
            if fpd is not None and bp is not None:
                ws.write_formula(r, 9, f'={G}/({B}^(1/3))', F['s3'], _num(m.get('fpdc_s')))
            if amp is not None:
                ws.write(r, 12, amp, F['uv'])
            if base_row is not None:
                bx = base_row + 1
                bm = (conds[BASELINE].get('measures') or {})
                for col, cell, ref_cell, ok in ((4, B, f'$B${bx}', bp is not None and _num(bm.get('bp_s'))),
                                                (7, G, f'$G${bx}', fpd is not None and _num(bm.get('fpd_s'))),
                                                (10, J, f'$J${bx}', fpd is not None and bp is not None and _num(bm.get('fpdc_s'))),
                                                (13, M, f'$M${bx}', amp is not None and _num(bm.get('amp_uV')))):
                    if ok:
                        ws.write_formula(r, col, f'={cell}/{ref_cell}*100', F['p1'])
                        ws.write_formula(r, col + 1, f'={_col(col)}{x}-100', F['p1'])
            ws.write(r, 15, _note(rec), F['txt'])
            ws.write(r, 16, rec.get('name'), F['cell'])
            ws.write(r, 17, rec.get('electrode'), F['cell'])
            r += 2
        row = r + 2
    ws.freeze_panes(0, 1)


def _write_raw_sheet(wb, F, name, key, chambers, title):
    ws = wb.add_worksheet(name)
    allrec = [r for conds in chambers.values() for r in conds.values()]
    _header_lines(ws, F, title, key, allrec)
    order = sorted(chambers, key=_chamber_key)
    # conditions aligned by position across channels (baseline, 1st dose, 2nd dose, …)
    per_ch = {ch: [chambers[ch][lab] for lab in sorted(chambers[ch], key=condition_sort_key)] for ch in order}
    n_cond = max((len(v) for v in per_ch.values()), default=0)
    for g, ch in enumerate(order):
        c0 = g * RAW_GROUP_WIDTH
        ws.merge_range(4, c0 + 1, 4, c0 + 7, _channel_title(ch, chambers[ch]), F['bold'])
        ws.set_column(c0, c0, 11)
        ws.set_column(c0 + 1, c0 + 7, 15)
        ws.set_column(c0 + 8, c0 + 8, 30)
    row = 6
    for ci in range(n_cond):
        n_max = max((len(per_ch[ch][ci]['beats']) for ch in order if ci < len(per_ch[ch])), default=0)
        for g, ch in enumerate(order):
            if ci >= len(per_ch[ch]):
                continue
            rec = per_ch[ch][ci]
            c0 = g * RAW_GROUP_WIDTH
            ws.write(row, c0, rec['condition'], F['lab'])
            for j, h in enumerate(RAW_HEADERS):
                ws.write(row, c0 + 1 + j, h, F['hdr'])
            ws.write(row, c0 + 8, 'Note', F['hdr'])
            beats = rec['beats']
            first = row + 1
            for i, b in enumerate(beats):
                r = first + i
                x = r + 1
                dep, up, dn = _col(c0 + 1), _col(c0 + 5), _col(c0 + 6)
                ws.write(r, c0 + 1, _num(b['dep_s']), F['s3'])
                if i + 1 < len(beats):
                    ws.write_formula(r, c0 + 2, f'={dep}{x + 1}-{dep}{x}', F['s3'], _num(b.get('bp_s')))
                if b.get('rep_s') is not None:
                    ws.write(r, c0 + 3, _num(b['rep_s']), F['s3'])
                    ws.write_formula(r, c0 + 4, f'={_col(c0 + 3)}{x}-{dep}{x}', F['s3'], _num(b.get('fpd_s')))
                ws.write(r, c0 + 5, _num(b.get('v_up')), F['v6'])
                ws.write(r, c0 + 6, _num(b.get('v_down')), F['v6'])
                ws.write_formula(r, c0 + 7, f'=({up}{x}-{dn}{x})*100', F['uv'], _num(b.get('amp_uV')))
                notes = [t for t in (_BEAT_NOTE.get(b.get('stato')), _REPOL_NOTE.get(b.get('stato_rip'))) if t]
                if notes:
                    ws.write(r, c0 + 8, '; '.join(notes), F['grey'])
            if beats:
                last = first + len(beats) - 1
                mean_r = first + n_max + 1
                ws.write(mean_r, c0, 'Mean', F['bold'])
                ws.write(mean_r + 1, c0, 'St. dev', F['bold'])
                for col, fmt in ((c0 + 2, F['s3']), (c0 + 4, F['s3']), (c0 + 7, F['uv'])):
                    rng = f'{_col(col)}{first + 1}:{_col(col)}{last + 1}'
                    ws.write_formula(mean_r, col, f'=IFERROR(AVERAGE({rng}),"")', fmt)
                    ws.write_formula(mean_r + 1, col, f'=IFERROR(STDEV({rng}),"")', fmt)
        row += n_max + 1 + 1 + 2 + 3        # header + beats + blank + mean/sd + gap
    ws.freeze_panes(6, 0)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Export recordings in the lab\'s Results workbook format')
    ap.add_argument('folder', help='folder with the recordings (samples.csv honoured, .review.json applied)')
    ap.add_argument('-o', '--output', help='workbook path (default: <folder>_FP_analysis.xlsx next to the folder)')
    ap.add_argument('--config', help='analysis configuration JSON')
    a = ap.parse_args(argv)
    from .config import AnalysisConfig
    cfg = AnalysisConfig.from_json(a.config) if a.config else AnalysisConfig()
    folder = Path(a.folder)
    out = Path(a.output) if a.output else folder.parent / f'{folder.name.replace(" ", "")}_FP_analysis.xlsx'
    recs = records_from_folder(folder, cfg, progress=lambda i, n, f: print(f'  [{i + 1}/{n}] {Path(f).name}'))
    write_results_workbook(recs, out)
    print(f'{len(recs)} recordings → {out}')
    return 0


if __name__ == '__main__':      # pragma: no cover
    raise SystemExit(main())
