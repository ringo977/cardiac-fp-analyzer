#!/usr/bin/env python3
"""Extract the published ground truth from the Toxicol.Science result Excels.

Each ``Total Results Drugs Exp<n>_*.xlsx`` holds one sheet per microtissue,
named like ``Exp8_day6_chipD_Ch1_DOFETILIDE``. Inside, one row per drug
concentration with the values the authors measured:

    Spike Amplitude (µV) | RR interval (s) | Frequency (Hz)
    QT interval (s)      | QT Fridericia (s)

``QT interval`` is their FPD and ``QT Fridericia`` their FPDc, so these are
directly comparable with our ``fpd_ms_mean`` / ``fpdc_ms_mean``. The sheets
also record which side of the microtissue (``sx``/``dx``) the parameters
were taken from — which removes the guesswork our automatic channel
selection introduces — and free-text notes ("Arrh?", "Difficile
riconoscere") that match Roberta's handwritten annotations.

61 sheets across the workbooks; the paper reports n=51 of 60 microtissues.

The layout is irregular — merged headers, several blocks per sheet, notes
in arbitrary columns — so rather than assuming fixed positions this scans
for the row that names both "RR interval" and "QT interval" and reads the
block beneath it. Sheets where that row is absent are reported as skipped
rather than silently dropped.

Usage
-----
    python3 tools/extract_reference_excel.py \\
        --root "Experiments-Toxicol.Science" \\
        --out data_reference/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Workbooks holding the per-microtissue result sheets. Excel lock files
# (~$...) are skipped.
_BOOK_HINTS = ('Total Results', 'Results_Accelera', 'results_drug',
               'Accelera_iPSC')

_COL_PATTERNS = {
    # The concentration column is named inconsistently across workbooks:
    # "Dofetilide conc (nM)" in Exp8, but plain "TERFENADINE (uM)" or
    # "QUINIDINE (uM)" in Exp4/Exp5 — the drug name with a unit and no
    # "conc" at all. Match either shape, and fall back to position (see
    # _map_columns): it always sits immediately before Spike Amplitude.
    'conc': r'conc|\((?:n|u|µ)M\)',
    'spike_amplitude_uV': r'spike\s*ampl',
    'rr_s': r'RR\s*interval',
    'freq_hz': r'frequency',
    'qt_s': r'QT\s*interval',
    'qtc_fridericia_s': r'QT\s*fridericia',
}


def _norm(v):
    return re.sub(r'\s+', ' ', str(v)).strip()


def _find_header(df):
    """Row index whose cells name both the RR and the QT interval."""
    for i in range(min(len(df), 40)):
        cells = [_norm(c).lower() for c in df.iloc[i].tolist()]
        joined = ' | '.join(cells)
        if re.search(r'rr\s*interval', joined) and re.search(r'qt\s*interval', joined):
            return i
    return None


def _side_markers(df):
    """Locate the 'sx' / 'dx' block markers.

    These are NOT a per-sheet "which side was used" flag — an early
    version of this script read them that way and reported 'sx' for all
    67 microtissues. They are the headers of two parallel measurement
    blocks, one per side of the microtissue, at fixed columns (25 and 49
    in the sheets checked). Both sides were recorded; which one was
    actually used has to be inferred from which block is populated.

    Returns ``{'sx': col, 'dx': col}`` for whichever markers are found.
    """
    out = {}
    for i in range(min(len(df), 40)):
        for j, cell in enumerate(df.iloc[i].tolist()):
            c = _norm(cell).lower()
            if c in ('sx', 'dx') and c not in out:
                out[c] = j
    return out


def _map_columns(header_row, lo=0, hi=None):
    """Column index per logical field, restricted to ``[lo, hi)``."""
    hi = len(header_row) if hi is None else hi
    out = {}
    for j in range(lo, min(hi, len(header_row))):
        c = _norm(header_row[j]).lower()
        if not c or c == 'nan':
            continue
        for key, pat in _COL_PATTERNS.items():
            if key in out:
                continue
            if re.search(pat, c, re.I):
                out[key] = j
    # Positional fallback: in every layout seen, the concentration column
    # is the one immediately left of Spike Amplitude. More reliable than
    # guessing at drug names.
    if 'conc' not in out and 'spike_amplitude_uV' in out:
        j = out['spike_amplitude_uV'] - 1
        if j >= lo:
            out['conc'] = j
    return out


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def parse_sheet(df):
    """Return (sides, note) for one microtissue sheet.

    ``sides`` maps 'sx'/'dx' (or 'unknown' when no marker is present) to
    the list of per-concentration records for that block.
    """
    h = _find_header(df)
    if h is None:
        return None, 'nessuna riga di intestazione con RR+QT'
    header = df.iloc[h].tolist()

    markers = _side_markers(df)
    if markers:
        bounds = sorted(markers.items(), key=lambda kv: kv[1])
        spans = []
        for k, (name, start) in enumerate(bounds):
            end = bounds[k + 1][1] if k + 1 < len(bounds) else None
            spans.append((name, start, end))
    else:
        spans = [('unknown', 0, None)]

    sides = {}
    for name, lo, hi in spans:
        cols = _map_columns(header, lo, hi)
        if 'qt_s' not in cols or 'conc' not in cols:
            continue
        rows = _read_block(df, h, cols)
        if rows:
            sides[name] = rows
    if not sides:
        cols = _map_columns(header)
        if 'qt_s' not in cols or 'conc' not in cols:
            return None, f'colonne mancanti (trovate: {sorted(cols)})'
        rows = _read_block(df, h, cols)
        if not rows:
            return None, 'nessuna riga dati'
        sides['unknown'] = rows
    return sides, ''


def _read_block(df, h, cols):
    """Rows of one measurement block, starting below header row *h*."""
    rows = []
    for i in range(h + 1, len(df)):
        raw = df.iloc[i].tolist()
        conc = _num(raw[cols['conc']]) if cols['conc'] < len(raw) else None
        qt = _num(raw[cols['qt_s']]) if cols['qt_s'] < len(raw) else None
        if conc is None and qt is None:
            # two blank rows in a row end the block
            if rows and all(_norm(c) in ('', 'nan') for c in raw):
                break
            continue
        rec = {'conc': conc}
        for key, j in cols.items():
            if key == 'conc' or j >= len(raw):
                continue
            rec[key] = _num(raw[j])
        # free-text note anywhere on the row (Arrh?, Difficile riconoscere…)
        notes = [_norm(c) for c in raw
                 if _norm(c) and _num(c) is None and _norm(c).lower() != 'nan']
        rec['note'] = '; '.join(notes[:3]) if notes else ''
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    import pandas as pd

    books = [p for p in args.root.rglob('*.xls*')
             if not p.name.startswith('~$')
             and any(h in p.name for h in _BOOK_HINTS)]
    if not books:
        print('nessun workbook trovato', file=sys.stderr)
        return 1

    out, skipped = {}, []
    for b in sorted(books):
        try:
            xl = pd.ExcelFile(b)
        except Exception as exc:                    # noqa: BLE001
            skipped.append((str(b), '', f'apertura fallita: {exc}'))
            continue
        for sh in xl.sheet_names:
            try:
                df = pd.read_excel(b, sheet_name=sh, header=None)
            except Exception as exc:                # noqa: BLE001
                skipped.append((b.name, sh, str(exc)))
                continue
            sides, why = parse_sheet(df)
            if not sides:
                skipped.append((b.name, sh, why))
                continue
            out[f'{b.name}::{sh}'] = {
                'workbook': str(b.relative_to(args.root)),
                'sheet': sh,
                'sides': {k: {'n_points': len(v),
                              'n_qt_measured': sum(
                                  1 for r in v if r.get('qt_s') is not None),
                              'points': v}
                          for k, v in sides.items()},
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {'microtissues': out,
         'skipped': [{'workbook': w, 'sheet': s, 'reason': r}
                     for w, s, r in skipped]},
        indent=1, default=str))

    print(f'estratti  : {len(out)} microtessuti')
    print(f'saltati   : {len(skipped)}')
    both = sum(1 for v in out.values() if len(v['sides']) > 1)
    print(f'con 2 blocchi (sx+dx): {both}/{len(out)}')
    print(f'output    : {args.out}')
    if skipped:
        print('\nprimi saltati:')
        for w, s, r in skipped[:8]:
            print(f'  {w} :: {s[:34]:<36} {r[:46]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
