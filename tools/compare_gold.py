#!/usr/bin/env python
"""Compare the pipeline against the manual gold standard (GG).

Gold standard: one workbook per experiment, produced by the analyst:
  * sheet 'CHIP X'      — per tissue channel (Ch1/2/3), per concentration
                          (baseline + 3 doses): mean beating period, CV, FPD,
                          FPDc Fridericia, spike amplitude, free-text notes
                          ("STOP BEATING", "impossible to analyse", "n.a.").
  * sheet 'CHIP X raw'  — per beat: depolarisation time, beating period,
                          repolarisation time, FPD, voltages.

Recordings: WaveForms CSVs named <Exp>_<chipX_chN>[_<chipY_chM>]_<...>_<cond>
where chipX_chN is a TISSUE channel (1-3 within a chip), Channel 1 of the
oscilloscope is the first token and Channel 2 the second; <cond> is the
folder: baseline / A / B / C (= 1st / 2nd / 3rd dose in the gold order).

For every matched (recording, electrode):
  * beat-by-beat: missed / spurious vs the analyst's depolarisation times
    (after estimating the constant time offset between the two clocks);
  * beating period, FPD, FPDc, amplitude: pipeline vs analyst means;
  * analysability: analyst's "not analysable" vs pipeline grade / counts.

Writes <gold_dir>/comparison.xlsx (+ .csv) and prints a summary. Nothing in
the pipeline is tuned here.

Usage:
    python tools/compare_gold.py <gold_xlsx_dir> <recordings_dir>
"""

import contextlib
import io
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardiac_fp_analyzer import __version__  # noqa: E402
from cardiac_fp_analyzer.analyze import analyze_single_file  # noqa: E402
from cardiac_fp_analyzer.config import AnalysisConfig  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

NOT_ANALYSABLE = re.compile(r'stop beating|impossible|non riconoscibile|not analy', re.I)
RAW_COL_OFFSETS = (1, 10, 19)   # Dep. time column for Ch1 / Ch2 / Ch3 in 'raw' sheets


# ──────────────────────────────────────────────────────────────────────────
#   Gold-standard parsing
# ──────────────────────────────────────────────────────────────────────────

def _num(v):
    try:
        if v is None or (isinstance(v, str) and v.strip().startswith('#')):
            return None
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _ms(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(x * 1000)


def parse_summary_sheet(ws, exp):
    """Yield dicts: one per (chip, channel, dose_index)."""
    chip = None
    ch = None
    ch_label = None
    idx = -1
    for r in ws.iter_rows(values_only=True):
        a = r[0]
        if isinstance(a, str) and a.upper().startswith('CHIP'):
            chip = a.split()[-1].upper()
        if isinstance(a, str) and re.match(r'Ch\d', a):
            ch = int(a[2])
            ch_label = a
            idx = -1
            continue
        if ch is None or a in (None, ''):
            continue
        if isinstance(a, str) and a.startswith('Beating'):
            continue
        # a dose row (baseline or concentration): col B = BP s, D = CV, G = FPD s, J = FPDc s, M = amp µV, P = notes
        idx += 1
        note = r[15] if len(r) > 15 else None
        na_channel = bool(re.search(r'n\.?a\.?\)', ch_label or '', re.I))
        yield {
            'exp': f'Exp{exp}', 'chip': chip, 'ch': ch, 'dose_idx': idx,
            'dose_label': str(a).strip(), 'ch_label': ch_label,
            'gold_bp_s': _num(r[1]), 'gold_cv_pct': _num(r[3]), 'gold_fpd_s': _num(r[6]),
            'gold_fpdc_s': _num(r[9]), 'gold_amp_uV': _num(r[12]),
            'gold_note': (str(note).strip() if note else ''),
            'gold_not_analysable': bool(na_channel or (note and NOT_ANALYSABLE.search(str(note)))
                                         or (_num(r[1]) is None and _num(r[6]) is None)),
            'gold_channel_na': na_channel,
        }


def parse_raw_sheet(ws, exp, chip):
    """Return {(ch, dose_idx): {'dep_times': [...], 'fpd_s': [...]}}."""
    rows = list(ws.iter_rows(values_only=True))
    out = {}
    block_idx = -1
    i = 0
    while i < len(rows):
        r = rows[i]
        a = r[0]
        if isinstance(a, str) and a not in ('Mean', 'St. dev') and len(r) > 1 and r[1] == 'Dep. time s ' or (
                isinstance(a, str) and len(r) > 1 and isinstance(r[1], str) and r[1].startswith('Dep. time')):
            block_idx += 1
            j = i + 1
            while j < len(rows) and rows[j][0] not in ('Mean', 'St. dev') and not (
                    isinstance(rows[j][0], str) and j > i + 1 and len(rows[j]) > 1
                    and isinstance(rows[j][1], str) and rows[j][1].startswith('Dep. time')):
                j += 1
            for ch, c0 in enumerate(RAW_COL_OFFSETS, start=1):
                dep, fpd = [], []
                for k in range(i + 1, j):
                    rr = rows[k]
                    if len(rr) <= c0 + 3:
                        continue
                    t = _num(rr[c0])
                    if t is not None and t > 0:
                        dep.append(t)
                        f = _num(rr[c0 + 3])
                        fpd.append(f if f is not None and f > 0 else np.nan)
                out[(ch, block_idx)] = {'dep_times': np.array(dep), 'fpd_s': np.array(fpd)}
            i = j
        else:
            i += 1
    return out


def load_gold(gold_dir: Path):
    summary, raw = [], {}
    for f in sorted(gold_dir.glob('*.xlsx')):
        exp = re.search(r'Exp(\d+)', f.name).group(1)
        wb = openpyxl.load_workbook(f, data_only=True)
        for sn in wb.sheetnames:
            if sn == 'samples':
                continue
            if sn.endswith('raw'):
                chip = sn.split()[1].upper()
                for k, v in parse_raw_sheet(wb[sn], exp, chip).items():
                    raw[(f'Exp{exp}', chip, k[0], k[1])] = v
            else:
                summary.extend(parse_summary_sheet(wb[sn], exp))
    return pd.DataFrame(summary), raw


# ──────────────────────────────────────────────────────────────────────────
#   Recording → gold key
# ──────────────────────────────────────────────────────────────────────────

COND_IDX = {'baseline': 0, 'a': 1, 'b': 2, 'c': 3}


def recording_keys(csv: Path, base: Path):
    """Return [(electrode, key)] where key = (exp, chip, ch, dose_idx)."""
    parts = csv.relative_to(base).parts
    exp = next((p.replace(' ', '') for p in parts if re.match(r'Exp\s*\d+', p, re.I)), None)
    cond = parts[-2].lower() if len(parts) >= 2 else ''
    if cond not in COND_IDX:
        return []
    tokens = re.findall(r'chip([A-Za-z])_?ch(\d)', csv.stem, re.I)
    if not tokens or exp is None:
        return []
    with open(csv, errors='ignore') as fh:
        for _ in range(40):
            line = fh.readline()
            if line.startswith('Time'):
                phys = re.findall(r'Channel (\d)', line)
                break
        else:
            phys = ['1', '2']
    out = []
    if len(phys) == 1:
        # loader maps a single-column file to el1; it is the first token (or the only one)
        tok = tokens[0] if (phys[0] == '1' or len(tokens) == 1) else tokens[1]
        out.append(('el1', (exp, tok[0].upper(), int(tok[1]), COND_IDX[cond])))
    else:
        out.append(('el1', (exp, tokens[0][0].upper(), int(tokens[0][1]), COND_IDX[cond])))
        tok2 = tokens[1] if len(tokens) > 1 else tokens[0]
        out.append(('el2', (exp, tok2[0].upper(), int(tok2[1]), COND_IDX[cond])))
    return out


# ──────────────────────────────────────────────────────────────────────────
#   Beat-by-beat matching
# ──────────────────────────────────────────────────────────────────────────

def match_beats(mine_s, gold_s, tol_s=0.05):
    """Estimate constant offset (gold − mine), then count missed / spurious."""
    if len(mine_s) == 0 or len(gold_s) == 0:
        return {'offset_s': np.nan, 'missed': len(gold_s), 'spurious': len(mine_s), 'matched': 0}
    # offset = mode of pairwise nearest differences, searched on a coarse grid
    diffs = (gold_s[:, None] - mine_s[None, :]).ravel()
    diffs = diffs[np.abs(diffs) < 2.0]
    if len(diffs) == 0:
        return {'offset_s': np.nan, 'missed': len(gold_s), 'spurious': len(mine_s), 'matched': 0}
    hist, edges = np.histogram(diffs, bins=np.arange(-2.0, 2.0001, 0.01))
    off = edges[np.argmax(hist)] + 0.005
    near = diffs[np.abs(diffs - off) < 0.05]
    off = float(np.median(near)) if len(near) else off
    m = mine_s + off
    matched_g = sum(1 for g in gold_s if np.min(np.abs(m - g)) <= tol_s)
    matched_m = sum(1 for x in m if np.min(np.abs(gold_s - x)) <= tol_s)
    return {'offset_s': round(off, 3), 'missed': int(len(gold_s) - matched_g),
            'spurious': int(len(m) - matched_m), 'matched': int(matched_g)}


# ──────────────────────────────────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────────────────────────────────

DEV_EXPS = ('Exp5', 'Exp8', 'Exp10')
TEST_EXPS = ('Exp6', 'Exp7', 'Exp9')


def scorecard(df):
    """Compact summary used to compare pipeline versions on a split."""
    m = df[df.matched_gold == True].copy()  # noqa: E712
    na = m[m.gold_not_analysable == True]  # noqa: E712
    ok = m[m.gold_not_analysable == False].copy()  # noqa: E712
    ok['bp_err'] = (ok.sw_bp_median_ms - ok.gold_bp_ms).abs() / ok.gold_bp_ms * 100
    ok['fpd_err'] = (ok.sw_fpd_median_ms - ok.gold_fpd_ms).abs() / ok.gold_fpd_ms * 100
    b = ok[ok.gold_n_beats_raw.fillna(0) > 0]
    miss = b.beats_missed / b.gold_n_beats_raw * 100
    spur = b.beats_spurious / b.gold_n_beats_raw * 100
    sw_na = m.get('sw_not_analysable', pd.Series(False, index=m.index)).fillna(False).astype(bool)
    lines = [
        f"electrodes matched: {len(m)}  (gold not-analysable: {len(na)}, analysable: {len(ok)})",
        f"NOT-ANALYSABLE verdict: caught {int((sw_na & m.gold_not_analysable).sum())}/{len(na)}, "
        f"false alarms {int((sw_na & ~m.gold_not_analysable.astype(bool)).sum())}/{len(ok)} "
        f"(of which grade A: {int((sw_na & ~m.gold_not_analysable.astype(bool) & (m.sw_grade == 'A')).sum())})",
        f"BEATS (n={len(b)}): missed total {b.beats_missed.sum() / b.gold_n_beats_raw.sum() * 100:.1f} %, "
        f"spurious total {b.beats_spurious.sum() / b.gold_n_beats_raw.sum() * 100:.1f} %; "
        f"per-electrode median missed {miss.median():.0f} % / spurious {spur.median():.0f} %; "
        f"both ≤5 %: {int(((miss <= 5) & (spur <= 5)).sum())}/{len(b)}",
    ]
    rep = ok[~sw_na.reindex(ok.index).fillna(False).astype(bool)]   # electrodes the software reports on
    bp_r = rep[rep.sw_bp_median_ms.notna() & rep.gold_bp_ms.notna()]
    fpd_r = rep[rep.sw_fpd_median_ms.notna() & rep.gold_fpd_ms.notna()]
    lines += [
        f"COVERAGE: software reports on {len(rep)}/{len(ok)} analysable electrodes "
        f"({len(rep) / max(len(ok), 1) * 100:.0f} %); FPD given on {len(fpd_r)}",
        f"BP   (among reported, n={len(bp_r)}) within ±5 %: {(bp_r.bp_err <= 5).mean() * 100:.0f} %, "
        f"±10 %: {(bp_r.bp_err <= 10).mean() * 100:.0f} %  "
        f"(median signed err {((bp_r.sw_bp_median_ms - bp_r.gold_bp_ms) / bp_r.gold_bp_ms * 100).median():+.1f} %)",
        f"FPD  (among reported, n={len(fpd_r)}) within ±10 %: {(fpd_r.fpd_err <= 10).mean() * 100:.0f} %, "
        f"±20 %: {(fpd_r.fpd_err <= 20).mean() * 100:.0f} %  "
        f"(median signed err {((fpd_r.sw_fpd_median_ms - fpd_r.gold_fpd_ms) / fpd_r.gold_fpd_ms * 100).median():+.1f} %)",
        f"ALL analysable (unreported counted as wrong): BP±10 %: {(ok.bp_err <= 10).mean() * 100:.0f} %, "
        f"FPD±10 %: {(ok.fpd_err <= 10).mean() * 100:.0f} %",
    ]
    for grade in ('A', 'B', 'C', 'D', 'F'):
        g = ok[ok.sw_grade == grade]
        if len(g):
            lines.append(f"   grade {grade}: n={len(g):3d}  BP±10 %: {(g.bp_err <= 10).mean() * 100:3.0f} %  "
                         f"FPD±10 %: {(g.fpd_err <= 10).mean() * 100:3.0f} %  "
                         f"gold-NA with this grade: {int((na.sw_grade == grade).sum())}")
    return '\n'.join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('gold_dir')
    ap.add_argument('rec_dir')
    ap.add_argument('--split', choices=['dev', 'test', 'all'], default='dev',
                    help='dev = Exp5/8/10 (development), test = Exp6/7/9 (held out, run once at the end)')
    ap.add_argument('--tag', default='', help='label for the output files (e.g. pipeline version)')
    args = ap.parse_args()
    gold_dir = Path(args.gold_dir).resolve()
    rec_dir = Path(args.rec_dir).resolve()
    exps = {'dev': DEV_EXPS, 'test': TEST_EXPS, 'all': DEV_EXPS + TEST_EXPS}[args.split]
    gold, raw = load_gold(gold_dir)
    gold_by_key = {(r.exp, r.chip, r.ch, r.dose_idx): r for r in gold.itertuples(index=False)}
    print(f"gold: {len(gold)} (chip, channel, dose) entries, {len(raw)} raw beat blocks")

    cfg = AnalysisConfig()
    rows = []
    csvs = sorted(p for p in rec_dir.rglob('*.csv') if '_report' not in p.parts)
    n_unmatched = 0
    for csv in csvs:
        for el, key in recording_keys(csv, rec_dir):
            if key[0] not in exps:
                continue
            g = gold_by_key.get(key)
            if g is None:
                n_unmatched += 1
                rows.append({'recording_id': f'{csv.stem}__{el}', 'file': str(csv.relative_to(rec_dir)),
                             'electrode': el, 'gold_key': str(key), 'matched_gold': False})
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                r = analyze_single_file(str(csv), channel=el, verbose=False, config=cfg)
            s = r['summary']
            fs = r['metadata']['sample_rate']
            bi_raw = np.asarray(r['beat_indices_raw'])
            bi = np.asarray(r['beat_indices'])
            rr = np.diff(bi_raw) / fs if len(bi_raw) > 1 else np.array([])
            qc = r.get('qc_report')
            det = r.get('detection_info') or {}
            mf = det.get('matched_filter') or {}
            row = {
                'recording_id': f'{csv.stem}__{el}', 'file': str(csv.relative_to(rec_dir)), 'electrode': el,
                'gold_key': str(key), 'matched_gold': True,
                'exp': g.exp, 'chip': g.chip, 'ch': g.ch, 'dose_idx': g.dose_idx, 'dose_label': g.dose_label,
                'gold_note': g.gold_note, 'gold_not_analysable': g.gold_not_analysable,
                'gold_bp_ms': _ms(g.gold_bp_s),
                'gold_cv_pct': g.gold_cv_pct,
                'gold_fpd_ms': _ms(g.gold_fpd_s),
                'gold_fpdc_ms': _ms(g.gold_fpdc_s),
                'gold_amp_uV': g.gold_amp_uV,
                'sw_n_beats': int(len(bi_raw)), 'sw_n_qc': int(len(bi)),
                'sw_bp_mean_ms': round(float(np.mean(rr)) * 1000) if len(rr) else None,
                'sw_bp_median_ms': round(float(np.median(rr)) * 1000) if len(rr) else None,
                'sw_cv_pct': round(float(np.std(rr) / np.mean(rr) * 100), 1) if len(rr) > 1 else None,
                'sw_fpd_median_ms': None if s.get('fpd_ms_median') is None or np.isnan(s.get('fpd_ms_median', np.nan)) else round(s['fpd_ms_median']),
                'sw_fpd_mean_ms': None if np.isnan(s.get('fpd_ms_mean', np.nan)) else round(s['fpd_ms_mean']),
                'sw_fpdc_mean_ms': None if np.isnan(s.get('fpdc_fridericia_ms_mean', s.get('fpdc_ms_mean', np.nan))) else round(s.get('fpdc_fridericia_ms_mean', s.get('fpdc_ms_mean'))),
                'sw_fpd_measurable_pct': round((s.get('fpd_valid_ratio') or 0) * 100),
                'sw_fpd_reliable': bool(s.get('fpd_reliable', True)),
                'sw_amp_uV': None if np.isnan(s.get('spike_amplitude_mV_mean', np.nan)) else round(s['spike_amplitude_mV_mean'] * 1000, 1),
                'sw_grade': getattr(qc, 'grade', None),
                'sw_polarity': det.get('polarity'),
                'sw_matched_filter': mf.get('matched_filter'), 'sw_mf_candidates': mf.get('n_candidates'),
                'sw_class': getattr(r.get('arrhythmia_report'), 'classification', None),
                'sw_not_analysable': bool(s.get('not_analysable', getattr(qc, 'grade', None) == 'F')),
                'sw_na_reason': s.get('not_analysable_reason', ''),
            }
            rb = raw.get(key)
            if rb is not None and len(rb['dep_times']):
                mm = match_beats(bi_raw / fs, rb['dep_times'])
                row.update({'gold_n_beats_raw': int(len(rb['dep_times'])), 'beat_offset_s': mm['offset_s'],
                            'beats_missed': mm['missed'], 'beats_spurious': mm['spurious'], 'beats_matched': mm['matched']})
                gf = rb['fpd_s'][~np.isnan(rb['fpd_s'])]
                row['gold_fpd_median_raw_ms'] = round(float(np.median(gf)) * 1000) if len(gf) else None
                row['gold_fpd_measured_pct'] = round(len(gf) / len(rb['dep_times']) * 100)
            else:
                row.update({'gold_n_beats_raw': 0 if g.gold_not_analysable else None})
            rows.append(row)

    df = pd.DataFrame(rows)
    # Several recordings can map to one gold entry (repeat acquisitions:
    # _v2, _registrazione2, post_rec). The analyst's per-beat times belong
    # to ONE of them; for the others, position metrics are meaningless.
    # Keep beat-by-beat metrics only on the best-aligned recording per key
    # (most matched beats); BP/FPD comparisons stay valid for all.
    if 'beats_matched' in df.columns:
        df['position_ref'] = 'single'
        for key, g in df[df.matched_gold == True].groupby('gold_key'):  # noqa: E712
            if len(g) > 1:
                best = g.beats_matched.fillna(-1).idxmax()
                for i in g.index:
                    if i != best:
                        df.loc[i, 'position_ref'] = 'other recording (positions not compared)'
                        df.loc[i, ['beats_missed', 'beats_spurious', 'beats_matched', 'beat_offset_s']] = np.nan
                    else:
                        df.loc[i, 'position_ref'] = 'best of duplicates'
    out = gold_dir / 'comparison'
    out.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ''
    stem = f"comparison_{args.split}{tag}"
    df.to_csv(out / f'{stem}.csv', index=False)
    with pd.ExcelWriter(out / f'{stem}.xlsx', engine='xlsxwriter') as xw:
        df.to_excel(xw, sheet_name='per_electrode', index=False)
        gold.to_excel(xw, sheet_name='gold_parsed', index=False)
        xw.sheets['per_electrode'].freeze_panes(1, 1)
    card = scorecard(df)
    (out / f'{stem}_scorecard.txt').write_text(
        f"pipeline v{__version__} — split={args.split} ({', '.join(exps)}) — {out / stem}\n{card}\n")
    print(f"\n== SCORECARD  split={args.split} ({', '.join(exps)})  pipeline v{__version__}  unmatched={n_unmatched}")
    print(card)


if __name__ == '__main__':
    main()
