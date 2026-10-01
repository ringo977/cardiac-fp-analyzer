#!/usr/bin/env python
"""Blind analysis report for a folder of WaveForms CSV recordings.

Runs the full pipeline on EVERY electrode of every CSV (many recordings
hold two different microtissues on Channel 1 / Channel 2, so "auto"
channel selection would silently drop one of them) and writes:

  <folder>/_report/blind_report.xlsx   one row per (file, electrode) + per-file sheet + README
  <folder>/_report/blind_report.csv    same rows, plain CSV
  <folder>/_report/figures/<id>.png    signal with beats marked (accepted / rejected / FPD end)
  <folder>/_report/run.log             progress and errors

Intended use: hand this to whoever produces the manual gold standard,
then compare. Nothing in the folder other than the CSVs is read.

Usage:
    python tools/blind_report.py "Experiments-GG" [--limit N] [--no-figures]
"""

import argparse
import contextlib
import io
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardiac_fp_analyzer import __version__  # noqa: E402
from cardiac_fp_analyzer.analyze import analyze_single_file  # noqa: E402
from cardiac_fp_analyzer.config import AnalysisConfig  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)


# ──────────────────────────────────────────────────────────────────────────
#   File-name / path parsing (this dataset's conventions)
# ──────────────────────────────────────────────────────────────────────────

def parse_path(csv: Path, base: Path):
    rel = csv.relative_to(base)
    parts = rel.parts
    exp = next((p for p in parts if re.match(r'Exp\s*\d+', p, re.I)), '')
    day = next((p for p in parts if re.match(r'Day\s*\d+', p, re.I)), '')
    cond = parts[-2] if len(parts) >= 2 else ''
    if re.match(r'Exp\s*\d+', cond, re.I):
        cond = '(root)'
    name = csv.stem
    # chip/channel tokens in order of appearance: chipB_ch3dx, chipF_ch1, ...
    tissues = [f"chip{c.upper()}_ch{n}{(s or '').lower()}"
               for c, n, s in re.findall(r'chip([A-Za-z])_?ch(\d)(sx|dx)?', name, re.I)]
    compounds = sorted(set(t.upper() for t in re.findall(r'(Ti\d{2})', name, re.I)))
    flags = [w for w in ('vehicle', 'post_rec', 'elsopra', 'baseline') if w in name.lower()]
    return {
        'exp': exp.replace(' ', ''), 'day': day.replace(' ', ''), 'condition': cond,
        'tissues_in_name': ' | '.join(tissues), 'n_tissue_tokens': len(tissues),
        'compounds': ' | '.join(compounds), 'flags': ' | '.join(flags),
    }


def physical_channels(csv: Path):
    """Which oscilloscope channels are in the file, in column order."""
    with open(csv, errors='ignore') as fh:
        for _ in range(40):
            line = fh.readline()
            if line.startswith('Time'):
                return re.findall(r'Channel (\d)', line)
    return []


# ──────────────────────────────────────────────────────────────────────────
#   One electrode → one row
# ──────────────────────────────────────────────────────────────────────────

def _f(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _quiet_analyze(csv, el, cfg):
    """analyze_single_file still print()s its trace even with verbose=False."""
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze_single_file(str(csv), channel=el, verbose=False, config=cfg)


def analyze_electrode(csv: Path, el: str, cfg: AnalysisConfig):
    r = _quiet_analyze(csv, el, cfg)
    s = r['summary']
    qc = r.get('qc_report')
    ar = r.get('arrhythmia_report')
    det = r.get('detection_info', {}) or {}
    fs = r['metadata']['sample_rate']
    bi_raw = np.asarray(r.get('beat_indices_raw', []))
    bi = np.asarray(r.get('beat_indices', []))
    rr_raw = np.diff(bi_raw) / fs * 1000 if len(bi_raw) > 1 else np.array([])
    dur = len(r['filtered_signal']) / fs
    mfd = det.get('matched_filter') or {}
    mf = mfd.get('matched_filter')
    gate = (det.get('noise_floor_gate') or {}).get('n_rejected_total')
    rc = (det.get('rhythm_classification') or {}).get('rhythm_type')
    cess = r.get('cessation_report')
    row = {
        'duration_s': _f(dur, 1), 'fs_hz': fs,
        'n_beats_detected': int(len(bi_raw)), 'n_beats_qc_accepted': int(len(bi)),
        'qc_rejected_pct': _f((1 - len(bi) / len(bi_raw)) * 100 if len(bi_raw) else None, 0),
        'beat_rate_bpm': _f(len(bi_raw) / dur * 60 if dur else None, 1),
        'rr_median_ms': _f(np.median(rr_raw) if len(rr_raw) else None, 0),
        'rr_mean_ms': _f(np.mean(rr_raw) if len(rr_raw) else None, 0),
        'rr_cv_pct': _f(np.std(rr_raw) / np.mean(rr_raw) * 100 if len(rr_raw) > 1 else None, 1),
        'stv_ms': _f(s.get('stv_ms'), 1),
        'fpd_median_ms': _f(s.get('fpd_ms_median'), 0), 'fpd_mean_ms': _f(s.get('fpd_ms_mean'), 0),
        'fpd_sd_ms': _f(s.get('fpd_ms_std'), 0),
        'fpd_n_measured': int(s.get('fpd_ms_n', 0) or 0),
        'fpd_measurable_pct': _f((s.get('fpd_valid_ratio') or 0) * 100, 0),
        'fpd_reliable': bool(s.get('fpd_reliable', True)),
        'fpd_confidence': _f(s.get('fpd_confidence'), 2),
        'fpd_method': s.get('fpd_method'),
        'fpdcf_median_ms': _f(s.get('fpdc_fridericia_ms_median', s.get('fpdc_ms_median')), 0),
        'fpdcf_mean_ms': _f(s.get('fpdc_fridericia_ms_mean', s.get('fpdc_ms_mean')), 0),
        'fpdcb_mean_ms': _f(s.get('fpdc_bazett_ms_mean'), 0),
        'spike_amplitude_uV': _f((s.get('spike_amplitude_mV_mean') or np.nan) * 1000, 0),
        'not_analysable': bool(s.get('not_analysable', False)),
        'not_analysable_reason': s.get('not_analysable_reason', ''),
        'qc_grade': getattr(qc, 'grade', None),
        'qc_global_snr': _f(getattr(qc, 'global_snr', None), 1),
        'detector_polarity': det.get('polarity'),
        'rhythm_type': rc,
        'noise_gate_rejected': gate,
        'matched_filter': mf,
        'mf_median_snr': _f(mfd.get('median_snr'), 2),
        'mf_n_candidates': mfd.get('n_candidates'),
        'mf_count_ratio': _f(mfd.get('count_ratio'), 2),
        'arrhythmia_class': getattr(ar, 'classification', None),
        'risk_score': getattr(ar, 'risk_score', None),
        'flags': '; '.join(f"{f['type']}" for f in getattr(ar, 'flags', [])[:6]),
        'cessation': bool(getattr(cess, 'has_cessation', False)) if cess is not None else None,
    }
    return row, r


# ──────────────────────────────────────────────────────────────────────────
#   Figure
# ──────────────────────────────────────────────────────────────────────────

def make_figure(r, title, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs = r['metadata']['sample_rate']
    x = r['filtered_signal'] * 1e6  # µV
    t = np.arange(len(x)) / fs
    bi_raw = np.asarray(r.get('beat_indices_raw', []))
    bi = set(np.asarray(r.get('beat_indices', [])).tolist())
    s = r['summary']
    fpd_ms = s.get('fpd_ms_median')
    fig, ax = plt.subplots(2, 1, figsize=(18, 7.5), gridspec_kw={'height_ratios': [1, 1.3]})
    for a, (t0, t1) in zip(ax, [(0, t[-1]), (max(0, t[-1] / 2 - 5), min(t[-1], t[-1] / 2 + 5))]):
        m = (t >= t0) & (t <= t1)
        a.plot(t[m], x[m], lw=0.5, color='k')
        for b in bi_raw:
            tb = b / fs
            if t0 <= tb <= t1:
                a.axvline(tb, color='tab:blue' if b in bi else 'tab:red', alpha=0.55, lw=1)
        a.set_xlim(t0, t1)
        a.set_ylabel('µV')
        a.grid(alpha=0.25)
    # FPD end markers on the zoom panel (accepted beats only)
    if fpd_ms is not None and not np.isnan(fpd_ms):
        t0, t1 = ax[1].get_xlim()
        for b in bi_raw:
            tb = b / fs
            if b in bi and t0 <= tb <= t1:
                ax[1].axvline(tb + fpd_ms / 1000, color='tab:green', alpha=0.5, lw=1, ls=':')
    ax[0].set_title(title, fontsize=10, loc='left')
    ax[1].set_title('zoom 10 s — blue: accepted beat · red: rejected by QC · green dotted: beat + median FPD',
                    fontsize=9, loc='left')
    ax[1].set_xlabel('s')
    fig.tight_layout()
    fig.savefig(out_png, dpi=72)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────────────────────────────────

README = """Blind analysis report — cardiac_fp_analyzer v{ver}

One row per (file, electrode). Files whose name lists two chip/channel tokens
hold two different microtissues (Channel 1 = first token, Channel 2 = second);
files with one token have both electrodes on the same tissue — both are still
reported, 'auto_pick' says which one the automatic channel selector would use.

Columns
  recording_id            file stem + electrode (also the figure file name)
  electrode / physical_channel   el1/el2 as analysed; oscilloscope channel it maps to
  tissue_guess            chip/channel token assigned to this electrode from the name
  exp, day, condition     folder structure (condition: baseline / A / B / C)
  compounds, flags        tokens found in the file name (TiNN codes, vehicle, ...)
  n_beats_detected        beats in the raw detection train (used for RR / CV / rate)
  n_beats_qc_accepted     beats surviving quality control (used for FPD)
  beat_rate_bpm           n_beats_detected / duration
  rr_median_ms, rr_mean_ms, rr_cv_pct   from the raw detection train
  stv_ms                  short-term variability of RR
  fpd_median_ms, fpd_mean_ms, fpd_sd_ms  field potential duration (≈ QT), ms
  fpd_n_measured, fpd_measurable_pct     beats with a measurable repolarisation
  fpd_reliable            False when < 50 % of accepted beats had a measurable FPD
  fpd_confidence          template-level repolarisation confidence (0–1)
  fpdcf_*                 Fridericia-corrected FPD (FPD / RR^(1/3)); fpdcb = Bazett
  spike_amplitude_uV      mean depolarisation spike amplitude (gain 1e4 applied)
  not_analysable          True = the pipeline declares it cannot analyse this electrode
                          (no depolarisation pattern above noise, or too sparse and
                          irregular); FPD/FPDc are withheld, grade is F. Beat counts kept.
  not_analysable_reason   why
  qc_grade                A (excellent) … F (not analysable)
  detector_polarity       positive / negative / mixed
  rhythm_type             rhythm topology classifier output
  noise_gate_rejected     detections removed as noise-level
  matched_filter          'applied' = low-SNR regime, beats re-detected by template correlation;
                          'rejected_count_ratio' = low-SNR regime but the correlation detector
                          disagreed with the derivative detector by more than 0.7–1.5× and was
                          NOT used (derivative result kept). For these, mf_n_candidates is the
                          alternative beat count — the gold standard will say which was right.
  mf_median_snr           median beat amplitude / noise floor (regime < 3.0 = low-SNR)
  mf_n_candidates, mf_count_ratio   matched-filter beat count and its ratio to the derivative count
  arrhythmia_class, risk_score, flags    arrhythmia module output
  cessation               cessation detector verdict
  error                   non-empty if the pipeline failed on this electrode

Values the gold standard is expected to provide for comparison:
  which electrode was used, RR (or beat count in a stated window), FPD,
  and 'not measurable' where applicable.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()
    base = Path(args.folder).resolve()
    out = base / '_report'
    figdir = out / 'figures'
    out.mkdir(exist_ok=True)
    figdir.mkdir(exist_ok=True)
    log_path = out / 'run.log'

    def say(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as log:
            log.write(msg + '\n')

    files = sorted(p for p in base.rglob('*.csv') if '_report' not in p.parts)
    if args.limit:
        files = files[:args.limit]
    say(f"# blind_report v{__version__} — {len(files)} CSV in {base.name} — {time.strftime('%Y-%m-%d %H:%M')}")
    cfg = AnalysisConfig()
    rows = []
    t_start = time.time()
    for k, csv in enumerate(files, 1):
        meta = parse_path(csv, base)
        phys = physical_channels(csv)
        tokens = meta['tissues_in_name'].split(' | ') if meta['tissues_in_name'] else []
        # loader maps a single-column file to el1 whatever the physical channel
        electrodes = [('el1', phys[0] if phys else '1')] if len(phys) == 1 else [('el1', '1'), ('el2', '2')]
        auto_pick = None
        if len(electrodes) == 2 and len(tokens) <= 1:
            try:
                ra = _quiet_analyze(csv, 'auto', cfg)
                auto_pick = ra['file_info'].get('analyzed_channel')
            except Exception as e:  # noqa: BLE001 — report must not stop
                auto_pick = f'error: {type(e).__name__}'
        for el, pch in electrodes:
            rid = f"{csv.stem}__{el}"
            if len(tokens) >= 2:
                tissue = tokens[0] if pch == '1' else tokens[1]
            else:
                tissue = tokens[0] if tokens else ''
            row = {'recording_id': rid, 'file': str(csv.relative_to(base)), 'electrode': el,
                   'physical_channel': f'Channel {pch}', 'tissue_guess': tissue,
                   'auto_pick': auto_pick if len(tokens) <= 1 else 'n/a (two tissues)', **meta}
            try:
                metrics, r = analyze_electrode(csv, el, cfg)
                row.update(metrics)
                row['error'] = ''
                if not args.no_figures:
                    title = (f"{rid}   [{meta['exp']} {meta['day']} {meta['condition']}]  tissue={tissue}  "
                             f"beats={metrics['n_beats_detected']} (QC {metrics['n_beats_qc_accepted']})  "
                             f"RR={metrics['rr_median_ms']} ms  CV={metrics['rr_cv_pct']} %  "
                             f"FPD={metrics['fpd_median_ms']} ms ({metrics['fpd_measurable_pct']} % measurable)  "
                             f"grade={metrics['qc_grade']}  {metrics['arrhythmia_class']}")
                    make_figure(r, title, figdir / f'{rid}.png')
            except Exception as e:  # noqa: BLE001 — report must not stop
                row['error'] = f'{type(e).__name__}: {e}'[:200]
                say(f"  !! {rid}: {row['error']}")
            rows.append(row)
        if k % 10 == 0 or k == len(files):
            el_ = time.time() - t_start
            say(f"  {k}/{len(files)} files  ({el_ / 60:.1f} min, ~{el_ / k * (len(files) - k) / 60:.1f} min left)")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'blind_report.csv', index=False)
    per_file = (df.groupby('file')
                  .agg(n_electrodes=('electrode', 'count'),
                       tissues=('tissues_in_name', 'first'), condition=('condition', 'first'),
                       exp=('exp', 'first'), day=('day', 'first'),
                       rr_el1=('rr_median_ms', 'first'), rr_el2=('rr_median_ms', 'last'),
                       fpd_el1=('fpd_median_ms', 'first'), fpd_el2=('fpd_median_ms', 'last'),
                       grade_el1=('qc_grade', 'first'), grade_el2=('qc_grade', 'last'),
                       errors=('error', lambda s: '; '.join(x for x in s if x)))
                  .reset_index())
    with pd.ExcelWriter(out / 'blind_report.xlsx', engine='xlsxwriter') as xw:
        df.to_excel(xw, sheet_name='per_electrode', index=False)
        per_file.to_excel(xw, sheet_name='per_file', index=False)
        pd.DataFrame({'README': README.format(ver=__version__).splitlines()}).to_excel(
            xw, sheet_name='README', index=False, header=False)
        for sh, d in (('per_electrode', df), ('per_file', per_file)):
            ws = xw.sheets[sh]
            ws.freeze_panes(1, 1)
            for i, col in enumerate(d.columns):
                width = min(60, max(10, int(d[col].astype(str).str.len().quantile(0.9)) + 2))
                ws.set_column(i, i, width)
    (out / 'config_used.json').write_text(json.dumps(cfg.to_dict() if hasattr(cfg, 'to_dict') else {}, indent=2, default=str))
    say(f"# done: {len(df)} electrode rows, {int((df['error'] != '').sum())} errors — {out}")


if __name__ == '__main__':
    main()
