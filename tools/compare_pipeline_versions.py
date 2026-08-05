#!/usr/bin/env python3
"""Compare pipeline output between two versions of ``cardiac_fp_analyzer``.

Purpose
-------
Sprint 0 (2026-08-05) fixed a per-beat repolarization polarity bug that
changes measured FPD on real recordings.  Before trusting any historical
result, we need to know *which conclusions move* — not just that numbers
changed.

This harness runs the full batch pipeline twice, from two different
package roots, over the same folder of CSVs, and diffs:

  * per-file scalars (FPD, FPDc, beat count, QC grade, risk score)
  * per-file normalization (%ΔFPDcF vs baseline)
  * drug-level classification (positive/negative, max %change)

The last one is what matters: a 5% FPD shift is only important if it
flips a drug call or moves a dose-response curve.

Usage
-----
    python3 tools/compare_pipeline_versions.py \\
        --old-root /tmp/orig \\
        --new-root . \\
        --data "µECG-Pharma Calibration - EXCEL DATA/EXP 5" \\
        --out-dir /tmp/cmp

Each version runs in its own subprocess so the two copies of the package
never share module state.  Results are cached as JSON, so re-running to
tweak the report does not re-analyse.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Scalars pulled from each file's ``summary`` dict.
SUMMARY_KEYS = [
    'fpd_ms_mean', 'fpd_ms_std', 'fpd_ms_median', 'fpd_ms_n',
    'fpdc_ms_mean', 'fpdc_ms_std', 'fpdc_ms_median',
    'beat_period_ms_mean', 'bpm_mean', 'spike_amplitude_mV_mean',
    'stv_ms', 'fpd_confidence', 'fpd_valid_ratio', 'fpd_reliable',
    'pct_beats_no_repol',
]

# Scalars pulled from each file's ``normalization`` dict.
NORM_KEYS = [
    'has_baseline', 'baseline_file', 'baseline_fpdc_ms',
    'pct_fpdc_change', 'pct_bp_change', 'pct_amp_change',
    'tdp_score', 'exceeds_LOW', 'exceeds_MID', 'exceeds_HIGH',
]


# ── Worker: runs inside a subprocess with one package root on sys.path ──

_WORKER = r'''
import sys, json, warnings
warnings.filterwarnings("ignore")
root, data_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
report_dir = sys.argv[6]
sys.path.insert(0, root)

from cardiac_fp_analyzer.analyze import batch_analyze

# Redirect reports away from the source data folder: batch_analyze
# defaults output_dir to <data_dir>/analysis_results, which would write
# xlsx/pdf into the read-only calibration dataset.
results = batch_analyze(data_dir, verbose=False, output_dir=report_dir)

SUMMARY_KEYS = json.loads(sys.argv[4])
NORM_KEYS = json.loads(sys.argv[5])

def scalar(v):
    if isinstance(v, (bool, str, type(None))):
        return v
    try:
        return round(float(v), 9)
    except (TypeError, ValueError):
        return str(v)

def safe_len(v):
    """len() that tolerates None and numpy arrays.

    ``x or []`` triggers numpy's ambiguous-truth-value error on arrays,
    so the emptiness check must not go through bool().
    """
    if v is None:
        return 0
    try:
        return int(len(v))
    except TypeError:
        return 0

per_file = {}
for r in results:
    if r is None:
        continue
    name = r.get("metadata", {}).get("filename") or "?"
    s = r.get("summary", {}) or {}
    n = r.get("normalization", {}) or {}
    qc = r.get("qc_report")
    ar = r.get("arrhythmia_report")
    inc = r.get("inclusion", {}) or {}
    per_file[name] = {
        **{k: scalar(s.get(k)) for k in SUMMARY_KEYS},
        **{"norm." + k: scalar(n.get(k)) for k in NORM_KEYS},
        "_n_beats": safe_len(r.get("beat_indices")),
        "_qc_grade": getattr(qc, "grade", None),
        "_risk_score": scalar(getattr(ar, "risk_score", None)),
        "_included": inc.get("passed", True),
    }

# Drug-level classification
try:
    from cardiac_fp_analyzer.normalization import classify_drug
    classification = classify_drug(results)
except Exception as exc:            # noqa: BLE001 - diagnostic harness
    classification = {"_error": str(exc)}

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return scalar(o)

with open(out_path, "w") as fh:
    json.dump({"per_file": per_file,
               "classification": clean(classification)},
              fh, indent=2, sort_keys=True, default=str)
print(f"analysed {len(per_file)} file(s)")
'''


def run_version(root: Path, data_dir: Path, out_path: Path) -> dict:
    """Run the pipeline from *root* and return the parsed result."""
    if out_path.exists():
        print(f"  [cached] {out_path}")
        return json.loads(out_path.read_text())

    print(f"  running {root} ...", flush=True)
    report_dir = out_path.parent / f'{out_path.stem}_reports'
    proc = subprocess.run(
        [sys.executable, '-c', _WORKER, str(root), str(data_dir),
         str(out_path), json.dumps(SUMMARY_KEYS), json.dumps(NORM_KEYS),
         str(report_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"analysis failed for {root}")
    print(f"  {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else 'done'}")
    return json.loads(out_path.read_text())


# ── Reporting ───────────────────────────────────────────────────────────

def _pct_delta(a, b):
    """Relative change from *a* to *b*, or None when undefined."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return (b - a) / abs(a) * 100.0


def report_per_file(old, new, fh):
    names = sorted(set(old) | set(new))
    print("\n" + "=" * 100, file=fh)
    print("PER-FILE SCALARS", file=fh)
    print("=" * 100, file=fh)
    hdr = (f"{'file':<40} {'FPD before':>11} {'FPD after':>11} {'Δ%':>7} "
           f"{'FPDc before':>12} {'FPDc after':>11} {'Δ%':>7}")
    print(hdr, file=fh)
    print("-" * 100, file=fh)

    moved = []
    for n in names:
        o, w = old.get(n), new.get(n)
        if o is None or w is None:
            print(f"{n[:39]:<40} {'MISSING in ' + ('new' if w is None else 'old'):>60}",
                  file=fh)
            moved.append((n, None))
            continue
        d_fpd = _pct_delta(o.get('fpd_ms_mean'), w.get('fpd_ms_mean'))
        d_fpdc = _pct_delta(o.get('fpdc_ms_mean'), w.get('fpdc_ms_mean'))

        def f(v, spec='11.1f'):
            try:
                return format(float(v), spec)
            except (TypeError, ValueError):
                return f"{'n/a':>11}"

        print(f"{n[:39]:<40} {f(o.get('fpd_ms_mean'))} {f(w.get('fpd_ms_mean'))} "
              f"{(f'{d_fpd:+6.1f}%' if d_fpd is not None else '     —'):>7} "
              f"{f(o.get('fpdc_ms_mean'), '12.1f')} {f(w.get('fpdc_ms_mean'))} "
              f"{(f'{d_fpdc:+6.1f}%' if d_fpdc is not None else '     —'):>7}",
              file=fh)
        if d_fpdc is not None:
            moved.append((n, d_fpdc))

    # Structural changes worth calling out separately.
    print("\nStructural changes (beat count / QC grade / inclusion):", file=fh)
    any_struct = False
    for n in names:
        o, w = old.get(n), new.get(n)
        if not o or not w:
            continue
        for key, label in [('_n_beats', 'beats'), ('_qc_grade', 'QC'),
                           ('_included', 'included'),
                           ('fpd_reliable', 'fpd_reliable')]:
            if o.get(key) != w.get(key):
                any_struct = True
                print(f"  {n[:45]:<46} {label}: {o.get(key)!r} -> {w.get(key)!r}",
                      file=fh)
    if not any_struct:
        print("  none — beat detection, QC and inclusion are unchanged", file=fh)

    return [m for m in moved if m[1] is not None]


def report_normalization(old, new, fh):
    print("\n" + "=" * 100, file=fh)
    print("NORMALIZATION — %ΔFPDcF vs baseline (the reported endpoint)", file=fh)
    print("=" * 100, file=fh)
    print(f"{'file':<40} {'%Δ before':>11} {'%Δ after':>11} {'shift (pp)':>11} "
          f"{'crosses 15%?':>14}", file=fh)
    print("-" * 100, file=fh)

    flips = []
    for n in sorted(set(old) | set(new)):
        o, w = old.get(n) or {}, new.get(n) or {}
        po, pw = o.get('norm.pct_fpdc_change'), w.get('norm.pct_fpdc_change')
        try:
            po_f, pw_f = float(po), float(pw)
        except (TypeError, ValueError):
            continue
        if po_f != po_f or pw_f != pw_f:      # NaN
            continue
        shift = pw_f - po_f
        # 15% is the default classification threshold (THRESHOLD_MID).
        was, now = po_f >= 15.0, pw_f >= 15.0
        mark = ''
        if was != now:
            mark = 'YES -> ' + ('above' if now else 'below')
            flips.append((n, po_f, pw_f))
        print(f"{n[:39]:<40} {po_f:>10.1f}% {pw_f:>10.1f}% {shift:>+10.1f} "
              f"{mark:>14}", file=fh)

    if flips:
        print(f"\n  !! {len(flips)} recording(s) crossed the 15% threshold", file=fh)
    else:
        print("\n  No recording crossed the 15% classification threshold.", file=fh)
    return flips


def report_classification(old, new, fh):
    print("\n" + "=" * 100, file=fh)
    print("DRUG-LEVEL CLASSIFICATION — does any conclusion change?", file=fh)
    print("=" * 100, file=fh)

    drugs = sorted(set(old) | set(new))
    changed = []
    for d in drugs:
        if d.startswith('_'):
            continue
        o, w = old.get(d) or {}, new.get(d) or {}
        po, pw = o.get('positive'), w.get('positive')
        mo, mw = o.get('max_pct_change'), w.get('max_pct_change')
        flag = ''
        if po != pw:
            flag = f"  <<< CALL CHANGED: {po} -> {pw}"
            changed.append(d)
        try:
            mo_s, mw_s = f"{float(mo):.1f}%", f"{float(mw):.1f}%"
        except (TypeError, ValueError):
            mo_s, mw_s = str(mo), str(mw)
        print(f"\n  {d}", file=fh)
        print(f"    positive        : {po}  ->  {pw}{flag}", file=fh)
        print(f"    max %ΔFPDcF     : {mo_s}  ->  {mw_s}", file=fh)
        print(f"    n concentrations: {o.get('n_concentrations')}  ->  "
              f"{w.get('n_concentrations')}", file=fh)

        co = {str(c): v for c, v in (o.get('concentrations') or [])}
        cw = {str(c): v for c, v in (w.get('concentrations') or [])}
        for c in sorted(set(co) | set(cw)):
            a, b = co.get(c), cw.get(c)
            try:
                print(f"      {c:<18} {float(a):>8.1f}%  ->  {float(b):>8.1f}%",
                      file=fh)
            except (TypeError, ValueError):
                print(f"      {c:<18} {a}  ->  {b}", file=fh)

    print("", file=fh)
    if changed:
        print(f"  !! {len(changed)} drug call(s) CHANGED: {', '.join(changed)}",
              file=fh)
    else:
        print("  No drug-level call changed.", file=fh)
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old-root', required=True, type=Path)
    ap.add_argument('--new-root', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.label or args.data.name.replace(' ', '_')

    print(f"Comparing on: {args.data}")
    old = run_version(args.old_root, args.data, args.out_dir / f'{tag}_old.json')
    new = run_version(args.new_root, args.data, args.out_dir / f'{tag}_new.json')

    report_path = args.out_dir / f'{tag}_report.txt'
    with report_path.open('w') as fh:
        print(f"Pipeline comparison — {tag}", file=fh)
        print(f"  old: {args.old_root}", file=fh)
        print(f"  new: {args.new_root}", file=fh)
        print(f"  data: {args.data}", file=fh)

        moved = report_per_file(old['per_file'], new['per_file'], fh)
        flips = report_normalization(old['per_file'], new['per_file'], fh)
        changed = report_classification(old['classification'],
                                        new['classification'], fh)

        print("\n" + "=" * 100, file=fh)
        print("SUMMARY", file=fh)
        print("=" * 100, file=fh)
        if moved:
            deltas = sorted(abs(d) for _n, d in moved)
            n = len(deltas)
            print(f"  files compared          : {n}", file=fh)
            print(f"  median |ΔFPDc|          : {deltas[n // 2]:.1f}%", file=fh)
            print(f"  max |ΔFPDc|             : {deltas[-1]:.1f}%", file=fh)
            print(f"  files with |ΔFPDc| > 5% : "
                  f"{sum(1 for d in deltas if d > 5)}", file=fh)
        print(f"  recordings crossing 15% : {len(flips)}", file=fh)
        print(f"  drug calls changed      : {len(changed)}", file=fh)

    print(report_path.read_text())
    print(f"\nReport written to {report_path}")


if __name__ == '__main__':
    main()
