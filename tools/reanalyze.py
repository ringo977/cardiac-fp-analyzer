#!/usr/bin/env python3
"""Re-run the analysis pipeline over datasets, with a provenance manifest.

Why this exists
---------------
Sprint 2 corrected the RR used for rate correction (commit 927568f).
Results produced before it have FPDc deflated by an amount that depends
on how many beats QC rejected in each recording — median 4%, above 10%
on a third of files, up to 55%. Because %dFPDcF is a ratio of two FPDc
values with different rejection fractions, the error does not cancel in
normalization. Historical results therefore need re-analysing.

Re-running is easy; knowing *what a set of results was produced with* is
the part that usually gets lost. Every run here writes a manifest
recording the git commit, whether the tree was dirty, the non-default
config, the timestamp, and the key scalars per file. Two manifests can
then be diffed directly, so the next correction does not require
reconstructing the previous state from memory.

Usage
-----
    # re-analyse the calibration dataset
    python3 tools/reanalyze.py \\
        --out reanalysis/2026-08-05 \\
        "µECG-Pharma Calibration - EXCEL DATA/EXP 5" \\
        "µECG-Pharma Calibration - EXCEL DATA/EXP 7/ChipC"

    # every leaf folder containing CSVs, under a root
    python3 tools/reanalyze.py --out reanalysis/2026-08-05 --walk \\
        "µECG-Pharma Calibration - EXCEL DATA"

    # compare two runs
    python3 tools/reanalyze.py --diff reanalysis/OLD reanalysis/NEW

Reports (xlsx/pdf) are written under ``--out``, never into the source
data folders.

Caveats
-------
* The recorded commit comes from ``git`` in the current working tree, so
  it describes *the checkout you ran from*. Running an out-of-tree copy
  of the package (a ``git archive`` extraction, say) still records the
  checkout's commit, not the copy's. Fine for the intended use — two runs
  of the same repository at different times — but do not trust it when
  deliberately mixing versions; use ``compare_pipeline_versions.py`` for
  that.
* ``git_dirty`` is the field that matters most. A result set produced
  from a dirty tree cannot be reproduced from its commit alone, and the
  run prints a warning saying so.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

# Running as ``python3 tools/reanalyze.py`` puts tools/ on sys.path, not the
# repo root, so the package would not be importable from a plain checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Scalars captured per file. Deliberately small: enough to detect a
# scientific change, not a full dump.
_KEYS = [
    'fpd_ms_mean', 'fpd_ms_median', 'fpd_ms_n',
    'fpdc_ms_mean', 'fpdc_ms_median', 'fpdc_ms_std',
    'fpdc_fridericia_ms_mean', 'fpdc_bazett_ms_mean',
    'beat_period_ms_mean', 'beat_period_ms_cv', 'bpm_mean',
    'spike_amplitude_mV_mean', 'stv_ms',
    'fpd_confidence', 'fpd_valid_ratio', 'fpd_reliable',
    'pct_beats_no_repol', 'correction',
]

_NORM_KEYS = [
    'has_baseline', 'baseline_file', 'pct_fpdc_change', 'pct_bp_change',
    'tdp_score', 'fpd_reliable',
]


# ── Provenance ──────────────────────────────────────────────────────────

def _git(*args, default=''):
    """Read-only git query, asking git not to take the index lock.

    ``git status`` refreshes the index cache as an optimisation, which
    means briefly acquiring .git/index.lock. Normally it removes it again
    and nothing is left behind. But a caller that cannot unlink inside
    .git — a sandbox mount with restricted permissions, say — leaves the
    lock in place, and every later commit then fails with "Impossibile
    creare '.git/index.lock': File exists".

    ``GIT_OPTIONAL_LOCKS=0`` tells git to skip locking where it is only an
    optimisation; ``--no-optional-locks`` is the command-level equivalent
    for versions that ignore the variable. Nothing here needs a refreshed
    index, so not taking the lock costs nothing and removes the failure
    mode entirely.
    """
    import os

    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
    try:
        return subprocess.run(['git', '--no-optional-locks', *args],
                              capture_output=True, text=True,
                              check=True, env=env).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return default


def _provenance(cfg):
    """Everything needed to know what a result set was produced with."""
    from cardiac_fp_analyzer import __version__ as pkg_version

    # Only *tracked* modifications matter for reproducibility: untracked
    # files are not part of the package and cannot change its behaviour.
    # Counting them too would make this flag true in any working repo,
    # which is exactly how a warning stops being read.
    modified = _git('status', '--porcelain', '--untracked-files=no')
    dirty = bool(modified)
    untracked = _git('ls-files', '--others', '--exclude-standard')
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'package_version': pkg_version,
        'git_commit': _git('rev-parse', 'HEAD', default='unknown'),
        'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD',
                           default='unknown'),
        'git_dirty': dirty,
        'git_modified_files': modified.splitlines() if modified else [],
        'git_untracked_count': len(untracked.splitlines()) if untracked else 0,
        'git_dirty_note': (
            'Tracked files were modified: this result set is NOT '
            'reproducible from the commit alone.' if dirty else ''
        ),
        'python': sys.version.split()[0],
        # Only the non-default config, so the diff between two manifests
        # shows the analysis decisions rather than 200 unchanged fields.
        'config_non_default': _config_diff(cfg),
    }


def _config_diff(cfg):
    """Fields differing from the packaged defaults, as a flat dict."""
    from dataclasses import asdict, fields

    from cardiac_fp_analyzer.config import AnalysisConfig

    default = AnalysisConfig()
    out = {}
    for f in fields(cfg):
        cur, dfl = getattr(cfg, f.name), getattr(default, f.name, None)
        if hasattr(cur, '__dataclass_fields__'):
            for sub in fields(cur):
                a = getattr(cur, sub.name)
                b = getattr(dfl, sub.name, None) if dfl is not None else None
                if a != b:
                    out[f'{f.name}.{sub.name}'] = a
        elif cur != dfl:
            try:
                out[f.name] = asdict(cur)
            except TypeError:
                out[f.name] = cur
    return out


# ── Capture ─────────────────────────────────────────────────────────────

def _scalar(v):
    if isinstance(v, (bool, str, type(None))):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return None if f != f else round(f, 9)      # NaN → None


def _capture(results):
    out = {}
    for r in results:
        if r is None:
            continue
        name = r.get('metadata', {}).get('filename') or '?'
        s = r.get('summary', {}) or {}
        n = r.get('normalization', {}) or {}
        inc = r.get('inclusion', {}) or {}
        qc = r.get('qc_report')
        ar = r.get('arrhythmia_report')
        out[name] = {
            **{k: _scalar(s.get(k)) for k in _KEYS},
            **{f'norm.{k}': _scalar(n.get(k)) for k in _NORM_KEYS},
            'qc_grade': getattr(qc, 'grade', None),
            'risk_score': _scalar(getattr(ar, 'risk_score', None)),
            'included': inc.get('passed', True),
            'exclusion_reason': inc.get('reason', '') or '',
            'exclusion_criterion': inc.get('criterion') or '',
        }
    return out


def run_dataset(data_dir: Path, out_root: Path, cfg):
    from cardiac_fp_analyzer.analyze import batch_analyze
    from cardiac_fp_analyzer.normalization import classify_drug

    tag = data_dir.name.replace(' ', '_')
    report_dir = out_root / 'reports' / tag
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"  analysing {data_dir} ...", flush=True)
    results = batch_analyze(str(data_dir), verbose=False, config=cfg,
                            output_dir=str(report_dir))

    classification = {}
    try:
        for drug, info in classify_drug(results, cfg=cfg.normalization).items():
            if drug.startswith('_'):
                continue
            classification[drug] = {
                'positive': bool(info.get('positive')),
                'max_pct_change': _scalar(info.get('max_pct_change')),
                'n_concentrations': info.get('n_concentrations'),
                'concentrations': [
                    [str(c), _scalar(p)]
                    for c, p in (info.get('concentrations') or [])
                ],
            }
    except Exception as exc:                     # noqa: BLE001 - diagnostic
        classification = {'_error': str(exc)}

    inc_report = {}
    for r in results:
        if r and r.get('inclusion_report'):
            inc_report = r['inclusion_report']
            break
    removed = {
        g: {
            'baseline_file': i.get('baseline_file'),
            'criterion': i.get('criterion'),
            'reason': i.get('reason'),
            'n_drug_recordings_lost': i.get('n_drug_recordings_lost'),
        }
        for g, i in (inc_report.get('excluded_groups') or {}).items()
    }

    per_file = _capture(results)
    return {
        'dataset': str(data_dir),
        'n_files': len(per_file),
        'n_included': sum(1 for v in per_file.values() if v['included']),
        'per_file': per_file,
        'classification': classification,
        'removed_groups': removed,
    }


# ── Diff ────────────────────────────────────────────────────────────────

def _pct(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    return None if a == 0 else (b - a) / abs(a) * 100.0


def diff_manifests(old_path: Path, new_path: Path):
    old = json.loads((old_path / 'manifest.json').read_text())
    new = json.loads((new_path / 'manifest.json').read_text())

    print("=" * 78)
    print("PROVENIENZA")
    print("=" * 78)
    for lbl, m in (('prima', old), ('dopo ', new)):
        p = m['provenance']
        dirty = ('  [FILE TRACCIATI MODIFICATI — non riproducibile]'
                 if p.get('git_dirty') else '')
        print(f"  {lbl}: {p['git_commit'][:10]} ({p['git_branch']})  "
              f"v{p['package_version']}  {p['timestamp_utc']}{dirty}")
    cd_o = old['provenance'].get('config_non_default', {})
    cd_n = new['provenance'].get('config_non_default', {})
    if cd_o != cd_n:
        print("\n  config non-default cambiata:")
        for k in sorted(set(cd_o) | set(cd_n)):
            if cd_o.get(k) != cd_n.get(k):
                print(f"    {k}: {cd_o.get(k)!r} -> {cd_n.get(k)!r}")

    for ds in sorted(set(old['datasets']) | set(new['datasets'])):
        o, n = old['datasets'].get(ds), new['datasets'].get(ds)
        print("\n" + "=" * 78)
        print(f"DATASET  {ds}")
        print("=" * 78)
        if o is None or n is None:
            print(f"  presente solo in {'dopo' if o is None else 'prima'}")
            continue

        print(f"  file inclusi: {o['n_included']}/{o['n_files']}  ->  "
              f"{n['n_included']}/{n['n_files']}")

        deltas = []
        for f in sorted(set(o['per_file']) & set(n['per_file'])):
            d = _pct(o['per_file'][f].get('fpdc_ms_mean'),
                     n['per_file'][f].get('fpdc_ms_mean'))
            if d is not None:
                deltas.append((abs(d), f, d))
        if deltas:
            deltas.sort(reverse=True)
            vals = sorted(x[0] for x in deltas)
            print(f"  |ΔFPDc|: mediana {vals[len(vals) // 2]:.1f}%   "
                  f"max {vals[-1]:.1f}%   "
                  f">5%: {sum(1 for v in vals if v > 5)}/{len(vals)}")
            worst = [x for x in deltas if x[0] > 5][:5]
            for _a, f, d in worst:
                print(f"      {f[:46]:<48}{d:+7.1f}%")

        for f in sorted(set(o['per_file']) & set(n['per_file'])):
            a, b = o['per_file'][f]['included'], n['per_file'][f]['included']
            if a != b:
                why = n['per_file'][f]['exclusion_reason'] or '-'
                print(f"  inclusione {f[:40]:<42} {a} -> {b}   {why[:40]}")

        co, cn = o['classification'], n['classification']
        for drug in sorted(set(co) | set(cn)):
            po = co.get(drug, {}).get('positive')
            pn = cn.get(drug, {}).get('positive')
            if drug not in co:
                print(f"  >>> FARMACO RECUPERATO: {drug} "
                      f"(positive={pn}, max={cn[drug]['max_pct_change']})")
            elif drug not in cn:
                print(f"  >>> farmaco perso     : {drug}")
            elif po != pn:
                print(f"  >>> CALL CAMBIATA     : {drug}  {po} -> {pn}")


# ── CLI ─────────────────────────────────────────────────────────────────

def _leaf_dirs(root: Path):
    return sorted({p.parent for p in root.rglob('*.csv')})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('datasets', nargs='*', type=Path)
    ap.add_argument('--out', type=Path, help='output folder for this run')
    ap.add_argument('--walk', action='store_true',
                    help='treat each argument as a root and analyse every '
                         'leaf folder containing CSVs')
    ap.add_argument('--diff', nargs=2, type=Path, metavar=('OLD', 'NEW'),
                    help='compare two previous runs and exit')
    args = ap.parse_args()

    if args.diff:
        diff_manifests(*args.diff)
        return 0

    if not args.datasets or not args.out:
        ap.error('serve --out e almeno un dataset (oppure --diff)')

    dirs = []
    for d in args.datasets:
        if not d.is_dir():
            print(f"  salto (non è una cartella): {d}", file=sys.stderr)
            continue
        dirs.extend(_leaf_dirs(d) if args.walk else [d])
    if not dirs:
        print("nessun dataset da analizzare", file=sys.stderr)
        return 1

    from cardiac_fp_analyzer.config import AnalysisConfig
    cfg = AnalysisConfig()

    args.out.mkdir(parents=True, exist_ok=True)
    prov = _provenance(cfg)
    if prov['git_dirty']:
        print("  ATTENZIONE: file tracciati modificati e non committati.")
        print("  Questo set di risultati non è riproducibile dal commit.")
        for line in prov['git_modified_files'][:10]:
            print(f"    {line}")
        extra = len(prov['git_modified_files']) - 10
        if extra > 0:
            print(f"    (+{extra} altri)")
        print()

    manifest = {'provenance': prov, 'datasets': {}}
    for d in dirs:
        manifest['datasets'][str(d)] = run_dataset(d, args.out, cfg)
        (args.out / 'manifest.json').write_text(
            json.dumps(manifest, indent=1, default=str))

    print(f"\nManifest: {args.out / 'manifest.json'}")
    print(f"Report:   {args.out / 'reports'}")
    tot = sum(v['n_files'] for v in manifest['datasets'].values())
    inc = sum(v['n_included'] for v in manifest['datasets'].values())
    print(f"{len(dirs)} dataset, {tot} file, {inc} inclusi")
    return 0


if __name__ == '__main__':
    sys.exit(main())
