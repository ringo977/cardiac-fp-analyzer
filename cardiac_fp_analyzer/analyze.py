#!/usr/bin/env python3
"""
analyze.py — Main entry point for cardiac FP analysis.

Usage:
  python analyze.py /path/to/data/folder [--channel auto] [--output /path/to/output]
"""

import sys, argparse, warnings, traceback
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cardiac_fp_analyzer.loader import load_csv, parse_filename
from cardiac_fp_analyzer.filtering import full_filter_pipeline
from cardiac_fp_analyzer.beat_detection import detect_beats, segment_beats, compute_beat_periods
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia
from cardiac_fp_analyzer.quality_control import validate_beats, estimate_global_snr
from cardiac_fp_analyzer.report import generate_excel_report, generate_pdf_report

warnings.filterwarnings('ignore', category=RuntimeWarning)


def _select_best_channel(df, fs):
    best_ch, best_score = 'ch1', -999
    details = {}
    for ch in ['ch1', 'ch2']:
        try:
            filt = full_filter_pipeline(df[ch].values, fs)
            bi, bt, info = detect_beats(filt, fs, method='auto', min_distance_ms=400)
            bp = compute_beat_periods(bi, fs)
            score = 0
            if len(bp) > 2:
                mbp, cv = np.mean(bp), np.std(bp)/np.mean(bp) if np.mean(bp)>0 else 999
                if 0.3 <= mbp <= 4.0: score += 30
                if cv < 0.10: score += 40
                elif cv < 0.20: score += 30
                elif cv < 0.35: score += 15
                rate = len(bi) / (len(df)/fs)
                if 0.3 <= rate <= 3.5: score += 20
                p5, p95 = np.percentile(filt, [5, 95])
                nm = (filt >= p5) & (filt <= p95)
                ns = np.std(filt[nm]) if np.sum(nm)>100 else np.std(filt)
                snr = np.mean(np.abs(filt[bi]))/ns if ns>0 else 0
                if snr > 5: score += 20
                elif snr > 3: score += 10
                details[ch] = f'{len(bi)} beats, BP={mbp*1000:.0f}ms, CV={cv*100:.1f}%, score={score}'
            else:
                details[ch] = f'{len(bi)} beats (too few)'
            if score > best_score:
                best_score, best_ch = score, ch
        except:
            details[ch] = 'error'
    return best_ch, details


def analyze_single_file(filepath, channel='auto', verbose=True):
    filepath = Path(filepath)
    if verbose:
        print(f"\n{'='*60}\n  Analyzing: {filepath.name}\n{'='*60}")
    try:
        metadata, df = load_csv(filepath)
        fs = metadata['sample_rate']
        if verbose: print(f"  Loaded: {len(df)} samples, Fs={fs} Hz, Duration={len(df)/fs:.1f}s")

        file_info = parse_filename(filepath.name)
        for p in filepath.parts:
            if p.startswith('EXP'): file_info['experiment'] = p; break

        actual_ch = channel
        if channel == 'auto':
            actual_ch, ch_det = _select_best_channel(df, fs)
            if verbose:
                print(f"  Channel selection:")
                for ch, d in ch_det.items():
                    print(f"    {ch}: {d}{' *' if ch==actual_ch else ''}")
        file_info['analyzed_channel'] = actual_ch

        raw = df[actual_ch].values
        filtered = full_filter_pipeline(raw, fs)
        if verbose: print(f"  Filtered ({actual_ch})")

        bi, bt, det = detect_beats(filtered, fs, method='auto', min_distance_ms=400)
        if verbose: print(f"  Beats: {det['n_beats']} ({det['method']})")

        if det['n_beats'] < 5 and len(df) > fs*10:
            bi, bt, det = detect_beats(filtered, fs, method='auto', min_distance_ms=300, threshold_factor=3.0)
            if verbose: print(f"  Retry: {det['n_beats']} beats")

        bd, btm, vi = segment_beats(filtered, df['time'].values, bi, fs, pre_ms=50, post_ms=600)

        # ─── Quality Control: validate beats ───
        qc_report, bi_clean, bd_clean, btm_clean = validate_beats(
            filtered, bi, bd, btm, fs,
            snr_threshold=3.0, morphology_threshold=0.5,
            use_morphology=True
        )
        if verbose:
            n_rej = qc_report.n_beats_input - qc_report.n_beats_accepted
            print(f"  QC: Grade {qc_report.grade} | SNR={qc_report.global_snr:.1f} | "
                  f"Accepted {qc_report.n_beats_accepted}/{qc_report.n_beats_input} "
                  f"(-{n_rej}: {qc_report.n_beats_rejected_snr} SNR, "
                  f"{qc_report.n_beats_rejected_morphology} morph)")
            for note in qc_report.notes:
                print(f"    >> {note}")

        # Use cleaned beats for parameter extraction
        all_p, summary = extract_all_parameters(bd_clean, btm_clean, bi_clean, fs)
        bp = compute_beat_periods(bi_clean, fs)

        if verbose and len(bp) > 0:
            print(f"  BP: {np.mean(bp)*1000:.0f}ms ({60/np.mean(bp):.1f} BPM)")

        ar = analyze_arrhythmia(bi_clean, bp, all_p, summary, fs)
        if verbose:
            print(f"  {ar.classification} (Risk: {ar.risk_score}/100)")

        return {'metadata': metadata, 'file_info': file_info, 'summary': summary,
                'all_params': all_p, 'arrhythmia_report': ar,
                'beat_indices': bi_clean, 'beat_indices_raw': bi,
                'beat_periods': bp, 'filtered_signal': filtered,
                'time_vector': df['time'].values,
                'beats_data': bd_clean, 'beats_time': btm_clean,
                'qc_report': qc_report}
    except Exception as e:
        print(f"  ERROR: {e}")
        if verbose: traceback.print_exc()
        return None


def batch_analyze(data_dir, channel='auto', output_dir=None, verbose=True):
    data_dir = Path(data_dir)
    if output_dir is None: output_dir = data_dir / 'analysis_results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.rglob('*.csv'))
    print(f"\n{'#'*60}\n  CARDIAC FP ANALYZER\n  Files: {len(csv_files)} | Channel: {channel}\n{'#'*60}")

    results, errors = [], []
    for i, f in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] {f.name}")
        r = analyze_single_file(f, channel=channel, verbose=verbose)
        if r: results.append(r)
        else: errors.append(str(f))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n  Generating reports... ({len(results)}/{len(csv_files)} OK)")
    generate_excel_report(results, output_dir / f'cardiac_fp_analysis_{ts}.xlsx')
    generate_pdf_report(results, output_dir / f'cardiac_fp_analysis_{ts}.pdf', str(data_dir))
    print(f"\n  DONE! Results in: {output_dir}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description='Cardiac FP Analyzer for hiPSC-CM µECG')
    parser.add_argument('data_dir')
    parser.add_argument('--channel', default='auto', choices=['auto','ch1','ch2'])
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()
    batch_analyze(args.data_dir, args.channel, args.output, not args.quiet)


if __name__ == '__main__':
    main()
