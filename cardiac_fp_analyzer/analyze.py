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


def _select_best_channel(df, fs, cfg=None):
    """Select the best channel based on beat detection quality. cfg = AnalysisConfig."""
    if cfg is not None:
        cs = cfg.channel_selection
        fc = cfg.filtering
    else:
        from .config import ChannelSelectionConfig, FilterConfig
        cs = ChannelSelectionConfig()
        fc = FilterConfig()

    best_ch, best_score = 'ch1', -999
    details = {}
    for ch in ['ch1', 'ch2']:
        try:
            filt = full_filter_pipeline(df[ch].values, fs, cfg=fc)
            bi, bt, info = detect_beats(filt, fs, method='auto', min_distance_ms=400)
            bp = compute_beat_periods(bi, fs)
            score = 0
            if len(bp) > 2:
                mbp, cv = np.mean(bp), np.std(bp)/np.mean(bp) if np.mean(bp)>0 else 999
                if cs.bp_ideal_range_s[0] <= mbp <= cs.bp_ideal_range_s[1]: score += 30
                if cv*100 < cs.cv_excellent: score += 40
                elif cv*100 < cs.cv_good: score += 30
                elif cv*100 < cs.cv_fair: score += 15
                rate = len(bi) / (len(df)/fs)
                if cs.rate_range_per_s[0] <= rate <= cs.rate_range_per_s[1]: score += 20
                p5, p95 = np.percentile(filt, [5, 95])
                nm = (filt >= p5) & (filt <= p95)
                ns = np.std(filt[nm]) if np.sum(nm)>100 else np.std(filt)
                snr = np.mean(np.abs(filt[bi]))/ns if ns>0 else 0
                if snr > cs.snr_good: score += 20
                elif snr > cs.snr_fair: score += 10
                details[ch] = f'{len(bi)} beats, BP={mbp*1000:.0f}ms, CV={cv*100:.1f}%, score={score}'
            else:
                details[ch] = f'{len(bi)} beats (too few)'
            if score > best_score:
                best_score, best_ch = score, ch
        except:
            details[ch] = 'error'
    return best_ch, details


def analyze_single_file(filepath, channel='auto', verbose=True, config=None):
    """Analyze a single CSV file through the full pipeline.

    Parameters
    ----------
    filepath : path to CSV file
    channel : 'auto', 'ch1', or 'ch2'
    verbose : print progress
    config : AnalysisConfig or None — controls all pipeline parameters
    """
    if config is None:
        from .config import AnalysisConfig
        config = AnalysisConfig()

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
            actual_ch, ch_det = _select_best_channel(df, fs, cfg=config)
            if verbose:
                print(f"  Channel selection:")
                for ch, d in ch_det.items():
                    print(f"    {ch}: {d}{' *' if ch==actual_ch else ''}")
        file_info['analyzed_channel'] = actual_ch

        raw = df[actual_ch].values
        filtered = full_filter_pipeline(raw, fs, cfg=config.filtering)
        if verbose: print(f"  Filtered ({actual_ch})")

        bd_cfg = config.beat_detection
        bi, bt, det = detect_beats(filtered, fs, cfg=bd_cfg)
        if verbose: print(f"  Beats: {det['n_beats']} ({det['method']})")

        if det['n_beats'] < 5 and len(df) > fs*10:
            bi, bt, det = detect_beats(
                filtered, fs,
                method=bd_cfg.method,
                min_distance_ms=bd_cfg.retry_min_distance_ms,
                threshold_factor=bd_cfg.retry_threshold_factor
            )
            if verbose: print(f"  Retry: {det['n_beats']} beats")

        rep_cfg = config.repolarization
        bd, btm, vi = segment_beats(filtered, df['time'].values, bi, fs,
                                     pre_ms=rep_cfg.segment_pre_ms,
                                     post_ms=max(850, rep_cfg.search_end_ms + 50))

        # ─── Quality Control: validate beats ───
        qc_report, bi_clean, bd_clean, btm_clean = validate_beats(
            filtered, bi, bd, btm, fs, cfg=config.quality
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
        all_p, summary = extract_all_parameters(bd_clean, btm_clean, bi_clean, fs,
                                                 cfg=rep_cfg)
        bp = compute_beat_periods(bi_clean, fs)

        if verbose and len(bp) > 0:
            print(f"  BP: {np.mean(bp)*1000:.0f}ms ({60/np.mean(bp):.1f} BPM)")

        ar = analyze_arrhythmia(bi_clean, bp, all_p, summary, fs, cfg=config.arrhythmia)
        if verbose:
            print(f"  {ar.classification} (Risk: {ar.risk_score}/100)")

        result = {'metadata': metadata, 'file_info': file_info, 'summary': summary,
                'all_params': all_p, 'arrhythmia_report': ar,
                'beat_indices': bi_clean, 'beat_indices_raw': bi,
                'beat_periods': bp, 'filtered_signal': filtered,
                'time_vector': df['time'].values,
                'beats_data': bd_clean, 'beats_time': btm_clean,
                'qc_report': qc_report}

        # ─── Cessation detection ───
        if config.enable_cessation:
            from .cessation import detect_cessation
            cess = detect_cessation(filtered, fs, bi, all_p,
                                     beat_indices_clean=bi_clean,
                                     qc_report=qc_report)
            result['cessation_report'] = cess
            if verbose and cess.has_cessation:
                print(f"  CESSATION: {cess.cessation_type} "
                      f"(conf={cess.cessation_confidence:.2f}, "
                      f"silent={cess.total_silent_s:.1f}s, "
                      f"max_gap={cess.max_gap_s:.1f}s)")

        # ─── Spectral analysis ───
        if config.enable_spectral:
            from .spectral import analyze_spectral
            spec = analyze_spectral(filtered, fs, bi_clean, bd_clean)
            result['spectral_report'] = spec
            if verbose:
                parts = []
                if not np.isnan(spec.spectral_entropy):
                    parts.append(f"entropy={spec.spectral_entropy:.2f}")
                if not np.isnan(spec.fundamental_freq_hz):
                    parts.append(f"f0={spec.fundamental_freq_hz:.2f}Hz")
                if spec.n_harmonics_detected > 0:
                    parts.append(f"harmonics={spec.n_harmonics_detected}")
                if parts:
                    print(f"  Spectral: {', '.join(parts)}")

        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        if verbose: traceback.print_exc()
        return None


def _apply_inclusion_criteria(results, verbose=True, cfg=None):
    """
    Apply quality-based inclusion criteria (as in Visone et al. 2023).

    Parameters
    ----------
    results : list of result dicts
    verbose : print progress
    cfg : InclusionConfig or None — if None, uses defaults

    Criteria applied:
      1. Baseline CV of BP must be < max_cv_bp (paper: 25%)
      2. FPDcF plausibility: recordings outside range are flagged
      3. FPD confidence: baselines with low confidence are excluded
    """
    if cfg is None:
        from .config import InclusionConfig
        cfg = InclusionConfig()

    from .normalization import _get_group_key, _is_baseline

    max_cv_bp = cfg.max_cv_bp if cfg.enabled_cv else 999.0
    fpdc_range = (cfg.fpdc_range_min, cfg.fpdc_range_max) if cfg.enabled_fpdc_range else (0, 99999)
    min_fpd_confidence = cfg.min_fpd_confidence if cfg.enabled_confidence else 0.0

    # Step 1: identify failing baselines and flag their groups
    excluded_groups = set()
    n_bl_ok = 0
    n_bl_fail = 0
    n_bl_conf_fail = 0
    for r in results:
        if not _is_baseline(r):
            continue
        cv = r.get('summary', {}).get('beat_period_ms_cv', np.nan)
        conf = r.get('summary', {}).get('fpd_confidence', np.nan)
        group = _get_group_key(r)

        fail_reason = None
        if cfg.enabled_cv and (np.isnan(cv) or cv >= max_cv_bp):
            fail_reason = f'Baseline CV={cv:.1f}% >= {max_cv_bp}%'
        elif cfg.enabled_confidence and not np.isnan(conf) and conf < min_fpd_confidence:
            fail_reason = f'Baseline FPD confidence={conf:.2f} < {min_fpd_confidence}'
            n_bl_conf_fail += 1

        if fail_reason:
            excluded_groups.add(group)
            r['inclusion'] = {'passed': False, 'reason': fail_reason}
            n_bl_fail += 1
        else:
            r['inclusion'] = {'passed': True, 'reason': ''}
            n_bl_ok += 1

    if verbose:
        print(f"  Inclusion criteria (CV BP < {max_cv_bp}%, FPD conf >= {min_fpd_confidence}): "
              f"{n_bl_ok} baselines OK, {n_bl_fail} excluded "
              f"({len(excluded_groups)} groups removed)"
              + (f" [{n_bl_conf_fail} low confidence]" if n_bl_conf_fail > 0 else ""))

    # Step 2: flag drug recordings in excluded groups
    n_drug_excl = 0
    for r in results:
        if _is_baseline(r):
            continue
        group = _get_group_key(r)
        if group in excluded_groups:
            r['inclusion'] = {'passed': False, 'reason': f'Baseline of group {group} failed inclusion'}
            n_drug_excl += 1
        else:
            r.setdefault('inclusion', {'passed': True, 'reason': ''})

    # Step 3: FPDcF plausibility check
    n_fpdc_fail = 0
    if cfg.enabled_fpdc_range:
        for r in results:
            fpdc = r.get('summary', {}).get('fpdc_ms_mean', np.nan)
            if not np.isnan(fpdc) and (fpdc < fpdc_range[0] or fpdc > fpdc_range[1]):
                inc = r.setdefault('inclusion', {'passed': True, 'reason': ''})
                inc['fpdc_plausible'] = False
                inc['fpdc_note'] = f'FPDcF={fpdc:.0f}ms outside [{fpdc_range[0]}-{fpdc_range[1]}]ms'
                n_fpdc_fail += 1
            else:
                inc = r.setdefault('inclusion', {'passed': True, 'reason': ''})
                inc['fpdc_plausible'] = True

    if verbose and n_fpdc_fail > 0:
        print(f"  FPDcF plausibility: {n_fpdc_fail} recordings outside {fpdc_range[0]}-{fpdc_range[1]}ms range")

    # Step 4: Flag drug recordings with low FPD confidence
    n_drug_conf = 0
    if cfg.enabled_confidence:
        for r in results:
            if _is_baseline(r):
                continue
            conf = r.get('summary', {}).get('fpd_confidence', np.nan)
            inc = r.setdefault('inclusion', {'passed': True, 'reason': ''})
            if not np.isnan(conf) and conf < min_fpd_confidence:
                inc['fpd_reliable'] = False
                n_drug_conf += 1
            else:
                inc['fpd_reliable'] = True

    if verbose and n_drug_conf > 0:
        print(f"  FPD reliability: {n_drug_conf} drug recordings with confidence < {min_fpd_confidence}")

    return results


def batch_analyze(data_dir, channel='auto', output_dir=None, verbose=True,
                  config=None,
                  # Legacy parameters (ignored when config is provided)
                  inclusion_cv=25.0, fpdc_range=(100, 1200),
                  min_fpd_confidence=0.68):
    """
    Batch analysis of all CSV files in a directory.

    Parameters
    ----------
    data_dir : path to directory with CSV files
    channel : 'auto', 'ch1', or 'ch2'
    output_dir : output directory (default: data_dir/analysis_results)
    verbose : print progress
    config : AnalysisConfig or None — controls all pipeline parameters.
             When provided, legacy parameters (inclusion_cv, fpdc_range,
             min_fpd_confidence) are ignored.
    """
    if config is None:
        from .config import AnalysisConfig
        config = AnalysisConfig()
        # Apply legacy overrides if they differ from defaults
        if inclusion_cv is None:
            config.inclusion.enabled_cv = False
        elif inclusion_cv != 25.0:
            config.inclusion.max_cv_bp = inclusion_cv
        if fpdc_range is None:
            config.inclusion.enabled_fpdc_range = False
        elif fpdc_range != (100, 1200):
            config.inclusion.fpdc_range_min = fpdc_range[0]
            config.inclusion.fpdc_range_max = fpdc_range[1]
        if min_fpd_confidence is None or min_fpd_confidence == 0:
            config.inclusion.enabled_confidence = False
        elif min_fpd_confidence != 0.68:
            config.inclusion.min_fpd_confidence = min_fpd_confidence

    data_dir = Path(data_dir)
    if output_dir is None: output_dir = data_dir / 'analysis_results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.rglob('*.csv'))
    print(f"\n{'#'*60}\n  CARDIAC FP ANALYZER\n  Files: {len(csv_files)} | Channel: {channel}\n{'#'*60}")

    # Save config alongside results
    config.to_json(str(output_dir / 'analysis_config.json'))

    results, errors = [], []
    for i, f in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] {f.name}")
        r = analyze_single_file(f, channel=channel, verbose=verbose, config=config)
        if r: results.append(r)
        else: errors.append(str(f))

    # ─── Inclusion criteria ───
    results = _apply_inclusion_criteria(results, verbose=verbose, cfg=config.inclusion)

    # ─── Baseline normalization ───
    from .normalization import normalize_all_results
    results = normalize_all_results(results, cfg=config.normalization)
    n_with_bl = sum(1 for r in results if r.get('normalization', {}).get('has_baseline'))
    if verbose and n_with_bl > 0:
        print(f"  Baseline normalization: {n_with_bl}/{len(results)} recordings paired")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n  Generating reports... ({len(results)}/{len(csv_files)} OK)")
    generate_excel_report(results, output_dir / f'cardiac_fp_analysis_{ts}.xlsx')
    generate_pdf_report(results, output_dir / f'cardiac_fp_analysis_{ts}.pdf', str(data_dir))
    print(f"\n  DONE! Results in: {output_dir}\n")
    return results


def main():
    from .config import AnalysisConfig

    parser = argparse.ArgumentParser(description='Cardiac FP Analyzer for hiPSC-CM µECG')
    parser.add_argument('data_dir')
    parser.add_argument('--channel', default='auto', choices=['auto','ch1','ch2'])
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--config', default=None,
                        help='Path to JSON config file (overrides all other flags)')
    parser.add_argument('--preset', default=None,
                        choices=['default', 'conservative', 'sensitive', 'peak_method', 'no_filters'],
                        help='Named config preset')
    # Legacy flags (applied if no --config or --preset)
    parser.add_argument('--inclusion-cv', type=float, default=25.0,
                        help='Max CV%% of baseline BP for inclusion (default 25, 0=disabled)')
    parser.add_argument('--no-fpdc-filter', action='store_true',
                        help='Disable FPDcF plausibility filter')
    parser.add_argument('--correction', default='fridericia',
                        choices=['fridericia', 'bazett', 'none'],
                        help='QT correction formula (default: fridericia)')
    parser.add_argument('--fpd-method', default=None,
                        choices=['tangent', 'peak', 'max_slope', '50pct', 'baseline_return', 'consensus'],
                        help='FPD measurement method')
    args = parser.parse_args()

    # Build config
    if args.config:
        config = AnalysisConfig.from_json(args.config)
    elif args.preset:
        config = AnalysisConfig.preset(args.preset)
    else:
        config = AnalysisConfig()

    # Apply CLI overrides
    if args.correction != 'fridericia':
        config.repolarization.correction = args.correction
    if args.fpd_method:
        config.repolarization.fpd_method = args.fpd_method
    if args.inclusion_cv == 0:
        config.inclusion.enabled_cv = False
    elif args.inclusion_cv != 25.0:
        config.inclusion.max_cv_bp = args.inclusion_cv
    if args.no_fpdc_filter:
        config.inclusion.enabled_fpdc_range = False

    # Show non-default config
    desc = config.describe()
    if desc != '(all defaults)' and not args.quiet:
        print(f"\n  Config overrides:\n{desc}")

    batch_analyze(args.data_dir, args.channel, args.output, not args.quiet,
                  config=config)


if __name__ == '__main__':
    main()
