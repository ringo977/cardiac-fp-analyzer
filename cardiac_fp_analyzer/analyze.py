#!/usr/bin/env python3
"""
analyze.py — Main entry point for cardiac FP analysis.

Usage:
  python analyze.py /path/to/data/folder [--channel auto] [--output /path/to/output]
"""

import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia
from cardiac_fp_analyzer.beat_detection import compute_beat_periods, detect_beats, segment_beats
from cardiac_fp_analyzer.channel_selection import select_best_channel
from cardiac_fp_analyzer.filtering import full_filter_pipeline
from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria
from cardiac_fp_analyzer.loader import load_csv, parse_filename
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.quality_control import validate_beats
from cardiac_fp_analyzer.report import generate_excel_report, generate_pdf_report
from cardiac_fp_analyzer.rhythm_integration import (
    apply_rhythm_filter,
    apply_rhythm_qc_downgrade,
    apply_rr_outlier_filter,
    build_rhythm_summary_fields,
)

logger = logging.getLogger(__name__)


# Back-compat alias for internal callers
_select_best_channel = select_best_channel
_apply_inclusion_criteria = apply_inclusion_criteria


# ──────────────────────────────────────────────────────────────────
#  Batch error-handling whitelist
#
#  These are exceptions that a single bad CSV should *not* be allowed
#  to abort the whole batch over.  ``analyze_single_file`` catches
#  these internally and returns ``None``; ``_safe_analyze`` is a thin
#  outer guard that handles the corner case where an error escapes
#  from a path the inner ``try`` doesn't cover (e.g. a child import
#  fault), so that the serial and parallel batch loops have a single,
#  unified safety net.
#
#  Pandas-specific entries (``ParserError``, ``EmptyDataError``) and
#  ``UnicodeError`` are explicitly listed because pandas raises them
#  when the CSV header is malformed, the file is empty, or the file's
#  encoding doesn't match the expected one — and prior to this fix
#  those would propagate uncaught from the serial branch and crash
#  the whole batch.
# ──────────────────────────────────────────────────────────────────
_BATCH_SAFE_EXCEPTIONS: tuple = (
    KeyError, ValueError, IndexError, RuntimeError, AssertionError,
    FileNotFoundError, OSError, UnicodeError,
    pd.errors.ParserError, pd.errors.EmptyDataError,
)


def _safe_analyze(filepath, channel='auto', verbose=True, config=None):
    """Run :func:`analyze_single_file` with batch-level error handling.

    Returns
    -------
    (result, error_message) : tuple
        ``result`` is the analyze_single_file return value (a dict on
        success, ``None`` on failure).  ``error_message`` is ``None`` on
        success, otherwise a short string suitable for the batch error
        log.

    This is the single entry point used by both the serial and parallel
    batch loops, so error semantics stay symmetric across the two.
    """
    try:
        result = analyze_single_file(
            filepath, channel=channel, verbose=verbose, config=config
        )
    except _BATCH_SAFE_EXCEPTIONS as e:
        logger.warning("Batch item failed %s: %s", filepath, e, exc_info=True)
        return None, f"{type(e).__name__}: {e}"
    if result is None:
        # analyze_single_file caught an error internally and logged it
        # already; no extra detail to surface here.
        return None, "analysis_returned_none"
    return result, None


def analyze_single_file(filepath, channel='auto', verbose=True, config=None):
    """Analyze a single CSV file through the full pipeline.

    Parameters
    ----------
    filepath : path to CSV file
    channel : 'auto', 'el1', or 'el2'
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
        # Normalize time to start at 0 (hardware may use pre-trigger negative times)
        if len(df) > 0 and df['time'].iloc[0] != 0:
            df['time'] = df['time'] - df['time'].iloc[0]
        if verbose: print(f"  Loaded: {len(df)} samples, Fs={fs} Hz, Duration={len(df)/fs:.1f}s")

        file_info = parse_filename(filepath.name)
        for p in filepath.parts:
            if p.startswith('EXP'): file_info['experiment'] = p; break

        actual_ch = channel
        if channel == 'auto':
            actual_ch, ch_det = select_best_channel(df, fs, cfg=config)
            if verbose:
                print(f"  Channel selection:")
                for ch, d in ch_det.items():
                    print(f"    {ch}: {d}{' *' if ch==actual_ch else ''}")
        file_info['analyzed_channel'] = actual_ch

        raw = df[actual_ch].values

        # ── Amplifier gain correction ──
        # Divide by gain to obtain real voltage.  Default gain=1 (no-op).
        gain = config.amplifier_gain
        if gain != 1.0:
            raw = raw / gain
            if verbose:
                print(f"  Gain correction: ÷{gain:.0e} → amplitude in V")

        raw_signal = raw.copy()  # keep unfiltered signal for display
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

        # ── Detailed pipeline tracing ──
        val_info = det.get('beat_validation', {})
        rec_info = det.get('beat_recovery', {})
        if verbose:
            print(f"  ── Pipeline trace ──")
            print(f"     Detection output: {len(bi)} beats")
            print(f"     Validation: input={val_info.get('n_input', '?')}, "
                  f"accepted={val_info.get('n_accepted', '?')}, "
                  f"rej_amp={val_info.get('n_rejected_amplitude', '?')}, "
                  f"rej_morph={val_info.get('n_rejected_morphology', '?')}, "
                  f"readmitted={val_info.get('n_readmitted', 0)}")
            print(f"     Recovery: {rec_info.get('n_recovered', 0)} recovered")

        rep_cfg = config.repolarization
        # Adaptive post_ms for segmentation — MUST cover BOTH the fixed
        # search_end_ms AND the adaptive search_end_pct_rr × RR window
        # applied downstream in repolarization.py. Otherwise segment_beats
        # produces a template shorter than the adaptive search window,
        # and repolarization.py silently clips `search_end = len(template)`,
        # missing the real T-wave on slow rhythms (e.g. dofetilide, long BP).
        _bp_pre = compute_beat_periods(bi, fs)
        _median_bp_s = float(np.median(_bp_pre)) if len(_bp_pre) > 0 else 0.0
        _pct_rr = getattr(rep_cfg, 'search_end_pct_rr', 0.0)
        _adaptive_end_ms = (_pct_rr * _median_bp_s * 1000.0
                            if (_pct_rr > 0 and _median_bp_s > 0) else 0.0)
        # 50 ms margin after the effective search end so the repolarization
        # tail is never clipped at the template boundary.
        _post_ms = max(850.0,
                       rep_cfg.search_end_ms + 50.0,
                       _adaptive_end_ms + 50.0)
        bd, btm, vi = segment_beats(filtered, df['time'].values, bi, fs,
                                     pre_ms=rep_cfg.segment_pre_ms,
                                     post_ms=_post_ms)
        if verbose and _adaptive_end_ms > rep_cfg.search_end_ms:
            print(f"     Segmentation: adaptive post_ms={_post_ms:.0f} ms "
                  f"(median BP={_median_bp_s*1000:.0f} ms, "
                  f"{_pct_rr*100:.0f}% RR={_adaptive_end_ms:.0f} ms "
                  f"> fixed search_end_ms={rep_cfg.search_end_ms:.0f})")

        # Use only successfully segmented beats (edge-truncated beats removed).
        # vi contains indices into bi of beats that fit the pre/post window.
        bi_seg = bi[np.array(vi)] if len(vi) > 0 else bi[:0]
        n_seg_dropped = len(bi) - len(bi_seg)
        if verbose and n_seg_dropped > 0:
            print(f"     Segmentation: {n_seg_dropped} edge beats dropped "
                  f"({len(bi)} → {len(bi_seg)})")
        assert len(bi_seg) == len(bd) == len(btm), (
            f"Segmentation alignment error: bi_seg={len(bi_seg)}, "
            f"beats_data={len(bd)}, beats_time={len(btm)}"
        )

        # ─── Quality Control: validate beats ───
        qc_report, bi_clean, bd_clean, btm_clean = validate_beats(
            filtered, bi_seg, bd, btm, fs, cfg=config.quality
        )
        if verbose:
            n_rej = qc_report.n_beats_input - qc_report.n_beats_accepted
            print(f"  QC: Grade {qc_report.grade} | SNR={qc_report.global_snr:.1f} | "
                  f"Accepted {qc_report.n_beats_accepted}/{qc_report.n_beats_input} "
                  f"(-{n_rej}: {qc_report.n_beats_rejected_snr} SNR, "
                  f"{qc_report.n_beats_rejected_morphology} morph)")
            for note in qc_report.notes:
                print(f"    >> {note}")

        # ─── Rhythm-topology-aware beat filtering (Sprint 2 #3) ───
        # For rhythm types where the "secondary" / "noise" amplitude
        # clusters would contaminate the template (alternans 2:1,
        # ectopics, noise, trimodal), restrict parameter extraction to
        # the dominant cluster only. Regular/chaotic/ambiguous signals
        # pass through unchanged.
        rc = det.get('rhythm_classification', {}) if isinstance(det, dict) else {}
        bd_fpd, btm_fpd, bi_fpd, rhythm_filter_info = apply_rhythm_filter(
            bd_clean, btm_clean, bi_clean, bi,
            rhythm_classification=rc,
            enable=getattr(config.beat_detection, 'enable_rhythm_aware_fpd', True),
            min_retention_ratio=getattr(config.beat_detection,
                                         'rhythm_filter_min_retention_ratio', 0.5),
            min_retention_beats=getattr(config.beat_detection,
                                         'rhythm_filter_min_retention_beats', 5),
        )
        if verbose and rhythm_filter_info.get('filter_applied'):
            print(f"  Rhythm filter ({rhythm_filter_info['rhythm_type']}): "
                  f"kept {rhythm_filter_info['n_kept']}/{rhythm_filter_info['n_input']} "
                  f"beats ({rhythm_filter_info['kept_role']} cluster only)")
        elif verbose and rhythm_filter_info.get('reason') == 'safety_bail_low_retention':
            sb = rhythm_filter_info.get('safety_bail', {})
            print(f"  Rhythm filter ({rhythm_filter_info['rhythm_type']}): "
                  f"SAFETY BAIL — would keep {sb.get('n_would_keep')}/{rhythm_filter_info['n_input']} "
                  f"({sb.get('retention_ratio', 0)*100:.0f}% < "
                  f"{sb.get('min_retention_ratio', 0)*100:.0f}%), passthrough")

        # Apply QC downgrade for noise-contaminated signals (mutates qc_report.grade).
        _qc_downgrade_info = apply_rhythm_qc_downgrade(
            qc_report, rc,
            downgrade_threshold=getattr(config.beat_detection,
                                         'rhythm_qc_downgrade_threshold', 0.30),
            downgrade_steps=getattr(config.beat_detection,
                                     'rhythm_qc_downgrade_steps', 1),
            enable=getattr(config.beat_detection, 'enable_rhythm_aware_fpd', True),
        )
        if verbose and _qc_downgrade_info.get('applied'):
            print(f"  QC downgrade: {_qc_downgrade_info['grade_before']} → "
                  f"{_qc_downgrade_info['grade_after']} "
                  f"(noise_ratio={_qc_downgrade_info['noise_ratio']:.2f})")

        # ─── RR-outlier filter (Sprint 3 #1) ───
        # Drop beats whose preceding RR is pathologically long (likely
        # dropout-plus-reactivation artefact). Genuinely bradycardic
        # recordings are passed through unchanged.
        bd_fpd, btm_fpd, bi_fpd, rr_filter_info = apply_rr_outlier_filter(
            bd_fpd, btm_fpd, bi_fpd, fs,
            max_rr_ratio=getattr(config.beat_detection,
                                  'max_rr_outlier_ratio', 5.0),
            enable=getattr(config.beat_detection,
                            'enable_rr_outlier_filter', True),
        )
        if verbose and rr_filter_info.get('filter_applied'):
            print(f"  RR filter: dropped {rr_filter_info['n_dropped']} beat(s) "
                  f"with RR > {rr_filter_info['max_rr_ratio']:.1f}× median "
                  f"(median RR={rr_filter_info['median_rr_ms']:.0f} ms)")

        # ─── Re-segmentation guard (Sprint 3 #2) ───
        # The initial segmentation used a post_ms derived from the *pre-filter*
        # median RR. On signals where beat detection picks up many false
        # positives between the true (bradycardic) beats, that median is
        # 3-5× shorter than the real one. After the rhythm + RR filters trim
        # the false positives, the kept beats may have a true median RR that
        # pushes the T-wave search window (0.70 × RR) beyond the segment
        # length. When that happens the T-wave is silently clipped at the
        # segment boundary and the repolarization detector falls back to an
        # afterpotential or fails the SNR gate.
        # Guard: if the post-filter median RR would require a meaningfully
        # longer segment, re-segment the kept beats with the correct post_ms
        # before parameter extraction. On signals where the filter changes
        # little (normal case), this is a no-op.
        _resegmented_info = None
        if len(bi_fpd) >= 3:
            _bp_post = compute_beat_periods(bi_fpd, fs)
            _median_bp_post_s = (float(np.median(_bp_post))
                                 if len(_bp_post) > 0 else 0.0)
            if _pct_rr > 0 and _median_bp_post_s > 0:
                _adaptive_end_post_ms = _pct_rr * _median_bp_post_s * 1000.0
            else:
                _adaptive_end_post_ms = 0.0
            _post_ms_needed = max(850.0,
                                   rep_cfg.search_end_ms + 50.0,
                                   _adaptive_end_post_ms + 50.0)
            # Only re-segment if the pre-filter segment is meaningfully
            # shorter than what the post-filter RR now demands.
            if _post_ms_needed > _post_ms + 100.0:
                bd_re, btm_re, vi_re = segment_beats(
                    filtered, df['time'].values, bi_fpd, fs,
                    pre_ms=rep_cfg.segment_pre_ms,
                    post_ms=_post_ms_needed,
                )
                if len(bd_re) > 0:
                    bd_fpd = bd_re
                    btm_fpd = btm_re
                    bi_fpd = (bi_fpd[np.array(vi_re)]
                              if len(vi_re) < len(bi_fpd) else bi_fpd)
                    _resegmented_info = {
                        'applied': True,
                        'post_ms_before': float(_post_ms),
                        'post_ms_after': float(_post_ms_needed),
                        'median_rr_pre_ms': float(_median_bp_s * 1000.0),
                        'median_rr_post_ms': float(_median_bp_post_s * 1000.0),
                        'n_beats_after': int(len(bd_fpd)),
                    }
                    if verbose:
                        print(f"  Re-segmentation: post_ms "
                              f"{_post_ms:.0f}→{_post_ms_needed:.0f} ms "
                              f"(pre-filter median RR={_median_bp_s*1000:.0f} ms "
                              f"→ true median RR={_median_bp_post_s*1000:.0f} ms)")

        # Use (possibly filtered and re-segmented) beats for parameter extraction.
        all_p, summary = extract_all_parameters(bd_fpd, btm_fpd, bi_fpd, fs,
                                                 cfg=rep_cfg)
        if _resegmented_info is not None:
            summary['resegmentation_info'] = _resegmented_info
        # Merge rhythm-classification-derived fields into summary (additive).
        summary.update(build_rhythm_summary_fields(rc, rhythm_filter_info))
        summary['rr_outlier_filter'] = rr_filter_info

        # Beat period from ALL detected beats (timing is reliable even for
        # morphologically marginal beats) — avoids artificial gaps from QC rejection.
        bp = compute_beat_periods(bi, fs)

        if verbose and len(bp) > 0:
            print(f"  BP: {np.mean(bp)*1000:.0f}ms ({60/np.mean(bp):.1f} BPM)")

        ar = analyze_arrhythmia(bi, bp, all_p, summary, fs,
                               cfg=config.arrhythmia, beats_data=bd_clean)
        if verbose:
            print(f"  {ar.classification} (Risk: {ar.risk_score}/100)")

        result = {'metadata': metadata, 'file_info': file_info, 'summary': summary,
                'all_params': all_p, 'arrhythmia_report': ar,
                'beat_indices': bi_clean, 'beat_indices_raw': bi,
                # Post rhythm/RR filter + re-segmentation: the beats that
                # actually fed parameter extraction. Used by the UI to mark
                # the "real" included beats on the signal plot.
                'beat_indices_fpd': np.asarray(bi_fpd, dtype=int),
                'beat_periods': bp, 'filtered_signal': filtered,
                'raw_signal': raw_signal,
                'time_vector': df['time'].values,
                'beats_data': bd_clean, 'beats_time': btm_clean,
                'qc_report': qc_report,
                'detection_info': det}

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
    except _BATCH_SAFE_EXCEPTIONS as e:
        logger.error("Analysis failed for %s: %s", filepath, e, exc_info=True)
        if verbose:
            print(f"  ERROR: {e}")
            traceback.print_exc()
        return None


def batch_analyze(data_dir, channel='auto', output_dir=None, verbose=True,
                  config=None, n_workers=1,
                  # Legacy parameters (ignored when config is provided)
                  inclusion_cv=25.0, fpdc_range=(100, 1200),
                  min_fpd_confidence=0.68):
    """
    Batch analysis of all CSV files in a directory.

    Parameters
    ----------
    data_dir : path to directory with CSV files
    channel : 'auto', 'el1', 'el2', or 'both'
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

    # When channel='both', analyze each file for el1 and el2 separately
    channels_to_run = ['el1', 'el2'] if channel == 'both' else [channel]

    if n_workers > 1 and len(csv_files) > 1:
        # Parallel pass 1: each file is independent.  We submit the
        # bare analyze_single_file (it cannot be a closure because
        # ProcessPoolExecutor pickles it), and apply the _safe_analyze
        # exception whitelist when awaiting .result() — this keeps the
        # parallel branch's error semantics aligned with the serial
        # branch without forcing _safe_analyze itself to be picklable.
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n_workers = min(n_workers, len(csv_files), _mp.cpu_count() or 4)
        if verbose:
            print(f"  Parallel processing: {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for f in csv_files:
                for ch in channels_to_run:
                    fut = executor.submit(analyze_single_file, f,
                                          channel=ch, verbose=False, config=config)
                    futures[fut] = (f, ch)
            total = len(futures)
            for i, future in enumerate(as_completed(futures)):
                f, ch = futures[future]
                ch_tag = f" ({ch})" if channel == 'both' else ""
                if verbose:
                    print(f"\n[{i+1}/{total}] {f.name}{ch_tag}")
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                    else:
                        errors.append(f"{f}{ch_tag}")
                except _BATCH_SAFE_EXCEPTIONS as e:
                    logger.warning("Batch item failed %s%s: %s", f, ch_tag, e)
                    errors.append(f"{f}{ch_tag}: {type(e).__name__}: {e}")
    else:
        # Serial branch: route every file through _safe_analyze so that
        # a malformed CSV (pandas ParserError, encoding mismatch, empty
        # file, etc.) becomes a logged error entry instead of crashing
        # the whole batch.
        total = len(csv_files) * len(channels_to_run)
        idx = 0
        for f in csv_files:
            for ch in channels_to_run:
                idx += 1
                ch_tag = f" ({ch})" if channel == 'both' else ""
                print(f"\n[{idx}/{total}] {f.name}{ch_tag}")
                r, err = _safe_analyze(f, channel=ch, verbose=verbose,
                                        config=config)
                if r:
                    results.append(r)
                else:
                    errors.append(f"{f}{ch_tag}: {err}" if err else f"{f}{ch_tag}")

    # ─── Baseline risk reset ───
    # Baselines are reference recordings (no drug applied). The arrhythmia
    # risk score measures drug-induced proarrhythmic risk, so it's not
    # meaningful for baselines. We keep the flags (useful for QC) but
    # reset risk_score and classification.
    from .normalization import is_baseline
    for r in results:
        if is_baseline(r):
            ar = r.get('arrhythmia_report')
            if ar is not None:
                ar.risk_score = 0
                ar.classification = 'Baseline (reference)'

    # ─── Inclusion criteria ───
    results = apply_inclusion_criteria(results, verbose=verbose, cfg=config.inclusion)

    # ─── Baseline-relative residual analysis (pass 2) ───
    # Collect baseline templates per group (chip+channel), then re-run
    # arrhythmia analysis for drug recordings using the baseline template.
    # This captures drug-induced morphology changes vs. normal baseline.
    from .arrhythmia import analyze_arrhythmia as _analyze_arrhythmia
    from .arrhythmia import compute_template
    from .normalization import get_group_key

    baseline_templates = {}
    for r in results:
        if not is_baseline(r):
            continue
        # Only use baselines that passed inclusion criteria
        inc = r.get('inclusion', {})
        if not inc.get('passed', True):
            continue
        bd = r.get('beats_data')
        if bd is None or len(bd) < 5:
            continue
        group = get_group_key(r)
        tmpl = compute_template(bd)
        if tmpl is not None:
            baseline_templates[group] = tmpl

    if baseline_templates:
        n_reanalyzed = 0
        for r in results:
            if is_baseline(r):
                continue
            group = get_group_key(r)
            bl_tmpl = baseline_templates.get(group)
            if bl_tmpl is None:
                continue
            bd = r.get('beats_data')
            if bd is None or len(bd) < 5:
                continue
            # Re-run arrhythmia analysis with baseline template.
            # Use cleaned beat_indices and recompute beat_periods from them
            # to ensure n_beats and CV denominators are consistent.
            # (beat_periods in the result dict are from raw/all detected beats,
            # which can diverge from cleaned beat_indices when QC rejects many.)
            bi = r.get('beat_indices', np.array([]))
            bp = compute_beat_periods(bi, r.get('metadata', {}).get('sample_rate', 1000.0))
            all_p = r.get('all_params', [])
            summary = r.get('summary', {})
            fs_val = r.get('metadata', {}).get('sample_rate', 1000.0)
            ar = _analyze_arrhythmia(
                bi, bp, all_p, summary, fs_val,
                cfg=config.arrhythmia, beats_data=bd,
                baseline_template=bl_tmpl
            )
            r['arrhythmia_report'] = ar
            n_reanalyzed += 1
        if verbose:
            print(f"  Baseline-relative residual analysis: {n_reanalyzed} drug recordings "
                  f"re-analyzed with {len(baseline_templates)} baseline template(s)")

    # Compact memory: replace raw beats_data (~100KB/rec) with
    # precomputed template (~1KB/rec) for downstream waveform display.
    # Also drop raw_signal (only needed in single-file interactive mode).
    for r in results:
        r.pop('raw_signal', None)
        bd = r.pop('beats_data', None)
        if bd is not None and len(bd) >= 5 and 'beat_template' not in r:
            tmpl = compute_template(bd)
            if tmpl is not None:
                r['beat_template'] = tmpl
    del baseline_templates

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

    # Configure logging for CLI use
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='Cardiac FP Analyzer for hiPSC-CM µECG')
    parser.add_argument('data_dir')
    parser.add_argument('--channel', default='auto', choices=['auto','el1','el2','both'])
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
