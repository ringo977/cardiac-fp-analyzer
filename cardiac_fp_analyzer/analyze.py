#!/usr/bin/env python3
"""
analyze.py — Main entry point for cardiac FP analysis.

Usage:
  python analyze.py /path/to/data/folder [--channel auto] [--output /path/to/output]
"""

import argparse
import logging
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia
from cardiac_fp_analyzer.beat_detection import (
    compute_beat_periods,
    detect_beats,
    rhythm_train,
    segment_beats,
)
from cardiac_fp_analyzer.channel_selection import select_best_channel
from cardiac_fp_analyzer.filtering import full_filter_pipeline
from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria
from cardiac_fp_analyzer.loader import describe_recording, load_csv
from cardiac_fp_analyzer.overrides import apply_overrides, load_overrides
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.quality_control import assess_analysability, validate_beats
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


def _safe_analyze(filepath, channel='auto', verbose=True, config=None, file_info_update=None):
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
        kw = {'file_info_update': file_info_update} if file_info_update else {}
        result = analyze_single_file(
            filepath, channel=channel, verbose=verbose, config=config, **kw
        )
    except _BATCH_SAFE_EXCEPTIONS as e:
        logger.warning("Batch item failed %s: %s", filepath, e, exc_info=True)
        return None, f"{type(e).__name__}: {e}"
    if result is None:
        # analyze_single_file caught an error internally and logged it
        # already; no extra detail to surface here.
        return None, "analysis_returned_none"
    return result, None


# ──────────────────────────────────────────────────────────────────
#  Post-detection pipeline (extracted from analyze_single_file)
#
#  This helper runs everything downstream of beat detection:
#  segmentation, QC, rhythm-topology filter, RR-outlier filter,
#  parameter extraction, repolarization / FPD, arrhythmia analysis,
#  and the optional cessation + spectral add-ons.
#
#  It is called by:
#    • analyze_single_file  — the full pipeline (I/O → detect → here).
#    • recompute_from_beats — the UI "Ricalcola" button, when the user
#      has manually corrected the beat set in the PySide6 viewer and
#      wants the Parametri/Aritmie tabs refreshed without re-running
#      detection.  The signal / filter / time arrays from the original
#      run are reused; only ``bi`` is replaced.
#
#  The split is a pure refactor: for any fixed ``bi`` the output is
#  bit-identical to the pre-refactor analyze_single_file result.  A
#  non-regression test in tests/test_recompute_from_beats.py pins this.
# ──────────────────────────────────────────────────────────────────

def _analyze_from_beats(
    bi,
    *,
    filtered,
    raw_signal,
    time_vector,
    fs,
    metadata,
    file_info,
    config,
    detection_info=None,
    verbose=True,
    rhythm_bi=None,
):
    """Run the post-detection pipeline on an already-known beat set.

    ``rhythm_bi`` (optional) is the rhythm train among the detections
    (beat_detection.rhythm_train). When given, beat period, CV, the local RR
    of the rate correction and the repolarisation search window come from
    it; segmentation, QC, FPD and the arrhythmia analysis keep every
    detection.

    Parameters
    ----------
    bi : 1-D int array
        Beat sample indices.  Either the output of ``detect_beats`` (full
        pipeline) or user-corrected indices from the PySide6 viewer.
    filtered : 1-D float array
        Band-pass filtered signal from ``full_filter_pipeline``.
    raw_signal : 1-D float array
        Gain-corrected unfiltered signal — stored in the result so the
        UI can overlay it on the plot; not used for analysis.
    time_vector : 1-D float array
        Time axis (seconds), normalised to start at 0.  Same length as
        ``filtered``.
    fs : float
        Sample rate (Hz).
    metadata, file_info : dict
        Loader outputs — forwarded verbatim into the returned result.
    config : AnalysisConfig
        Pipeline configuration.
    detection_info : dict or None
        Output of ``detect_beats``.  Used for ``rhythm_classification``
        in the rhythm-aware FPD filter and for verbose pipeline-trace
        logging.  ``None`` (e.g. UI recompute with no re-detection) is
        treated as an empty dict — the rhythm filter then skips its
        cluster-based trimming and passes every beat through.
    verbose : bool
        Print progress to stdout (kept symmetric with
        ``analyze_single_file``).

    Returns
    -------
    dict
        Full result dict, same shape as ``analyze_single_file`` returns.
        Raises exceptions normally; the outer wrapper is responsible
        for batch-level error handling.
    """
    bi = np.asarray(bi, dtype=int)
    det = detection_info or {}
    bi_stats = bi if rhythm_bi is None else np.asarray(rhythm_bi, dtype=int)

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
    _bp_pre = compute_beat_periods(bi_stats, fs)
    _median_bp_s = float(np.median(_bp_pre)) if len(_bp_pre) > 0 else 0.0
    _pct_rr = getattr(rep_cfg, 'search_end_pct_rr', 0.0)
    _adaptive_end_ms = (_pct_rr * _median_bp_s * 1000.0
                        if (_pct_rr > 0 and _median_bp_s > 0) else 0.0)
    # 50 ms margin after the effective search end so the repolarization
    # tail is never clipped at the template boundary.
    _post_ms = max(850.0,
                   rep_cfg.search_end_ms + 50.0,
                   _adaptive_end_ms + 50.0)
    bd, btm, vi = segment_beats(filtered, time_vector, bi, fs,
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

    # ─── Analysability verdict (Oct 2026) ───
    # Decided on the full detected train against the recording's own noise
    # floor; see QualityConfig.enable_analysability_verdict. Applied to the
    # outputs after parameter extraction (below) so beat counts stay
    # available for audit while FPD/FPDc are withheld.
    _verdict = assess_analysability(filtered, fs, bi, cfg=config.quality)
    if _verdict['not_analysable']:
        qc_report.grade = 'F'
        qc_report.not_analysable = True
        qc_report.not_analysable_reason = _verdict['reason']
        if verbose:
            print(f"  NOT ANALYSABLE: {_verdict['reason']}")

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
                filtered, time_vector, bi_fpd, fs,
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
    # ``bi`` is the full detected train; ``bi_fpd`` is what survived QC and
    # the rhythm/RR filters. Passing both lets each accepted beat be
    # rate-corrected against its real predecessor instead of against the
    # next surviving beat, and lets the beat-period summary describe the
    # actual rhythm rather than the QC rejection pattern.
    all_p, summary = extract_all_parameters(bd_fpd, btm_fpd, bi_fpd, fs,
                                             cfg=rep_cfg,
                                             all_beat_indices=bi_stats)
    if _resegmented_info is not None:
        summary['resegmentation_info'] = _resegmented_info
    # Merge rhythm-classification-derived fields into summary (additive).
    summary.update(build_rhythm_summary_fields(rc, rhythm_filter_info))
    summary['rr_outlier_filter'] = rr_filter_info
    summary['beat_snr_median'] = _verdict['beat_snr_median']
    summary['not_analysable'] = bool(_verdict['not_analysable'])
    summary['not_analysable_reason'] = _verdict['reason'] if _verdict['not_analysable'] else ''
    if _verdict['not_analysable']:
        # Withhold repolarisation outputs: they would be numbers without a
        # signal behind them. Beat-period fields are kept (audit) but the
        # reliability flags make the state explicit to every consumer.
        for _k in list(summary.keys()):
            if _k.startswith(('fpd_ms', 'fpdc_', 'fpd_confidence', 'template_fpd')) \
                    and isinstance(summary[_k], (int, float, np.floating)):
                summary[_k] = np.nan
        summary['fpd_reliable'] = False
        summary['fpd_valid_ratio'] = 0.0
        summary['fpd_note'] = 'not analysable: ' + _verdict['reason']

    # Beat period from ALL detected beats (timing is reliable even for
    # morphologically marginal beats) — avoids artificial gaps from QC rejection.
    bp = compute_beat_periods(bi, fs)

    if verbose and len(bp) > 0:
        print(f"  BP: {np.mean(bp)*1000:.0f}ms ({60/np.mean(bp):.1f} BPM)")

    ar = analyze_arrhythmia(bi, bp, all_p, summary, fs,
                           cfg=config.arrhythmia, beats_data=bd_clean)
    if summary.get('not_analysable'):
        ar.classification = 'Not analysable'
        ar.risk_score = 0
        ar.add_flag('not_analysable', 'critical', _verdict['reason'])
    if verbose:
        print(f"  {ar.classification} (Risk: {ar.risk_score}/100)")

    result = {'metadata': metadata, 'file_info': file_info, 'summary': summary,
            'all_params': all_p, 'arrhythmia_report': ar,
            'beat_indices': bi_clean, 'beat_indices_raw': bi,
            # The rhythm train (beat_detection.rhythm_train): equal to the
            # raw detections unless that option removed beats from them.
            'beat_indices_rhythm': np.asarray(bi_stats, dtype=int),
            # Post rhythm/RR filter + re-segmentation: the beats that
            # actually fed parameter extraction. Used by the UI to mark
            # the "real" included beats on the signal plot.
            'beat_indices_fpd': np.asarray(bi_fpd, dtype=int),
            'beat_periods': bp, 'filtered_signal': filtered,
            'raw_signal': raw_signal,
            'time_vector': time_vector,
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


def recompute_from_beats(result, bi_edited, config=None, verbose=False):
    """Re-run the post-detection pipeline with a user-corrected beat set.

    Used by the PySide6 UI "Ricalcola" button: the user adds / removes
    beats in the viewer and asks for fresh Parametri + Aritmie tabs
    without paying the cost of re-loading the CSV and re-running beat
    detection.  The filtered signal, raw signal, and time vector are
    reused from the original analysis.

    Parameters
    ----------
    result : dict
        A previous result produced by ``analyze_single_file``.  Must
        contain ``filtered_signal``, ``raw_signal``, ``time_vector``,
        ``metadata``, ``file_info``, and (preferably) ``detection_info``.
    bi_edited : array-like of int
        Sample indices of the beat set to analyse.  Typically produced
        by the viewer's ``get_beat_indices()`` after manual edits.
    config : AnalysisConfig or None
        Configuration.  ``None`` builds defaults (matches the UI's
        "Apri CSV" path).  Pass an explicit config only when the caller
        knows the original run used non-default settings.
    verbose : bool
        Echo pipeline trace to stdout.  Off by default — the UI does
        not normally want console noise.

    Returns
    -------
    dict
        A fresh result dict.  Signal / time arrays are shared with the
        input, so updating the returned dict does NOT mutate the
        original.

    Raises
    ------
    Any exception the inner pipeline raises (no batch-safe wrapper).
    """
    if config is None:
        from .config import AnalysisConfig
        config = AnalysisConfig()

    # Prefer the sample rate that the loader captured; fall back to
    # deriving it from the time vector.  The derived value can drift
    # by floating-point noise, so only use it when the metadata is
    # unavailable.
    fs = float(result.get('metadata', {}).get('sample_rate') or 0.0)
    if fs <= 0:
        tv = result['time_vector']
        if len(tv) >= 2:
            fs = 1.0 / float(tv[1] - tv[0])
        else:
            raise ValueError(
                "recompute_from_beats: cannot determine sample rate "
                "(metadata['sample_rate'] missing and time_vector too short)."
            )

    return _analyze_from_beats(
        np.asarray(bi_edited, dtype=int),
        filtered=result['filtered_signal'],
        raw_signal=result['raw_signal'],
        time_vector=result['time_vector'],
        fs=fs,
        metadata=result.get('metadata', {}),
        file_info=result.get('file_info', {}),
        config=config,
        detection_info=result.get('detection_info', {}),
        verbose=verbose,
    )


def analyze_single_file(filepath, channel='auto', verbose=True, config=None, file_info_update=None):
    """Analyze a single CSV file through the full pipeline.

    Parameters
    ----------
    filepath : path to CSV file
    channel : 'auto', 'el1', or 'el2'
    verbose : print progress
    config : AnalysisConfig or None — controls all pipeline parameters
    file_info_update : dict or None — fields that replace those read from
        the path (tissue, item, dose, role …) when the recording is one input
        of a two-tissue file or is described in samples.csv
        (sample_sheet.plan_batch)
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

        # Experiment, day, chip and chamber from the folder layout; 'tissue'
        # is the key normalisation pairs on (see loader.describe_recording).
        file_info = describe_recording(filepath)
        if file_info_update:
            file_info.update(file_info_update)

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

        # Snapshot the pure (pre-override) detection so the PySide UI can
        # reconstruct the "automatic" baseline when writing a new sidecar
        # from Ricalcola (task #79).  Without this, a user who Ricalcola-s
        # after loading a CSV that already had a sidecar would overwrite
        # it with a diff against the *post-override* set — silently losing
        # earlier corrections.
        det = dict(det)
        det['bi_detection'] = [int(x) for x in bi]

        # ── Manual beat overrides (sidecar .overrides.json) ──
        # If the user has previously corrected this file in the PySide6
        # viewer (add/remove beats) a ``<csv>.overrides.json`` sits next
        # to the CSV.  Apply it here so the correction survives re-opens
        # and batch re-runs without the UI having to redo the diff.
        # Overrides are in seconds so they are robust to re-sampling.
        # The sidecar holds one beat list per file: with two tissues in the
        # file it cannot say which input it was made on, so it is not used.
        if config.use_overrides and not file_info.get('two_tissue_file'):
            ov = load_overrides(filepath)
            if ov is not None and not ov.is_empty():
                bi, ov_info = apply_overrides(bi, ov, fs)
                # bt (time-domain beats) is not passed to the post-detection
                # pipeline — only bi reaches segmentation — so we do not
                # recompute it here.  Keep detection_info in sync so any
                # downstream log line that reads ``det['n_beats']`` sees
                # the post-override count.
                det['overrides_applied'] = ov_info
                det['n_beats'] = int(bi.size)
                if verbose:
                    n_unm = len(ov_info['unmatched_removals'])
                    extras = f", {n_unm} unmatched" if n_unm else ""
                    print(f"  Overrides: +{ov_info['n_added']}, "
                          f"-{ov_info['n_removed']}{extras} "
                          f"→ {bi.size} beats")

        # Rhythm train among the detections (off by default); not after a
        # manual beat correction, which is the user's own beat set.
        rhythm_bi = None
        if not det.get('overrides_applied'):
            bi_r, r_info = rhythm_train(filtered, fs, bi, cfg=bd_cfg)
            det['rhythm_train'] = r_info
            if r_info.get('rhythm_train') == 'applied':
                rhythm_bi = bi_r
                if verbose:
                    print(f"  Rhythm train: {r_info['n_kept']}/{r_info['n_input']} detections "
                          f"(period {r_info['period_ms']:.0f} ms) + {r_info['n_recovered']} recovered")

        return _analyze_from_beats(
            bi,
            filtered=filtered,
            raw_signal=raw_signal,
            time_vector=df['time'].values,
            fs=fs,
            metadata=metadata,
            file_info=file_info,
            config=config,
            detection_info=det,
            verbose=verbose,
            rhythm_bi=rhythm_bi,
        )
    except _BATCH_SAFE_EXCEPTIONS as e:
        logger.error("Analysis failed for %s: %s", filepath, e, exc_info=True)
        if verbose:
            print(f"  ERROR: {e}")
            traceback.print_exc()
        return None


def _run_batch_jobs(jobs, config, n_workers, verbose, show_channel, counter, updates=None):
    """Analyse jobs; return ({job: result}, [errors]).

    A job is (path, channel) or (path, channel, uid); ``updates`` maps a uid
    to the file_info fields of that recording (one input of a two-tissue
    file, or a recording described in samples.csv — sample_sheet.plan_batch).

    Serial jobs go through _safe_analyze; parallel ones apply the same
    exception whitelist when awaiting the future, so both branches report
    a malformed CSV as an error entry instead of aborting the batch.
    """
    out, errors = {}, []
    updates = updates or {}

    def tag(ch):
        return f" ({ch})" if show_channel else ""

    def extra(job):
        upd = updates.get(job[2]) if len(job) > 2 else None
        return {'file_info_update': upd} if upd else {}

    if n_workers > 1 and len(jobs) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n = min(n_workers, len(jobs), _mp.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=n) as executor:
            futures = {executor.submit(analyze_single_file, job[0], channel=job[1],
                                       verbose=False, config=config, **extra(job)): job
                       for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                f, ch = job[0], job[1]
                counter['done'] += 1
                if verbose:
                    print(f"\n[{counter['done']}/{counter['total']}] {f.name}{tag(ch)}")
                try:
                    r = future.result()
                    if r:
                        out[job] = r
                    else:
                        errors.append(f"{f}{tag(ch)}")
                except _BATCH_SAFE_EXCEPTIONS as e:
                    logger.warning("Batch item failed %s%s: %s", f, tag(ch), e)
                    errors.append(f"{f}{tag(ch)}: {type(e).__name__}: {e}")
    else:
        for job in jobs:
            f, ch = job[0], job[1]
            counter['done'] += 1
            print(f"\n[{counter['done']}/{counter['total']}] {f.name}{tag(ch)}")
            r, err = _safe_analyze(f, channel=ch, verbose=verbose, config=config, **extra(job))
            if r:
                out[job] = r
            else:
                errors.append(f"{f}{tag(ch)}: {err}" if err else f"{f}{tag(ch)}")
    return out, errors


def _choose_tissue_electrodes(baseline_out, unit_info, dose_jobs):
    """Electrode for each tissue = the one used on its reference.

    ``baseline_out`` maps each reference job (path, channel, uid) to its
    result, ``unit_info`` every job to the recording's file_info as planned
    (sample_sheet.plan_batch), ``dose_jobs`` lists the other jobs.

    The reference is the last one recorded before the tissue's first dose
    (acquisition time from the file header; Oct 2026), the same rule
    normalization.pair_with_baselines applies. Without times: t0 first,
    then the one in the folder holding most of the tissue's other
    recordings (a 'new baseline' sub-folder or a '_bis' copy loses to the
    baseline recorded with the dose series), then the better QC grade,
    then the file name. References the analysability verdict rejected are
    used only when the tissue has no other. A reference whose input is
    fixed (two-tissue file, samples.csv) gives its input.
    """
    from .loader import recording_datetime
    from .normalization import (
        canonical_drug_name,
        last_reference_before,
        recording_time,
        reference_kind,
    )

    grade_rank = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
    others = defaultdict(list)
    first_dose = {}
    for job in dose_jobs:
        f, info = job[0], unit_info[job]
        tissue = info.get('tissue')
        if not tissue:
            continue
        others[tissue].append(f.parent)
        drug = canonical_drug_name(info.get('drug'))
        text = f"{info.get('drug') or ''} {f.name}".lower()
        if drug.startswith('ctr') or 'wash' in text or 'recovery' in text or info.get('role') == 'washout':
            continue
        ts = recording_datetime(f)
        if ts is not None and (tissue not in first_dose or ts < first_dose[tissue]):
            first_dose[tissue] = ts
    by_tissue = defaultdict(list)
    for job, r in baseline_out.items():
        by_tissue[unit_info[job]['tissue']].append((job, r))
    chosen = {}
    for tissue, cands in by_tissue.items():
        def score(jr):
            job, r = jr
            f = job[0]
            same_dir = sum(1 for d in others.get(tissue, []) if d == f.parent)
            grade = getattr(r.get('qc_report'), 'grade', 'F')
            return (reference_kind(r) != 't0', -same_dir, grade_rank.get(grade, 5), f.name, str(job[1]))
        usable = [jr for jr in cands if not (jr[1].get('summary') or {}).get('not_analysable', False)] or cands
        picked, _ = last_reference_before([(jr, recording_time(jr[1])) for jr in usable],
                                          first_dose.get(tissue), score)
        job, r = picked if picked is not None else min(usable, key=score)
        el = r['file_info'].get('analyzed_channel', 'el1')
        name = job[0].name + (f' [{el}]' if unit_info[job].get('two_tissue_file') else '')
        chosen[tissue] = {'electrode': el, 'baseline_file': name}
    return chosen


def _warn_tissue_collisions(infos, data_dir, verbose):
    """Warn when one tissue key spans several top-level folders and no
    experiment folder was recognised: experiments may be mixed."""
    tops = defaultdict(set)
    for key, info in infos.items():
        f = key[0] if isinstance(key, tuple) else key
        t = info.get('tissue')
        if t and t.startswith('-/'):
            try:
                tops[t].add(f.relative_to(data_dir).parts[0])
            except (ValueError, IndexError):
                continue
    mixed = {t: sorted(d) for t, d in tops.items() if len(d) > 1}
    if mixed:
        msg = (f"{len(mixed)} tissue key(s) span several top-level folders and no "
               f"experiment folder ('Exp<N>') was found — recordings of different "
               f"experiments may be paired. Example: {next(iter(mixed))} in {next(iter(mixed.values()))}")
        logger.warning(msg)
        if verbose:
            print(f"  WARNING: {msg}")
    return mixed


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

    from .sample_sheet import describe_plan, is_sheet_file, plan_batch
    csv_files = sorted(p for p in data_dir.rglob('*.csv') if not is_sheet_file(p))
    print(f"\n{'#'*60}\n  CARDIAC FP ANALYZER\n  Files: {len(csv_files)} | Channel: {channel}\n{'#'*60}")

    # Save config alongside results
    config.to_json(str(output_dir / 'analysis_config.json'))

    results, errors = [], []

    # When channel='both', analyze each file for el1 and el2 separately
    channels_to_run = ['el1', 'el2'] if channel == 'both' else [channel]
    counter = {'done': 0, 'total': len(csv_files) * len(channels_to_run)}
    if n_workers > 1 and len(csv_files) > 1 and verbose:
        import multiprocessing as _mp
        print(f"  Parallel processing: {min(n_workers, len(csv_files), _mp.cpu_count() or 4)} workers")

    if channel == 'auto':
        # One electrode per tissue (Oct 2026). Choosing the electrode file by
        # file made a dose series mix el1 and el2 of the same microtissue (39
        # of 75 recordings on the Visone 2023 data). Baselines are analysed
        # first; each tissue then keeps its baseline's electrode.
        # A file holding two tissues (one per input) gives one recording per
        # input, and a samples.csv in the folder overrides the names for the
        # files it lists (sample_sheet.plan_batch).
        infos = {f: describe_recording(f) for f in csv_files}
        units, plan = plan_batch(csv_files, data_dir, infos)
        for line in describe_plan(plan):
            logger.info(line)
            if verbose:
                print(f"  {line}")
        job_of, unit_info, updates = {}, {}, {}
        for u in units:
            job = (u['file'], u['electrode'], u['uid'])
            job_of[u['uid']] = job
            unit_info[job] = {**infos[u['file']], **(u['update'] or {})}
            if u['update']:
                updates[u['uid']] = u['update']
        _warn_tissue_collisions(unit_info, data_dir, verbose)
        counter['total'] = len(units)
        bl_jobs = [j for j, info in unit_info.items() if info.get('is_baseline') and info.get('tissue')]
        out, errors = _run_batch_jobs(bl_jobs, config, n_workers, verbose, False, counter, updates)
        bl_set = set(bl_jobs)
        dose_jobs = [j for j in unit_info if j not in bl_set]
        tissue_el = _choose_tissue_electrodes(out, unit_info, dose_jobs)
        redo = [(j[0], tissue_el[unit_info[j]['tissue']]['electrode'], j[2]) for j, r in out.items()
                if j[1] == 'auto' and unit_info[j]['tissue'] in tissue_el
                and r['file_info'].get('analyzed_channel') != tissue_el[unit_info[j]['tissue']]['electrode']]
        if redo:
            counter['total'] += len(redo)
            out_redo, err_redo = _run_batch_jobs(redo, config, n_workers, verbose, True, counter, updates)
            errors += err_redo
            for job, r in out_redo.items():
                out[job_of[job[2]]] = r
        rest = [(j[0], j[1] if j[1] != 'auto'
                 else tissue_el.get(unit_info[j].get('tissue'), {}).get('electrode', 'auto'), j[2])
                for j in dose_jobs]
        out2, err2 = _run_batch_jobs(rest, config, n_workers, verbose, False, counter, updates)
        errors += err2
        for job, r in out2.items():
            out[job_of[job[2]]] = r
        results = []
        for u in units:
            r = out.get(job_of[u['uid']])
            if r is None:
                continue
            fi = r['file_info']
            if u['electrode'] != 'auto':
                fi['tissue_electrode'] = fi.get('analyzed_channel')
                fi['tissue_electrode_from'] = u['source']
            elif fi.get('tissue') in tissue_el:
                t = tissue_el[fi['tissue']]
                fi['tissue_electrode'] = t['electrode']
                fi['tissue_electrode_from'] = t['baseline_file']
            # one electrode (or one given input) per tissue: the tissue is
            # one normalisation group whatever input its files used
            fi['group_electrode'] = ''
            if u['multi']:
                r['metadata']['filename'] = f"{r['metadata'].get('filename', '')} [{fi.get('analyzed_channel')}]"
            r['batch_plan'] = plan
            results.append(r)
    else:
        jobs = [(f, ch) for f in csv_files for ch in channels_to_run]
        out, errors = _run_batch_jobs(jobs, config, n_workers, verbose, channel == 'both', counter)
        results = [out[j] for j in jobs if j in out]

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
    # Capture the structured exclusion provenance and attach it to every
    # result, so downstream consumers (Excel/PDF report, PySide study
    # panel, CDISC export) can show *which* dose-response groups were
    # dropped and why. A removed group produces no %ΔFPDcF at all, and
    # that absence is otherwise invisible in the outputs.
    inclusion_report = {}
    results = apply_inclusion_criteria(results, verbose=verbose,
                                       cfg=config.inclusion,
                                       report_out=inclusion_report)
    for r in results:
        if r is not None:
            r['inclusion_report'] = inclusion_report

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
    print(f"\n  Generating reports... ({len(results)} recordings from {len(csv_files)} files)")
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
