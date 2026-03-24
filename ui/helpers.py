"""
Shared helper functions for the Streamlit GUI.
"""

import numpy as np

from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia
from cardiac_fp_analyzer.beat_detection import compute_beat_periods, segment_beats
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.quality_control import validate_beats
from ui.i18n import T


def reanalyze_with_modified_beats(result, new_beat_indices, config):
    """Re-run analysis pipeline from beat segmentation onwards using modified beat indices.

    This keeps the original filtered signal but recomputes segmentation,
    QC, parameters, and arrhythmia analysis with the user-edited beats.
    """
    filtered = result['filtered_signal']
    t = result['time_vector']
    fs = result['metadata']['sample_rate']
    rep_cfg = config.repolarization

    bi = np.sort(new_beat_indices)

    # Segment beats
    bd, btm, vi = segment_beats(filtered, t, bi, fs,
                                 pre_ms=rep_cfg.segment_pre_ms,
                                 post_ms=max(850, rep_cfg.search_end_ms + 50))

    # Use only successfully segmented beats (edge-truncated beats removed)
    bi_seg = bi[np.array(vi)] if len(vi) > 0 else bi[:0]

    # QC
    qc_report, bi_clean, bd_clean, btm_clean = validate_beats(
        filtered, bi_seg, bd, btm, fs, cfg=config.quality
    )

    # Parameters
    all_p, summary = extract_all_parameters(bd_clean, btm_clean, bi_clean, fs,
                                             cfg=rep_cfg)
    bp = compute_beat_periods(bi_clean, fs)

    # Arrhythmia
    ar = analyze_arrhythmia(bi_clean, bp, all_p, summary, fs,
                            cfg=config.arrhythmia, beats_data=bd_clean)

    # Build updated result (preserving original signal data)
    updated = dict(result)
    updated.update({
        'beat_indices': bi_clean,
        'beat_indices_raw': bi,
        'beat_periods': bp,
        'all_params': all_p,
        'summary': summary,
        'arrhythmia_report': ar,
        'beats_data': bd_clean,
        'beats_time': btm_clean,
        'qc_report': qc_report,
    })
    return updated


def amplitude_scale(sig):
    """Choose the right display unit for a signal array.

    Returns (multiplier, y_axis_label) so that sig * multiplier is in a
    human-friendly range.  With amplifier gain = 10^4 the filtered signal
    is in the uV range; with gain = 1 (raw) it is in the mV range.
    """
    sig_range = np.ptp(sig) if len(sig) > 0 else 0
    if sig_range < 0.001:        # < 1 mV peak-to-peak -> display in uV
        return 1e6, T('amplitude_uV')
    else:                         # >= 1 mV -> display in mV
        return 1e3, T('amplitude_mV')
