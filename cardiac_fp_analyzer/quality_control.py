"""
quality_control.py — Signal quality assessment and beat validation.

The core problem: when drug effects suppress cardiac activity (e.g. high-dose
quinidine), the signal becomes mostly noise. Beat detectors then pick up noise
spikes as "beats", generating massive false positives. This module addresses
this by:

  1. Global SNR estimation — is this recording analyzable at all?
  2. Amplitude consistency — do beats have physiologically consistent amplitude,
     or are some just noise spikes?
  3. Morphological consistency — does this beat look like the others?
  4. Beat-by-beat validation with rejection of low-confidence detections.

Quality grades:
  A — Excellent: clear spikes, high SNR, consistent morphology
  B — Good: clear spikes with some noise, minor morphology variation
  C — Fair: spikes detectable but noisy, some questionable beats
  D — Poor: low SNR, many questionable beats, results unreliable
  F — Unanalyzable: signal indistinguishable from noise
"""

import numpy as np

# ─── Module-level defaults (used when no config is provided) ───
GLOBAL_SNR_EXCELLENT = 8.0
GLOBAL_SNR_GOOD = 5.0
GLOBAL_SNR_FAIR = 3.0
GLOBAL_SNR_POOR = 2.0

AMPLITUDE_REJECT_FRACTION = 0.25
MORPHOLOGY_CORR_ACCEPT = 0.4
MORPHOLOGY_CORR_MARGINAL = 0.2
MAX_REJECTION_RATE = 0.40


def _get_qc_cfg(cfg=None):
    """Return a QualityConfig, creating default if needed."""
    if cfg is not None:
        return cfg
    from .config import QualityConfig
    return QualityConfig()


class QualityReport:
    """Quality assessment report for a single recording."""

    def __init__(self):
        self.grade = 'A'
        self.global_snr = np.nan
        self.mean_local_snr = np.nan
        self.mean_morphology_corr = np.nan
        self.n_beats_input = 0
        self.n_beats_accepted = 0
        self.n_beats_rejected_snr = 0        # kept as "snr" label for report compat
        self.n_beats_rejected_morphology = 0
        self.rejection_rate = 0.0
        self.notes = []
        self.per_beat_snr = []     # actually per-beat amplitude ratios
        self.per_beat_corr = []
        self.accepted_mask = []

    def summary_text(self):
        lines = [
            f"Quality Grade: {self.grade}",
            f"Global SNR: {self.global_snr:.1f}",
            f"Beats: {self.n_beats_accepted}/{self.n_beats_input} accepted "
            f"({self.rejection_rate*100:.1f}% rejected)",
            f"  - Rejected (amplitude): {self.n_beats_rejected_snr}",
            f"  - Rejected (morphology): {self.n_beats_rejected_morphology}",
            f"Mean local SNR: {self.mean_local_snr:.1f}",
            f"Mean morphology correlation: {self.mean_morphology_corr:.2f}",
        ]
        for n in self.notes:
            lines.append(f"  Note: {n}")
        return '\n'.join(lines)


def estimate_global_snr(data, beat_indices, fs, window_ms=30):
    """
    Estimate global signal-to-noise ratio.

    Signal power: mean peak amplitude at detected beat locations.
    Noise power: std of inter-beat "quiet" regions (excluding ±window around beats).

    Returns SNR (ratio, not dB).
    """
    if len(beat_indices) < 2:
        return 0.0

    window = int(window_ms * fs / 1000)

    # Build a mask of "beat regions" (±window around each beat)
    beat_mask = np.zeros(len(data), dtype=bool)
    for bi in beat_indices:
        s = max(0, bi - window)
        e = min(len(data), bi + window)
        beat_mask[s:e] = True

    # Signal: peak-to-peak amplitude around each beat
    beat_amplitudes = []
    for bi in beat_indices:
        s = max(0, bi - window)
        e = min(len(data), bi + window)
        seg = data[s:e]
        beat_amplitudes.append(np.max(seg) - np.min(seg))

    signal_power = np.mean(beat_amplitudes) if beat_amplitudes else 0

    # Noise: std of non-beat regions
    noise_data = data[~beat_mask]
    if len(noise_data) > 100:
        noise_power = np.std(noise_data)
    else:
        # Not enough quiet regions — use interquartile range of full signal
        q25, q75 = np.percentile(data, [25, 75])
        noise_power = (q75 - q25) / 1.349  # IQR-based std estimate

    if noise_power == 0:
        return 100.0  # perfect signal (unlikely)

    return signal_power / noise_power


def compute_beat_amplitudes(data, beat_indices, fs, window_ms=20):
    """
    Compute peak-to-peak amplitude for each beat in a tight window
    around the depolarization spike.
    """
    window = int(window_ms * fs / 1000)
    amplitudes = []
    for bi in beat_indices:
        s = max(0, bi - window)
        e = min(len(data), bi + window)
        seg = data[s:e]
        amplitudes.append(np.max(seg) - np.min(seg))
    return np.array(amplitudes)


def compute_amplitude_ratios(amplitudes):
    """
    Compute each beat's amplitude as a ratio to the robust reference.

    The reference is the median of the upper 50% of amplitudes — this is
    robust against noise-beats (which pull the median down) and against
    a few very large artefacts (which pull the mean up).

    Returns:
        ratios : array of amplitude / reference for each beat
        reference_amplitude : the reference value
    """
    if len(amplitudes) == 0:
        return np.array([]), 0.0

    # Reference: median of the top 50% of beat amplitudes
    sorted_amps = np.sort(amplitudes)
    upper_half = sorted_amps[len(sorted_amps) // 2:]
    reference = np.median(upper_half) if len(upper_half) > 0 else np.median(amplitudes)

    if reference == 0 or reference < 1e-10:
        return np.ones(len(amplitudes)), 0.0

    ratios = amplitudes / reference
    return ratios, reference


def compute_beat_template(beats_data, amplitudes=None, max_beats=50):
    """
    Compute a median beat template from the highest-amplitude beats.
    The median is robust to outliers (noise, artefacts).
    """
    if len(beats_data) == 0:
        return None

    n = min(len(beats_data), max_beats)

    if amplitudes is not None and len(amplitudes) == len(beats_data):
        # Select by amplitude (prefer strong, clear beats)
        order = np.argsort(amplitudes)[::-1]
        selected = [beats_data[i] for i in order[:n]]
    else:
        # Fallback: use beats from the middle of the recording
        if n < len(beats_data):
            mid = len(beats_data) // 2
            start = max(0, mid - n // 2)
            selected = beats_data[start:start + n]
        else:
            selected = list(beats_data[:n])

    # Ensure all beats have the same length
    min_len = min(len(b) for b in selected)
    if min_len < 10:
        return None
    aligned = np.array([b[:min_len] for b in selected])

    return np.median(aligned, axis=0)


def morphology_correlation(beat_data, template):
    """
    Compute correlation between a beat and the template.
    Uses Pearson correlation — 1.0 = identical shape, 0 = no similarity.
    """
    if template is None or len(beat_data) == 0:
        return 0.0

    # Align lengths
    n = min(len(beat_data), len(template))
    b = beat_data[:n]
    t = template[:n]

    # Guard against NaN propagation
    if np.any(np.isnan(b)) or np.any(np.isnan(t)):
        return 0.0

    # Pearson correlation
    b_c = b - np.mean(b)
    t_c = t - np.mean(t)
    denom = np.sqrt(np.sum(b_c**2) * np.sum(t_c**2))

    if denom == 0:
        return 0.0

    return np.sum(b_c * t_c) / denom


def validate_beats(data, beat_indices, beats_data, beats_time, fs,
                   snr_threshold=None, morphology_threshold=MORPHOLOGY_CORR_ACCEPT,
                   amplitude_threshold=AMPLITUDE_REJECT_FRACTION,
                   use_morphology=True, cfg=None):
    """
    Validate each detected beat using amplitude consistency and morphology.

    Strategy:
    - Compute each beat's peak-to-peak amplitude
    - Reject beats whose amplitude is far below the robust reference
      (these are noise spikes misidentified as beats)
    - Optionally reject beats with poor morphological correlation to template
    - Assign overall quality grade based on global SNR + rejection stats

    Parameters
    ----------
    data : array — full filtered signal
    beat_indices : array — detected beat sample indices
    beats_data : list of arrays — segmented beat waveforms
    beats_time : list of arrays — time vectors for each beat
    fs : float — sampling rate
    snr_threshold : float — (legacy, ignored — kept for API compatibility)
    morphology_threshold : float — minimum correlation with template
    amplitude_threshold : float — min amplitude ratio to reference (default 0.25)
    use_morphology : bool — whether to use morphological filtering
    cfg : QualityConfig or None — if provided, overrides threshold parameters

    Returns
    -------
    qc_report : QualityReport
    accepted_indices : array — indices of accepted beats (into beat_indices)
    accepted_beats_data : list — beat waveforms of accepted beats
    accepted_beats_time : list — time vectors of accepted beats
    """
    c = _get_qc_cfg(cfg)

    # When config is provided, use its values
    if cfg is not None:
        morphology_threshold = c.morphology_threshold
        amplitude_threshold = c.amplitude_reject_fraction
        use_morphology = c.use_morphology

    # Alignment guard: beat_indices must match segmented data arrays.
    # Callers must pass only successfully-segmented beat indices.
    assert len(beat_indices) == len(beats_data) == len(beats_time), (
        f"validate_beats alignment error: beat_indices={len(beat_indices)}, "
        f"beats_data={len(beats_data)}, beats_time={len(beats_time)}"
    )

    qc = QualityReport()
    qc.n_beats_input = len(beat_indices)

    if len(beat_indices) == 0:
        qc.grade = 'F'
        qc.notes.append('No beats detected')
        return qc, np.array([], dtype=int), [], []

    # ─── Step 1: Global SNR ───
    qc.global_snr = estimate_global_snr(data, beat_indices, fs)

    if qc.global_snr < c.snr_poor:
        qc.notes.append(f'Very low global SNR ({qc.global_snr:.1f}) — signal may be mostly noise')

    # ─── Step 2: Beat amplitudes and amplitude ratios ───
    amplitudes = compute_beat_amplitudes(data, beat_indices, fs, window_ms=c.morphology_window_ms)
    amp_ratios, ref_amplitude = compute_amplitude_ratios(amplitudes)

    # Store amplitude ratios as "per_beat_snr" for report compatibility
    qc.per_beat_snr = amp_ratios.tolist() if len(amp_ratios) > 0 else []

    # ─── Step 3: Morphological template and correlation ───
    template = None
    morphology_corrs = [1.0] * len(beats_data)  # default: all pass

    if use_morphology and len(beats_data) >= 5:
        # Build template from highest-amplitude beats (most likely real)
        # amplitudes and beats_data are guaranteed same length by alignment assert above
        template = compute_beat_template(beats_data, amplitudes=amplitudes, max_beats=c.morphology_max_beats)

        if template is not None:
            morphology_corrs = []
            for bd in beats_data:
                corr = morphology_correlation(bd, template)
                morphology_corrs.append(corr)
            qc.per_beat_corr = morphology_corrs

    # ─── Step 3b: Adaptive morphology threshold ───
    # If the fixed threshold would reject too many beats (>40%), the
    # signal likely has naturally variable morphology (common in µECG /
    # MEA recordings where amplitude fluctuates across beats).
    # Strategy: lower threshold to keep ≈80% of beats, but never below
    # the marginal floor.  If timing is also consistent (low CV of beat
    # periods), trust the beats even more and lower further.
    effective_morph_threshold = morphology_threshold
    if use_morphology and len(morphology_corrs) >= 5:
        n_pass = sum(1 for c_ in morphology_corrs if c_ >= morphology_threshold)
        if n_pass < 0.6 * len(morphology_corrs):
            # Check if beat timing is consistent (good proxy for real beats)
            bp_samples = np.diff(beat_indices)
            bp_cv = np.std(bp_samples) / np.mean(bp_samples) if len(bp_samples) > 1 and np.mean(bp_samples) > 0 else 999

            if bp_cv < 0.20:
                # Timing is very regular — beats are almost certainly real.
                # Use 5th percentile (keeps ~95%).
                adaptive_thresh = np.percentile(morphology_corrs, 5)
            elif bp_cv < 0.35:
                # Timing is reasonably regular. Use 15th percentile (~85%).
                adaptive_thresh = np.percentile(morphology_corrs, 15)
            else:
                # Timing is irregular — be more cautious, 30th percentile (~70%)
                adaptive_thresh = np.percentile(morphology_corrs, 30)

            marginal = getattr(c, 'morphology_marginal', 0.10)
            effective_morph_threshold = max(adaptive_thresh, marginal)

    # ─── Step 4: Accept/Reject decisions ───
    accepted_mask = []
    n_rej_amp = 0
    n_rej_morph = 0

    for i in range(len(beat_indices)):
        # Amplitude check
        amp_ok = True
        if i < len(amp_ratios):
            amp_ok = amp_ratios[i] >= amplitude_threshold

        # Morphology check
        morph_ok = True
        if use_morphology and i < len(morphology_corrs):
            morph_ok = morphology_corrs[i] >= effective_morph_threshold

        if not amp_ok:
            accepted_mask.append(False)
            n_rej_amp += 1
        elif not morph_ok:
            accepted_mask.append(False)
            n_rej_morph += 1
        else:
            accepted_mask.append(True)

    qc.accepted_mask = accepted_mask
    qc.n_beats_rejected_snr = n_rej_amp  # "snr" label kept for report compat
    qc.n_beats_rejected_morphology = n_rej_morph
    qc.n_beats_accepted = sum(accepted_mask)
    qc.rejection_rate = 1 - qc.n_beats_accepted / qc.n_beats_input if qc.n_beats_input > 0 else 0

    # Mean stats for accepted beats only
    accepted_ratios = [r for r, a in zip(amp_ratios, accepted_mask) if a]
    # Convert amplitude ratio to an SNR-like value (ratio * global_snr) for reporting
    qc.mean_local_snr = (np.mean(accepted_ratios) * qc.global_snr
                         if accepted_ratios else 0)

    accepted_corrs = [c for c, a in zip(morphology_corrs, accepted_mask)
                      if a and len(morphology_corrs) > 0]
    qc.mean_morphology_corr = np.mean(accepted_corrs) if accepted_corrs else 0

    # ─── Step 5: Assign quality grade ───
    qc.grade = _assign_grade(qc, cfg=c)

    # ─── Step 6: Build accepted outputs ───
    mask_arr = np.array(accepted_mask)
    accepted_beat_indices = beat_indices[mask_arr]
    accepted_beats_data = [bd for bd, a in zip(beats_data, accepted_mask) if a]
    accepted_beats_time = [bt for bt, a in zip(beats_time, accepted_mask) if a]

    assert len(accepted_beat_indices) == len(accepted_beats_data) == len(accepted_beats_time), (
        f"validate_beats output alignment error: indices={len(accepted_beat_indices)}, "
        f"data={len(accepted_beats_data)}, time={len(accepted_beats_time)}"
    )

    # Add notes
    if qc.rejection_rate > c.rejection_high_note:
        qc.notes.append(f'High rejection rate ({qc.rejection_rate*100:.0f}%) — signal quality is poor')
    if n_rej_amp > 0:
        qc.notes.append(f'{n_rej_amp} beats rejected for low amplitude '
                        f'(< {amplitude_threshold*100:.0f}% of reference)')
    if n_rej_morph > 0:
        thresh_note = f'corr < {effective_morph_threshold:.2f}'
        if effective_morph_threshold != morphology_threshold:
            thresh_note += f' (adapted from {morphology_threshold:.2f})'
        qc.notes.append(f'{n_rej_morph} beats rejected for abnormal morphology ({thresh_note})')

    return qc, accepted_beat_indices, accepted_beats_data, accepted_beats_time


def _assign_grade(qc, cfg=None):
    """Assign overall quality grade based on SNR and rejection metrics."""
    c = _get_qc_cfg(cfg)
    gsnr = qc.global_snr
    rej = qc.rejection_rate
    n_acc = qc.n_beats_accepted
    mean_corr = qc.mean_morphology_corr

    # F: unanalyzable
    if n_acc < c.min_beats_for_analysis:
        return 'F'
    if gsnr < c.snr_poor and rej > 0.6:
        return 'F'

    # D: poor
    if gsnr < c.snr_fair or rej > c.max_rejection_rate:
        return 'D'

    # C: fair
    if gsnr < c.snr_good or rej > c.rejection_grade_c:
        return 'C'
    if mean_corr < c.morphology_threshold:
        return 'C'

    # B: good
    if gsnr < c.snr_excellent or rej > c.rejection_grade_b:
        return 'B'

    # A: excellent
    return 'A'
