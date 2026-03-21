"""
quality_control.py — Signal quality assessment and beat validation.

The core problem: when drug effects suppress cardiac activity (e.g. high-dose
quinidine), the signal becomes mostly noise. Beat detectors then pick up noise
spikes as "beats", generating massive false positives. This module addresses
this by:

  1. Global SNR estimation — is this recording analyzable at all?
  2. Local SNR around each detected beat — does this beat stand out from
     the surrounding noise floor?
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
from scipy import signal as sig


# ─── Thresholds ───
GLOBAL_SNR_EXCELLENT = 8.0
GLOBAL_SNR_GOOD = 5.0
GLOBAL_SNR_FAIR = 3.0
GLOBAL_SNR_POOR = 2.0

LOCAL_SNR_ACCEPT = 3.0       # beat must be 3x above local noise
LOCAL_SNR_MARGINAL = 2.0     # below this → reject

MORPHOLOGY_CORR_ACCEPT = 0.6    # correlation with template ≥ 0.6 → accept
MORPHOLOGY_CORR_MARGINAL = 0.3  # below this → reject

# Maximum fraction of beats that can be rejected before downgrading quality
MAX_REJECTION_RATE = 0.40


class QualityReport:
    """Quality assessment report for a single recording."""

    def __init__(self):
        self.grade = 'A'
        self.global_snr = np.nan
        self.mean_local_snr = np.nan
        self.mean_morphology_corr = np.nan
        self.n_beats_input = 0
        self.n_beats_accepted = 0
        self.n_beats_rejected_snr = 0
        self.n_beats_rejected_morphology = 0
        self.rejection_rate = 0.0
        self.notes = []
        self.per_beat_snr = []
        self.per_beat_corr = []
        self.accepted_mask = []

    def summary_text(self):
        lines = [
            f"Quality Grade: {self.grade}",
            f"Global SNR: {self.global_snr:.1f}",
            f"Beats: {self.n_beats_accepted}/{self.n_beats_input} accepted "
            f"({self.rejection_rate*100:.1f}% rejected)",
            f"  - Rejected (low SNR): {self.n_beats_rejected_snr}",
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


def estimate_local_snr(data, beat_index, fs, signal_window_ms=20, noise_window_ms=200):
    """
    Estimate SNR around a single beat.

    Signal: peak-to-peak in a tight window (±signal_window) around the beat.
    Noise: std of a wider window (±noise_window) excluding the signal window.

    This catches beats in noisy regions that happen to cross the global threshold
    but don't actually stand out locally.
    """
    sig_win = int(signal_window_ms * fs / 1000)
    noise_win = int(noise_window_ms * fs / 1000)

    # Signal amplitude
    sig_s = max(0, beat_index - sig_win)
    sig_e = min(len(data), beat_index + sig_win)
    sig_seg = data[sig_s:sig_e]
    signal_amp = np.max(sig_seg) - np.min(sig_seg)

    # Noise: wider window excluding signal region
    noise_s = max(0, beat_index - noise_win)
    noise_e = min(len(data), beat_index + noise_win)
    noise_seg = np.concatenate([
        data[noise_s:sig_s],
        data[sig_e:noise_e]
    ])

    if len(noise_seg) < 20:
        # Fallback: use the edges of the noise window
        noise_seg = data[noise_s:noise_e]

    noise_std = np.std(noise_seg) if len(noise_seg) > 0 else 1e-10

    if noise_std == 0:
        return 100.0

    return signal_amp / (2 * noise_std)  # factor 2 because amplitude is peak-to-peak


def compute_beat_template(beats_data, max_beats=50):
    """
    Compute a median beat template from the first N beats.
    The median is robust to outliers (noise, artefacts).
    """
    if len(beats_data) == 0:
        return None

    # Use up to max_beats, preferring beats from the middle of the recording
    n = min(len(beats_data), max_beats)
    if n < len(beats_data):
        mid = len(beats_data) // 2
        start = max(0, mid - n // 2)
        selected = beats_data[start:start + n]
    else:
        selected = beats_data[:n]

    # Ensure all beats have the same length
    min_len = min(len(b) for b in selected)
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

    # Pearson correlation
    b_c = b - np.mean(b)
    t_c = t - np.mean(t)
    denom = np.sqrt(np.sum(b_c**2) * np.sum(t_c**2))

    if denom == 0:
        return 0.0

    return np.sum(b_c * t_c) / denom


def validate_beats(data, beat_indices, beats_data, beats_time, fs,
                   snr_threshold=LOCAL_SNR_ACCEPT,
                   morphology_threshold=MORPHOLOGY_CORR_ACCEPT,
                   use_morphology=True):
    """
    Validate each detected beat using local SNR and morphological consistency.

    Parameters
    ----------
    data : array — full filtered signal
    beat_indices : array — detected beat sample indices
    beats_data : list of arrays — segmented beat waveforms
    beats_time : list of arrays — time vectors for each beat
    fs : float — sampling rate
    snr_threshold : float — minimum local SNR to accept a beat
    morphology_threshold : float — minimum correlation with template

    Returns
    -------
    qc_report : QualityReport
    accepted_indices : array — indices of accepted beats (into beat_indices)
    accepted_beats_data : list — beat waveforms of accepted beats
    accepted_beats_time : list — time vectors of accepted beats
    """
    qc = QualityReport()
    qc.n_beats_input = len(beat_indices)

    if len(beat_indices) == 0:
        qc.grade = 'F'
        qc.notes.append('No beats detected')
        return qc, np.array([], dtype=int), [], []

    # ─── Step 1: Global SNR ───
    qc.global_snr = estimate_global_snr(data, beat_indices, fs)

    if qc.global_snr < GLOBAL_SNR_POOR:
        qc.notes.append(f'Very low global SNR ({qc.global_snr:.1f}) — signal may be mostly noise')

    # ─── Step 2: Local SNR for each beat ───
    local_snrs = []
    for bi in beat_indices:
        snr = estimate_local_snr(data, bi, fs)
        local_snrs.append(snr)
    qc.per_beat_snr = local_snrs

    # ─── Step 3: Morphological template and correlation ───
    template = None
    morphology_corrs = [1.0] * len(beats_data)  # default: all pass

    if use_morphology and len(beats_data) >= 5:
        # Build template from beats with highest local SNR
        snr_order = np.argsort(local_snrs)[::-1]
        # Map beat_indices positions to beats_data positions
        # beats_data may have fewer entries than beat_indices (edge beats excluded)
        n_template = min(30, len(beats_data))
        best_beats = []
        count = 0
        for idx in snr_order:
            if idx < len(beats_data):
                best_beats.append(beats_data[idx])
                count += 1
                if count >= n_template:
                    break

        if best_beats:
            template = compute_beat_template(best_beats, max_beats=30)

        if template is not None:
            morphology_corrs = []
            for bd in beats_data:
                corr = morphology_correlation(bd, template)
                morphology_corrs.append(corr)
            qc.per_beat_corr = morphology_corrs

    # ─── Step 4: Accept/Reject decisions ───
    accepted_mask = []
    n_rej_snr = 0
    n_rej_morph = 0

    for i in range(len(beat_indices)):
        snr_ok = local_snrs[i] >= snr_threshold
        morph_ok = True

        if use_morphology and i < len(morphology_corrs):
            morph_ok = morphology_corrs[i] >= morphology_threshold

        if not snr_ok:
            accepted_mask.append(False)
            n_rej_snr += 1
        elif not morph_ok:
            accepted_mask.append(False)
            n_rej_morph += 1
        else:
            accepted_mask.append(True)

    qc.accepted_mask = accepted_mask
    qc.n_beats_rejected_snr = n_rej_snr
    qc.n_beats_rejected_morphology = n_rej_morph
    qc.n_beats_accepted = sum(accepted_mask)
    qc.rejection_rate = 1 - qc.n_beats_accepted / qc.n_beats_input if qc.n_beats_input > 0 else 0

    # Mean stats for accepted beats only
    accepted_snrs = [s for s, a in zip(local_snrs, accepted_mask) if a]
    qc.mean_local_snr = np.mean(accepted_snrs) if accepted_snrs else 0

    accepted_corrs = [c for c, a in zip(morphology_corrs, accepted_mask) if a and len(morphology_corrs) > 0]
    qc.mean_morphology_corr = np.mean(accepted_corrs) if accepted_corrs else 0

    # ─── Step 5: Assign quality grade ───
    qc.grade = _assign_grade(qc)

    # ─── Step 6: Build accepted outputs ───
    accepted_beat_indices = beat_indices[np.array(accepted_mask)]

    # For beats_data/beats_time, we need the mask aligned to the segmented beats
    # (which may be fewer than beat_indices due to edge exclusion)
    accepted_beats_data = []
    accepted_beats_time = []
    for i in range(min(len(beats_data), len(accepted_mask))):
        if accepted_mask[i]:
            accepted_beats_data.append(beats_data[i])
            accepted_beats_time.append(beats_time[i])

    # Add notes
    if qc.rejection_rate > 0.5:
        qc.notes.append(f'High rejection rate ({qc.rejection_rate*100:.0f}%) — signal quality is poor')
    if n_rej_snr > 0:
        qc.notes.append(f'{n_rej_snr} beats rejected for low local SNR (< {snr_threshold:.1f}x noise)')
    if n_rej_morph > 0:
        qc.notes.append(f'{n_rej_morph} beats rejected for abnormal morphology (corr < {morphology_threshold:.2f})')

    return qc, accepted_beat_indices, accepted_beats_data, accepted_beats_time


def _assign_grade(qc):
    """Assign overall quality grade based on SNR and rejection metrics."""
    gsnr = qc.global_snr
    rej = qc.rejection_rate
    n_acc = qc.n_beats_accepted
    mean_lsnr = qc.mean_local_snr
    mean_corr = qc.mean_morphology_corr

    # F: unanalyzable
    if n_acc < 3:
        return 'F'
    if gsnr < GLOBAL_SNR_POOR and rej > 0.6:
        return 'F'

    # D: poor
    if gsnr < GLOBAL_SNR_FAIR or rej > MAX_REJECTION_RATE:
        return 'D'
    if mean_lsnr < LOCAL_SNR_ACCEPT:
        return 'D'

    # C: fair
    if gsnr < GLOBAL_SNR_GOOD or rej > 0.20:
        return 'C'
    if mean_corr < MORPHOLOGY_CORR_ACCEPT:
        return 'C'

    # B: good
    if gsnr < GLOBAL_SNR_EXCELLENT or rej > 0.05:
        return 'B'

    # A: excellent
    return 'A'
