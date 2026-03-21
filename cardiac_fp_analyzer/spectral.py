"""
spectral.py — Frequency domain analysis for hiPSC-CM field potentials.

Extracts spectral features that capture drug-induced morphological changes
not visible in standard time-domain parameters (FPD, amplitude, BP).

Key features:
  1. Beat-rate power: fundamental frequency and harmonics
  2. Spectral entropy: measure of signal complexity/disorder
  3. Band power ratios: low-freq vs high-freq energy distribution
  4. Morphology spectral signature: compare beat spectrum to baseline
  5. Wavelet coherence: time-frequency analysis for transient events

These features are especially useful for:
  - Drugs that alter waveform shape without changing FPD (e.g. Na+ channel blockers)
  - Detecting early signs of waveform deterioration before complete cessation
  - Distinguishing noise from genuine signal morphology changes
"""

import numpy as np
from scipy import signal as sig
from dataclasses import dataclass, field


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis."""

    # Welch PSD parameters
    welch_segment_s: float = 4.0     # segment duration for Welch method
    welch_overlap_frac: float = 0.5  # overlap fraction
    welch_window: str = 'hann'       # window function

    # Band definitions (Hz)
    band_low: tuple = (0.5, 5.0)     # low-frequency: breathing, drift
    band_beat: tuple = (0.3, 3.5)    # beat-rate band (auto-adjusted to actual rate)
    band_repol: tuple = (5.0, 30.0)  # repolarization-related
    band_high: tuple = (30.0, 200.0) # high-frequency: noise, spikes

    # Beat-rate fundamental search
    fundamental_search_range: tuple = (0.3, 4.0)  # Hz: 0.3-4 Hz = 18-240 BPM

    # Spectral entropy
    entropy_n_bins: int = 50
    entropy_freq_range: tuple = (0.5, 100.0)  # Hz range for entropy calc

    # Morphology comparison
    morph_freq_range: tuple = (1.0, 50.0)  # Hz range for morphology comparison
    morph_n_harmonics: int = 5  # how many harmonics to compare

    # Beat-level wavelet
    wavelet_freq_range: tuple = (1.0, 50.0)
    wavelet_n_freqs: int = 30


@dataclass
class SpectralReport:
    """Results of spectral analysis."""

    # Fundamental frequency
    fundamental_freq_hz: float = np.nan
    fundamental_power: float = np.nan

    # Band powers (absolute and relative)
    power_low: float = np.nan
    power_beat: float = np.nan
    power_repol: float = np.nan
    power_high: float = np.nan
    power_total: float = np.nan

    # Ratios
    ratio_beat_total: float = np.nan     # how much power is in beat-rate band
    ratio_repol_beat: float = np.nan     # repol vs beat power
    ratio_high_total: float = np.nan     # noise fraction

    # Spectral entropy (0 = pure tone, 1 = white noise)
    spectral_entropy: float = np.nan

    # Harmonic structure
    n_harmonics_detected: int = 0
    harmonic_ratio: float = np.nan  # power in harmonics vs noise floor

    # Beat morphology spectral features
    beat_spectral_centroid: float = np.nan   # center of mass of beat spectrum
    beat_spectral_bandwidth: float = np.nan  # spread of beat spectrum

    # Comparison with baseline (if available)
    spectral_correlation: float = np.nan     # corr between drug and baseline PSD
    spectral_divergence: float = np.nan      # KL divergence from baseline

    details: dict = field(default_factory=dict)


def analyze_spectral(filtered_signal, fs, beat_indices=None,
                     beats_data=None, baseline_spectral=None,
                     cfg=None):
    """
    Perform spectral analysis on a recording.

    Parameters
    ----------
    filtered_signal : array — filtered FP signal
    fs : float — sampling rate
    beat_indices : array — detected beat positions
    beats_data : list of arrays — segmented beats (for beat-level spectral)
    baseline_spectral : SpectralReport or None — baseline for comparison
    cfg : SpectralConfig or None

    Returns
    -------
    SpectralReport
    """
    if cfg is None:
        cfg = SpectralConfig()

    report = SpectralReport()

    if len(filtered_signal) < int(cfg.welch_segment_s * fs):
        return report

    # ═══════════════════════════════════════════════════════════════
    #  1. Global PSD (Welch method)
    # ═══════════════════════════════════════════════════════════════
    nperseg = int(cfg.welch_segment_s * fs)
    noverlap = int(nperseg * cfg.welch_overlap_frac)
    freqs, psd = sig.welch(filtered_signal, fs=fs, nperseg=nperseg,
                           noverlap=noverlap, window=cfg.welch_window)

    if len(freqs) == 0 or len(psd) == 0:
        return report

    # ═══════════════════════════════════════════════════════════════
    #  2. Fundamental frequency detection
    # ═══════════════════════════════════════════════════════════════
    f_mask = (freqs >= cfg.fundamental_search_range[0]) & (freqs <= cfg.fundamental_search_range[1])
    if np.any(f_mask):
        fund_idx = np.argmax(psd[f_mask])
        report.fundamental_freq_hz = freqs[f_mask][fund_idx]
        report.fundamental_power = psd[f_mask][fund_idx]

    # ═══════════════════════════════════════════════════════════════
    #  3. Band powers
    # ═══════════════════════════════════════════════════════════════
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(mask):
            return np.trapz(psd[mask], freqs[mask])
        return 0.0

    report.power_low = band_power(*cfg.band_low)
    report.power_repol = band_power(*cfg.band_repol)
    report.power_high = band_power(*cfg.band_high)

    # Auto-adjust beat band around detected fundamental
    if not np.isnan(report.fundamental_freq_hz):
        f0 = report.fundamental_freq_hz
        beat_low = max(0.1, f0 * 0.7)
        beat_high = f0 * 1.3
        report.power_beat = band_power(beat_low, beat_high)
    else:
        report.power_beat = band_power(*cfg.band_beat)

    report.power_total = band_power(freqs[1], freqs[-1])

    if report.power_total > 0:
        report.ratio_beat_total = report.power_beat / report.power_total
        report.ratio_high_total = report.power_high / report.power_total
    if report.power_beat > 0:
        report.ratio_repol_beat = report.power_repol / report.power_beat

    # ═══════════════════════════════════════════════════════════════
    #  4. Spectral entropy
    # ═══════════════════════════════════════════════════════════════
    ent_mask = (freqs >= cfg.entropy_freq_range[0]) & (freqs <= cfg.entropy_freq_range[1])
    if np.any(ent_mask):
        psd_ent = psd[ent_mask]
        psd_norm = psd_ent / np.sum(psd_ent) if np.sum(psd_ent) > 0 else psd_ent
        psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
        if len(psd_norm) > 1:
            entropy = -np.sum(psd_norm * np.log2(psd_norm))
            max_entropy = np.log2(len(psd_norm))
            report.spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # ═══════════════════════════════════════════════════════════════
    #  5. Harmonic structure
    # ═══════════════════════════════════════════════════════════════
    if not np.isnan(report.fundamental_freq_hz):
        f0 = report.fundamental_freq_hz
        harmonic_power = 0
        n_detected = 0
        for h in range(1, cfg.morph_n_harmonics + 1):
            fh = f0 * (h + 1)
            if fh > freqs[-1]:
                break
            h_mask = (freqs >= fh * 0.9) & (freqs <= fh * 1.1)
            if np.any(h_mask):
                hp = np.max(psd[h_mask])
                # A harmonic is "detected" if its power > 10% of fundamental
                if hp > 0.1 * report.fundamental_power:
                    n_detected += 1
                    harmonic_power += hp

        report.n_harmonics_detected = n_detected
        if report.power_total > 0 and harmonic_power > 0:
            noise_power = report.power_total - report.power_beat - harmonic_power
            report.harmonic_ratio = (harmonic_power /
                                     max(noise_power, 1e-12))

    # ═══════════════════════════════════════════════════════════════
    #  6. Beat-level spectral features
    # ═══════════════════════════════════════════════════════════════
    if beats_data is not None and len(beats_data) >= 5:
        _compute_beat_spectral(beats_data, fs, cfg, report)

    # ═══════════════════════════════════════════════════════════════
    #  7. Comparison with baseline
    # ═══════════════════════════════════════════════════════════════
    if baseline_spectral is not None:
        _compare_with_baseline(freqs, psd, baseline_spectral, cfg, report)

    # Store PSD for future baseline comparison
    report.details['freqs'] = freqs
    report.details['psd'] = psd

    return report


def _compute_beat_spectral(beats_data, fs, cfg, report):
    """Compute spectral features from individual beat morphology."""
    # Average beat spectrum
    beat_psds = []
    for beat in beats_data[:30]:  # limit for speed
        if len(beat) < 20:
            continue
        # Zero-pad to uniform length
        n = len(beat)
        freqs_b = np.fft.rfftfreq(n, d=1.0/fs)
        fft_mag = np.abs(np.fft.rfft(beat - np.mean(beat)))
        beat_psds.append(fft_mag)

    if not beat_psds:
        return

    # Uniform length
    min_len = min(len(p) for p in beat_psds)
    beat_psds = [p[:min_len] for p in beat_psds]
    mean_psd = np.mean(beat_psds, axis=0)
    freqs_b = np.fft.rfftfreq(len(beats_data[0]), d=1.0/fs)[:min_len]

    # Spectral centroid (center of mass)
    morph_mask = (freqs_b >= cfg.morph_freq_range[0]) & (freqs_b <= cfg.morph_freq_range[1])
    if np.any(morph_mask) and np.sum(mean_psd[morph_mask]) > 0:
        f_morph = freqs_b[morph_mask]
        p_morph = mean_psd[morph_mask]
        report.beat_spectral_centroid = np.sum(f_morph * p_morph) / np.sum(p_morph)
        report.beat_spectral_bandwidth = np.sqrt(
            np.sum((f_morph - report.beat_spectral_centroid)**2 * p_morph) / np.sum(p_morph)
        )

    report.details['beat_mean_spectrum'] = mean_psd
    report.details['beat_freqs'] = freqs_b


def _compare_with_baseline(freqs, psd, baseline_report, cfg, report):
    """Compare spectral features between drug recording and baseline."""
    bl_freqs = baseline_report.details.get('freqs')
    bl_psd = baseline_report.details.get('psd')

    if bl_freqs is None or bl_psd is None:
        return

    # Interpolate to common frequency grid
    common_mask = (freqs >= cfg.morph_freq_range[0]) & (freqs <= cfg.morph_freq_range[1])
    if not np.any(common_mask):
        return

    f_common = freqs[common_mask]
    psd_drug = psd[common_mask]

    # Interpolate baseline to same grid
    psd_bl = np.interp(f_common, bl_freqs, bl_psd)

    # Correlation
    if np.std(psd_drug) > 0 and np.std(psd_bl) > 0:
        report.spectral_correlation = np.corrcoef(
            np.log1p(psd_drug), np.log1p(psd_bl)
        )[0, 1]

    # KL divergence (symmetrized)
    p = psd_drug / np.sum(psd_drug) if np.sum(psd_drug) > 0 else psd_drug
    q = psd_bl / np.sum(psd_bl) if np.sum(psd_bl) > 0 else psd_bl
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    report.spectral_divergence = (kl_pq + kl_qp) / 2


def compute_morphology_change_score(drug_spectral, baseline_spectral):
    """
    Compute a single score (0-1) representing how much the beat morphology
    has changed from baseline based on spectral features.

    0.0 = identical to baseline
    1.0 = completely different morphology

    This score can be used as an additional drug effect indicator,
    especially for drugs that alter waveform shape without changing FPD.
    """
    if baseline_spectral is None or drug_spectral is None:
        return np.nan

    scores = []

    # 1. Spectral correlation (inverted)
    corr = drug_spectral.spectral_correlation
    if not np.isnan(corr):
        scores.append(1.0 - max(0, corr))

    # 2. Spectral entropy change
    bl_ent = baseline_spectral.spectral_entropy
    dr_ent = drug_spectral.spectral_entropy
    if not np.isnan(bl_ent) and not np.isnan(dr_ent):
        ent_change = abs(dr_ent - bl_ent)
        scores.append(min(1.0, ent_change * 3))  # scale: 0.33 change → score 1.0

    # 3. Beat spectral centroid shift
    bl_cent = baseline_spectral.beat_spectral_centroid
    dr_cent = drug_spectral.beat_spectral_centroid
    if not np.isnan(bl_cent) and not np.isnan(dr_cent) and bl_cent > 0:
        cent_change = abs(dr_cent - bl_cent) / bl_cent
        scores.append(min(1.0, cent_change * 2))

    # 4. Harmonic structure change
    bl_hr = baseline_spectral.harmonic_ratio
    dr_hr = drug_spectral.harmonic_ratio
    if not np.isnan(bl_hr) and not np.isnan(dr_hr) and bl_hr > 0:
        hr_change = abs(dr_hr - bl_hr) / max(bl_hr, 0.01)
        scores.append(min(1.0, hr_change))

    if not scores:
        return np.nan

    return np.mean(scores)
