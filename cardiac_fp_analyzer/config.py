"""
config.py — Central configuration for Cardiac FP Analyzer.

All tunable parameters are organized into logical groups using dataclasses.
The top-level AnalysisConfig holds all sub-configurations and provides:
  - JSON serialization / deserialization (for GUI persistence)
  - Named presets (e.g. "default", "conservative", "sensitive")
  - Method selection for key algorithms (FPD measurement, correction, etc.)

Every function in the pipeline accepts the relevant config section, so the
entire analysis behaviour can be controlled from a single config object.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

# Import sub-configs from their modules (lazy to avoid circular imports)
# CessationConfig and SpectralConfig are re-exported here for convenience


# ═════════════════════════════════════════════════════════════════════════
#   FILTERING
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class FilterConfig:
    """Signal preprocessing / filtering parameters."""

    # Notch filter (powerline removal)
    notch_freq_hz: float = 50.0
    notch_harmonics: int = 3
    notch_q: float = 30.0

    # Bandpass filter
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 500.0
    bandpass_order: int = 4

    # Final smoothing (Savitzky-Golay)
    savgol_window: int = 7
    savgol_polyorder: int = 3


# ═════════════════════════════════════════════════════════════════════════
#   BEAT DETECTION
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class BeatDetectionConfig:
    """Beat detection parameters."""

    # Detection method: 'auto', 'prominence', 'derivative', 'peak'
    method: str = 'auto'

    # Minimum inter-beat interval (ms)
    min_distance_ms: float = 400.0

    # Adaptive threshold multiplier for spike detection
    threshold_factor: float = 4.0

    # Retry parameters (used when first attempt gives poor results)
    retry_min_distance_ms: float = 300.0
    retry_threshold_factor: float = 3.0

    # Physiological plausibility scoring (thresholds)
    bp_ideal_range_s: tuple[float, float] = (0.4, 3.0)
    bp_extended_range_s: tuple[float, float] = (0.3, 5.0)
    # CV thresholds as FRACTIONS (0.15 = 15%). The `_frac` suffix is
    # intentional: ChannelSelectionConfig previously declared identically-
    # named fields in percent units, which was a semantic trap. Those
    # fields were dead code and have been removed.
    cv_good_frac: float = 0.15        # CV < 15% → good score
    cv_fair_frac: float = 0.30        # CV < 30% → fair
    cv_marginal_frac: float = 0.50    # CV < 50% → marginal

    # Auto-method scoring weights (points awarded per criterion)
    score_bp_ideal: float = 30.0       # beat period in ideal range
    score_bp_extended: float = 15.0    # beat period in extended range
    score_cv_good: float = 30.0        # CV below cv_good
    score_cv_fair: float = 20.0        # CV below cv_fair
    score_cv_marginal: float = 10.0    # CV below cv_marginal
    score_rate_ok: float = 20.0        # beat count in plausible range
    score_rate_low: float = 10.0       # >3 beats but outside plausible range
    score_rate_excess: float = -20.0   # too many beats (likely noise)
    score_too_few: float = -10.0       # <3 beats

    # Derivative method bonus: dV/dt is the physiologically correct way to
    # identify depolarisation (fastest deflection, regardless of polarity or
    # amplitude).  Prominence can confuse large repolarisation waves with
    # depolarisation spikes — especially in 3D constructs where the
    # repolarisation amplitude often exceeds the depolarisation spike.
    # Literature basis: CiPA MEA analysis uses dV/dt as primary criterion;
    # CardioMDA (Clements 2013) and EFP Analyzer (Patel 2025) confirm.
    score_derivative_bonus: float = 15.0

    # ── Post-detection morphological validation (CardioMDA approach) ──
    # After initial beat detection, build a template from the strongest
    # candidates and reject beats whose correlation with the template
    # falls below this threshold.  This removes noise spikes BEFORE
    # parameter extraction — unlike QC which runs after segmentation.
    # Reference: Clements & Thomas, PLOS ONE 2013 (CardioMDA, r ≥ 0.95–0.98).
    # Default 0.7 is intentionally lower than CardioMDA's 0.98 because
    # µECG signals have more morphological variability than planar MEA.
    enable_morphology_validation: bool = True
    morphology_min_corr: float = 0.7
    # Minimum amplitude ratio vs robust reference (median of top 50%).
    # Beats below this are likely noise, not depolarization spikes.
    min_amplitude_ratio: float = 0.25
    # Minimum number of beats required to build a validation template.
    # With fewer beats, morphological validation is skipped (amplitude
    # validation still applies).
    morphology_min_beats: int = 5
    # Accept inverted beats in morphology validation.
    # When True, use abs(correlation) so that beats with inverted polarity
    # (negative spike where template has positive spike, or vice versa)
    # still pass the morphology gate.  Physiological basis: FP polarity
    # depends on electrode-tissue geometry and can switch within a
    # recording (biphasic FP with alternating phase dominance).
    morphology_accept_inverted: bool = True
    # Jitter correction: before computing template correlation, shift each
    # beat by ±jitter_max_shift samples to find the lag that maximises
    # cross-correlation with the template.  This compensates for alignment
    # errors that destroy Pearson r on narrow spikes.
    # Standard practice in neural spike sorting.
    enable_jitter_correction: bool = True
    # Adaptive jitter: estimate spike half-width from the template and set
    # the jitter window to `jitter_adaptive_fraction` × half-width.
    # This scales automatically: ~2–3 ms for classic MEA (spike ~1–3 ms),
    # ~10–15 ms for 3D constructs (spike ~30–50 ms).
    # When False (or template width estimation fails), falls back to the
    # fixed `jitter_max_shift_ms` value.
    jitter_adaptive: bool = True
    jitter_adaptive_fraction: float = 0.5  # jitter = 50% of spike half-width
    jitter_max_shift_ms: float = 5.0       # fixed fallback (±ms)

    # ── Amplitude-cluster filter (Sprint 2 #1) ──
    # When the detected peaks show a clearly bimodal amplitude distribution
    # (e.g. 16 real depolarisation spikes at ~1.4 V + 33 T-wave / baseline
    # bumps at ~0.15 V — the Exp6_ChipD_ch2 pattern) the morphology validator
    # can fail to separate the two populations, especially in mixed-polarity
    # mode where correlation is not meaningful. This filter sorts the per-peak
    # local max-abs amplitudes and, if the largest adjacent gap ratio exceeds
    # ``cluster_gap_ratio``, drops the low-amplitude cluster — provided the
    # dominant (high-amp) cluster has at least ``cluster_min_dominant_count``
    # peaks (so a single artefact spike cannot eat the whole recording).
    # Operates on local max-abs in ±``cluster_window_ms`` around each peak, so
    # it is polarity-agnostic.
    enable_amplitude_cluster_filter: bool = True
    cluster_gap_ratio: float = 3.0         # min sorted-adjacent ratio to fire
    cluster_min_dominant_count: int = 3    # min peaks in dominant cluster
    cluster_window_ms: float = 50.0        # window around peak for max-abs
    # ── Safeguards (Sprint 2 #1b) ──
    # Before applying the filter, verify that the low-amp cluster looks
    # like an artefact pattern (T-wave residuals interlaced between beats)
    # rather than a legitimate cluster of real beats.
    #   (a) Topology: at least ``cluster_topology_min_interlaced`` fraction
    #       of the low-cluster peaks must sit *between* two adjacent
    #       high-cluster peaks (i.e. have a high peak immediately before
    #       and immediately after, with no other low peak of the same
    #       cluster in that high-RR interval required). A cluster of 3+
    #       weak beats at the start/end of the recording (contiguous) will
    #       fail this check and the filter aborts.
    #   (b) Alternans: if n_low / n_high falls in
    #       ``cluster_alternans_ratio_band`` the pattern is likely an
    #       alternance (big/small/big/small…) — applying the filter would
    #       halve the real beat count. Abort.
    # On abort, the diagnostics report ``'aborted_topology'`` or
    # ``'aborted_alternans'`` and the original beat set is returned.
    cluster_topology_min_interlaced: float = 0.7
    cluster_alternans_ratio_low: float = 0.85   # lower edge of alt band
    cluster_alternans_ratio_high: float = 1.15  # upper edge of alt band

    # ── Noise-floor SNR gate (Oct 2026, Exp8 2× over-detection) ──
    # A detected "beat" whose local peak-to-peak amplitude is not clearly
    # above the recording's own noise floor is noise, whatever the rhythm
    # says. The gap-filling passes and the lenient mixed-polarity amplitude
    # gate (10 % of a median already contaminated by the false positives)
    # let such detections through on Exp8/Day6/chipD_ch1: 156 beats where
    # the authors count ~75, one noise-level "beat" at the midpoint of every
    # true RR interval. The cluster filter above cannot catch it (no single
    # sorted-gap ≥ 3× on a continuum, and n_low ≈ n_high trips the alternans
    # safeguard).
    #
    # Noise floor = median peak-to-peak over non-overlapping
    # ``noise_floor_window_ms`` windows across the whole filtered signal
    # (beats occupy a small fraction of windows at any plausible rate, so
    # the median is noise-dominated). Beat "SNR" = peak-to-peak in
    # ±``noise_floor_beat_half_window_ms`` divided by the noise floor.
    #
    # Two rules, both anchored to the noise floor:
    #   (1) hard floor — SNR < ``noise_floor_min_snr`` is always rejected.
    #       1.0 means "quieter than a typical noise window": no real event
    #       can be below it. It is NOT a detection threshold.
    #   (2) noise-cluster — in the sorted SNRs, find the largest geometric
    #       gap such that everything below it is noise-compatible
    #       (median ≤ ``noise_cluster_max_median_snr``, max ≤
    #       ``noise_cluster_max_snr``), the gap is ≥ ``noise_cluster_min_gap``
    #       and the upper cluster's median is ≥ ``noise_cluster_min_separation``
    #       × the lower's. If such a split exists, reject the lower cluster.
    #
    # Why a cluster rule and not a threshold: a single detection cannot be
    # told apart from noise by its own amplitude OR its own dV/dt. On
    # studio_MR/Baseline/chipA_ch1 (el1, ~15 µV spikes) every real beat sits
    # at SNR 1.1–2.3 — the same range as the Exp8 noise insertions — and the
    # other electrode confirms them (220 beats, CV 6 %). What separates the
    # two cases is the ensemble: Exp8's fakes have median SNR 1.12, i.e.
    # indistinguishable from random noise windows, next to a cluster 4×
    # higher; the low-SNR real beats have median 1.8 and no higher cluster.
    # A fixed floor at 1.5 (first version of this gate) removed 44 of 208
    # real beats on that file. Hence: floor at 1.0, "noise" only when the
    # cluster median is ≤ 1.35 and a ≥3× cluster exists above it.
    #
    # Calibrated on 8 published baselines (Exp5/6/7/8) plus the low-SNR
    # studio file, full recordings and 60/90 s windows: Exp8 156→81 beats
    # (RR +10 % vs published), Exp6 windows 76→35, six clean files
    # untouched, low-SNR file 208→205.
    # Unlike the cluster filter there is deliberately NO "don't halve the
    # count" safeguard: on Exp8 the correct answer IS half the count.
    # The gate only skips when it would remove every beat (degenerate
    # signal — left to QC grading) or the noise floor is zero.
    enable_noise_floor_gate: bool = True
    noise_floor_min_snr: float = 1.0
    noise_floor_window_ms: float = 40.0
    noise_floor_beat_half_window_ms: float = 20.0
    noise_cluster_max_median_snr: float = 1.35
    noise_cluster_max_snr: float = 3.5
    noise_cluster_min_gap: float = 1.3
    noise_cluster_min_separation: float = 3.0

    # ── Matched-filter refinement for low-SNR recordings (Oct 2026) ──
    # When spikes are barely above the noise (studio_MR chipA_ch1 el1:
    # ~15 µV spikes on ~10 µV noise, every beat at amplitude-SNR 1.1–2.3),
    # the derivative detector misses ~1/3 of the real spikes and the
    # gap-filling passes insert guesses at "expected" positions: against
    # the clean other electrode, 73 beats missed and 58 detections on
    # nothing. Neither amplitude nor dV/dt of a single detection helps
    # (both are noise-level), but the spike SHAPE is constant, which is
    # exactly what a matched filter exploits: correlating the signal with
    # a unit-norm template gives noise a std of σ and a spike its full
    # energy — a gain of ~√(spike samples). Un-normalised on purpose: NCC
    # divides by the local window energy, which at this SNR is all noise.
    #
    # Procedure: template = median of the top ``mf_seed_top_frac`` detected
    # beats by max|dV/dt|, ±``mf_half_ms``; y = x ⋆ template; peaks of y
    # above median + ``mf_threshold_k`` × MAD, refractory max(250 ms,
    # 0.5 × median RR). Result replaces the detections only if the count is
    # within ``mf_count_ratio`` of the original.
    #
    # Only in the low-SNR regime (median amplitude-SNR of detected beats
    # < ``mf_low_snr_regime``): on high-SNR recordings the same filter
    # picks up T-waves and after-potentials (Exp7 chipE: 201 vs 109 true
    # beats). Clean corpus files sit at SNR ≥ 4.4, the lab file at 1.8.
    # Calibration on the lab file vs its clean electrode: 220 beats,
    # 3 missed, 3 spurious, CV 13.5 % (derivative detector: 205 beats,
    # 73 missed, 58 spurious, CV 23 %). k=4.0 → 8 missed; k=4.5 → 31.
    enable_matched_filter_refine: bool = True
    mf_low_snr_regime: float = 3.0
    mf_half_ms: float = 25.0
    mf_threshold_k: float = 3.5
    mf_seed_top_frac: float = 0.5
    mf_min_seeds: int = 10
    # Acceptance window for the matched-filter count relative to the
    # derivative detector's. Was (0.7, 1.5): on the GG gold
    # standard (DEV split) the filter was rejected 97 times for finding too
    # FEW beats and was closer to the analyst in 61 of them — on noisy
    # recordings the derivative detector over-detects 2-3×. Widening the
    # lower bound to 0.3 cut spurious beats from 58 % to 37 % of the
    # analyst's count with missed beats unchanged (16.5 %).
    mf_count_ratio: tuple = (0.3, 2.0)

    # ── Minor-amplitude-population rejection (Oct 2026, GG) ──
    # Recordings with a dominant population of large spikes and a second
    # population of 2-5× smaller deflections (incubator noise bursts, a
    # weaker asynchronous source) that the analyst does not count. Too far
    # above the noise floor for the noise gate; no single sorted-amplitude
    # jump for the cluster filter. Split the log-amplitudes in two (Otsu)
    # and drop the small population only if ALL of:
    #   median(big)/median(small) ≥ ``minor_pop_ratio_min``;
    #   the big population alone has CV(RR) ≤ ``minor_pop_cv_gain`` × CV(RR)
    #     of the union (removing the small ones makes the rhythm MORE
    #     regular — true extra beats would make it less);
    #   the small beats are not phase-locked at a fixed fraction of the
    #     big-big interval with ~1:1 count (that is amplitude alternans,
    #     real beats, kept).
    # DEV split, combined with the wider MF acceptance: spurious 58 → 34 %,
    # perfect electrodes 44 → 49, missed 17.3 → 16.7 %.
    enable_minor_population_reject: bool = True
    minor_pop_ratio_min: float = 2.5
    minor_pop_cv_gain: float = 0.8
    minor_pop_min_big: int = 5
    minor_pop_alternans_phase_std: float = 0.12

    # ── Rhythm topology classifier (Sprint 2 #3) ──
    # Characterises the detected beats into one of:
    #   'regular'                 — one amplitude cluster, CV(RR) low
    #   'chaotic'                 — one amplitude cluster, CV(RR) high
    #   'alternans_2_to_1'        — two clusters, low beats at phase 0.5 ±ε
    #                                of each high-high cycle (bigeminy-like)
    #   'regular_with_ectopics'   — two clusters, low cluster amplitudes
    #                                uniform (likely biological ectopic beats)
    #   'regular_with_noise'      — two clusters, low cluster amplitudes
    #                                disperse (likely noise spikes)
    #   'trimodal'                — three clusters (dominant + secondary + noise)
    #   'unimodal_insufficient'   — fewer than required beats
    #   'degenerate'              — zero-amplitude windows present
    # This is a PURE classifier: it never modifies the beat list. Downstream
    # code reads info['rhythm_classification'] to decide how to interpret
    # beats (e.g. FPD only on dominant cluster when alternans, flag ectopics
    # for manual review, etc.).
    enable_rhythm_topology: bool = True
    topology_gap_ratio: float = 2.5           # absolute-floor ratio to always split
    topology_secondary_gap_ratio: float = 1.3 # min ratio for statistical gap
    topology_gap_zscore: float = 3.0          # z-score on log-ratio for significance
    topology_regular_cv_max: float = 0.15     # CV(RR) below → regular
    topology_chaotic_cv_min: float = 0.25     # CV(RR) above → chaotic
    topology_alternans_phase_band: float = 0.1  # |phase_low - 0.5| must be ≤ this
    topology_alternans_phase_std_max: float = 0.08  # phase std must be ≤ this
    topology_amp_cv_biological_max: float = 0.25    # amp CV ≤ this → biological
    topology_amp_cv_noise_min: float = 0.40         # amp CV ≥ this → noise
    topology_min_beats: int = 5                     # minimum beats to classify

    # ── Rhythm-aware FPD / QC (Sprint 2 #3, integration PR) ──
    # When enabled, parameter extraction (FPD, spike amplitude) uses only the
    # DOMINANT amplitude cluster for rhythm types where secondary/noise beats
    # would contaminate the template (alternans_2_to_1, regular_with_ectopics,
    # regular_with_noise, trimodal). The other rhythm types pass through
    # unchanged, so clean recordings are bit-identical to the non-integrated
    # pipeline.
    enable_rhythm_aware_fpd: bool = True
    # QC grade is downgraded by `rhythm_qc_downgrade_steps` when the
    # noise-cluster fraction of total classified beats exceeds this threshold.
    rhythm_qc_downgrade_threshold: float = 0.30
    rhythm_qc_downgrade_steps: int = 1

    # Rhythm filter safety bail (Sprint 3 #3 — bradycardia trimodal fix)
    # If "dominant cluster only" filtering would retain fewer than
    # ``rhythm_filter_min_retention_ratio × len(bi_clean)`` beats OR fewer
    # than ``rhythm_filter_min_retention_beats`` in absolute terms, skip
    # the filter and keep all QC-accepted beats. This prevents the
    # trimodal/ectopic classifier from throwing away the majority of a
    # bradycardic recording when the R+T double-spike pattern confuses
    # the amplitude-cluster assignment (observed on
    # Exp6_chipD_ch1 EL1 where 34 QC-accepted beats were reduced to 7 by
    # the "trimodal" branch). A rhythm whose dominant cluster carries
    # less than half the QC-accepted signal is almost certainly a
    # classification artefact rather than a true ectopy pattern.
    rhythm_filter_min_retention_ratio: float = 0.5
    rhythm_filter_min_retention_beats: int = 3

    # RR-outlier filter (Sprint 3 #1 — bradycardia robustness)
    # Drops beats whose preceding RR exceeds ``max_rr_outlier_ratio ×
    # median RR``. Catches dropout-followed-by-reactivation artefacts
    # that otherwise contaminate FPD / amplitude statistics (observed
    # on Exp6_chipD_ch1 where a single 24-second gap produced the only
    # "valid" FPD in the recording). Genuinely bradycardic signals are
    # untouched: when every RR is uniformly long the ratio test never
    # fires.
    enable_rr_outlier_filter: bool = True
    max_rr_outlier_ratio: float = 5.0

    # Beat recovery: after initial detection, use the estimated beat period
    # to search for missed beats at expected locations with a lower threshold.
    # Recovered candidates are validated against the template before acceptance.
    # This improves detection on low-SNR signals where beat amplitude varies.
    enable_beat_recovery: bool = True
    recovery_search_tolerance: float = 0.25    # ±25% of median beat period
    recovery_min_corr: float = 0.15            # min template correlation (low: noisy signals)
    recovery_min_amplitude_ratio: float = 0.20 # OR accept if amplitude ≥ 20% of ref
    recovery_min_dvdt_ratio: float = 0.25      # OR accept if dV/dt ≥ 25% of ref

    # Derivative method
    deriv_smooth_ms: float = 2.0       # smoothing window for derivative (ms)
    peak_refine_window_ms: float = 10.0  # ±window for peak refinement (ms)

    # ── Rhythm train (Oct 2026) ──
    # Besides the beats, the detected train can hold noise peaks (bursts of
    # noise), sharp artefacts and repolarisation waves. On the GG recordings
    # (slow beating, noisy) there were 32 % more detections than beats marked
    # by the analyst on the development experiments, and the CV of the
    # detected train reached 25 % on 49 of 114 recordings (analyst: 4): the
    # inclusion criterion then removed the tissue. When the detected train's
    # CV reaches rhythm_min_cv, the rhythm train is used for beat period, CV,
    # the local RR of the rate correction and the repolarisation search
    # window: the most regular sequence among the detections (period from
    # the first lobe of the forward-match fraction; sequence by dynamic
    # programming, an interval costing rhythm_lambda × |log(interval /
    # k·period)| / rhythm_sigma plus rhythm_miss_penalty per skipped beat),
    # with the periodicity-guided recovery filling its gaps. Segmentation,
    # QC, FPD and the arrhythmia analysis keep every detection.
    # Development experiments: CV ≥ 25 % on 3 recordings (analyst 4), beat
    # period within 10 % of the analyst's on 86 of 114 (was 74); held-out
    # experiments: CV ≥ 25 % on 1 (analyst 2, was 30). Decisions: GG 10 of
    # 13 test items as the analyst (was 9), Visone 2023 8 of 12 compounds
    # (was 7).
    enable_rhythm_train: bool = True
    rhythm_min_cv: float = 25.0                # % — only trains at least this irregular
    rhythm_period_frac: float = 0.7
    rhythm_period_tol: float = 0.10
    rhythm_lambda: float = 0.3
    rhythm_sigma: float = 0.15
    rhythm_miss_penalty: float = 0.6
    rhythm_min_beats: int = 6


# ═════════════════════════════════════════════════════════════════════════
#   REPOLARIZATION / FPD MEASUREMENT
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class RepolarizationConfig:
    """Repolarization detection and FPD measurement parameters."""

    # --- FPD measurement method ---
    # 'peak'            : peak of the repolarisation wave (default since
    #                     v3.6.0)
    # 'tangent'         : max-downslope → tangent-baseline intersection
    #                     (default until v3.5.x)
    # 'max_slope'       : point of maximum downslope after peak
    # '50pct'           : 50% amplitude on descending side
    # 'baseline_return' : zero-crossing after peak
    # 'consensus'       : run all methods, pick by cluster agreement
    # Evidence (Oct 2026, manual gold standard GG, DEV split,
    # full pipeline): the analyst marks the PEAK of the repolarisation wave.
    # With the wave chosen correctly (repol_candidate_rule), 'peak' gives a
    # median signed error of 0.0 % and 61 % of electrodes within ±10 %
    # (grade A 74 %); 'tangent' gives +2.5 % and 54 % (grade A 65 %). The
    # older note that 'peak' "underestimates ~25 %" is not supported by
    # either reference. The paper corpus (Visone 2023) stays within
    # tolerance with both.
    fpd_method: str = 'peak'

    # --- Correction formula ---
    # 'fridericia' : FPDcF = FPD / RR^(1/3)
    # 'bazett'     : FPDcB = FPD / sqrt(RR)
    # 'none'       : no correction (raw FPD)
    correction: str = 'fridericia'

    # --- Template averaging ---
    max_beats_template: int = 60
    alignment_max_shift_ms: float = 50.0
    alignment_depol_region_ms: float = 100.0  # first N ms used for alignment

    # --- Spike detection on template ---
    spike_search_window_ms: float = 50.0

    # --- Repolarization search window ---
    search_start_ms: float = 150.0      # start searching N ms after spike
    search_end_ms: float = 900.0        # stop searching N ms after spike

    # --- Adaptive search window extension ---
    # For signals with long beat periods (e.g. dofetilide-induced bradycardia),
    # the fixed search_end_ms may be too short to reach the real T-wave.
    # When median_bp is available, the effective search end is:
    #   max(search_end_ms, search_end_pct_rr × RR_ms)
    # 70% of RR is safe because the T-wave always ends well before the next beat.
    # Set to 0.0 to disable (use fixed search_end_ms only).
    search_end_pct_rr: float = 0.70

    # --- Minimum FPD constraint ---
    # Physiological floor: FPD below this is almost certainly an
    # afterpotential (post-spike rebound), not the actual T-wave.
    # hiPSC-CM FPD is typically 200–500 ms; even fast cells rarely
    # go below 120 ms.  Any candidate peak yielding FPD < min_fpd_ms
    # is excluded from selection.
    min_fpd_ms: float = 120.0

    # --- Adaptive minimum FPD based on beat period ---
    # FPD must be at least this fraction of the RR interval.
    # Physiological basis: action potential duration is always a
    # substantial fraction of the cycle length (typically 30–60%
    # in hiPSC-CM).  An FPD < 20% of RR is almost certainly the
    # afterpotential or a baseline artefact, not the true T-wave.
    # The effective floor is max(min_fpd_ms, min(min_fpd_pct_rr × RR,
    # max_adaptive_min_fpd_ms)).
    # Set to 0.0 to disable (use fixed min_fpd_ms only).
    min_fpd_pct_rr: float = 0.20

    # --- Cap on the adaptive contribution to the minimum-FPD floor ---
    # On strongly bradycardic signals (RR > 2.0 s — e.g. dofetilide
    # overdose, quiescent chambers) the adaptive floor
    # ``min_fpd_pct_rr × RR`` can exceed the typical hiPSC-CM T-wave
    # latency (200–400 ms), pushing the search start past the real
    # repolarisation peak and forcing every beat into the argmax
    # fallback.  Capping the adaptive contribution keeps the floor
    # within a biologically sensible range regardless of how slow the
    # rhythm is.  The fixed ``min_fpd_ms`` floor (120 ms, covers the
    # afterpotential zone) is NOT affected by this cap.
    # 350 ms (Apr 2026) was chosen so the floor on a 3-second bradycardia
    # stays below a hypothetical 300–500 ms T-wave. Raised to 600 ms in
    # Oct 2026 on the manual gold standard GG: on slow,
    # noisy rhythms (RR 2–4 s) the template picked a sharp after-potential
    # at ~500 ms instead of the broad T-wave at ~0.5×RR in 12.5 % of
    # beats; with the cap at 600 ms this drops to 9.1 % and per-beat FPD
    # within ±20 % goes 76 → 80 %. Both real references available —
    # Visone 2023 (paper) and this analyst (199 blocks) — show FPD/RR
    # ≥ 0.3 almost always (2 blocks < 0.3, none < 0.13), so a floor of
    # min(0.2×RR, 600 ms) excludes no observed physiology. A T-wave at
    # 400 ms on a 3 s rhythm (ratio 0.13) would now be missed; no such
    # case exists in either reference.
    # Set to 0.0 to disable the cap (pre-v3.3.1 behaviour).
    max_adaptive_min_fpd_ms: float = 600.0

    # --- FPD reliability gate (Sprint 3 #1, Fix C) ---
    # Fraction of beats that must produce a valid (non-NaN) FPD for the
    # recording-level FPD mean / std to be considered clinically
    # reportable.  When the valid fraction falls below this threshold
    # the summary sets ``fpd_reliable=False`` and adds a diagnostic
    # ``fpd_note``; downstream UI / export layers should surface the
    # warning instead of reporting the numerical mean as if it were
    # reliable.  Exp6_chipD_ch1 motivated this gate: 1/7 beats valid
    # produced FPDcF 318.8 ± 0.0, a meaningless summary statistic.
    # Set to 0.0 to disable (always flag reliable).
    min_valid_fpd_ratio: float = 0.50

    # --- Minimum signal amplitude for FPD analysis ---
    # If the median spike amplitude across all beats is below this threshold,
    # the signal is considered too weak for a template-guided search: the
    # template FPD is skipped (template confidence 0) and the per-beat
    # search runs unguided, where the per-beat gate rejects what is noise.
    # This keeps meaningless template FPDs off signals that are essentially
    # noise (e.g. chipC_ch3_MEXIL_1uM).
    # Unit: µV (microvolts).  0 = disabled.
    min_signal_amplitude_uV: float = 10.0

    # --- Signal conditioning for repolarization ---
    repol_lowpass_hz: float = 20.0       # low-pass cutoff for repol region
    repol_filter_order: int = 3
    detrend_margin_frac: float = 0.08    # linear detrend: use first/last 8%

    # --- Peak detection ---
    peak_prominence_factor: float = 0.15  # min prominence = factor × std(segment)
    peak_min_distance_ms: float = 50.0    # min distance between candidate peaks

    # --- Which candidate wave is the repolarisation (Oct 2026, phase 2) ---
    # 'max_prominence'  : historical — the most prominent peak of either
    #                     polarity beyond the min-FPD floor.
    # 'prefer_positive' : the repolarisation of these recordings is often
    #                     BIPHASIC (a positive and a negative lobe 60–200 ms
    #                     apart). The manual reference (GG, 87
    #                     electrodes with a matching candidate) marks the
    #                     POSITIVE lobe in 76 of 87, whether it comes first
    #                     or second; the negative lobe is often the more
    #                     prominent one, so 'max_prominence' picked it. Rule:
    #                     among eligible candidates, take the most prominent
    #                     positive peak whose prominence is ≥
    #                     ``repol_positive_min_rel_prom`` × the overall
    #                     maximum and which lies within
    #                     ``repol_positive_max_offset_ms`` of it (same
    #                     repolarisation complex); otherwise fall back to the
    #                     overall maximum (monophasic negative T-waves).
    # Calibrated on the DEV split only; leave-one-experiment-out chose the
    # same parameters in every fold (f 0.5–0.6, W 400 ms) and held-out
    # accuracy (67.9 % within ±10 %) matched in-sample (68.8 %).
    # A polarity convention, not a latency prior: it does not pull FPD
    # towards any value, so drug-induced prolongation is measured as-is.
    # It also makes the choice deterministic where two lobes have similar
    # size, instead of flipping between them across concentrations.
    repol_candidate_rule: str = 'prefer_positive'
    repol_positive_min_rel_prom: float = 0.5
    repol_positive_max_offset_ms: float = 400.0

    # --- Rhythm used for the template search window (Oct 2026) ---
    # The adaptive window end (search_end_pct_rr × RR) and the adaptive
    # min-FPD floor need the TRUE cycle length. Since v3.4 the beat-period
    # summary (correctly) uses the full detected train for rate correction,
    # and the window inherited it — but on over-detected recordings that
    # train contains spurious beats and its median RR is ~half the true one,
    # so the window closed before the T-wave (22 of 25 DEV electrodes with
    # no candidate near the analyst's value). The re-segmentation guard in
    # analyze.py already lengthens the template using the RR of the beats
    # that survive QC; the window now uses it too: RR_window =
    # max(RR of full train, RR of the beats forming the template).
    # The extension beyond the old window is scanned for a depolarisation-
    # like event (next beat) and stops before it, so the new window always
    # contains the old one and never crosses a spike.
    window_rr_from_template_beats: bool = True
    next_spike_guard_min_corr: float = 0.9   # correlation with the main spike shape
    next_spike_guard_min_amp: float = 0.5    # peak-to-peak relative to the main spike
    next_spike_guard_margin_ms: float = 30.0

    # --- Tangent method ---
    tangent_max_slope_window_ms: float = 300.0  # max distance peak → max-slope
    tangent_max_extension_ms: float = 400.0     # max distance peak → tangent intersection

    # --- Per-beat detection ---
    per_beat_tolerance_ms: float = 150.0  # search window ±tolerance around template FPD
    per_beat_peak_distance_ms: float = 30.0
    per_beat_distance_penalty_ms: float = 50.0  # distance penalty scale
    per_beat_prominence_factor: float = 0.15  # same as template; sensitivity comes from MAD-based noise estimate
    # Per-beat polarity: search the template's repolarisation sign first and
    # consider the opposite sign only if no same-sign peak qualifies. With a
    # biphasic repolarisation the opposite lobe sits 60–200 ms away, inside
    # the per-beat window, and is often larger — scoring both signs together
    # lets individual beats jump to the other lobe that the template
    # deliberately did not choose.
    # Evaluated on the DEV split (Oct 2026): no gain (−1 electrode within
    # ±10 % with the other phase-2 changes on), so OFF by default; kept as
    # an option.
    per_beat_prefer_template_sign: bool = False
    # A beat counts as polarity-inverted (repolarisation sign flipped,
    # template peak guidance dropped) only if its spike window is
    # anti-correlated with the template's at or below this value.
    inversion_corr_threshold: float = -0.5

    # --- Repolarization detectability gate ---
    # If the best repolarization candidate has prominence < gate_min_snr × noise,
    # the repolarization is considered NOT DETECTABLE and FPD is set to NaN.
    # This prevents forcing FPD measurements on noise (flat T-wave / drug effect).
    # Reference: EFP Analyzer (Patel et al., Sci Rep 2025) excludes traces
    # with non-detectable repolarization rather than forcing a value.
    enable_repol_gate: bool = True
    repol_gate_min_snr: float = 2.0    # min prominence / noise_std for template
    repol_gate_min_snr_beat: float = 1.5  # min for template-guided per-beat (lowered from original via noise exclusion)
    repol_gate_min_snr_beat_unguided: float = 1.5  # min for unguided per-beat (stricter)

    # --- Confidence scoring ---
    confidence_prominence_scale: float = 3.0  # prominence / (scale × noise) → saturates at 1
    confidence_agreement_range_ms: float = 300.0  # endpoint spread → 0% confidence
    confidence_weight_prominence: float = 0.6
    confidence_weight_agreement: float = 0.4

    # FPD confidence: template vs consistency weights
    fpd_conf_weight_template: float = 0.5
    fpd_conf_weight_consistency: float = 0.5
    fpd_cv_max_for_confidence: float = 0.5  # CV = 50% → consistency confidence = 0

    # --- Spike region for amplitude ---
    spike_pre_ms: float = 10.0       # before spike for amplitude calc
    spike_post_ms: float = 20.0      # after spike
    segment_pre_ms: float = 50.0     # pre-spike in segmented beats


# ═════════════════════════════════════════════════════════════════════════
#   QUALITY CONTROL
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class QualityConfig:
    """Signal quality assessment and beat validation."""

    # SNR thresholds for grading
    snr_excellent: float = 8.0   # Grade A
    snr_good: float = 5.0       # Grade B
    snr_fair: float = 3.0       # Grade C
    snr_poor: float = 2.0       # Grade D / F boundary

    # Beat amplitude rejection
    amplitude_reject_fraction: float = 0.25  # reject if < 25% of reference

    # Morphology correlation
    morphology_threshold: float = 0.40    # min corr for acceptance
    morphology_marginal: float = 0.20     # floor of the adaptive threshold; below it only the
                                          # strict-rhythm readmission can keep a beat
    use_morphology: bool = True

    # ── Analysability verdict (Oct 2026, calibrated on GG DEV set) ──
    # A recording whose detected "beats" are not clearly above its own noise
    # floor cannot be analysed by this software, whatever numbers the
    # pipeline would otherwise print. Metric: median over detected beats of
    # (peak-to-peak in ±20 ms) / (median 40 ms-window ptp of the signal) —
    # the same quantity the beat detector uses for its noise gate.
    # On the development split (Exp5/8/10, 217 electrodes with a manual
    # reference) a threshold of 1.6 flags 25 of the 28 recordings the analyst
    # declared not analysable ("STOP BEATING", "impossible to analyse", n.a.)
    # and 55 more that the analyst did measure — but on 54 of those 55 the
    # pipeline's own output was wrong (BP within ±10 % in 9 %, FPD in 2 %,
    # 82 % of beats missed). So "not analysable" here means "not by this
    # software", and it is honest in 79/80 cases. The 3 analyst-NA cases it
    # misses are high-SNR rhythms judged "too irregular" — a rhythm verdict,
    # not a noise verdict. Held-out check on Exp6/7/9 in the changelog.
    # When it fires: QC grade F, FPD/FPDc set to NaN, fpd_reliable False,
    # arrhythmia class "Not analysable", excluded from normalisation both
    # as drug recording and as baseline. Beat counts are kept for audit.
    # Second criterion — rhythm: the analyst also declares recordings "too
    # irregular to analyse". After the detection fixes of Oct 2026 removed
    # most noise detections, 8 analyst-NA recordings passed the SNR test;
    # all had RR CV 45–93 %, most with < 16 beats in 60 s. Rule: fewer than
    # ``not_analysable_sparse_beats`` detections AND RR CV above
    # ``not_analysable_sparse_cv_pct`` → not analysable. DEV: catches 6 of
    # those 8; flags 7 analysed-by-analyst recordings on which the pipeline
    # was wrong in all 7 (BP ±10 % in 29 %, none with FPD ±20 %).
    enable_analysability_verdict: bool = True
    not_analysable_snr: float = 1.6
    not_analysable_min_beats: int = 3     # fewer detections → verdict by count, not SNR
    not_analysable_sparse_beats: int = 16
    not_analysable_sparse_cv_pct: float = 40.0

    # Rejection rate thresholds for grade downgrade
    max_rejection_rate: float = 0.40      # above → Grade D
    rejection_high_note: float = 0.50     # above → add warning note
    rejection_grade_c: float = 0.20       # above → limits to Grade C
    rejection_grade_b: float = 0.05       # above → limits to Grade B

    # Template building for morphology QC
    morphology_max_beats: int = 30
    morphology_window_ms: float = 20.0    # amplitude computation window

    # Depolarization-focused morphology correlation: restrict the correlation
    # to the first N ms of the beat segment (covering the depolarization spike
    # and early repolarization).  The full beat segment (800+ ms) includes the
    # entire repolarization wave, which varies a lot in 3D constructs and drags
    # down correlation even when the depolarization spike is perfectly clear.
    # Set to 0 to use the full segment (original behavior).
    morphology_corr_region_ms: float = 150.0  # first 150 ms of beat segment

    # Strict on-rhythm re-admission for morphology rejections:
    # when a morph-rejected beat is very tightly on-rhythm (timing residual
    # below *strict_rhythm_residual_ratio* × median RR) AND its amplitude
    # ratio is robust (≥ *strict_rhythm_amp_ratio*), admit it even if its
    # morphology correlation is below the marginal floor.  Rationale: on
    # bradycardic signals in 3D constructs, a missed beat may retain normal
    # amplitude but present a degraded / decorrelated shape; without this
    # path the QC loses genuinely real beats (observed on Exp6 Ti08 EL2).
    # Set *strict_rhythm_amp_ratio* ≥ 1 to disable the strict-rhythm path.
    strict_rhythm_residual_ratio: float = 0.10  # ±10% of median RR
    strict_rhythm_amp_ratio: float = 0.50       # ≥ 50% of reference amplitude

    # Minimum accepted beats
    min_beats_for_analysis: int = 3       # below → Grade F


# ═════════════════════════════════════════════════════════════════════════
#   INCLUSION CRITERIA
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class InclusionConfig:
    """Inclusion criteria for baseline normalization."""

    # Baseline CV of beat period
    max_cv_bp: float = 25.0            # % — paper standard
    enabled_cv: bool = True

    # ── Precision of the baseline FPDc reference ──
    # What a baseline is *for* is to supply a reference FPDc against which
    # the drug value is compared. What matters is therefore how precisely
    # that reference is determined — the relative standard error of the
    # mean, SD/(mean·sqrt(n)) — not how regular the rhythm happened to be.
    #
    # CV(RR) was standing in for this and does it badly. Measured on the 36
    # calibration baselines with corrected RR, CV(RR) explains only ~38% of
    # the variance in FPDc dispersion (r = 0.61), and the CV<25% gate has
    # 5 false positives against 0 false negatives — it discards precise
    # references and catches nothing a precision criterion would miss:
    #
    #   chipC_ch2_baseline   CV(RR) 29.3%   rSEM 1.68%   n=70   QC C
    #   chipD_ch2_baseline   CV(RR) 38.5%   rSEM 1.63%   n=17   QC B
    #   chipD_ch3_baseline   CV(RR) 29.2%   rSEM 1.25%   n=65   QC B
    #   chip4_ch3_BASELINE   CV(RR) 26.2%   rSEM 1.14%   n=84   QC B
    #
    # An irregular but well-sampled preparation gives a perfectly usable
    # reference; the averaging absorbs the irregularity. Rejecting it costs
    # an entire dose-response group.
    #
    # Threshold rationale: the classification threshold is a 15% change in
    # FPDc, so a reference uncertain to 3% contributes at most a fifth of
    # the effect being called. Distribution over the 36 baselines: median
    # 1.41%, p75 2.31%, p90 3.84%, max 6.37%.
    #
    # Caveat: rSEM falls with beat count, so a short recording of a healthy
    # preparation scores worse than a long one. That is arguably correct —
    # a shorter recording *does* give a less precise reference — but it
    # means this measures the reference, not the biology. Preparation
    # quality is the QC grade's job, and physiological plausibility the
    # FPD/RR ratio's.
    max_baseline_fpdc_rsem: float = 3.0    # %
    enabled_baseline_precision: bool = False   # opt-in until validated

    # FPDcF plausibility range (ms) — wide safety net
    fpdc_range_min: float = 100.0
    fpdc_range_max: float = 1200.0
    enabled_fpdc_range: bool = True

    # FPD confidence threshold for baselines (data-driven)
    # With physiological filter ON (criterion 4), the critical chipE_ch2
    # baseline (FPDcF=346ms) is already excluded by the physiol range,
    # so this threshold only needs to separate chipE_ch1 (0.659, bad)
    # from chipA_ch1 (0.691, good).  Gap = 0.032, any value in [0.66, 0.69]
    # works.  Lowered from 0.69 → 0.66 for generality (midpoint ≈ 0.675).
    min_fpd_confidence: float = 0.66
    enabled_confidence: bool = True

    # ── Physiological FPDcF range for baselines (literature-based) ──
    # hiPSC-CM FPDcF values from literature:
    #   Visone et al. 2023: 560 ± 150 ms (n=51 microtissues)
    #   Asakura et al. 2015: 400–700 ms (hiPSC-CM on MEA)
    #   Blinova et al. 2017 (CiPA): 350–800 ms typical range
    # Baselines outside this range likely have erroneous FPD detection.
    # More defensible than a data-driven confidence threshold because
    # the bounds come from published population data, not from fitting
    # to a specific dataset.
    fpdc_physiol_min: float = 350.0     # ms — lower bound
    fpdc_physiol_max: float = 800.0     # ms — upper bound
    # OFF since 2026-08-05: superseded by the FPD/RR ratio below.
    #
    # These bounds are sound literature values, but an *absolute* window on
    # FPDc misfires at the extremes of beat rate, because Fridericia does
    # not fully remove the rate dependence. Measured on the 36 calibration
    # baselines with corrected RR, this filter rejected:
    #   * chipD_ch3_baseline — 27 bpm, FPDc 871 ms, but FPD/RR only 51%:
    #     a bradycardic preparation with a long yet proportionate FPD;
    #   * chipD_ch1_baseline — 70 bpm, FPDc 277 ms, FPD/RR 31%: fast, with
    #     a correspondingly short FPD.
    # Both are QC-grade-B recordings rejected for being at the edges of the
    # rate range rather than for anything wrong with them, and rejecting a
    # baseline removes its whole dose-response group.
    enabled_fpdc_physiol: bool = False

    # ── FPD / RR ratio (physiological, rate-independent) ──
    # Repolarization cannot occupy the whole cardiac cycle: an FPD equal to
    # or longer than the beat interval would mean repolarization ending
    # after the next depolarization has already started. Values at or above
    # 100% are therefore not measurements but detection failures — typically
    # an afterpotential, or the following depolarization, mistaken for the
    # T wave.
    #
    # Distribution over the 36 calibration baselines (corrected RR):
    #   p25 = 32%, median = 41%, p75 = 55%, p90 = 83%, max = 109%.
    # Only 4 of 36 exceed 80%, and all four are implausible:
    #   chipE_ch2_baseline          109%   (FPD longer than RR)
    #   chip1_ch3_baseline_nosignal 105%   (operator marked it "no signal")
    #   chipA_ch1_basline            98%
    #   chipA_ch2_baseline           94%
    #
    # Being a ratio this is dimensionless and valid at any beat rate, which
    # is exactly what the absolute FPDc window could not manage.
    max_fpd_rr_ratio: float = 0.80
    enabled_fpd_rr_ratio: bool = True

    # ── Population-based outlier exclusion for baselines ──
    # Within a batch, exclude baselines whose FPDcF is > N standard
    # deviations from the median of all baselines in the same experiment.
    # This is a data-adaptive alternative to fixed thresholds: it lets
    # each experiment define its own "normal" range, accommodating
    # biological variability across preparations while still catching
    # outliers.  Requires ≥ min_baselines_for_stats baselines to compute
    # statistics; with fewer, this criterion is silently skipped.
    fpdc_outlier_n_sigma: float = 2.0          # reject if |FPDcF - median| > N × MAD
    fpdc_outlier_min_baselines: int = 3        # need at least 3 baselines to compute stats
    enabled_fpdc_outlier: bool = False          # off by default (opt-in)

    # ── Plausibility guardrails (independent of CV) ──
    # A low CV is evidence of *regularity*, not of quality — periodic noise
    # scores a better CV than a real preparation. Measured on the 36
    # calibration baselines: chip1_ch3_baseline_nosignal.csv (named
    # "nosignal" by the operator, BPM=106, 33% of beats with no detectable
    # repolarization, confidence 0.59) PASSES the CV<25% gate, while
    # QC-grade-B baselines with confidence 0.86-0.88 are excluded by it.
    #
    # These bounds catch what CV structurally cannot: a spontaneous hiPSC-CM
    # preparation beating at >120 bpm or <10 bpm is not a usable baseline
    # whatever its rhythm regularity, and neither is one where most beats
    # have no measurable repolarization.
    bpm_plausible_min: float = 10.0
    bpm_plausible_max: float = 120.0
    max_pct_beats_no_repol: float = 50.0   # %
    enabled_plausibility: bool = False       # OFF by default (opt-in)

    # ── Combined quality rule (alternative to CV-only) ──
    # Replaces the single CV gate with QC grade AND a much wider CV bound
    # AND the confidence threshold. On the calibration set this keeps the
    # same number of baselines (15/36) but swaps five of them: it admits
    # the QC-B/C baselines that only failed on rhythm irregularity — normal
    # in spontaneously beating hiPSC-CM — and rejects the "nosignal" file
    # plus four grade-D/F ones.
    #
    # See SPRINT1_soglia_CV_baseline.md for the full distribution.
    #
    # When enabled, this SUPERSEDES criterion 1 (max_cv_bp); the other
    # criteria still apply. OFF by default: turning it on changes which
    # recordings enter the analysis, which must be an explicit decision.
    combined_min_qc_grade: str = 'C'        # worst acceptable QC grade
    combined_max_cv_bp: float = 60.0        # % — wide bound, catches only the extremes
    # Still OFF by default, deliberately.
    #
    # With corrected RR the CV-only gate turns out not to discriminate at
    # all — median CV per QC grade is A 7.4%, B 24.9%, C 22.7%, D 27.9%,
    # F 26.1%: only grade A separates. The A→F gradient measured before the
    # RR fix was the artefact itself (worse QC → more rejected beats → more
    # inflated CV). So reading quality from the QC grade directly is the
    # right instinct.
    #
    # But the cutoff is not calibrated. At 'C' this rule re-excludes the
    # dofetilide baseline (QC=D, 49% of beats rejected) and loses the whole
    # dose-response again — 15/24 files included instead of 23/24. At 'D'
    # it keeps everything (24/24). The grade is driven largely by rejection
    # rate, which is itself sensitive to the adaptive morphology threshold,
    # and the FPD-measurement concern it stands for is already covered more
    # directly by ``min_valid_fpd_ratio`` / ``fpd_reliable``.
    #
    # Removing a baseline removes its entire dose-response group, so this
    # gate is expensive to get wrong. It stays opt-in until the cutoff is
    # calibrated against corrected data the way max_cv_bp was.
    enabled_combined_rule: bool = False


# ═════════════════════════════════════════════════════════════════════════
#   NORMALIZATION & TdP SCORING
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class NormalizationConfig:
    """Baseline normalization and TdP risk scoring."""

    # FPDcF change thresholds for TdP scoring (%)
    threshold_low: float = 10.0     # score 1 if ≥ LOW
    threshold_mid: float = 15.0     # score 2 if ≥ MID
    threshold_high: float = 20.0    # score 3 if ≥ HIGH

    # Sensitivity/Specificity classification threshold
    # (which threshold to use for positive/negative call)
    classification_threshold: str = 'mid'  # 'low', 'mid', 'high'

    # Classification method for drug-level decision
    # 'concentration' : tissue mean at each concentration (≥ min_tissues
    #                   tissues) reaches the threshold at `consecutive`
    #                   adjacent concentrations — default since Oct 2026
    # 'max'  : drug positive if any recording ≥ threshold (default until
    #          v3.7.0: on the Visone 2023 data it called every negative
    #          compound positive, see normalization.classify_drug)
    # 'mean' : drug positive if the mean of all recordings ≥ threshold
    # 'n_above' : drug positive if ≥ n recordings ≥ threshold
    classification_method: str = 'concentration'
    classification_n_above: int = 2  # used when method = 'n_above'
    # used when method = 'concentration'; min_tissues=1 and consecutive=1
    # give the rule of Visone et al. 2023
    classification_min_tissues: int = 2
    classification_consecutive: int = 2

    # Smart cessation override
    # When a drug causes cessation AND waveform destruction (low FPD confidence),
    # elevate the drug to positive even if FPDcF measurement failed.
    # Only triggers when min FPD confidence across concentrations < threshold.
    # Off by default since Oct 2026: on the Visone 2023 data the condition
    # holds for three compounds (ranolazine, mexiletine, aspirin) and the only
    # call it changes is aspirin, a negative, to positive. The condition is
    # always reported as 'cessation_flag' in the drug classification.
    enable_cessation_override: bool = False
    cessation_override_max_fpd_confidence: float = 0.60

    # ── QC filter for normalized recordings ──
    # When enabled, drug recordings with a QC grade below the minimum
    # are EXCLUDED from the drug-level classification (classify_drug).
    # They still appear in the normalization table (so the user can see
    # the data), but they do not contribute to the positive/negative call.
    #
    # Rationale: low-QC recordings often have unreliable FPDcF values
    # (high beat rejection, noisy morphology) that cause false positives.
    # For example, a single QC=D recording with +31% FPDcF can flip the
    # whole drug classification to "prolongation" even when all other
    # concentrations show shortening.
    #
    # Grade hierarchy: A > B > C > D > F
    # Default 'D' means only grades A, B, C are used for classification.
    norm_min_qc_grade: str = 'D'          # minimum QC grade to include
    norm_min_qc_enabled: bool = False      # OFF by default (opt-in)

    # ── Maximum CV for normalized recordings ──
    # Drug recordings with CV(BP) above this threshold are excluded from
    # classification, similar to the baseline inclusion CV filter.
    # Very irregular recordings (CV > 50%) often have unreliable FPDcF.
    norm_max_cv_bp: float = 50.0           # %
    norm_max_cv_enabled: bool = False       # OFF by default (opt-in)

    # ── FPD reliability filter ──
    # ``RepolarizationConfig.min_valid_fpd_ratio`` already makes
    # parameters.py stamp ``summary['fpd_reliable'] = False`` when the
    # repolarization wave was measurable on too few beats.  That flag was
    # computed but never consulted here, so a recording whose FPDcF came
    # from (say) 1 of 7 beats — the documented "FPDcF 318.8 ± 0.0" case —
    # still contributed a %ΔFPDcF and could flip a drug classification.
    #
    # Two independent switches, deliberately:
    #
    #   * the flag is ALWAYS propagated into the normalization dict
    #     (``baseline_fpd_reliable`` / ``drug_fpd_reliable`` /
    #     ``fpd_reliable``), so the condition is visible in reports and in
    #     the CDISC export whatever the setting;
    #   * exclusion from drug-level classification is opt-in, matching the
    #     QC-grade and CV filters above, so enabling it is an explicit,
    #     documented analysis decision rather than a silent change.
    #
    # Turning this ON is recommended for anything reported externally.
    norm_require_fpd_reliable: bool = False   # OFF by default (opt-in)

    # ── Near-cessation guard (Oct 2026) ──
    # Fridericia divides FPD by RR^(1/3): with a beating period of tens of
    # seconds (tissue almost stopped) the corrected FPD is meaningless — a
    # cisapride recording at RR = 40 s gave %ΔFPDcF = +580 % on the Visone
    # 2023 data and decided the drug call on its own. Above this period on
    # the baseline or the drug recording the %ΔFPDcF is withheld and the
    # reason recorded in normalization['fpdc_withheld']; BP and amplitude
    # changes are still reported and the cessation override still sees the
    # recording. 6000 ms = 10 beats/min, the same bound as
    # InclusionConfig.bpm_plausible_min.
    max_beat_period_for_fpdc_ms: float = 6000.0


# ═════════════════════════════════════════════════════════════════════════
#   ARRHYTHMIA DETECTION
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ArrhythmiaConfig:
    """Arrhythmia detection thresholds."""

    # Heart rate classification
    tachycardia_bp_ms: float = 300.0
    bradycardia_bp_ms: float = 2500.0

    # Rhythm regularity
    rr_irregularity_cv: float = 15.0        # % — CV threshold
    rr_critical_cv: float = 30.0            # % — critical severity
    fibrillation_cv: float = 40.0           # % — chaotic/fibrillation-like

    # Beat timing classification
    premature_beat_factor: float = 0.7      # < 70% of mean BP
    delayed_beat_factor: float = 1.5        # > 150% of mean BP
    cessation_factor: float = 3.0           # > 300% of mean BP

    # STV (short-term variability)
    stv_high_risk_ms: float = 10.0

    # FPD prolongation (ratio vs baseline)
    fpd_prolongation_threshold: float = 1.3   # > 130% of baseline
    fpd_critical_length_ms: float = 500.0     # FPD above which prolongation is critical

    # EAD detection (statistical — FPD outlier)
    ead_mad_factor: float = 3.0             # 3× median absolute deviation

    # EAD detection (residual; Visone et al. 2023 looked for irregular peaks
    # in the residual, the five criteria below are this software's)
    # Five criteria — ALL must be met for a residual peak to be EAD:
    #   1. Statistical:  peak > prominence × σ  of residual noise
    #   2. Absolute:     peak > min_amp_frac × template peak-to-peak
    #   3. Width:        half-max width ∈ [min_width, max_width] ms
    #   4. Polarity:     must be positive (secondary depolarisation)
    #   5. Location:     within 150-500 ms repol. window (plateau phase)
    ead_residual_prominence: float = 6.0    # peak > N × σ in repol. residual
    ead_residual_min_amp_frac: float = 0.08 # peak > 8% of template amplitude
    ead_residual_min_width_ms: float = 8.0  # peak width ≥ 8 ms at half-max
    ead_residual_max_width_ms: float = 150.0  # peak width ≤ 150 ms (wider = shape change, not EAD)

    # Amplitude instability
    amplitude_instability_cv: float = 30.0  # %

    # (v3.14.1: the fields ead_critical_count, premature_count_threshold and
    # tdp_require_severe_only were removed — nothing read them; severity
    # is decided on incidence, 10 % of beats, see analyze_arrhythmia.)

    # ── Risk score mode ──
    # 'manual'      : expert-assigned weights (default — physiological rationale)
    # 'data_driven' : weights fitted via logistic regression on CiPA dataset
    #                 (loaded from fitted_weights.json)
    # The manual weights are recommended as default because per-recording
    # risk scoring serves a different purpose than drug-level classification.
    # Data-driven weights are experimental and require the fitted_weights.json
    # file from the CiPA calibration analysis.
    risk_score_mode: str = 'manual'


# ═════════════════════════════════════════════════════════════════════════
#   CHANNEL SELECTION
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelSelectionConfig:
    """Parameters for automatic channel selection."""

    bp_ideal_range_s: tuple[float, float] = (0.3, 4.0)
    # NOTE: `cv_excellent/cv_good/cv_fair` (in percent units) were declared
    # here historically but never read by any channel-selection code.
    # Regularity scoring is computed via the linear formula
    # `w_regularity_max - cv_pct * w_regularity_slope` (see
    # channel_selection.select_best_channel), so these threshold fields
    # were dead code AND a semantic trap because BeatDetectionConfig has
    # fields with identical names but in FRACTION units. Removed.
    rate_range_per_s: tuple[float, float] = (0.3, 3.5)

    # Scoring weights (points awarded per criterion)
    w_bp_range: float = 15.0         # beat period in ideal range
    w_rate_ok: float = 10.0          # beat rate in plausible range
    w_corr_max: float = 40.0         # max points for template correlation
    w_corr_scale: float = 44.0       # linear scaling: score = corr * scale - offset
    w_corr_offset: float = 4.0       # offset for correlation scoring
    w_regularity_max: float = 20.0   # max points for beat-period regularity
    w_regularity_slope: float = 0.4  # points lost per 1% CV
    w_amplitude_max: float = 15.0    # max points for spike amplitude
    w_amplitude_ref_mV: float = 500.0  # amplitude (mV) for max points


# ═════════════════════════════════════════════════════════════════════════
#   TOP-LEVEL CONFIG
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisConfig:
    """
    Top-level configuration for the entire Cardiac FP Analyzer pipeline.

    Usage:
        config = AnalysisConfig()                     # defaults
        config = AnalysisConfig.from_json("my.json")  # load from file
        config = AnalysisConfig.preset("conservative") # named preset

        # Modify individual parameters
        config.repolarization.fpd_method = 'peak'
        config.inclusion.max_cv_bp = 30.0

        # Pass to pipeline
        batch_analyze(data_dir, config=config)
    """

    filtering: FilterConfig = field(default_factory=FilterConfig)
    beat_detection: BeatDetectionConfig = field(default_factory=BeatDetectionConfig)
    repolarization: RepolarizationConfig = field(default_factory=RepolarizationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    inclusion: InclusionConfig = field(default_factory=InclusionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    arrhythmia: ArrhythmiaConfig = field(default_factory=ArrhythmiaConfig)
    channel_selection: ChannelSelectionConfig = field(default_factory=ChannelSelectionConfig)

    # ── Signal scaling ──
    # Amplifier gain correction: raw_signal / amplifier_gain = real voltage.
    # For the µECG-Pharma Digilent system the amplifier gain is 10⁴, so the
    # CSV columns are in *amplified* volts and tissue voltage = raw / 1e4.
    # Default 1e4 (Oct 2026; was 1.0): both UIs already forced 1e4, but the
    # library default governed CLI, batch and Study runs, where every
    # ``spike_amplitude_mV`` came out 10 000× too large and the absolute
    # ``min_signal_amplitude_uV`` gate could never fire. Set to 1.0 only for
    # data already in physical volts (e.g. synthetic tests).
    # Applied once, in analyze_single_file, before filtering, and in the
    # el1/el2 channel selection (whose amplitude term, referenced to raw
    # mV, is therefore ≈0 with the default gain: the choice rests on the
    # template correlation, rhythm regularity and rate terms). Not applied
    # to MCS files, which are already in volts.
    amplifier_gain: float = 1e4

    # Advanced analysis modules (enabled by default)
    enable_cessation: bool = True
    enable_spectral: bool = True

    # ── Manual beat overrides ──
    # When a user corrects the automatic beat detection in the PySide6
    # UI we persist the correction as a ``.overrides.json`` sidecar next
    # to the CSV (see ``cardiac_fp_analyzer.overrides``).  When True the
    # pipeline loads that sidecar on every ``analyze_single_file`` run
    # and applies the saved add/remove list on top of the automatic
    # detection — so the same edits survive re-opens and batch runs.
    # Set to False to ignore any sidecars (e.g. for a pristine re-run).
    use_overrides: bool = True

    # ── Multi-chamber chips (MCS files) ──
    # 'auto': a file whose channel labels match a known layout (chambers.py:
    # 'uheart_mvp_64', E1…E64) is analysed as one recording per chamber,
    # each restricted to the chamber's electrodes, with the stimulation
    # electrodes left out of the electrode choice. A layout name forces
    # it; 'none' treats the file as one tissue.
    chamber_layout: str = 'auto'
    # Chamber consensus (v3.14): on a chamber of a multi-electrode chip the
    # beat period comes from the beats seen by several electrodes, the
    # rhythm is classified (regular / irregular / conduction lost / silent)
    # and the FPD is the consensus of the electrodes (same wave as the
    # baseline on the doses); the single electrode's values stay in the
    # summary as *_electrode. False keeps the single-electrode measures.
    chamber_consensus: bool = True

    # ── Serialization ──

    def to_dict(self) -> dict:
        """Convert entire config to a nested dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Serialize to JSON string. Optionally write to file."""
        d = self.to_dict()
        s = json.dumps(d, indent=indent, ensure_ascii=False)
        if path:
            with open(path, 'w') as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> 'AnalysisConfig':
        """Create config from a (possibly partial) nested dictionary.

        Missing keys keep their defaults — so you can provide only
        the parameters you want to override.
        """
        cfg = cls()
        section_map = {
            'filtering': (FilterConfig, 'filtering'),
            'beat_detection': (BeatDetectionConfig, 'beat_detection'),
            'repolarization': (RepolarizationConfig, 'repolarization'),
            'quality': (QualityConfig, 'quality'),
            'inclusion': (InclusionConfig, 'inclusion'),
            'normalization': (NormalizationConfig, 'normalization'),
            'arrhythmia': (ArrhythmiaConfig, 'arrhythmia'),
            'channel_selection': (ChannelSelectionConfig, 'channel_selection'),
        }
        # Legacy field name migrations — applied BEFORE the generic loop
        # so values written by older versions still take effect instead of
        # being silently dropped by the `hasattr` guard below.
        _legacy_renames = {
            # BeatDetectionConfig: cv thresholds got `_frac` suffix
            # (they are fractions, not percentages).
            'beat_detection': {
                'cv_good': 'cv_good_frac',
                'cv_fair': 'cv_fair_frac',
                'cv_marginal': 'cv_marginal_frac',
            },
        }
        for sect_name, renames in _legacy_renames.items():
            if sect_name in d and isinstance(d[sect_name], dict):
                for old, new in renames.items():
                    if old in d[sect_name] and new not in d[sect_name]:
                        d[sect_name][new] = d[sect_name].pop(old)

        for key, (klass, attr) in section_map.items():
            if key in d:
                section = getattr(cfg, attr)
                for k, v in d[key].items():
                    if hasattr(section, k):
                        # Handle tuple fields
                        current = getattr(section, k)
                        if isinstance(current, tuple) and isinstance(v, list):
                            v = tuple(v)
                        setattr(section, k, v)

        # Top-level flags
        if 'amplifier_gain' in d:
            cfg.amplifier_gain = float(d['amplifier_gain'])
        if 'enable_cessation' in d:
            cfg.enable_cessation = d['enable_cessation']
        if 'enable_spectral' in d:
            cfg.enable_spectral = d['enable_spectral']
        if 'use_overrides' in d:
            cfg.use_overrides = bool(d['use_overrides'])
        if 'chamber_layout' in d:
            cfg.chamber_layout = str(d['chamber_layout'] or 'auto')
        if 'chamber_consensus' in d:
            cfg.chamber_consensus = bool(d['chamber_consensus'])

        return cfg

    @classmethod
    def from_json(cls, path: str) -> 'AnalysisConfig':
        """Load config from a JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def preset(cls, name: str) -> 'AnalysisConfig':
        """
        Named presets for common use cases.

        Available presets:
          - 'default'       : standard parameters (peak method, Fridericia, paper criteria)
          - 'conservative'  : stricter inclusion, higher confidence thresholds
          - 'sensitive'     : looser thresholds, catches more but more FP risk
          - 'peak_method'   : fpd_method='peak' — identical to 'default' since
                              peak became the default (kept for old configs)
          - 'no_filters'    : disable all inclusion criteria
        """
        cfg = cls()

        if name == 'default':
            pass  # all defaults

        elif name == 'conservative':
            cfg.inclusion.max_cv_bp = 20.0
            cfg.inclusion.min_fpd_confidence = 0.75
            cfg.quality.morphology_threshold = 0.50
            cfg.normalization.classification_threshold = 'high'
            cfg.arrhythmia.rr_irregularity_cv = 12.0

        elif name == 'sensitive':
            cfg.inclusion.max_cv_bp = 30.0
            cfg.inclusion.min_fpd_confidence = 0.50
            cfg.quality.morphology_threshold = 0.30
            cfg.normalization.classification_threshold = 'low'

        elif name == 'peak_method':
            cfg.repolarization.fpd_method = 'peak'

        elif name == 'no_filters':
            cfg.inclusion.enabled_cv = False
            cfg.inclusion.enabled_fpdc_range = False
            cfg.inclusion.enabled_confidence = False

        else:
            raise ValueError(f"Unknown preset: {name!r}. "
                           f"Available: default, conservative, sensitive, "
                           f"peak_method, no_filters")

        return cfg

    def describe(self) -> str:
        """Human-readable summary of non-default parameters."""
        default = AnalysisConfig()
        lines = []
        for section_name in ['filtering', 'beat_detection', 'repolarization',
                             'quality', 'inclusion', 'normalization',
                             'arrhythmia', 'channel_selection']:
            section = getattr(self, section_name)
            default_section = getattr(default, section_name)
            diffs = []
            for k in vars(section):
                if getattr(section, k) != getattr(default_section, k):
                    diffs.append(f"  {k} = {getattr(section, k)!r}  (default: {getattr(default_section, k)!r})")
            if diffs:
                lines.append(f"[{section_name}]")
                lines.extend(diffs)
        return '\n'.join(lines) if lines else '(all defaults)'
