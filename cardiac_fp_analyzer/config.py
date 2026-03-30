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

    # Baseline drift removal (highpass)
    highpass_cutoff_hz: float = 0.5
    highpass_order: int = 4

    # Anti-alias / smoothing (lowpass)
    lowpass_cutoff_hz: float = 200.0
    lowpass_order: int = 4

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
    cv_good: float = 0.15        # CV < 15% → good score
    cv_fair: float = 0.30        # CV < 30% → fair
    cv_marginal: float = 0.50    # CV < 50% → marginal

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

    # Derivative method
    deriv_smooth_ms: float = 2.0       # smoothing window for derivative (ms)
    peak_refine_window_ms: float = 10.0  # ±window for peak refinement (ms)


# ═════════════════════════════════════════════════════════════════════════
#   REPOLARIZATION / FPD MEASUREMENT
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class RepolarizationConfig:
    """Repolarization detection and FPD measurement parameters."""

    # --- FPD measurement method ---
    # 'tangent'         : max-downslope → tangent-baseline intersection (gold standard)
    # 'peak'            : repolarization peak only (simpler, underestimates ~25%)
    # 'max_slope'       : point of maximum downslope after peak
    # '50pct'           : 50% amplitude on descending side
    # 'baseline_return' : zero-crossing after peak (overestimates ~5%)
    # 'consensus'       : run all methods, pick by cluster agreement (most robust)
    fpd_method: str = 'tangent'

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

    # --- Signal conditioning for repolarization ---
    repol_lowpass_hz: float = 20.0       # low-pass cutoff for repol region
    repol_filter_order: int = 3
    detrend_margin_frac: float = 0.08    # linear detrend: use first/last 8%

    # --- Peak detection ---
    peak_prominence_factor: float = 0.15  # min prominence = factor × std(segment)
    peak_min_distance_ms: float = 50.0    # min distance between candidate peaks

    # --- Tangent method ---
    tangent_max_slope_window_ms: float = 300.0  # max distance peak → max-slope
    tangent_max_extension_ms: float = 400.0     # max distance peak → tangent intersection

    # --- Per-beat detection ---
    per_beat_tolerance_ms: float = 150.0  # search window ±tolerance around template FPD
    per_beat_peak_distance_ms: float = 30.0
    per_beat_distance_penalty_ms: float = 50.0  # distance penalty scale

    # --- Repolarization detectability gate ---
    # If the best repolarization candidate has prominence < gate_min_snr × noise,
    # the repolarization is considered NOT DETECTABLE and FPD is set to NaN.
    # This prevents forcing FPD measurements on noise (flat T-wave / drug effect).
    # Reference: EFP Analyzer (Patel et al., Sci Rep 2025) excludes traces
    # with non-detectable repolarization rather than forcing a value.
    enable_repol_gate: bool = True
    repol_gate_min_snr: float = 2.5    # min prominence / noise_std for template
    repol_gate_min_snr_beat: float = 2.0  # min for per-beat (more lenient)

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
    morphology_marginal: float = 0.20     # below this → forced rejection
    use_morphology: bool = True

    # Rejection rate thresholds for grade downgrade
    max_rejection_rate: float = 0.40      # above → Grade D
    rejection_high_note: float = 0.50     # above → add warning note
    rejection_grade_c: float = 0.20       # above → limits to Grade C
    rejection_grade_b: float = 0.05       # above → limits to Grade B

    # Template building for morphology QC
    morphology_max_beats: int = 30
    morphology_window_ms: float = 20.0    # amplitude computation window

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
    enabled_fpdc_physiol: bool = True    # ON by default (literature-based)

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
    # 'max'  : drug positive if any concentration > threshold
    # 'mean' : drug positive if mean across concentrations > threshold
    # 'n_above' : drug positive if ≥ n concentrations > threshold
    classification_method: str = 'max'
    classification_n_above: int = 2  # used when method = 'n_above'

    # Smart cessation override
    # When a drug causes cessation AND waveform destruction (low FPD confidence),
    # elevate the drug to positive even if FPDcF measurement failed.
    # This catches drugs like dofetilide that destroy waveform morphology.
    # Only triggers when min FPD confidence across concentrations < threshold,
    # preventing false positives for drugs with cessation at extreme doses
    # but good FPD data (e.g. ranolazine at 100µM).
    enable_cessation_override: bool = True
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
    ead_critical_count: int = 3             # ≥ 3 EADs → critical severity

    # EAD detection (residual — paper approach, Visone et al. 2023)
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

    # Premature beat classification threshold
    premature_count_threshold: int = 5

    # TdP scoring — restrict to severe events only
    # (EADs with critical severity + positive FPDcF trend, or cessation)
    tdp_require_severe_only: bool = True

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
    cv_excellent: float = 10.0    # CV < 10% → +40 score
    cv_good: float = 20.0        # CV < 20% → +30
    cv_fair: float = 35.0        # CV < 35% → +15
    rate_range_per_s: tuple[float, float] = (0.3, 3.5)
    snr_good: float = 5.0        # +20
    snr_fair: float = 3.0        # +10

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
    # For µECG-Pharma Digilent system the amplifier gain is 10⁴ (×10 000).
    # Set to 1.0 to skip correction (raw units preserved).
    amplifier_gain: float = 1.0

    # Advanced analysis modules (enabled by default)
    enable_cessation: bool = True
    enable_spectral: bool = True

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
          - 'default'       : standard parameters (tangent method, paper criteria)
          - 'conservative'  : stricter inclusion, higher confidence thresholds
          - 'sensitive'     : looser thresholds, catches more but more FP risk
          - 'peak_method'   : use peak instead of tangent (backwards compatibility)
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
