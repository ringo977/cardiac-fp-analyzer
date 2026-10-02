"""
parameters.py — Extraction of electrophysiological parameters from FP beats.

Key parameters:
  - Beat Period (BP) / RR interval
  - Signal Amplitude (Vmax)
  - FPD (Field Potential Duration) ~ QT
  - FPDc (Fridericia): FPDc = FPD / (RR)^(1/3)
  - FPDc (Bazett): FPDc = FPD / sqrt(RR)
  - Rise Time
  - Repolarization amplitude
  - STV (short-term variability)

FPD measurement strategy (inspired by Visone et al., Tox Sci 2023):
  1. Build an averaged beat template via cross-correlation alignment
  2. Detect the repolarization wave on the clean template → reference FPD
  3. Refine FPD per-beat using a template-guided search window

All parameters are configurable via RepolarizationConfig (see config.py).
"""

import logging

import numpy as np

from .repolarization import (
    _get_repol_cfg,
)
from .repolarization import (
    find_repolarization_on_template as _find_repolarization_on_template,
)
from .repolarization import (
    find_repolarization_per_beat as _find_repolarization_per_beat,
)

logger = logging.getLogger(__name__)


# ─── Rate-correction selection ───

#: Correction formulas understood by :func:`_select_corrected_fpd`.
#: Kept in sync with ``RepolarizationConfig.correction`` (config.py) and with
#: the ``--correction`` CLI choices in analyze.py.
VALID_CORRECTIONS = ('fridericia', 'bazett', 'none')


def _select_corrected_fpd(fpd_ms, fpdc_fridericia_ms, fpdc_bazett_ms,
                          correction):
    """Return the rate-corrected FPD selected by *correction*.

    Parameters
    ----------
    fpd_ms : float
        Uncorrected FPD in milliseconds (used when ``correction='none'``).
    fpdc_fridericia_ms, fpdc_bazett_ms : float
        Pre-computed corrected values, in milliseconds.
    correction : str
        One of :data:`VALID_CORRECTIONS`.

    Returns
    -------
    float
        The selected value.

    Notes
    -----
    An unrecognised *correction* falls back to Fridericia (the package
    default) and logs a warning rather than raising, so that a typo in a
    hand-edited sidecar degrades to the documented default instead of
    aborting a batch run. ``AnalysisConfig.from_dict`` does not validate
    this field, so a typo is reachable in practice.
    """
    if correction == 'fridericia':
        return fpdc_fridericia_ms
    if correction == 'bazett':
        return fpdc_bazett_ms
    if correction == 'none':
        return fpd_ms
    logger.warning(
        "Unknown correction %r (expected one of %s) — falling back to "
        "Fridericia.", correction, ', '.join(VALID_CORRECTIONS),
    )
    return fpdc_fridericia_ms


# ─── True local RR ───

#: Two genuine depolarizations cannot be closer than the tissue's
#: refractory period, so anything nearer than this is the *same* beat
#: seen at a slightly different index — which happens when
#: re-segmentation shifts a position by a few samples. Well below any
#: real RR, well above the jitter.
_SAME_BEAT_TOL_S = 0.05


def compute_local_rr(accepted_indices, all_indices, fs,
                     same_beat_tol_s=_SAME_BEAT_TOL_S):
    """Interval from each accepted beat to its *real* predecessor.

    Rate correction is a per-beat, physiological relationship:
    repolarization duration depends on the cycle that preceded it
    (restitution). The correct RR for beat *N* is therefore the time since
    the depolarization immediately before it — whatever the analysis later
    decided to do with that depolarization.

    QC rejects beats on spike amplitude and on template correlation, i.e.
    on *waveform shape*. A rejected beat still depolarized the tissue and
    still set the diastolic interval for the next one. Computing RR from
    the surviving beats alone therefore measures intervals that span the
    gaps and inflates them — up to 3.7x on real recordings, which deflates
    FPDc by up to 55% and inflates CV(BP) by up to 250 percentage points.

    Parameters
    ----------
    accepted_indices : sequence of int
        Sample positions of the beats being characterised.
    all_indices : sequence of int
        Sample positions of *every* detected beat, QC-rejected included.
        Must be sorted ascending.
    fs : float
        Sampling rate in Hz.
    same_beat_tol_s : float
        Raw beats closer than this to the accepted position are treated
        as the *same* beat rather than as its predecessor. Without it,
        a beat whose index moved by a few samples during re-segmentation
        would be rate-corrected against itself, yielding an RR of a few
        milliseconds and an absurdly large FPDc.

    Returns
    -------
    list of (float or None)
        One entry per accepted beat: seconds since its true predecessor,
        or ``None`` when there is none (the first beat of the recording,
        whose FPDc is undefined by construction).

    Notes
    -----
    Indexing trap this exists to avoid: ``beat_periods[i-1]`` walks the
    *accepted* list while a raw period array walks the *raw* list. Once
    anything is rejected the two stop lining up and each beat silently
    receives another beat's RR. The alignment must go through sample
    positions, never through list positions.
    """
    accepted = np.asarray(accepted_indices)
    allb = np.asarray(all_indices)
    if len(accepted) == 0:
        return []
    if len(allb) == 0:
        return [None] * len(accepted)

    # Index of the first raw beat at or after each accepted position.
    pos = np.searchsorted(allb, accepted, side='left')

    tol_samples = max(0.0, float(same_beat_tol_s) * fs)

    out = []
    for acc_idx, p in zip(accepted, pos):
        # Walk back to the last raw beat that is genuinely *before* this
        # one. Skip any raw beat at or after the accepted position, and any
        # within the same-beat tolerance — those are this beat itself, not
        # its predecessor.
        prev_pos = int(p) - 1
        while prev_pos >= 0 and (acc_idx - allb[prev_pos]) <= tol_samples:
            prev_pos -= 1
        if prev_pos < 0:
            out.append(None)          # no predecessor: first beat
        else:
            out.append(float(acc_idx - allb[prev_pos]) / fs)
    return out


# ─── FPD reliability gate (Sprint 3 #1, Fix C) ───

def apply_fpd_reliability_gate(summary, all_params, cfg):
    """Attach ``fpd_valid_ratio`` / ``fpd_reliable`` / ``fpd_note`` to
    ``summary`` based on the fraction of beats that produced a valid
    FPD.

    Motivation
    ----------
    On Exp6_chipD_ch1 only 1/7 beats produced a valid FPD (the rest
    had no detectable T-wave) yet the summary reported
    ``FPDcF 318.8 ± 0.0`` — a number derived from a single pathological
    artefact.  The gate adds an explicit flag so the UI / export
    layers can tell the user when the FPD mean is too sparse to be
    trusted.

    Parameters
    ----------
    summary : dict
        Summary dict to mutate — new keys ``fpd_valid_ratio``,
        ``fpd_reliable``, ``fpd_note`` are added.
    all_params : list of dict
        Per-beat parameter dicts containing ``fpd_ms`` (NaN when the
        beat's repolarisation was not detectable).
    cfg : RepolarizationConfig
        Must expose ``min_valid_fpd_ratio`` (default 0.50).
    """
    rc = _get_repol_cfg(cfg)
    min_valid_ratio = getattr(rc, 'min_valid_fpd_ratio', 0.50)

    n_total = len(all_params)
    n_no_repol = sum(1 for p in all_params if np.isnan(p.get('fpd_ms', np.nan)))

    if n_total > 0:
        valid_ratio = 1.0 - n_no_repol / n_total
    else:
        valid_ratio = 0.0

    summary['fpd_valid_ratio'] = float(valid_ratio)
    # A 0-beat recording cannot be reliable.  A non-positive threshold
    # disables the gate (always flag as reliable, pre-v3.3.1 behaviour).
    if n_total == 0:
        summary['fpd_reliable'] = False
    elif min_valid_ratio <= 0:
        summary['fpd_reliable'] = True
    else:
        summary['fpd_reliable'] = bool(valid_ratio >= min_valid_ratio)

    if not summary['fpd_reliable']:
        summary['fpd_note'] = (
            "FPD non misurabile in modo affidabile: "
            f"{int(round(valid_ratio * 100))}% dei battiti ha T-wave "
            f"rilevabile (soglia {int(round(min_valid_ratio * 100))}%). "
            "I valori medi di FPD / FPDcF vanno considerati indicativi "
            "e ispezionati sul singolo battito."
        )
    else:
        summary['fpd_note'] = None


# ─── Template averaging ───

def beat_spike_inverted(template_spike_shape, beat_spike, threshold=-0.5):
    """True if a beat's depolarisation is polarity-inverted w.r.t. the template.

    ``template_spike_shape`` : demeaned template samples around the spike
    (or None). ``beat_spike`` : the beat's samples over the same window.
    Inverted ⇔ Pearson correlation ≤ ``threshold`` (anti-correlated). Robust
    to biphasic spikes with lobes of similar size, where comparing the
    largest deflections flips on noise.
    """
    if template_spike_shape is None:
        return False
    b = np.asarray(beat_spike, dtype=float)
    m = min(len(b), len(template_spike_shape))
    if m < 3:
        return False
    w = b[:m] - b[:m].mean()
    k = np.asarray(template_spike_shape[:m], dtype=float)
    den = float(np.linalg.norm(w) * np.linalg.norm(k))
    if den <= 0:
        return False
    return float(np.dot(w, k)) / den <= threshold


def _align_beats_xcorr(beats_data, fs, cfg=None):
    """
    Align beats via cross-correlation to a reference beat.

    The reference is initially the median beat, then refined iteratively.
    Returns aligned beat waveforms (all same length).
    """
    rc = _get_repol_cfg(cfg)
    if len(beats_data) < 3:
        return beats_data

    max_shift = int(rc.alignment_max_shift_ms * fs / 1000)

    # Ensure uniform length
    min_len = min(len(b) for b in beats_data)
    beats = [b[:min_len].copy() for b in beats_data]

    # Reference: median of all beats (robust starting point)
    ref = np.median(np.array(beats), axis=0)

    # Align each beat to reference via cross-correlation.
    #
    # For every candidate shift s ∈ [−max_shift, +max_shift] compare the
    # reference depolarisation region ref[max_shift : max_shift + dep_len]
    # with beat[max_shift + s : max_shift + s + dep_len]; the best s is how
    # much the beat lags the reference, and the beat is moved back by s.
    #
    # Fixed Oct 2026. The previous code correlated ref[:dep_len+2M] with
    # beat[:dep_len], so zero lag landed at index 0 instead of M: an already
    # aligned beat came out shifted by exactly max_shift (50 ms), beats that
    # lagged the reference were pinned at the boundary, and beats that led
    # it had their offset doubled. Every template was therefore 50 ms late
    # relative to the beats it summarised, with smeared spikes — and the
    # per-beat inversion test, which reads the template's spike at
    # ``segment_pre_ms``, was reading baseline noise (see
    # extract_all_parameters).
    dep_len = int(rc.alignment_depol_region_ms / 1000 * fs)
    if dep_len + 2 * max_shift > min_len:
        dep_len = max(0, min_len - 2 * max_shift)
    if dep_len < 3:
        return beats
    ref_seg = ref[max_shift:max_shift + dep_len]
    aligned = []
    for beat in beats:
        span = beat[:dep_len + 2 * max_shift]
        corr = np.correlate(span, ref_seg, mode='valid')
        if len(corr) == 0:
            aligned.append(beat)
            continue
        # Normalise by the energy of each beat window so that a shift is not
        # preferred merely because its window captures more of the spike.
        energy = np.sqrt(np.convolve(span * span, np.ones(dep_len), mode='valid'))
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.where(energy > 0, corr / energy, -np.inf)
        shift = int(np.argmax(corr)) - max_shift
        shift = int(np.clip(shift, -max_shift, max_shift))

        if shift > 0:
            padded = np.concatenate([beat[shift:], np.full(shift, beat[-1])])
        elif shift < 0:
            padded = np.concatenate([np.full(-shift, beat[0]), beat[:shift]])
        else:
            padded = beat

        aligned.append(padded[:min_len])

    return aligned


def build_beat_template(beats_data, fs, cfg=None):
    """
    Build a clean averaged beat template.

    1. Select up to max_beats from the recording (evenly spaced)
    2. Align via cross-correlation
    3. Compute robust median template

    Returns: template (array), or None if too few beats.
    """
    rc = _get_repol_cfg(cfg)
    if len(beats_data) < 5:
        return None

    n = min(len(beats_data), rc.max_beats_template)
    if n < len(beats_data):
        indices = np.linspace(0, len(beats_data)-1, n, dtype=int)
        selected = [beats_data[i] for i in indices]
    else:
        selected = list(beats_data)

    aligned = _align_beats_xcorr(selected, fs, cfg=cfg)

    if not aligned:
        return None

    # Invariant: _align_beats_xcorr returns beats that all share the same
    # length as its input (it truncates to min_len internally and each
    # aligned beat is re-sliced to that length at its return site).  The
    # per-beat slice that used to appear here was therefore a no-op.
    # Kept as an assertion so the precondition is documented and
    # enforced (and so np.array() doesn't silently produce an object
    # array if the invariant ever breaks under a future refactor).
    beat_len = len(aligned[0])
    assert all(len(b) == beat_len for b in aligned), (
        "build_beat_template: _align_beats_xcorr must return "
        "equal-length beats; got lengths " +
        repr({len(b) for b in aligned})
    )
    mat = np.asarray(aligned)

    template = np.median(mat, axis=0)
    return template


# ─── Per-beat parameter extraction ───

def extract_beat_parameters(beat_data, beat_time, fs, rr_interval=None,
                           template_fpd_samples=None, template_peak_samples=None,
                           template_repol_sign=1, cfg=None,
                           beat_period_s=None):
    """
    Extract parameters from a single segmented beat.

    If template_fpd_samples is provided, uses template-guided FPD detection.
    """
    rc = _get_repol_cfg(cfg)
    params = {}
    data = np.array(beat_data, dtype=np.float64)
    t = np.array(beat_time, dtype=np.float64)
    zero_idx = np.argmin(np.abs(t))

    # Spike amplitude
    sp = max(0, zero_idx - int(rc.spike_pre_ms / 1000 * fs))
    ep = min(len(data), zero_idx + int(rc.spike_post_ms / 1000 * fs))
    spike_region = data[sp:ep]
    spike_max, spike_min = np.max(spike_region), np.min(spike_region)
    params['spike_amplitude_mV'] = (spike_max - spike_min) * 1000

    # Rise time (10-90%)
    spike_amp = spike_max - spike_min
    if spike_amp > 0:
        t10, t90 = spike_min + 0.1 * spike_amp, spike_min + 0.9 * spike_amp
        a10 = np.where(spike_region >= t10)[0]
        a90 = np.where(spike_region >= t90)[0]
        params['rise_time_ms'] = (max(0, (a90[0] - a10[0]) / fs * 1000)
                                  if len(a10) > 0 and len(a90) > 0 else np.nan)
    else:
        params['rise_time_ms'] = np.nan

    # FPD (template-guided, configured method)
    fpd, repol_amp, repol_peak_i, fpd_end_i = _find_repolarization_per_beat(
        data, t, zero_idx, fs,
        template_fpd_samples=template_fpd_samples,
        template_peak_samples=template_peak_samples,
        template_repol_sign=template_repol_sign,
        cfg=cfg,
        beat_period_s=beat_period_s
    )
    params['fpd_ms'] = fpd * 1000 if fpd is not None else np.nan
    params['repol_amplitude_mV'] = repol_amp * 1000 if not np.isnan(repol_amp) else np.nan
    params['repol_peak_idx_in_beat'] = repol_peak_i
    params['fpd_endpoint_idx_in_beat'] = fpd_end_i

    # ── Rate correction ────────────────────────────────────────────────
    # Both named formulas are ALWAYS computed and stored under their own
    # explicit keys (``fpdc_fridericia_ms`` / ``fpdc_bazett_ms``) so that a
    # consumer that needs a specific formula never has to guess which one
    # ``fpdc_ms`` currently holds.
    #
    # ``fpdc_ms`` is the *configured* correction — this is what ``rc.correction``
    # selects, and ``summary['correction']`` records which formula it is.
    #
    # History: before this fix ``fpdc_ms`` was hard-coded to Fridericia while
    # ``rc.correction`` was only stamped into the summary as a label, so
    # ``--correction bazett`` produced Fridericia values labelled "bazett".
    # The default is still 'fridericia', so default-config results are
    # bit-identical to the previous behaviour.
    if fpd is not None and rr_interval is not None and rr_interval > 0:
        fpdc_fridericia = (fpd / (rr_interval ** (1 / 3))) * 1000
        fpdc_bazett = (fpd / np.sqrt(rr_interval)) * 1000
        params['fpdc_fridericia_ms'] = fpdc_fridericia
        params['fpdc_bazett_ms'] = fpdc_bazett
        params['fpdc_ms'] = _select_corrected_fpd(
            fpd_ms=fpd * 1000,
            fpdc_fridericia_ms=fpdc_fridericia,
            fpdc_bazett_ms=fpdc_bazett,
            correction=rc.correction,
        )
    else:
        params['fpdc_fridericia_ms'] = np.nan
        params['fpdc_bazett_ms'] = np.nan
        params['fpdc_ms'] = np.nan

    # Max dV/dt
    deriv = np.gradient(data, 1.0 / fs)
    params['max_dvdt'] = np.max(np.abs(deriv[sp:ep]))

    return params


def extract_all_parameters(beats_data, beats_time, beat_indices, fs, cfg=None,
                           all_beat_indices=None):
    """
    Extract parameters for all beats and compute summary statistics.

    Uses template averaging for robust FPD measurement:
    1. Build averaged template from aligned beats
    2. Detect repolarization on template → reference FPD
    3. Guide per-beat FPD with template reference

    Parameters
    ----------
    beat_indices : sequence of int
        Sample positions of the beats to characterise (post-QC).
    cfg : RepolarizationConfig or None
    all_beat_indices : sequence of int or None
        Sample positions of *every* detected beat, QC-rejected included.

        When given, each beat's RR is measured against its real
        predecessor (see :func:`compute_local_rr`) and the beat-period
        summary describes the full detected rhythm. This is the correct
        behaviour: rate correction depends on the physical cycle length,
        and a QC-rejected beat still happened.

        When ``None`` the function falls back to deriving RR from
        ``beat_indices`` alone. That is the legacy path — it measures
        intervals that span QC gaps, inflating RR and CV and deflating
        FPDc — and is kept only so that callers which genuinely have no
        raw train (unit tests, ``recompute_from_beats`` on an edited set)
        keep working.
    """
    rc = _get_repol_cfg(cfg)
    from .beat_detection import compute_beat_periods

    # Alignment guard: beats_data, beats_time and beat_indices must be in sync
    if not (len(beats_data) == len(beats_time) == len(beat_indices)):
        raise ValueError(
            f"Parameter extraction alignment error: beats_data={len(beats_data)}, "
            f"beats_time={len(beats_time)}, beat_indices={len(beat_indices)}"
        )

    # Guard against empty input
    if len(beats_data) == 0:
        logger.warning("extract_all_parameters: no beats to process")
        return [], {}

    # ─── Beat periods and per-beat RR ───
    # Two distinct quantities that the legacy code conflated:
    #   * ``beat_periods``  — the rhythm of the preparation, used for the
    #     summary statistics and for the adaptive FPD window. Must describe
    #     every detected beat, otherwise QC rejections masquerade as
    #     bradycardia and as rhythm irregularity.
    #   * ``local_rr[i]``   — the cycle length preceding accepted beat *i*,
    #     used to rate-correct that beat's FPD.
    if all_beat_indices is not None and len(all_beat_indices) > 0:
        beat_periods = compute_beat_periods(all_beat_indices, fs)
        local_rr = compute_local_rr(beat_indices, all_beat_indices, fs)
    else:
        # Legacy fallback — see the ``all_beat_indices`` docstring.
        beat_periods = compute_beat_periods(beat_indices, fs)
        local_rr = [
            beat_periods[i - 1] if 0 < i <= len(beat_periods) else None
            for i in range(len(beat_indices))
        ]

    # ─── Template averaging for FPD reference ───
    template = build_beat_template(beats_data, fs, cfg=cfg)
    template_fpd_samples = None
    template_repol_sign = 1

    repol_confidence = 0.0
    template_peak_samples = None

    consensus_info = None
    # Compute median beat period for adaptive min FPD
    median_bp_s = None
    if len(beat_periods) > 0:
        median_bp_s = float(np.median(beat_periods))

    # ─── Minimal signal amplitude gate ───
    # If median spike amplitude is below threshold, skip FPD detection entirely.
    min_amp_uV = getattr(rc, 'min_signal_amplitude_uV', 0.0)
    signal_too_weak = False
    if min_amp_uV > 0 and len(beats_data) > 0:
        pre_samples = int(rc.segment_pre_ms / 1000 * fs)
        spike_amps = []
        for bd in beats_data:
            sp = max(0, pre_samples - int(rc.spike_pre_ms / 1000 * fs))
            ep = min(len(bd), pre_samples + int(rc.spike_post_ms / 1000 * fs))
            if ep > sp:
                spike_region = bd[sp:ep]
                spike_amps.append((np.max(spike_region) - np.min(spike_region)) * 1e6)  # V → µV
        if spike_amps:
            median_amp_uV = float(np.median(spike_amps))
            if median_amp_uV < min_amp_uV:
                signal_too_weak = True
                logger.info("Signal amplitude gate: median %.1f µV < %.1f µV → "
                            "FPD analysis skipped (signal too weak)",
                            median_amp_uV, min_amp_uV)

    # ─── Rhythm for the template search window ───
    # See RepolarizationConfig.window_rr_from_template_beats. The window
    # must reach the T-wave of the beats that FORM the template; on
    # over-detected recordings the full-train median RR is ~half the true
    # cycle and the window closed before the T-wave. Use the longer of the
    # two estimates and let the template search guard the extension.
    window_bp_s = median_bp_s
    window_guard_after_ms = None
    if getattr(rc, 'window_rr_from_template_beats', False) and len(beat_indices) >= 3:
        kept_bp = np.diff(np.asarray(beat_indices, dtype=float)) / fs
        kept_bp = kept_bp[kept_bp > 0]
        if len(kept_bp) > 0:
            kept_med = float(np.median(kept_bp))
            if median_bp_s is None or kept_med > median_bp_s:
                old_end_ms = rc.search_end_ms
                if getattr(rc, 'search_end_pct_rr', 0.0) > 0 and median_bp_s:
                    old_end_ms = max(old_end_ms, rc.search_end_pct_rr * median_bp_s * 1000)
                window_guard_after_ms = old_end_ms
                window_bp_s = kept_med

    if template is not None and not signal_too_weak:
        pre_ms = rc.segment_pre_ms
        fpd_result = _find_repolarization_on_template(
            template, fs, pre_ms=pre_ms, cfg=cfg,
            median_bp_s=window_bp_s, guard_after_ms=window_guard_after_ms)
        if fpd_result[0] is not None:
            template_fpd_samples = fpd_result[0]
            template_repol_sign = fpd_result[1]
            repol_confidence = fpd_result[3] if len(fpd_result) > 3 else 0.5
            template_peak_samples = fpd_result[4] if len(fpd_result) > 4 else None
            consensus_info = fpd_result[5] if len(fpd_result) > 5 else None
            _template_fpd_ms = template_fpd_samples / fs * 1000  # noqa: F841

    # ─── Per-beat extraction ───
    all_params = []
    fpd_vals, fpdc_vals, fpdc_bazett_vals, amp_vals = [], [], [], []
    # Explicit per-formula accumulator, kept alongside ``fpdc_vals`` (which
    # holds whichever formula ``rc.correction`` selected) so that consumers
    # needing a named formula — e.g. the CDISC ``FPDCF`` test code — can read
    # it without inspecting the config.
    fpdc_fridericia_vals = []
    rise_time_vals, rr_interval_vals = [], []
    pre_samples = int(rc.segment_pre_ms / 1000 * fs)

    # Template spike shape for per-beat inversion detection.
    # A beat whose depolarisation is inverted relative to the template must
    # have its repolarisation sign flipped. "Inverted" = the beat's spike
    # window is ANTI-correlated with the template's (corr ≤
    # ``inversion_corr_threshold``). Until Oct 2026 the test compared which
    # deflection (max or min) was larger; on biphasic spikes of similar
    # lobes that verdict flips on noise, and — with the template 50 ms late
    # because of the alignment bug fixed in _align_beats_xcorr — the
    # template window held baseline, so in ~40 % of recordings most beats
    # were declared inverted and lost the template's guidance.
    t_spike_shape = None
    if template is not None:
        t_pre = int(rc.segment_pre_ms / 1000 * fs)
        t_sp = max(0, t_pre - int(rc.spike_pre_ms / 1000 * fs))
        t_ep = min(len(template), t_pre + int(rc.spike_post_ms / 1000 * fs))
        t_spike = np.asarray(template[t_sp:t_ep], dtype=float)
        if len(t_spike) > 2 and np.ptp(t_spike) > 0:
            t_spike_shape = t_spike - t_spike.mean()
    inversion_thr = float(getattr(rc, 'inversion_corr_threshold', -0.5))

    for i, (bd, bt) in enumerate(zip(beats_data, beats_time)):
        rr = local_rr[i] if i < len(local_rr) else None
        # Use per-beat RR for adaptive min FPD; fall back to median
        bp_for_beat = rr if rr is not None else median_bp_s

        # Per-beat polarity detection: check if this beat's spike is inverted
        # relative to the template.  If so, flip the repol sign and drop the
        # template guidance (peak position) since the morphology differs.
        beat_repol_sign = template_repol_sign
        beat_tpl_fpd = template_fpd_samples
        beat_tpl_peak = template_peak_samples
        zero_i = np.argmin(np.abs(np.array(bt)))
        b_sp = max(0, zero_i - int(rc.spike_pre_ms / 1000 * fs))
        b_ep = min(len(bd), zero_i + int(rc.spike_post_ms / 1000 * fs))
        beat_inverted = beat_spike_inverted(t_spike_shape, bd[b_sp:b_ep], inversion_thr)
        if beat_inverted:
            # Inverted beat: flip repol sign but KEEP template FPD
            # timing as a guide — the repolarization timing is similar
            # even when the morphology is inverted.  Drop peak_samples
            # (exact peak position depends on morphology) but keep
            # fpd_samples (approximate timing window).
            beat_repol_sign = -template_repol_sign
            beat_tpl_peak = None
            # beat_tpl_fpd stays as template_fpd_samples

        params = extract_beat_parameters(
            bd, bt, fs, rr_interval=rr,
            template_fpd_samples=beat_tpl_fpd,
            template_peak_samples=beat_tpl_peak,
            template_repol_sign=beat_repol_sign,
            cfg=cfg,
            beat_period_s=bp_for_beat
        )
        params['beat_number'] = i + 1
        params['rr_interval_ms'] = rr * 1000 if rr is not None else np.nan
        bi_g = beat_indices[i]
        rp = params.get('repol_peak_idx_in_beat')
        fe = params.get('fpd_endpoint_idx_in_beat')
        if rp is not None:
            params['repol_peak_global_idx'] = int(bi_g - pre_samples + rp)
        else:
            params['repol_peak_global_idx'] = None
        if fe is not None:
            params['fpd_endpoint_global_idx'] = int(bi_g - pre_samples + fe)
        else:
            params['fpd_endpoint_global_idx'] = None
        all_params.append(params)
        if not np.isnan(params['fpd_ms']):
            fpd_vals.append(params['fpd_ms'])
        if not np.isnan(params['fpdc_ms']):
            fpdc_vals.append(params['fpdc_ms'])
        if not np.isnan(params.get('fpdc_bazett_ms', np.nan)):
            fpdc_bazett_vals.append(params['fpdc_bazett_ms'])
        if not np.isnan(params.get('fpdc_fridericia_ms', np.nan)):
            fpdc_fridericia_vals.append(params['fpdc_fridericia_ms'])
        amp_vals.append(params['spike_amplitude_mV'])
        if not np.isnan(params.get('rise_time_ms', np.nan)):
            rise_time_vals.append(params['rise_time_ms'])
        if not np.isnan(params.get('rr_interval_ms', np.nan)):
            rr_interval_vals.append(params['rr_interval_ms'])

    # ── Repolarization diagnostic trace ──
    n_repol_ok = sum(1 for p in all_params if not np.isnan(p['fpd_ms']))
    n_repol_fail = sum(1 for p in all_params if np.isnan(p['fpd_ms']))
    fpd_measured = [p['fpd_ms'] for p in all_params if not np.isnan(p['fpd_ms'])]
    fpd_stats = ""
    if fpd_measured:
        fpd_arr_diag = np.array(fpd_measured)
        fpd_stats = (f", FPD: median={np.median(fpd_arr_diag):.0f}ms "
                     f"range={np.min(fpd_arr_diag):.0f}-{np.max(fpd_arr_diag):.0f}ms")
    print(f"     Repol: {n_repol_ok}/{len(all_params)} detected, "
          f"{n_repol_fail} not detectable"
          f" (template FPD={'%.0f ms' % (template_fpd_samples / fs * 1000) if template_fpd_samples else 'None'}"
          f"{fpd_stats})")

    # ─── Summary statistics ───
    summary = {}
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])
    for name, vals in [('beat_period_ms', bp_ms), ('spike_amplitude_mV', amp_vals),
                       ('fpd_ms', fpd_vals), ('fpdc_ms', fpdc_vals),
                       ('fpdc_bazett_ms', fpdc_bazett_vals),
                       ('fpdc_fridericia_ms', fpdc_fridericia_vals),
                       ('rise_time_ms', rise_time_vals),
                       ('rr_interval_ms', rr_interval_vals)]:
        v = np.array(vals)
        v = v[~np.isnan(v)]
        if len(v) > 0:
            summary[f'{name}_mean'] = np.mean(v)
            summary[f'{name}_std'] = np.std(v)
            summary[f'{name}_median'] = np.median(v)
            summary[f'{name}_cv'] = np.std(v) / np.mean(v) * 100 if np.mean(v) != 0 else np.nan
            summary[f'{name}_min'] = np.min(v)
            summary[f'{name}_max'] = np.max(v)
            summary[f'{name}_n'] = len(v)
        else:
            for s in ['_mean', '_std', '_median', '_cv', '_min', '_max', '_n']:
                summary[f'{name}{s}'] = np.nan

    summary['bpm_mean'] = 60000 / np.mean(bp_ms) if len(bp_ms) > 0 and np.mean(bp_ms) > 0 else np.nan
    summary['stv_ms'] = np.mean(np.abs(np.diff(bp_ms))) / np.sqrt(2) if len(bp_ms) > 1 else np.nan
    summary['beat_periods'] = beat_periods
    summary['fpd_values'] = np.array([p['fpd_ms'] / 1000 for p in all_params if not np.isnan(p['fpd_ms'])])
    summary['fpdc_values'] = np.array([p['fpdc_ms'] / 1000 for p in all_params if not np.isnan(p['fpdc_ms'])])

    # ─── Repolarization detectability statistics ───
    # Count beats where FPD could not be measured (repolarization not detectable).
    # This is a clinically meaningful parameter: a drug that abolishes visible
    # repolarization is high-risk for proarrhythmic effects.
    n_total_beats = len(all_params)
    n_no_repol = sum(1 for p in all_params if np.isnan(p['fpd_ms']))
    summary['n_beats_no_repol'] = n_no_repol
    summary['pct_beats_no_repol'] = (n_no_repol / n_total_beats * 100
                                      if n_total_beats > 0 else 0.0)

    # ─── FPD reliability gate (Sprint 3 #1, Fix C) ───
    apply_fpd_reliability_gate(summary, all_params, rc)

    # ─── FPD confidence score ───
    fpd_arr = np.array(fpd_vals)
    fpd_arr = fpd_arr[~np.isnan(fpd_arr)]
    if len(fpd_arr) > 3 and np.mean(fpd_arr) > 0:
        fpd_cv = np.std(fpd_arr) / np.mean(fpd_arr)
        consistency_conf = max(0, 1.0 - fpd_cv / rc.fpd_cv_max_for_confidence)
    else:
        consistency_conf = 0.0

    summary['fpd_confidence'] = (rc.fpd_conf_weight_template * repol_confidence +
                                  rc.fpd_conf_weight_consistency * consistency_conf)

    # Add template info and config to summary
    if template_fpd_samples is not None:
        summary['template_fpd_ms'] = template_fpd_samples / fs * 1000
        summary['template_repol_sign'] = int(template_repol_sign)
    summary['fpd_method'] = rc.fpd_method
    summary['repol_candidate_rule'] = getattr(rc, 'repol_candidate_rule', 'max_prominence')
    summary['correction'] = rc.correction
    # Rhythm actually used for the template search window (diagnostic).
    if window_bp_s is not None:
        summary['repol_window_rr_ms'] = window_bp_s * 1000
        summary['repol_window_extended'] = window_guard_after_ms is not None

    # Multi-method consensus info
    if consensus_info is not None:
        summary['consensus'] = consensus_info

    return all_params, summary
