"""
template_quality.py — Is the averaged beat template representative?

Policy shared by every UI (Streamlit ``ui/display.py`` and PySide
``pyside_app/main.py``). Until Oct 2026 the two constants below lived as
identical copies in both modules, with the predicate re-implemented in
each; changing one silently made the two UIs disagree on signal quality.
"""

# Rhythm types that make a single averaged template a poor summary of the
# underlying beats. ``chaotic`` / ``ambiguous`` already carry that meaning;
# ``alternans_2_to_1`` and ``trimodal`` have multiple morphological families
# that should not be collapsed into one template.
TEMPLATE_RISKY_RHYTHM_TYPES = frozenset({
    'chaotic',
    'ambiguous',
    'alternans_2_to_1',
    'trimodal',
})

# FPD-dispersion threshold above which the T-waves of individual beats fall
# at wildly different offsets relative to the depolarisation spike, so
# point-wise aggregation (mean *or* median) cancels them. 20 % is a
# conservative cut: below it, visual T-wave alignment in the overlay is
# preserved on the real-signal fixtures tested so far.
FPD_CV_TEMPLATE_WARN = 0.20


def template_representativity(result):
    """Assess whether the mean template of ``result`` can be trusted.

    Returns
    -------
    dict with keys
        representative : bool — False if either trigger fired
        risky_rhythm   : bool — rhythm type in TEMPLATE_RISKY_RHYTHM_TYPES
        dispersive_fpd : bool — FPD CV above FPD_CV_TEMPLATE_WARN
        rhythm_type    : str
        fpd_cv         : float (fraction, not percent)
    """
    if result is None:
        return {'representative': True, 'risky_rhythm': False,
                'dispersive_fpd': False, 'rhythm_type': '', 'fpd_cv': 0.0}
    rc = (result.get('detection_info') or {}).get('rhythm_classification') or {}
    rhythm_type = str(rc.get('rhythm_type') or '')
    summary = result.get('summary') or {}
    try:
        fpd_mean = float(summary.get('fpd_ms_mean', 0) or 0.0)
        fpd_std = float(summary.get('fpd_ms_std', 0) or 0.0)
    except (TypeError, ValueError):
        fpd_mean, fpd_std = 0.0, 0.0
    fpd_cv = (fpd_std / fpd_mean) if fpd_mean > 0 else 0.0

    risky = rhythm_type in TEMPLATE_RISKY_RHYTHM_TYPES
    dispersive = fpd_cv > FPD_CV_TEMPLATE_WARN
    return {'representative': not (risky or dispersive),
            'risky_rhythm': risky, 'dispersive_fpd': dispersive,
            'rhythm_type': rhythm_type, 'fpd_cv': fpd_cv}
