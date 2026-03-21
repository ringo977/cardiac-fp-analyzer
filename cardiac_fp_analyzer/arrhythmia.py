"""
arrhythmia.py — Arrhythmia detection and classification for hiPSC-CM FP.

Detects: irregular beating, brady/tachycardia, premature/delayed beats,
beat cessation, EAD-like events, triggered activity, fibrillation-like.
"""

import numpy as np

TACHYCARDIA_BP_MS = 300
BRADYCARDIA_BP_MS = 2500
RR_IRREGULARITY_CV = 15.0
PREMATURE_BEAT_FACTOR = 0.7
DELAYED_BEAT_FACTOR = 1.5
CESSATION_FACTOR = 3.0
STV_HIGH_RISK_MS = 10.0
FPD_PROLONGATION_THRESHOLD = 1.3


class ArrhythmiaReport:
    def __init__(self):
        self.flags = []
        self.events = []
        self.classification = 'Normal Sinus Rhythm'
        self.risk_score = 0
        self.details = {}

    def add_flag(self, flag_type, severity, description):
        self.flags.append({'type': flag_type, 'severity': severity, 'description': description})
        if severity == 'critical': self.risk_score = min(100, self.risk_score + 30)
        elif severity == 'warning': self.risk_score = min(100, self.risk_score + 15)
        else: self.risk_score = min(100, self.risk_score + 5)

    def add_event(self, beat_number, event_type, details=''):
        self.events.append({'beat': beat_number, 'type': event_type, 'details': details})

    def summary_text(self):
        lines = [f"Classification: {self.classification}", f"Risk Score: {self.risk_score}/100"]
        for f in self.flags:
            lines.append(f"  [{f['severity'].upper()}] {f['type']}: {f['description']}")
        return '\n'.join(lines)


def analyze_arrhythmia(beat_indices, beat_periods, all_params, summary, fs, baseline_summary=None, cfg=None):
    """Analyze arrhythmia. cfg = ArrhythmiaConfig or None (uses module defaults)."""
    if cfg is None:
        from .config import ArrhythmiaConfig
        cfg = ArrhythmiaConfig()

    report = ArrhythmiaReport()
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])

    if len(bp_ms) < 3:
        report.add_flag('insufficient_data', 'warning', f'Only {len(bp_ms)} beat periods')
        report.classification = 'Insufficient Data'
        return report

    mean_bp, std_bp = np.mean(bp_ms), np.std(bp_ms)
    cv_bp = std_bp / mean_bp * 100 if mean_bp > 0 else 0
    report.details.update({'mean_bp_ms': mean_bp, 'cv_bp_pct': cv_bp, 'n_beats': len(beat_indices),
                           'bpm': 60000/mean_bp if mean_bp > 0 else 0})

    if mean_bp < cfg.tachycardia_bp_ms:
        report.add_flag('tachycardia', 'warning', f'Mean BP = {mean_bp:.0f} ms ({60000/mean_bp:.0f} BPM)')
    elif mean_bp > cfg.bradycardia_bp_ms:
        report.add_flag('bradycardia', 'warning', f'Mean BP = {mean_bp:.0f} ms ({60000/mean_bp:.0f} BPM)')

    if cv_bp > cfg.rr_irregularity_cv:
        sev = 'critical' if cv_bp > cfg.rr_critical_cv else 'warning'
        report.add_flag('rr_irregular', sev, f'RR CV = {cv_bp:.1f}%')

    n_prem, n_del = 0, 0
    for i, bp in enumerate(bp_ms):
        if bp < mean_bp * cfg.premature_beat_factor:
            n_prem += 1
            report.add_event(i+1, 'premature_beat', f'BP={bp:.0f}ms')
        elif bp > mean_bp * cfg.delayed_beat_factor:
            n_del += 1
            report.add_event(i+1, 'delayed_beat', f'BP={bp:.0f}ms')

    if n_prem > 0:
        pct = n_prem/len(bp_ms)*100
        report.add_flag('premature_beats', 'critical' if pct>10 else 'warning', f'{n_prem} premature ({pct:.1f}%)')
    if n_del > 0:
        pct = n_del/len(bp_ms)*100
        report.add_flag('delayed_beats', 'warning' if pct>5 else 'info', f'{n_del} delayed ({pct:.1f}%)')

    pauses = bp_ms[bp_ms > mean_bp * cfg.cessation_factor]
    if len(pauses) > 0:
        report.add_flag('beat_cessation', 'critical', f'{len(pauses)} pause(s), max={np.max(pauses):.0f}ms')

    stv = summary.get('stv_ms', np.nan)
    if not np.isnan(stv) and stv > cfg.stv_high_risk_ms:
        report.add_flag('high_stv', 'warning', f'STV = {stv:.1f} ms')

    fpd_vals = [p['fpd_ms'] for p in all_params if not np.isnan(p.get('fpd_ms', np.nan))]
    if fpd_vals:
        mean_fpd = np.mean(fpd_vals)
        if baseline_summary and 'fpd_ms_mean' in baseline_summary:
            ratio = mean_fpd / baseline_summary['fpd_ms_mean']
            if ratio > cfg.fpd_prolongation_threshold:
                report.add_flag('fpd_prolongation', 'critical', f'FPD {ratio:.0%} of baseline')
        if mean_fpd > cfg.fpd_critical_length_ms:
            report.add_flag('fpd_very_long', 'critical', f'FPD = {mean_fpd:.0f} ms')

    # EAD detection
    n_ead = 0
    if len(fpd_vals) > 5:
        med_fpd = np.median(fpd_vals)
        mad_fpd = np.median(np.abs(np.array(fpd_vals) - med_fpd))
        for i, p in enumerate(all_params):
            fpd = p.get('fpd_ms', np.nan)
            if not np.isnan(fpd) and mad_fpd > 0 and (fpd - med_fpd) > cfg.ead_mad_factor * mad_fpd * 1.4826:
                n_ead += 1
                report.add_event(i+1, 'ead_suspect', f'FPD={fpd:.0f}ms')
        if n_ead > 0:
            report.add_flag('ead_events',
                          'critical' if n_ead > cfg.ead_critical_count else 'warning',
                          f'{n_ead} EAD-like event(s)')

    # Amplitude instability
    amps = [p['spike_amplitude_mV'] for p in all_params if not np.isnan(p.get('spike_amplitude_mV', np.nan))]
    if len(amps) > 5:
        amp_cv = np.std(amps) / np.mean(amps) * 100
        if amp_cv > cfg.amplitude_instability_cv:
            report.add_flag('amplitude_instability', 'warning', f'Amplitude CV = {amp_cv:.1f}%')

    # Classification
    crit = [f for f in report.flags if f['severity'] == 'critical']
    if len(pauses) > 0 and cv_bp > cfg.fibrillation_cv:
        report.classification = 'Fibrillation-like / Chaotic Rhythm'
    elif n_ead > cfg.ead_critical_count:
        report.classification = 'EAD with Triggered Activity'
    elif n_ead > 0 and crit:
        report.classification = 'Proarrhythmic (EAD-prone)'
    elif len(pauses) > 0:
        report.classification = 'Intermittent Cessation'
    elif cv_bp > cfg.rr_critical_cv:
        report.classification = 'Highly Irregular Rhythm'
    elif n_prem > cfg.premature_count_threshold:
        report.classification = 'Frequent Premature Beats'
    elif cv_bp > cfg.rr_irregularity_cv:
        report.classification = 'Irregular Rhythm'
    elif mean_bp < cfg.tachycardia_bp_ms:
        report.classification = 'Tachycardia'
    elif mean_bp > cfg.bradycardia_bp_ms:
        report.classification = 'Bradycardia'
    elif [f for f in report.flags if f['severity'] == 'warning']:
        report.classification = 'Borderline / Mild Abnormalities'

    return report
