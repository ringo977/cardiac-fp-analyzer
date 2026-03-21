"""
risk_map.py — CiPA-style 2D risk map for drug proarrhythmic classification.

Generates a scatter plot placing each drug on two axes:
  X-axis : max ΔFPDcF (%) — repolarisation prolongation
  Y-axis : proarrhythmic index (0-100) — composite of beat irregularity,
           cessation, and spectral morphology change

The plot is divided into three risk zones (Low / Intermediate / High)
following the CiPA framework philosophy (Blinova et al. 2017, Strauss
et al. 2019).

Usage
-----
    from cardiac_fp_analyzer.risk_map import generate_risk_map

    # results_list comes from batch_analyze()
    fig = generate_risk_map(results_list, config=config)
    fig.savefig('risk_map.png', dpi=150)

    # Or pass ground-truth for colouring by known class:
    gt = {'terfenadine': True, 'quinidine': True, ...}
    fig = generate_risk_map(results_list, config=config, ground_truth=gt)
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe


# ── Drug-name normalisation (same aliases used across the pipeline) ────

_DRUG_ALIASES: Dict[str, str] = {
    'terfe': 'terfenadine',
    'quinidine': 'quinidine',
    'dofe': 'dofetilide', 'dofetilide': 'dofetilide',
    'alfus': 'alfuzosin', 'alfuso': 'alfuzosin', 'alfu': 'alfuzosin',
    'mexi': 'mexiletine', 'mexil': 'mexiletine', 'mexilitine': 'mexiletine',
    'nifedipine': 'nifedipine', 'nife': 'nifedipine',
    'ranolazine': 'ranolazine',
}


def _canonical_drug(raw: str) -> str:
    """Map raw drug name from filename to canonical name."""
    raw_l = raw.lower().strip()
    if raw_l in _DRUG_ALIASES:
        return _DRUG_ALIASES[raw_l]
    for prefix, canon in _DRUG_ALIASES.items():
        if raw_l.startswith(prefix):
            return canon
    return raw_l


# ── Per-drug metric aggregation ────────────────────────────────────────

@dataclass
class DrugRiskMetrics:
    """Aggregated risk metrics for a single drug."""
    name: str
    # FPDcF prolongation
    max_pct_fpdc_change: float = np.nan      # max ΔFPDcF (%)
    mean_pct_fpdc_change: float = np.nan
    # Beat irregularity (CV of beat period, %)
    max_bp_cv: float = 0.0
    mean_bp_cv: float = 0.0
    # Cessation
    max_cessation_conf: float = 0.0
    has_cessation: bool = False
    # Spectral morphology change (0-1)
    max_spectral_change: float = 0.0
    # Residual-based arrhythmia metrics (paper approach)
    max_morphology_instability: float = 0.0   # 0-1 from residual RMS (all)
    mean_morphology_instability: float = 0.0
    max_morph_inst_bl: float = 0.0           # 0-1 baseline-relative only
    has_bl_morph: bool = False               # True if any BL-relative morph data
    max_ead_incidence_pct: float = 0.0        # % beats with EAD
    max_poincare_stv_fpdc: float = 0.0        # Poincaré STV of FPDcF (ms)
    max_risk_score: int = 0                   # arrhythmia risk_score 0-100
    # FPD reliability
    min_fpd_confidence: float = 1.0
    # Counts
    n_concentrations: int = 0
    n_with_normalization: int = 0
    # Classification
    cessation_override: bool = False


def aggregate_drug_metrics(results_list) -> Dict[str, DrugRiskMetrics]:
    """
    Walk the full results list and aggregate risk-relevant metrics per
    canonical drug name.
    """
    from .normalization import _is_baseline, _is_control

    buckets = defaultdict(lambda: {
        'pct': [], 'bp_cv': [], 'cess_conf': [], 'spec': [], 'fpd_conf': [],
        'has_cess': False,
        'morph_inst': [], 'morph_inst_bl': [],  # bl = baseline-relative only
        'ead_pct': [], 'stv_fpdc': [], 'risk_scores': [],
    })

    for r in results_list:
        if _is_baseline(r) or _is_control(r):
            continue
        fi = r.get('file_info', {})
        drug_raw = str(fi.get('drug', '') or '').lower()
        if not drug_raw:
            continue
        drug = _canonical_drug(drug_raw)
        b = buckets[drug]

        s = r.get('summary', {})
        norm = r.get('normalization', {})
        cess = r.get('cessation_report')
        ar = r.get('arrhythmia_report')

        # ΔFPDcF
        pct = norm.get('pct_fpdc_change', np.nan)
        if not np.isnan(pct):
            b['pct'].append(pct)

        # Beat period irregularity
        bp_cv = s.get('beat_period_ms_cv', np.nan)
        if not np.isnan(bp_cv):
            b['bp_cv'].append(bp_cv)

        # Cessation
        if cess is not None:
            b['cess_conf'].append(cess.cessation_confidence)
            if cess.has_cessation:
                b['has_cess'] = True

        # Spectral change
        spec = norm.get('spectral_change_score', np.nan)
        if not np.isnan(spec):
            b['spec'].append(spec)

        # FPD confidence
        fpd_c = s.get('fpd_confidence', np.nan)
        if not np.isnan(fpd_c):
            b['fpd_conf'].append(fpd_c)

        # Residual-based arrhythmia metrics
        if ar is not None:
            b['risk_scores'].append(ar.risk_score)
            det = ar.details if hasattr(ar, 'details') else {}
            rd = ar.residual_details if hasattr(ar, 'residual_details') else {}
            is_bl_rel = rd.get('baseline_relative', False) if rd else False
            mi = det.get('morphology_instability', 0)
            if mi > 0:
                b['morph_inst'].append(mi)
                if is_bl_rel:
                    b['morph_inst_bl'].append(mi)
            ep = det.get('ead_incidence_pct', 0)
            if ep > 0:
                b['ead_pct'].append(ep)
            sv = det.get('poincare_stv_fpdc_ms', np.nan)
            if not np.isnan(sv):
                b['stv_fpdc'].append(sv)

    # Build DrugRiskMetrics
    out = {}
    for drug, b in buckets.items():
        m = DrugRiskMetrics(name=drug)
        if b['pct']:
            m.max_pct_fpdc_change = max(b['pct'])
            m.mean_pct_fpdc_change = float(np.mean(b['pct']))
            m.n_with_normalization = len(b['pct'])
        if b['bp_cv']:
            m.max_bp_cv = max(b['bp_cv'])
            m.mean_bp_cv = float(np.mean(b['bp_cv']))
        m.max_cessation_conf = max(b['cess_conf']) if b['cess_conf'] else 0.0
        m.has_cessation = b['has_cess']
        if b['spec']:
            m.max_spectral_change = max(b['spec'])
        if b['morph_inst']:
            m.max_morphology_instability = max(b['morph_inst'])
            m.mean_morphology_instability = float(np.mean(b['morph_inst']))
        if b['morph_inst_bl']:
            m.max_morph_inst_bl = max(b['morph_inst_bl'])
            m.has_bl_morph = True
        if b['ead_pct']:
            m.max_ead_incidence_pct = max(b['ead_pct'])
        if b['stv_fpdc']:
            m.max_poincare_stv_fpdc = max(b['stv_fpdc'])
        if b['risk_scores']:
            m.max_risk_score = max(b['risk_scores'])
        if b['fpd_conf']:
            m.min_fpd_confidence = min(b['fpd_conf'])
        m.n_concentrations = len(b['bp_cv'])
        out[drug] = m
    return out


# ── Proarrhythmic index ────────────────────────────────────────────────

def compute_proarrhythmic_index(m: DrugRiskMetrics,
                                 w_spec: float = 0.70,
                                 w_morph: float = 0.25,
                                 w_ead: float = 0.05) -> float:
    """
    Composite proarrhythmic index (0-100).

    Three-component index (v3.3), weighted by **specificity** for
    hERG-related proarrhythmic risk:

      - **Spectral change** (70%) : frequency-domain morphology score —
        the single best discriminator between positive and negative drugs
        on the 7-drug validation set.  hERG blockers alter the
        repolarisation waveform shape (T-wave broadening, secondary
        humps, U-waves), which manifests as spectral differences vs
        baseline.  Validation: pos mean=0.654, neg mean=0.214 — 3× ratio.

      - **Morphology instability** (25%) : residual-based morphological
        instability (0-1) from the baseline-relative residual analysis.
        With baseline templates (v3.3), this metric is now discriminatory:
        pos mean=0.581, neg mean=0.296 — 2.0× ratio.  hERG blockers
        cause progressive beat-to-beat morphology shifts (AP prolongation,
        EAD-like bumps) that differ from the baseline template, while
        non-proarrhythmic drugs produce smaller deviations.

      - **EAD incidence** (5%) : % beats with residual-based EAD
        detection.  Small weight; primarily contributes for pure hERG
        blockers (dofetilide, quinidine) at high concentrations.

    Components still excluded (anti-discriminatory even with baseline-
    relative analysis):
      - Cessation : ranolazine(−) 0.77 > dofetilide(+) 0.36.
      - Waveform degradation : nifedipine(−) FPDc=0 ≈ dofetilide(+) 0.
      - Repol. STV : nifedipine(−) 72ms ≈ dofetilide(+) 65ms.
    """
    # Spectral morphology change (0-1 → 0-100)
    spec = m.max_spectral_change * 100.0

    # Morphology instability (0-1 → 0-100)
    # Use baseline-relative value if available; intra-recording values
    # are anti-discriminatory (nifedipine− 0.915 > terfenadine+ 0.262)
    # so we only include morph when we have a reliable baseline reference.
    if m.has_bl_morph:
        morph = m.max_morph_inst_bl * 100.0
    else:
        morph = 0.0  # no baseline → don't contribute anti-discriminatory noise

    # EAD incidence: cap at 30% for normalisation
    ead = min(m.max_ead_incidence_pct, 30.0) * (100.0 / 30.0)

    idx = w_spec * spec + w_morph * morph + w_ead * ead
    return min(idx, 100.0)


# ── Risk zone thresholds ───────────────────────────────────────────────

@dataclass
class RiskZoneConfig:
    """Thresholds for risk zone boundaries.

    The Y-axis boundaries are calibrated for the spectral-dominant
    proarrhythmic index (v3.2).  Spectral change 0-1 maps to 0-100,
    with positive drugs typically showing scores 49-73 and negative
    drugs 0-51.  The 40/20 split creates three zones that achieve
    6/7 accuracy on the CiPA validation set (7 drugs).
    """
    # X-axis: ΔFPDcF (%)
    fpdc_low_mid: float = 10.0      # < 10% → low prolongation risk
    fpdc_mid_high: float = 20.0     # > 20% → high prolongation risk
    # Y-axis: proarrhythmic index
    proarrh_low_mid: float = 20.0   # baseline noise / non-specific effects
    proarrh_mid_high: float = 40.0  # spectral change ≈ 0.4 → suspicious


# ── Main plotting function ─────────────────────────────────────────────

def generate_risk_map(
    results_list,
    config=None,
    ground_truth: Optional[Dict[str, bool]] = None,
    zone_cfg: Optional[RiskZoneConfig] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = 'CiPA-style 2D Risk Map',
) -> plt.Figure:
    """
    Generate a CiPA-style 2D risk map from analysis results.

    Parameters
    ----------
    results_list : list
        Output of batch_analyze().
    config : AnalysisConfig or None
        Pipeline config (used to read thresholds for annotation).
    ground_truth : dict or None
        {drug_name: True/False} — if provided, colours points by known
        positive (red) / negative (blue) class.
    zone_cfg : RiskZoneConfig or None
        Custom risk-zone thresholds.  Defaults to standard values.
    figsize : tuple
        Figure size in inches.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if zone_cfg is None:
        zone_cfg = RiskZoneConfig()

    metrics = aggregate_drug_metrics(results_list)

    # Compute coordinates
    drugs, xs, ys = [], [], []
    for drug, m in sorted(metrics.items()):
        x = m.max_pct_fpdc_change if not np.isnan(m.max_pct_fpdc_change) else 0.0
        y = compute_proarrhythmic_index(m)
        drugs.append(drug)
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=figsize)

    # Risk zone backgrounds
    x_min = min(xs.min() - 10, -15)
    x_max = max(xs.max() + 10, zone_cfg.fpdc_mid_high + 15)
    y_min = -2
    y_max = max(ys.max() + 10, zone_cfg.proarrh_mid_high + 15)

    # Low risk (green)
    ax.axhspan(y_min, zone_cfg.proarrh_low_mid, xmin=0, xmax=1,
               color='#d4edda', alpha=0.5, zorder=0)
    # Intermediate (yellow) — middle band
    ax.axhspan(zone_cfg.proarrh_low_mid, zone_cfg.proarrh_mid_high, xmin=0, xmax=1,
               color='#fff3cd', alpha=0.5, zorder=0)
    # High risk (red)
    ax.axhspan(zone_cfg.proarrh_mid_high, y_max + 20, xmin=0, xmax=1,
               color='#f8d7da', alpha=0.5, zorder=0)

    # FPDcF prolongation threshold lines
    ax.axvline(zone_cfg.fpdc_low_mid, color='#888', ls='--', lw=0.8, alpha=0.7)
    ax.axvline(zone_cfg.fpdc_mid_high, color='#888', ls='--', lw=0.8, alpha=0.7)
    ax.axhline(zone_cfg.proarrh_low_mid, color='#888', ls='--', lw=0.8, alpha=0.7)
    ax.axhline(zone_cfg.proarrh_mid_high, color='#888', ls='--', lw=0.8, alpha=0.7)

    # Zone labels
    ax.text(x_min + 2, zone_cfg.proarrh_low_mid / 2, 'LOW RISK',
            fontsize=9, color='#155724', alpha=0.6, fontweight='bold')
    ax.text(x_min + 2, (zone_cfg.proarrh_low_mid + zone_cfg.proarrh_mid_high) / 2,
            'INTERMEDIATE', fontsize=9, color='#856404', alpha=0.6, fontweight='bold')
    ax.text(x_min + 2, zone_cfg.proarrh_mid_high + 5, 'HIGH RISK',
            fontsize=9, color='#721c24', alpha=0.6, fontweight='bold')

    # ── Scatter points ──
    for i, drug in enumerate(drugs):
        x, y = xs[i], ys[i]

        # Colour by ground truth if available
        if ground_truth is not None:
            gt = ground_truth.get(drug)
            if gt is True:
                color, marker = '#dc3545', 'D'     # red diamond = hERG+
            elif gt is False:
                color, marker = '#0d6efd', 'o'     # blue circle = hERG-
            else:
                color, marker = '#6c757d', 's'     # grey square = unknown
        else:
            # Colour by computed risk zone
            if y >= zone_cfg.proarrh_mid_high or x >= zone_cfg.fpdc_mid_high:
                color, marker = '#dc3545', 'D'
            elif y >= zone_cfg.proarrh_low_mid or x >= zone_cfg.fpdc_low_mid:
                color, marker = '#ffc107', 's'
            else:
                color, marker = '#198754', 'o'

        m_obj = metrics[drug]
        # Larger point if cessation detected
        size = 180 if m_obj.has_cessation else 120

        ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='white',
                   linewidths=1.2, zorder=5)

        # Drug label with white outline for readability
        label = drug.capitalize()
        txt = ax.annotate(label, (x, y), textcoords="offset points",
                          xytext=(8, 6), fontsize=9.5, fontweight='bold',
                          color='#212529')
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

        # Cessation indicator (⚡)
        if m_obj.has_cessation:
            txt2 = ax.annotate('⚡', (x, y), textcoords="offset points",
                               xytext=(-12, -12), fontsize=12)
            txt2.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    # ── Axes ──
    ax.set_xlabel('Max ΔFPDcF (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proarrhythmic Index (0–100)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    # ── Legend ──
    legend_elements = []
    if ground_truth is not None:
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker='D', color='w',
                               markerfacecolor='#dc3545', markersize=10,
                               label='hERG+ (QT prolonger)'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='#0d6efd', markersize=10,
                               label='hERG− (negative)'))
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], marker='$⚡$', color='w',
                           markerfacecolor='#212529', markersize=12,
                           label='Cessation detected'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#dee2e6')

    # ── Annotation: axis explanation ──
    ax.text(0.98, 0.02,
            'X = max FPDcF change across concentrations\n'
            'Y = composite: spectral (70%) + morph. instability (25%) + EAD (5%)',
            transform=ax.transAxes, fontsize=7.5, color='#6c757d',
            ha='right', va='bottom')

    fig.tight_layout()
    return fig
