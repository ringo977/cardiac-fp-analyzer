"""
inclusion.py — Quality-based inclusion criteria for batch analysis.

Implements the multi-tier inclusion workflow inspired by Visone et al. 2023:
  1. Baseline CV of beat-period must be < threshold
  2. FPDcF plausibility: wide safety-net range
  3. FPD confidence: low-confidence baselines excluded
  4. Physiological FPDcF range (opt-in)
  5. Population outlier detection via MAD (opt-in)

Baselines that fail any criterion are excluded together with all
drug recordings belonging to the same group (chip + channel).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_inclusion_criteria(results, verbose=True, cfg=None):
    """
    Apply quality-based inclusion criteria.

    Parameters
    ----------
    results : list of result dicts from analyze_single_file
    verbose : print summary
    cfg : InclusionConfig or None

    Returns
    -------
    results : same list, with 'inclusion' dict added to each entry
    """
    if cfg is None:
        from .config import InclusionConfig
        cfg = InclusionConfig()

    from .normalization import get_group_key, is_baseline

    # ── Pre-compute population statistics for outlier detection (criterion 5) ──
    bl_fpdc_by_exp = {}
    if getattr(cfg, 'enabled_fpdc_outlier', False):
        from collections import defaultdict
        _exp_vals = defaultdict(list)
        for r in results:
            if not is_baseline(r):
                continue
            fpdc = r.get('summary', {}).get('fpdc_ms_mean', np.nan)
            exp = r.get('file_info', {}).get('experiment', 'unknown')
            if not np.isnan(fpdc):
                _exp_vals[exp].append(fpdc)
        for exp, vals in _exp_vals.items():
            n_min = getattr(cfg, 'fpdc_outlier_min_baselines', 3)
            if len(vals) >= n_min:
                med = np.median(vals)
                mad = np.median(np.abs(np.array(vals) - med))
                # Scale MAD to σ-equivalent for normal distributions
                mad_s = mad * 1.4826 if mad > 0 else np.std(vals)
                bl_fpdc_by_exp[exp] = (med, mad_s)

    # ── Step 1: identify failing baselines and flag their groups ──
    excluded_groups = set()
    n_bl_ok = 0
    n_bl_fail = 0
    n_bl_conf_fail = 0
    n_bl_physiol_fail = 0
    n_bl_outlier_fail = 0

    for r in results:
        if not is_baseline(r):
            continue
        summary = r.get('summary', {})
        cv = summary.get('beat_period_ms_cv', np.nan)
        conf = summary.get('fpd_confidence', np.nan)
        fpdc = summary.get('fpdc_ms_mean', np.nan)
        group = get_group_key(r)
        exp = r.get('file_info', {}).get('experiment', 'unknown')

        fail_reason = None

        # Criterion 1: CV of beat period (paper standard)
        if cfg.enabled_cv and (np.isnan(cv) or cv >= cfg.max_cv_bp):
            fail_reason = f'Baseline CV={cv:.1f}% >= {cfg.max_cv_bp}%'

        # Criterion 2: wide FPDcF plausibility range (safety net)
        if fail_reason is None and cfg.enabled_fpdc_range and not np.isnan(fpdc):
            if fpdc < cfg.fpdc_range_min or fpdc > cfg.fpdc_range_max:
                fail_reason = f'Baseline FPDcF={fpdc:.0f}ms outside [{cfg.fpdc_range_min:.0f}, {cfg.fpdc_range_max:.0f}]'

        # Criterion 3: FPD confidence (data-driven threshold)
        if fail_reason is None and cfg.enabled_confidence and not np.isnan(conf) and conf < cfg.min_fpd_confidence:
            fail_reason = f'Baseline FPD confidence={conf:.3f} < {cfg.min_fpd_confidence}'
            n_bl_conf_fail += 1

        # Criterion 4: physiological FPDcF range (literature-based, opt-in)
        if fail_reason is None and getattr(cfg, 'enabled_fpdc_physiol', False) and not np.isnan(fpdc):
            physiol_min = getattr(cfg, 'fpdc_physiol_min', 350.0)
            physiol_max = getattr(cfg, 'fpdc_physiol_max', 800.0)
            if fpdc < physiol_min or fpdc > physiol_max:
                fail_reason = (f'Baseline FPDcF={fpdc:.0f}ms outside physiological range '
                               f'[{physiol_min:.0f}, {physiol_max:.0f}]ms')
                n_bl_physiol_fail += 1

        # Criterion 5: population outlier (data-adaptive, opt-in)
        if fail_reason is None and getattr(cfg, 'enabled_fpdc_outlier', False) and not np.isnan(fpdc):
            if exp in bl_fpdc_by_exp:
                med, mad_s = bl_fpdc_by_exp[exp]
                if mad_s > 0:
                    n_sigma = getattr(cfg, 'fpdc_outlier_n_sigma', 2.0)
                    z = abs(fpdc - med) / mad_s
                    if z > n_sigma:
                        fail_reason = (f'Baseline FPDcF={fpdc:.0f}ms is outlier in {exp} '
                                       f'(median={med:.0f}ms, {z:.1f}σ > {n_sigma}σ)')
                        n_bl_outlier_fail += 1

        if fail_reason:
            excluded_groups.add(group)
            r['inclusion'] = {'passed': False, 'reason': fail_reason}
            n_bl_fail += 1
        else:
            r['inclusion'] = {'passed': True, 'reason': ''}
            n_bl_ok += 1

    if verbose:
        parts = [f"CV BP < {cfg.max_cv_bp}%"]
        if cfg.enabled_confidence:
            parts.append(f"conf >= {cfg.min_fpd_confidence}")
        if getattr(cfg, 'enabled_fpdc_physiol', False):
            parts.append(f"FPDcF ∈ [{cfg.fpdc_physiol_min:.0f}-{cfg.fpdc_physiol_max:.0f}]ms")
        if getattr(cfg, 'enabled_fpdc_outlier', False):
            parts.append(f"outlier < {cfg.fpdc_outlier_n_sigma}σ")
        detail_parts = []
        if n_bl_conf_fail > 0:
            detail_parts.append(f"{n_bl_conf_fail} low confidence")
        if n_bl_physiol_fail > 0:
            detail_parts.append(f"{n_bl_physiol_fail} outside physiol. range")
        if n_bl_outlier_fail > 0:
            detail_parts.append(f"{n_bl_outlier_fail} population outlier")
        detail_str = f" [{', '.join(detail_parts)}]" if detail_parts else ""
        print(f"  Inclusion criteria ({', '.join(parts)}): "
              f"{n_bl_ok} baselines OK, {n_bl_fail} excluded "
              f"({len(excluded_groups)} groups removed){detail_str}")

    # Step 2: flag drug recordings in excluded groups
    n_drug_excl = 0
    for r in results:
        if is_baseline(r):
            continue
        group = get_group_key(r)
        if group in excluded_groups:
            r['inclusion'] = {'passed': False, 'reason': f'Baseline of group {group} failed inclusion'}
            n_drug_excl += 1
        else:
            r.setdefault('inclusion', {'passed': True, 'reason': ''})

    # Step 3: FPDcF plausibility check
    n_fpdc_fail = 0
    if cfg.enabled_fpdc_range:
        for r in results:
            fpdc = r.get('summary', {}).get('fpdc_ms_mean', np.nan)
            if not np.isnan(fpdc) and (fpdc < cfg.fpdc_range_min or fpdc > cfg.fpdc_range_max):
                inc = r.setdefault('inclusion', {'passed': True, 'reason': ''})
                inc['fpdc_plausible'] = False
                inc['fpdc_note'] = f'FPDcF={fpdc:.0f}ms outside [{cfg.fpdc_range_min:.0f}-{cfg.fpdc_range_max:.0f}]ms'
                n_fpdc_fail += 1
            else:
                inc = r.setdefault('inclusion', {'passed': True, 'reason': ''})
                inc['fpdc_plausible'] = True

    if verbose and n_fpdc_fail > 0:
        print(f"  FPDcF plausibility: {n_fpdc_fail} recordings outside {cfg.fpdc_range_min:.0f}-{cfg.fpdc_range_max:.0f}ms range")

    # Step 4: Flag drug recordings with low FPD confidence
    n_drug_conf = 0
    if cfg.enabled_confidence:
        for r in results:
            if is_baseline(r):
                continue
            conf = r.get('summary', {}).get('fpd_confidence', np.nan)
            inc = r.setdefault('inclusion', {'passed': True, 'reason': ''})
            if not np.isnan(conf) and conf < cfg.min_fpd_confidence:
                inc['fpd_reliable'] = False
                n_drug_conf += 1
            else:
                inc['fpd_reliable'] = True

    if verbose and n_drug_conf > 0:
        print(f"  FPD reliability: {n_drug_conf} drug recordings with confidence < {cfg.min_fpd_confidence}")

    return results
