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


def apply_inclusion_criteria(results, verbose=True, cfg=None, report_out=None):
    """
    Apply quality-based inclusion criteria.

    Parameters
    ----------
    results : list of result dicts from analyze_single_file
    verbose : print summary, including the removed-groups block
    cfg : InclusionConfig or None
    report_out : dict or None
        When a dict is passed it is cleared and filled with structured
        provenance for the exclusions::

            {'excluded_groups': {group_key: {...}},
             'n_groups_removed': int,
             'n_drug_recordings_excluded': int,
             'n_baselines_ok': int,
             'n_baselines_failed': int}

        Each ``excluded_groups`` entry records the offending baseline, the
        criterion that fired (``cv_bp`` / ``fpdc_range`` /
        ``fpd_confidence`` / ``fpdc_physiol`` / ``fpdc_outlier``), the
        measured values, and which drug recordings were taken down with
        it. An out-parameter is used rather than a second return value
        because the return type is relied on by the batch pipeline.

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
    # ``excluded_groups`` maps group key -> provenance dict rather than being
    # a bare set: when a baseline fails, its whole dose-response group is
    # removed, and that removal has to be explainable afterwards. Losing an
    # entire drug silently is worse than losing it loudly — the failure mode
    # produces *absence* of numbers, which is far harder to notice than
    # wrong ones.
    excluded_groups = {}
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
        # Machine-readable counterpart of ``fail_reason`` so a report can
        # group by criterion without parsing prose.
        fail_criterion = None

        # Criterion 0: plausibility guardrails (opt-in, independent of CV).
        # Checked first because these are "this is not a usable recording"
        # conditions, not "this recording is marginal" ones — and because
        # CV structurally cannot catch them (periodic noise has a good CV).
        if getattr(cfg, 'enabled_plausibility', False):
            bpm = summary.get('bpm_mean', np.nan)
            pct_no_repol = summary.get('pct_beats_no_repol', np.nan)
            bpm_min = getattr(cfg, 'bpm_plausible_min', 10.0)
            bpm_max = getattr(cfg, 'bpm_plausible_max', 120.0)
            max_no_repol = getattr(cfg, 'max_pct_beats_no_repol', 50.0)
            if not np.isnan(bpm) and (bpm < bpm_min or bpm > bpm_max):
                fail_reason = (f'Baseline BPM={bpm:.0f} outside plausible '
                               f'range [{bpm_min:.0f}, {bpm_max:.0f}]')
                fail_criterion = 'bpm_plausible'
            elif not np.isnan(pct_no_repol) and pct_no_repol > max_no_repol:
                fail_reason = (f'Baseline has {pct_no_repol:.0f}% of beats '
                               f'without detectable repolarization '
                               f'(> {max_no_repol:.0f}%)')
                fail_criterion = 'no_repol'

        # Criterion 1: beat-period regularity.
        #
        # Two mutually exclusive forms — the combined rule supersedes the
        # CV-only gate when enabled, rather than stacking with it, because
        # stacking would keep the very false-positives the combined rule
        # exists to remove.
        if fail_reason is None and getattr(cfg, 'enabled_combined_rule', False):
            grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
            qc_grade = getattr(r.get('qc_report'), 'grade', None)
            worst = getattr(cfg, 'combined_min_qc_grade', 'C')
            max_cv = getattr(cfg, 'combined_max_cv_bp', 60.0)
            qc_rank = grade_order.get(str(qc_grade), 4)
            if qc_rank > grade_order.get(worst, 2):
                fail_reason = (f'Baseline QC grade={qc_grade} worse than '
                               f'{worst}')
                fail_criterion = 'qc_grade'
            elif np.isnan(cv) or cv >= max_cv:
                fail_reason = f'Baseline CV={cv:.1f}% >= {max_cv}% (combined rule)'
                fail_criterion = 'cv_bp'
        elif fail_reason is None and cfg.enabled_cv and (np.isnan(cv) or cv >= cfg.max_cv_bp):
            fail_reason = f'Baseline CV={cv:.1f}% >= {cfg.max_cv_bp}%'
            fail_criterion = 'cv_bp'

        # Criterion 2: wide FPDcF plausibility range (safety net)
        if fail_reason is None and cfg.enabled_fpdc_range and not np.isnan(fpdc):
            if fpdc < cfg.fpdc_range_min or fpdc > cfg.fpdc_range_max:
                fail_reason = f'Baseline FPDcF={fpdc:.0f}ms outside [{cfg.fpdc_range_min:.0f}, {cfg.fpdc_range_max:.0f}]'
                fail_criterion = 'fpdc_range'

        # Criterion 3: FPD confidence (data-driven threshold)
        if fail_reason is None and cfg.enabled_confidence and not np.isnan(conf) and conf < cfg.min_fpd_confidence:
            fail_reason = f'Baseline FPD confidence={conf:.3f} < {cfg.min_fpd_confidence}'
            fail_criterion = 'fpd_confidence'
            n_bl_conf_fail += 1

        # Criterion 4: physiological FPDcF range (literature-based, opt-in)
        if fail_reason is None and getattr(cfg, 'enabled_fpdc_physiol', False) and not np.isnan(fpdc):
            physiol_min = getattr(cfg, 'fpdc_physiol_min', 350.0)
            physiol_max = getattr(cfg, 'fpdc_physiol_max', 800.0)
            if fpdc < physiol_min or fpdc > physiol_max:
                fail_reason = (f'Baseline FPDcF={fpdc:.0f}ms outside physiological range '
                               f'[{physiol_min:.0f}, {physiol_max:.0f}]ms')
                fail_criterion = 'fpdc_physiol'
                n_bl_physiol_fail += 1

        # Criterion 1b: precision of the FPDc reference (opt-in).
        # Measures what the baseline is actually for — see
        # ``max_baseline_fpdc_rsem`` in config.py. Placed after the rhythm
        # criteria so that, when both are on, the more specific reason wins
        # the report.
        if fail_reason is None and getattr(cfg, 'enabled_baseline_precision', False):
            sd = summary.get('fpdc_ms_std', np.nan)
            n_fpd = summary.get('fpd_ms_n', np.nan)
            max_rsem = getattr(cfg, 'max_baseline_fpdc_rsem', 3.0)
            if (not np.isnan(fpdc) and fpdc > 0 and not np.isnan(sd)
                    and not np.isnan(n_fpd) and n_fpd >= 1):
                rsem = (sd / fpdc * 100.0) / np.sqrt(n_fpd)
                if rsem > max_rsem:
                    fail_reason = (
                        f'Baseline FPDc reference imprecise: rSEM={rsem:.2f}% '
                        f'> {max_rsem}% (SD={sd:.0f}ms, mean={fpdc:.0f}ms, '
                        f'n={int(n_fpd)})'
                    )
                    fail_criterion = 'baseline_precision'

        # Criterion 4b: FPD / RR ratio (physiological, rate-independent).
        # Repolarization cannot occupy the whole cycle. A ratio at or near
        # 100% means the detected "T wave" is an afterpotential or the next
        # depolarization, not a measurement — and unlike the absolute FPDc
        # window this holds at any beat rate.
        if fail_reason is None and getattr(cfg, 'enabled_fpd_rr_ratio', False):
            fpd_ms = summary.get('fpd_ms_median',
                                 summary.get('fpd_ms_mean', np.nan))
            bp_ms = summary.get('beat_period_ms_median',
                                summary.get('beat_period_ms_mean', np.nan))
            max_ratio = getattr(cfg, 'max_fpd_rr_ratio', 0.80)
            if (not np.isnan(fpd_ms) and not np.isnan(bp_ms) and bp_ms > 0):
                ratio = fpd_ms / bp_ms
                if ratio > max_ratio:
                    fail_reason = (
                        f'Baseline FPD/RR={ratio * 100:.0f}% > '
                        f'{max_ratio * 100:.0f}% (FPD={fpd_ms:.0f}ms vs '
                        f'RR={bp_ms:.0f}ms) — repolarization cannot fill '
                        f'the cycle'
                    )
                    fail_criterion = 'fpd_rr_ratio'

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
                        fail_criterion = 'fpdc_outlier'
                        n_bl_outlier_fail += 1

        if fail_reason:
            excluded_groups[group] = {
                'group': group,
                'baseline_file': r.get('metadata', {}).get('filename', '?'),
                'criterion': fail_criterion,
                'reason': fail_reason,
                'cv_bp': None if np.isnan(cv) else float(cv),
                'fpd_confidence': None if np.isnan(conf) else float(conf),
                'fpdc_ms_mean': None if np.isnan(fpdc) else float(fpdc),
                'qc_grade': getattr(r.get('qc_report'), 'grade', None),
                'experiment': exp,
                'n_drug_recordings_lost': 0,   # filled in step 2
                'drug_recordings_lost': [],
            }
            r['inclusion'] = {'passed': False, 'reason': fail_reason,
                              'criterion': fail_criterion}
            n_bl_fail += 1
        else:
            r['inclusion'] = {'passed': True, 'reason': ''}
            n_bl_ok += 1

    if verbose:
        if getattr(cfg, 'enabled_combined_rule', False):
            parts = [f"QC <= {getattr(cfg, 'combined_min_qc_grade', 'C')}",
                     f"CV BP < {getattr(cfg, 'combined_max_cv_bp', 60.0)}%"]
        else:
            parts = [f"CV BP < {cfg.max_cv_bp}%"]
        if getattr(cfg, 'enabled_plausibility', False):
            parts.append(
                f"BPM ∈ [{getattr(cfg, 'bpm_plausible_min', 10.0):.0f}-"
                f"{getattr(cfg, 'bpm_plausible_max', 120.0):.0f}]"
            )
        if cfg.enabled_confidence:
            parts.append(f"conf >= {cfg.min_fpd_confidence}")
        if getattr(cfg, 'enabled_fpdc_physiol', False):
            parts.append(f"FPDcF ∈ [{cfg.fpdc_physiol_min:.0f}-{cfg.fpdc_physiol_max:.0f}]ms")
        if getattr(cfg, 'enabled_fpd_rr_ratio', False):
            parts.append(f"FPD/RR <= {getattr(cfg, 'max_fpd_rr_ratio', 0.80) * 100:.0f}%")
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
            fname = r.get('metadata', {}).get('filename', '?')
            r['inclusion'] = {
                'passed': False,
                'reason': f'Baseline of group {group} failed inclusion',
                'criterion': 'group_removed',
                'excluded_group': group,
                # Carry the *root* cause, not just "the group failed", so a
                # single recording is self-explanatory in a report.
                'root_cause': excluded_groups[group]['reason'],
            }
            excluded_groups[group]['n_drug_recordings_lost'] += 1
            excluded_groups[group]['drug_recordings_lost'].append(fname)
            n_drug_excl += 1
        else:
            r.setdefault('inclusion', {'passed': True, 'reason': ''})

    # ── Removed-groups report ──
    # Printed after step 2 because only then is the drug count known.
    if verbose and excluded_groups:
        print()
        print("  " + "=" * 68)
        print("  GRUPPI RIMOSSI — nessun %ΔFPDcF verrà calcolato per questi")
        print("  " + "=" * 68)
        # Worst first: a group taking down 7 concentrations matters more
        # than one taking down 1.
        for info in sorted(excluded_groups.values(),
                           key=lambda i: -i['n_drug_recordings_lost']):
            lost = info['n_drug_recordings_lost']
            print(f"  ▸ {info['group']}")
            print(f"      baseline : {info['baseline_file']}")
            print(f"      motivo   : {info['reason']}")
            qc = info['qc_grade']
            extra = []
            if info['cv_bp'] is not None:
                extra.append(f"CV={info['cv_bp']:.1f}%")
            if info['fpd_confidence'] is not None:
                extra.append(f"conf={info['fpd_confidence']:.2f}")
            if info['fpdc_ms_mean'] is not None:
                extra.append(f"FPDc={info['fpdc_ms_mean']:.0f}ms")
            if qc:
                extra.append(f"QC={qc}")
            if extra:
                print(f"      misure   : {', '.join(extra)}")
            if lost:
                names = info['drug_recordings_lost']
                shown = ', '.join(n[:34] for n in names[:4])
                more = f" (+{len(names) - 4} altri)" if len(names) > 4 else ""
                print(f"      perde    : {lost} registrazioni → {shown}{more}")
            else:
                print("      perde    : nessuna registrazione farmaco associata")
        print(f"  {'-' * 68}")
        print(f"  Totale: {len(excluded_groups)} gruppi, "
              f"{n_drug_excl} registrazioni farmaco escluse")
        print()

    if excluded_groups:
        logger.warning(
            "Inclusion removed %d group(s) and %d drug recording(s). "
            "Criteria: %s",
            len(excluded_groups), n_drug_excl,
            ', '.join(sorted({i['criterion'] or '?'
                              for i in excluded_groups.values()})),
        )

    # Make the provenance available to callers (batch report, UI) without
    # changing the return type, which has one call site plus tests.
    if report_out is not None:
        report_out.clear()
        report_out.update({
            'excluded_groups': excluded_groups,
            'n_groups_removed': len(excluded_groups),
            'n_drug_recordings_excluded': n_drug_excl,
            'n_baselines_ok': n_bl_ok,
            'n_baselines_failed': n_bl_fail,
        })

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
