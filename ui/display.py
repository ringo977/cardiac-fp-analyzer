"""
Shared display components for signal, beats, parameters, and arrhythmia.

These are used by both single_file.py and batch.py to avoid
cross-module private imports.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cardiac_fp_analyzer.arrhythmia import compute_template
from cardiac_fp_analyzer.config import AnalysisConfig
from ui.helpers import amplitude_scale, reanalyze_with_modified_beats
from ui.i18n import T


def plot_signal(result, key_suffix=""):
    """Plot filtered signal with beat markers and interactive beat editor."""
    sig_filt = result['filtered_signal']
    scale, y_label = amplitude_scale(sig_filt)
    sig = sig_filt * scale
    t = result['time_vector']
    _fs = result['metadata']['sample_rate']  # noqa: F841 — reserved for future per-beat time conversion
    ks = key_suffix

    # ── Build unified beat set with inclusion state ──
    bi_raw = result['beat_indices_raw']

    # Initialize beat editor state on first run or when result changes
    result_id = id(result['filtered_signal'])
    rid_key = f'_beat_editor_result_id{ks}'
    all_key = f'beat_all_indices{ks}'
    excl_key = f'beat_excluded{ks}'

    if (st.session_state.get(rid_key) != result_id
            or all_key not in st.session_state):
        st.session_state[rid_key] = result_id
        st.session_state[all_key] = sorted(set(bi_raw))
        st.session_state[excl_key] = set()

    all_bi = sorted(st.session_state[all_key])
    excluded = st.session_state[excl_key]

    # FPD-included set: beats that survived rhythm/RR filter + re-segmentation
    # and actually fed parameter extraction. When available, we visually
    # distinguish them from beats detected-but-filtered-out. Fallback to
    # beat_indices (post-QC) for backward compatibility with older results.
    bi_fpd_arr = result.get('beat_indices_fpd')
    if bi_fpd_arr is None:
        bi_fpd_arr = result.get('beat_indices', [])
    fpd_set = set(int(x) for x in np.asarray(bi_fpd_arr, dtype=int).tolist())

    included_bi = np.array([i for i in all_bi
                             if i in fpd_set and i not in excluded],
                            dtype=int)
    filtered_bi = np.array([i for i in all_bi
                             if i not in fpd_set and i not in excluded],
                            dtype=int)
    excluded_bi = np.array([i for i in all_bi if i in excluded], dtype=int)

    # Toggle for raw signal overlay
    show_raw = st.checkbox(T('show_raw_signal'), value=False, key=f'show_raw{ks}')

    fig = go.Figure()
    step = max(1, len(sig) // 50000)

    # Raw signal (faded, behind filtered)
    if show_raw and 'raw_signal' in result:
        raw_sig = result['raw_signal'] * scale
        fig.add_trace(go.Scatter(
            x=t[::step], y=raw_sig[::step],
            mode='lines', name=T('raw_signal'),
            line=dict(color='#ff7f0e', width=0.6),
            opacity=0.4
        ))

    fig.add_trace(go.Scatter(
        x=t[::step], y=sig[::step],
        mode='lines', name=T('filtered_signal'),
        line=dict(color='#1f77b4', width=0.8)
    ))

    # Included beats (post rhythm/RR filter — fed FPD) — red triangles
    if len(included_bi) > 0:
        fig.add_trace(go.Scatter(
            x=t[included_bi], y=sig[included_bi],
            mode='markers', name=T('beats_included'),
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
    # Detected but filtered out by rhythm/RR logic — small faded greys.
    # Shown so the user can see what the detector fired on, without
    # confusing them with the beats that actually produced FPD.
    if len(filtered_bi) > 0:
        fig.add_trace(go.Scatter(
            x=t[filtered_bi], y=sig[filtered_bi],
            mode='markers', name=T('beats_filtered_out'),
            marker=dict(color='#666666', size=4, symbol='triangle-down'),
            opacity=0.35
        ))
    # Excluded beats (manual exclusions via beat editor) — grey with outline
    if len(excluded_bi) > 0:
        fig.add_trace(go.Scatter(
            x=t[excluded_bi], y=sig[excluded_bi],
            mode='markers', name=T('beats_excluded'),
            marker=dict(color='#888888', size=7, symbol='triangle-down',
                        line=dict(width=1, color='white')),
            opacity=0.5
        ))

    # Repolarization peaks: all_params is indexed over the FPD beat set
    # (bi_fpd), so we iterate over that set, not over bi_clean.
    bi_fpd_for_map = np.asarray(bi_fpd_arr, dtype=int)
    all_params = result.get('all_params') or []
    bi_to_repol = {}
    for k, bi_dep in enumerate(bi_fpd_for_map):
        if k >= len(all_params):
            break
        g = all_params[k].get('repol_peak_global_idx')
        if g is not None:
            bi_to_repol[int(bi_dep)] = int(g)
    repol_gi = []
    for bi in included_bi:
        gi = bi_to_repol.get(int(bi))
        if gi is not None and 0 <= gi < len(sig):
            repol_gi.append(gi)
    if repol_gi:
        repol_gi = np.asarray(repol_gi, dtype=int)
        fig.add_trace(go.Scatter(
            x=t[repol_gi], y=sig[repol_gi],
            mode='markers', name=T('repol_peak_markers'),
            marker=dict(
                color='#1b9e77', size=8, symbol='triangle-up',
                line=dict(width=1, color='black'),
            ),
        ))

    fig.update_layout(
        xaxis_title=T('time_s'), yaxis_title=y_label,
        height=400, margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats: prefer post-filter (FPD) periods so BP / BPM reflect the beats
    # that actually drove the FPD calculation. Fallback to all-detected
    # beat_periods when FPD set is unavailable (e.g. legacy results).
    bp_all = result.get('beat_periods', np.array([]))
    if len(included_bi) >= 2:
        bp = np.diff(np.sort(included_bi)) / float(
            result['metadata'].get('sample_rate', 1.0))
    else:
        bp = bp_all
    cols = st.columns(4)
    cols[0].metric(T('n_beats_included'), f"{len(included_bi)}")
    cols[1].metric(T('n_beats_raw'), f"{len(all_bi)}")
    if len(bp) > 0 and np.mean(bp) > 0:
        cols[2].metric(T('beat_period'), f"{np.mean(bp)*1000:.0f} ms")
        cols[3].metric("BPM", f"{60/np.mean(bp):.1f}")

    # ── Detection pipeline diagnostics ──
    det_info = result.get('detection_info', {})
    val_info = det_info.get('beat_validation', {})
    rec_info = det_info.get('beat_recovery', {})
    qc_rep = result.get('qc_report', None)
    with st.expander("🔍 Detection diagnostics", expanded=False):
        st.markdown("**Beat detection pipeline**")
        dcols = st.columns(5)
        dcols[0].metric("Method", det_info.get('method', '?'))
        dcols[1].metric("Rej. amplitude", val_info.get('n_rejected_amplitude', '?'))
        dcols[2].metric("Rej. morphology", val_info.get('n_rejected_morphology', '?'))
        dcols[3].metric("Re-admitted", val_info.get('n_readmitted', 0))
        dcols[4].metric("Recovered", rec_info.get('n_recovered', 0))
        if qc_rep is not None:
            st.markdown("**Quality control (post-segmentation)**")
            qcols = st.columns(6)
            qcols[0].metric("QC input", qc_rep.n_beats_input)
            qcols[1].metric("QC accepted", qc_rep.n_beats_accepted)
            qcols[2].metric("QC rej. amp", qc_rep.n_beats_rejected_snr)
            qcols[3].metric("QC rej. morph", qc_rep.n_beats_rejected_morphology)
            qcols[4].metric("QC re-admitted", getattr(qc_rep, 'n_readmitted', 0))
            qcols[5].metric("Grade / SNR",
                            f"{qc_rep.grade} / {qc_rep.global_snr:.1f}")
            if qc_rep.notes:
                for note in qc_rep.notes:
                    st.caption(f"ℹ️ {note}")

    # ── Beat editor ──
    with st.expander(f"✏️ {T('beat_editor')}", expanded=False):
        st.caption(T('beat_editor_help'))

        # --- Beat table with toggle checkboxes ---
        beat_df = pd.DataFrame({
            T('beat_include'): [i not in excluded for i in all_bi],
            '#': list(range(1, len(all_bi) + 1)),
            T('time_s'): [f"{t[i]:.3f}" for i in all_bi],
            '_idx': all_bi,
        })

        edited = st.data_editor(
            beat_df[[T('beat_include'), '#', T('time_s')]],
            hide_index=True,
            use_container_width=True,
            height=min(400, 35 * len(all_bi) + 40),
            column_config={
                T('beat_include'): st.column_config.CheckboxColumn(
                    T('beat_include'), default=True, width='small'
                ),
                '#': st.column_config.NumberColumn('#', width='small'),
                T('time_s'): st.column_config.TextColumn(T('time_s'), width='medium'),
            },
            key=f'beat_editor_table{ks}'
        )

        # Sync checkbox changes back to session state
        new_excluded = set()
        for row_i, included in enumerate(edited[T('beat_include')]):
            if not included:
                new_excluded.add(all_bi[row_i])
        st.session_state[excl_key] = new_excluded

        st.divider()

        # --- Add new beat ---
        col_add, col_btn = st.columns([3, 1])
        with col_add:
            new_beat_t = st.number_input(
                T('add_beat_time'), min_value=float(t[0]),
                max_value=float(t[-1]),
                value=float(t[len(t)//2]),
                step=0.001, format="%.3f",
                key=f'add_beat_time_input{ks}'
            )
        with col_btn:
            st.write("")
            if st.button(T('add_beat_btn'), key=f'add_beat_btn{ks}'):
                new_idx = int(np.argmin(np.abs(t - new_beat_t)))
                if new_idx not in st.session_state[all_key]:
                    st.session_state[all_key].append(new_idx)
                    st.session_state[all_key].sort()
                # Make sure it's not excluded
                st.session_state[excl_key].discard(new_idx)
                st.rerun()

        # --- Summary & re-analyze ---
        n_inc = len(all_bi) - len(new_excluded)
        n_orig = len(bi_raw)
        has_changes = (set(all_bi) != set(bi_raw)) or (len(new_excluded) > 0)

        if has_changes:
            st.info(T('beats_modified', orig=n_orig, new=n_inc))

            if st.button(T('reanalyze_btn'), type='primary', key=f'reanalyze_btn{ks}'):
                final_bi = np.array([i for i in all_bi if i not in new_excluded], dtype=int)
                config = st.session_state.get('_analysis_config', AnalysisConfig())
                with st.spinner(T('analyzing')):
                    updated = reanalyze_with_modified_beats(result, final_bi, config)
                # Store updated result in the right place
                if ks and st.session_state.get('single_result_both'):
                    ch_name = ks.lstrip('_')
                    st.session_state['single_result_both'][ch_name] = updated
                else:
                    st.session_state['single_result'] = updated
                # Reset editor state so it re-syncs with new result
                st.session_state.pop(rid_key, None)
                st.session_state.pop(all_key, None)
                st.session_state.pop(excl_key, None)
                st.success(T('reanalysis_done', n=len(updated['beat_indices'])))
                st.rerun()


# Rhythm types that make a single averaged template a poor summary of the
# underlying beats.  ``chaotic`` / ``ambiguous`` already carry that meaning;
# ``alternans_2_to_1`` and ``trimodal`` have multiple morphological families
# that should not be collapsed into one template.
_TEMPLATE_RISKY_RHYTHM_TYPES = frozenset({
    'chaotic',
    'ambiguous',
    'alternans_2_to_1',
    'trimodal',
})

# FPD-dispersion threshold above which the T-wave of individual beats falls
# at wildly different offsets relative to the depolarisation spike, so
# point-wise aggregation (mean *or* median) cancels them.  20 % is a
# conservative cut: below this, visual T-wave alignment in the overlay is
# preserved on the real-signal fixtures we've tested.
_FPD_CV_TEMPLATE_WARN = 0.20


def _render_template_representativity_banner(result):
    """Warn in the Battiti tab when the averaged template is likely to mask
    the actual morphology.

    Two triggers (either is sufficient):

    * rhythm classifier reports a type that mixes multiple morphologies
      (``chaotic``, ``ambiguous``, ``alternans_2_to_1``, ``trimodal``);
    * FPD coefficient-of-variation exceeds ``_FPD_CV_TEMPLATE_WARN`` — the
      T-waves fall at different time offsets and cancel out when averaged.

    Silent (no banner) when the rhythm is regular and FPD dispersion is
    low, which is the common case.
    """
    rc = (result.get('detection_info') or {}).get('rhythm_classification') or {}
    rhythm_type = str(rc.get('rhythm_type') or '')
    summary = result.get('summary') or {}
    fpd_mean = float(summary.get('fpd_ms_mean', 0) or 0.0)
    fpd_std = float(summary.get('fpd_ms_std', 0) or 0.0)
    fpd_cv = (fpd_std / fpd_mean) if fpd_mean > 0 else 0.0

    risky_rhythm = rhythm_type in _TEMPLATE_RISKY_RHYTHM_TYPES
    dispersive_fpd = fpd_cv > _FPD_CV_TEMPLATE_WARN
    if not (risky_rhythm or dispersive_fpd):
        return

    reasons: list[str] = []
    if risky_rhythm:
        reasons.append(T('template_reason_rhythm').format(rhythm=rhythm_type))
    if dispersive_fpd:
        reasons.append(
            T('template_reason_fpd_cv').format(cv_pct=fpd_cv * 100.0)
        )

    title = T('template_unrepresentative_title')
    body = T('template_unrepresentative_body').format(
        reasons=' · '.join(reasons)
    )
    st.warning(f"**{title}**\n\n{body}")


def plot_beats(result):
    """Plot overlaid beat waveforms and template."""
    bd = result.get('beats_data', [])
    fs = result['metadata']['sample_rate']

    if not bd:
        st.warning(T('no_params'))
        return

    # ── Representativity banner (v3.4.x) ─────────────────────────────────
    # Flag cases where the overlaid template is unlikely to summarise the
    # beat morphology faithfully (chaotic/alternans rhythm, or wide FPD
    # dispersion causing T-wave cancellation in the mean/median).
    _render_template_representativity_banner(result)

    # Detect amplitude scale from first beat
    scale, y_label = amplitude_scale(bd[0])

    fig = go.Figure()
    # Plot up to 30 beats
    n_plot = min(30, len(bd))
    indices = np.linspace(0, len(bd)-1, n_plot, dtype=int)
    for i in indices:
        t_beat = np.arange(len(bd[i])) / fs * 1000  # ms
        fig.add_trace(go.Scatter(
            x=t_beat, y=bd[i] * scale,
            mode='lines', line=dict(color='rgba(100,149,237,0.3)', width=0.8),
            showlegend=False, hoverinfo='skip'
        ))

    # Template
    tmpl = compute_template(bd)
    if tmpl is not None:
        t_tmpl = np.arange(len(tmpl)) / fs * 1000
        fig.add_trace(go.Scatter(
            x=t_tmpl, y=tmpl * scale,
            mode='lines', name=T('template'),
            line=dict(color='red', width=2.5)
        ))

    fig.update_layout(
        xaxis_title=T('time_ms'), yaxis_title=y_label,
        height=400, margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # FPD annotation
    summary = result.get('summary', {})
    cols = st.columns(4)
    cols[0].metric(T('fpd'), f"{summary.get('fpd_ms_mean', 0):.1f} ± {summary.get('fpd_ms_std', 0):.1f}")
    cols[1].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.1f} ± {summary.get('fpdc_ms_std', 0):.1f}")
    cols[2].metric(T('spike_amp'), f"{summary.get('spike_amplitude_mV_mean', 0):.4f} mV")
    cols[3].metric(T('fpd_confidence'), f"{summary.get('fpd_confidence', 0):.2f}")


def show_params_table(result):
    """Show per-beat parameter table."""
    all_p = result.get('all_params', [])
    if not all_p:
        st.warning(T('no_params'))
        return

    rows = []
    for p in all_p:
        rows.append({
            'Beat': p.get('beat_number', ''),
            'RR (ms)': f"{p.get('rr_interval_ms', np.nan):.1f}",
            T('spike_amp') + ' (mV)': f"{p.get('spike_amplitude_mV', np.nan):.5f}",
            'FPD (ms)': f"{p.get('fpd_ms', np.nan):.1f}",
            'FPDcF (ms)': f"{p.get('fpdc_ms', np.nan):.1f}",
            'Rise time (ms)': f"{p.get('rise_time_ms', np.nan):.2f}",
            'Max dV/dt': f"{p.get('max_dvdt', np.nan):.5f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)

    # Summary stats
    summary = result.get('summary', {})
    st.subheader(T('summary'))
    summary_rows = []
    for key in ['spike_amplitude_mV', 'fpd_ms', 'fpdc_ms', 'rise_time_ms', 'rr_interval_ms']:
        m = summary.get(f'{key}_mean', np.nan)
        s = summary.get(f'{key}_std', np.nan)
        cv = summary.get(f'{key}_cv', np.nan)
        summary_rows.append({
            T('parameter'): key.replace('_mV', ' (mV)').replace('_ms', ' (ms)'),
            T('mean'): f"{m:.3f}" if not np.isnan(m) else "—",
            'SD': f"{s:.3f}" if not np.isnan(s) else "—",
            'CV%': f"{cv:.1f}" if not np.isnan(cv) else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def show_arrhythmia(result):
    """Show arrhythmia analysis results."""
    ar = result.get('arrhythmia_report')
    if ar is None:
        st.warning(T('no_params'))
        return

    # Classification banner
    risk_color = "🟢" if ar.risk_score < 30 else ("🟡" if ar.risk_score < 60 else "🔴")
    st.markdown(f"### {risk_color} {ar.classification} — Risk Score: {ar.risk_score}/100")

    # Flags
    if ar.flags:
        st.subheader("Flag rilevate")
        for f in ar.flags:
            sev_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}.get(f['severity'], "❓")
            st.markdown(f"{sev_icon} **{f['type']}** ({f['severity']}): {f['description']}")
    else:
        st.success("Nessuna flag aritmca rilevata.")

    # Residual details
    rd = ar.residual_details
    if rd:
        st.subheader("Analisi residuo")
        cols = st.columns(4)
        cols[0].metric("Morphology instability", f"{rd.get('morphology_instability', 0):.3f}")
        cols[1].metric("EAD incidence", f"{rd.get('ead_incidence_pct', 0):.1f}%")
        cols[2].metric("STV FPDcF (ms)", f"{rd.get('poincare_stv_fpdc_ms', np.nan):.1f}")
        cols[3].metric("Baseline-relative", "✓" if rd.get('baseline_relative') else "✗")

    # Events
    if ar.events:
        st.subheader("Eventi")
        ev_df = pd.DataFrame(ar.events)
        st.dataframe(ev_df, use_container_width=True, height=200)
