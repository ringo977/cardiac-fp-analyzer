"""
Single File Analysis page — upload, analyze, display, export.
"""

import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.analyze import analyze_single_file
from cardiac_fp_analyzer.arrhythmia import compute_template

from ui.i18n import T
from ui.helpers import reanalyze_with_modified_beats, amplitude_scale


# ═══════════════════════════════════════════════════════════════════════
#  Page entry point
# ═══════════════════════════════════════════════════════════════════════

def page_single_file(config: AnalysisConfig):
    st.header(f"🔬 {T('single_file')}")
    st.caption(T('single_file_desc'))

    uploaded = st.file_uploader(T('upload_csv'), type=['csv'], key='single_file')
    if uploaded is None:
        st.info(T('upload_csv_info'))
        return

    # Save to temp
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    channel = st.radio(T('channel'),
                       ['auto', 'el1', 'el2', T('both_channels')],
                       horizontal=True)

    if st.button(f"▶️ {T('analyze')}", type="primary", use_container_width=True):
        if channel == T('both_channels'):
            # Analyze both electrodes
            results_both = {}
            for ch in ['el1', 'el2']:
                with st.spinner(T('analyzing_ch', ch=ch.upper())):
                    r = analyze_single_file(tmp_path, channel=ch, verbose=False, config=config)
                if r is not None:
                    results_both[ch] = r
                else:
                    st.warning(T('channel_failed', ch=ch.upper()))
            if not results_both:
                st.error(T('no_valid_channel'))
                return
            st.session_state['single_result'] = None
            st.session_state['single_result_both'] = results_both
            channels_ok = list(results_both.keys())
            st.success(f"{T('analysis_complete')} — {', '.join(c.upper() for c in channels_ok)}")
        else:
            with st.spinner(T('analyzing')):
                result = analyze_single_file(tmp_path, channel=channel, verbose=False, config=config)
            if result is None:
                st.error(T('analysis_error'))
                return
            st.success(f"{T('analysis_complete')} — Elettrodo: {result['file_info'].get('analyzed_channel', '?')}")
            st.session_state['single_result'] = result
            st.session_state['single_result_both'] = None

    # ── Display results ──
    results_both = st.session_state.get('single_result_both')
    single_result = st.session_state.get('single_result')

    if results_both:
        _display_both_channels(results_both, config)
    elif single_result:
        _display_single_channel(single_result, config)


# ═══════════════════════════════════════════════════════════════════════
#  Display helpers
# ═══════════════════════════════════════════════════════════════════════

def _display_single_channel(result, config):
    """Display results for a single channel analysis."""
    fi = result['file_info']
    summary = result['summary']
    qc = result['qc_report']
    ar = result['arrhythmia_report']

    # ── Info cards ──
    cols = st.columns(5)
    cols[0].metric(T('chip_channel'), f"{fi.get('chip', '?')} / {fi.get('analyzed_channel', '?')}")
    cols[1].metric(T('drug'), fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade)
    cols[3].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.0f} ± {summary.get('fpdc_ms_std', 0):.0f}")
    cols[4].metric(T('risk_score'), f"{ar.risk_score}/100")

    # ── Tabs ──
    tab_signal, tab_beats, tab_params, tab_arrhythmia = st.tabs([
        f"📈 {T('signal')}", f"💓 {T('beats')}", f"📋 {T('params')}", f"⚡ {T('arrhythmia')}"
    ])

    with tab_signal:
        _plot_signal(result)

    with tab_beats:
        _plot_beats(result)

    with tab_params:
        _show_params_table(result)

    with tab_arrhythmia:
        _show_arrhythmia(result)

    # ── Export section ──
    _single_file_exports(result, config)


def _display_both_channels(results_both, config):
    """Display results for dual-channel analysis with comparison."""
    channels = list(results_both.keys())

    # ── Comparison table ──
    st.subheader(f"📊 {T('channel_comparison')}")
    comp_rows = []
    for ch in channels:
        r = results_both[ch]
        s, qc, ar = r['summary'], r['qc_report'], r['arrhythmia_report']
        comp_rows.append({
            T('channel'): ch.upper(),
            'QC Grade': qc.grade,
            'Beats': s.get('beat_period_ms_n', 0),
            'BP (ms)': f"{s.get('beat_period_ms_mean', 0):.0f} ± {s.get('beat_period_ms_std', 0):.0f}",
            'CV BP%': f"{s.get('beat_period_ms_cv', 0):.1f}",
            'FPDcF (ms)': f"{s.get('fpdc_ms_mean', 0):.0f} ± {s.get('fpdc_ms_std', 0):.0f}",
            'Spike (mV)': f"{s.get('spike_amplitude_mV_mean', 0):.4f}",
            'Risk': f"{ar.risk_score}/100",
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── Channel selector ──
    ch_labels = [ch.upper() for ch in channels]
    selected_label = st.radio(T('select_channel'), ch_labels, horizontal=True)
    selected_ch = selected_label.lower()
    result = results_both[selected_ch]

    fi = result['file_info']
    summary = result['summary']
    qc = result['qc_report']
    ar = result['arrhythmia_report']

    # ── Info cards ──
    cols = st.columns(5)
    cols[0].metric(T('chip_channel'), f"{fi.get('chip', '?')} / {selected_label}")
    cols[1].metric(T('drug'), fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade)
    cols[3].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.0f} ± {summary.get('fpdc_ms_std', 0):.0f}")
    cols[4].metric(T('risk_score'), f"{ar.risk_score}/100")

    # ── Analysis tabs for selected channel ──
    tab_signal, tab_beats, tab_params, tab_arrhythmia = st.tabs([
        f"📈 {T('signal')}", f"💓 {T('beats')}", f"📋 {T('params')}", f"⚡ {T('arrhythmia')}"
    ])

    with tab_signal:
        _plot_signal(result, key_suffix=f"_{selected_ch}")

    with tab_beats:
        _plot_beats(result)

    with tab_params:
        _show_params_table(result)

    with tab_arrhythmia:
        _show_arrhythmia(result)

    # ── Export section (selected channel) ──
    _single_file_exports(result, config)


# ═══════════════════════════════════════════════════════════════════════
#  Exports
# ═══════════════════════════════════════════════════════════════════════

def _single_file_exports(result, config):
    """Provide download buttons for single-file analysis results."""
    st.divider()
    st.subheader(f"📥 {T('export_section')}")

    fi = result['file_info']
    summary = result['summary']
    qc = result['qc_report']
    ar = result['arrhythmia_report']
    all_p = result.get('all_params', [])
    fname_base = fi.get('filename', 'recording').replace('.csv', '')

    cols = st.columns(3)

    # ── 1. Per-beat parameters CSV ──
    with cols[0]:
        if all_p:
            rows = []
            for p in all_p:
                rows.append({
                    'Beat': p.get('beat_number', ''),
                    'RR_interval_ms': p.get('rr_interval_ms', np.nan),
                    'Spike_amplitude_mV': p.get('spike_amplitude_mV', np.nan),
                    'FPD_ms': p.get('fpd_ms', np.nan),
                    'FPDcF_ms': p.get('fpdc_ms', np.nan),
                    'Rise_time_ms': p.get('rise_time_ms', np.nan),
                    'Max_dVdt': p.get('max_dvdt', np.nan),
                    'Morphology_corr': p.get('morphology_corr', np.nan),
                    'FPD_confidence': p.get('fpd_confidence', np.nan),
                })
            df_params = pd.DataFrame(rows)
            csv_bytes = df_params.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📋 {T('export_csv')}",
                csv_bytes,
                f"{fname_base}_parameters.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── 2. Summary CSV (one row with all summary stats) ──
    with cols[1]:
        summary_row = {
            'File': fi.get('filename', ''),
            'Chip': fi.get('chip', ''),
            'Electrode': fi.get('analyzed_channel', ''),
            'Drug': fi.get('drug', 'N/A'),
            'Concentration': fi.get('concentration', ''),
            'QC_Grade': qc.grade,
            'Beats_accepted': summary.get('beat_period_ms_n', 0),
            'Mean_BP_ms': summary.get('beat_period_ms_mean', np.nan),
            'CV_BP_pct': summary.get('beat_period_ms_cv', np.nan),
            'Mean_FPD_ms': summary.get('fpd_ms_mean', np.nan),
            'SD_FPD_ms': summary.get('fpd_ms_std', np.nan),
            'Mean_FPDcF_ms': summary.get('fpdc_ms_mean', np.nan),
            'SD_FPDcF_ms': summary.get('fpdc_ms_std', np.nan),
            'STV_FPDcF_ms': summary.get('stv_fpdc_ms', np.nan),
            'Mean_spike_mV': summary.get('spike_amplitude_mV_mean', np.nan),
            'Risk_score': ar.risk_score,
            'Classification': ar.classification,
        }
        df_summary = pd.DataFrame([summary_row])
        csv_summary = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"📊 {T('export_summary')}",
            csv_summary,
            f"{fname_base}_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── 3. Excel report (reuse batch report generator) ──
    with cols[2]:
        if st.button(f"📊 {T('export_excel_single')}", use_container_width=True):
            with st.spinner("Generazione Excel..."):
                try:
                    from cardiac_fp_analyzer.report import generate_excel_report
                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                        generate_excel_report([result], tmp.name)
                        with open(tmp.name, 'rb') as f:
                            buf = f.read()
                    st.download_button(
                        f"⬇️ {T('download_excel')}",
                        buf,
                        f"{fname_base}_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"{T('error')}: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Signal plot + beat editor
# ═══════════════════════════════════════════════════════════════════════

def _plot_signal(result, key_suffix=""):
    """Plot filtered signal with beat markers and interactive beat editor."""
    sig_filt = result['filtered_signal']
    scale, y_label = amplitude_scale(sig_filt)
    sig = sig_filt * scale
    t = result['time_vector']
    fs = result['metadata']['sample_rate']
    ks = key_suffix  # shorthand for widget key suffix

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

    included_bi = np.array([i for i in all_bi if i not in excluded], dtype=int)
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

    # Included beats — red triangles
    if len(included_bi) > 0:
        fig.add_trace(go.Scatter(
            x=t[included_bi], y=sig[included_bi],
            mode='markers', name=T('beats_included'),
            marker=dict(color='red', size=7, symbol='triangle-down')
        ))
    # Excluded beats — grey triangles
    if len(excluded_bi) > 0:
        fig.add_trace(go.Scatter(
            x=t[excluded_bi], y=sig[excluded_bi],
            mode='markers', name=T('beats_excluded'),
            marker=dict(color='#888888', size=7, symbol='triangle-down',
                        line=dict(width=1, color='white')),
            opacity=0.5
        ))

    fig.update_layout(
        xaxis_title=T('time_s'), yaxis_title=y_label,
        height=400, margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats (from current included beats)
    bp = result['beat_periods']
    cols = st.columns(4)
    cols[0].metric(T('n_beats_qc'), f"{len(result['beat_indices'])}")
    cols[1].metric(T('n_beats_raw'), f"{len(all_bi)}")
    if len(bp) > 0:
        cols[2].metric(T('beat_period'), f"{np.mean(bp)*1000:.0f} ms")
        cols[3].metric("BPM", f"{60/np.mean(bp):.1f}")

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
                    ch_name = ks.lstrip('_')  # e.g. '_ch1' -> 'ch1'
                    st.session_state['single_result_both'][ch_name] = updated
                else:
                    st.session_state['single_result'] = updated
                # Reset editor state so it re-syncs with new result
                st.session_state.pop(rid_key, None)
                st.session_state.pop(all_key, None)
                st.session_state.pop(excl_key, None)
                st.success(T('reanalysis_done', n=len(updated['beat_indices'])))
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  Beat overlay + template plot
# ═══════════════════════════════════════════════════════════════════════

def _plot_beats(result):
    """Plot overlaid beat waveforms and template."""
    bd = result.get('beats_data', [])
    fs = result['metadata']['sample_rate']

    if not bd:
        st.warning(T('no_params'))
        return

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
    summary = result['summary']
    cols = st.columns(4)
    cols[0].metric(T('fpd'), f"{summary.get('fpd_ms_mean', 0):.1f} ± {summary.get('fpd_ms_std', 0):.1f}")
    cols[1].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.1f} ± {summary.get('fpdc_ms_std', 0):.1f}")
    cols[2].metric(T('spike_amp'), f"{summary.get('spike_amplitude_mV_mean', 0):.4f} mV")
    cols[3].metric(T('fpd_confidence'), f"{summary.get('fpd_confidence', 0):.2f}")


# ═══════════════════════════════════════════════════════════════════════
#  Parameter table + arrhythmia display
# ═══════════════════════════════════════════════════════════════════════

def _show_params_table(result):
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
    summary = result['summary']
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


def _show_arrhythmia(result):
    """Show arrhythmia analysis results."""
    ar = result['arrhythmia_report']

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
