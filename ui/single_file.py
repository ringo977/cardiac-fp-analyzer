"""
Single File Analysis page — upload, analyze, display, export.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from cardiac_fp_analyzer.config import AnalysisConfig
from ui.cache import analyze_single_file_cached
from ui.display import plot_beats, plot_signal, show_arrhythmia, show_params_table
from ui.i18n import T

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

    # Save to temp ONCE per uploaded file (keyed by file_id), so that the
    # cache key `(tmp_path, mtime)` stays stable across Streamlit reruns.
    # Writing a fresh NamedTemporaryFile on every rerun would change the
    # path and defeat @st.cache_data.
    file_key = f'_single_file_tmp_{uploaded.file_id}'
    if file_key not in st.session_state:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            st.session_state[file_key] = tmp.name
    tmp_path = st.session_state[file_key]
    mtime = os.path.getmtime(tmp_path)
    config_json = config.to_json()

    channel = st.radio(T('channel'),
                       ['auto', 'el1', 'el2', T('both_channels')],
                       horizontal=True)

    if st.button(f"▶️ {T('analyze')}", type="primary", use_container_width=True):
        if channel == T('both_channels'):
            # Analyze both electrodes
            results_both = {}
            for ch in ['el1', 'el2']:
                with st.spinner(T('analyzing_ch', ch=ch.upper())):
                    r = analyze_single_file_cached(tmp_path, mtime, ch, config_json)
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
                result = analyze_single_file_cached(tmp_path, mtime, channel, config_json)
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
    fi = result.get('file_info', {})
    summary = result.get('summary', {})
    qc = result.get('qc_report')
    ar = result.get('arrhythmia_report')

    # ── Info cards ──
    cols = st.columns(5)
    cols[0].metric(T('chip_channel'), f"{fi.get('chip', '?')} / {fi.get('analyzed_channel', '?')}")
    cols[1].metric(T('drug'), fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade if qc else '?')
    cols[3].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.1f} ± {summary.get('fpdc_ms_std', 0):.1f}")
    cols[4].metric(T('risk_score'), f"{ar.risk_score}/100" if ar else '?')

    # ── Tabs ──
    tab_signal, tab_beats, tab_params, tab_arrhythmia = st.tabs([
        f"📈 {T('signal')}", f"💓 {T('beats')}", f"📋 {T('params')}", f"⚡ {T('arrhythmia')}"
    ])

    with tab_signal:
        plot_signal(result)

    with tab_beats:
        plot_beats(result)

    with tab_params:
        show_params_table(result)

    with tab_arrhythmia:
        show_arrhythmia(result)

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
        s = r.get('summary', {})
        qc = r.get('qc_report')
        ar = r.get('arrhythmia_report')
        # Use beat_periods from all detected beats (not QC-cleaned) for
        # accurate rhythm stats; beat count = number of QC-accepted beats.
        bp_all = r.get('beat_periods', np.array([]))
        n_beats = len(r.get('beat_indices', []))
        if len(bp_all) > 0:
            bp_str = f"{np.mean(bp_all)*1000:.0f} ± {np.std(bp_all)*1000:.0f}"
            cv_bp = np.std(bp_all) / np.mean(bp_all) * 100 if np.mean(bp_all) > 0 else 0
        else:
            bp_str = '?'
            cv_bp = 0
        comp_rows.append({
            T('channel'): ch.upper(),
            'QC Grade': qc.grade if qc else '?',
            'Beats': n_beats,
            'BP (ms)': bp_str,
            'CV BP%': f"{cv_bp:.1f}",
            'FPDcF (ms)': f"{s.get('fpdc_ms_mean', 0):.1f} ± {s.get('fpdc_ms_std', 0):.1f}",
            'Spike (mV)': f"{s.get('spike_amplitude_mV_mean', 0):.4f}",
            'Risk': f"{ar.risk_score}/100" if ar else '?',
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── Channel selector ──
    ch_labels = [ch.upper() for ch in channels]
    selected_label = st.radio(T('select_channel'), ch_labels, horizontal=True)
    selected_ch = selected_label.lower()
    result = results_both[selected_ch]

    fi = result.get('file_info', {})
    summary = result.get('summary', {})
    qc = result.get('qc_report')
    ar = result.get('arrhythmia_report')

    # ── Info cards ──
    cols = st.columns(5)
    cols[0].metric(T('chip_channel'), f"{fi.get('chip', '?')} / {selected_label}")
    cols[1].metric(T('drug'), fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade if qc else '?')
    cols[3].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.1f} ± {summary.get('fpdc_ms_std', 0):.1f}")
    cols[4].metric(T('risk_score'), f"{ar.risk_score}/100" if ar else '?')

    # ── Analysis tabs for selected channel ──
    tab_signal, tab_beats, tab_params, tab_arrhythmia = st.tabs([
        f"📈 {T('signal')}", f"💓 {T('beats')}", f"📋 {T('params')}", f"⚡ {T('arrhythmia')}"
    ])

    with tab_signal:
        plot_signal(result, key_suffix=f"_{selected_ch}")

    with tab_beats:
        plot_beats(result)

    with tab_params:
        show_params_table(result)

    with tab_arrhythmia:
        show_arrhythmia(result)

    # ── Export section (selected channel) ──
    _single_file_exports(result, config)


# ═══════════════════════════════════════════════════════════════════════
#  Exports
# ═══════════════════════════════════════════════════════════════════════

def _single_file_exports(result, config):
    """Provide download buttons for single-file analysis results."""
    st.divider()
    st.subheader(f"📥 {T('export_section')}")

    fi = result.get('file_info', {})
    summary = result.get('summary', {})
    qc = result.get('qc_report')
    ar = result.get('arrhythmia_report')
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
            'QC_Grade': qc.grade if qc else '?',
            'Beats_accepted': summary.get('beat_period_ms_n', 0),
            'Mean_BP_ms': summary.get('beat_period_ms_mean', np.nan),
            'CV_BP_pct': summary.get('beat_period_ms_cv', np.nan),
            'Mean_FPD_ms': summary.get('fpd_ms_mean', np.nan),
            'SD_FPD_ms': summary.get('fpd_ms_std', np.nan),
            'Mean_FPDcF_ms': summary.get('fpdc_ms_mean', np.nan),
            'SD_FPDcF_ms': summary.get('fpdc_ms_std', np.nan),
            'STV_FPDcF_ms': summary.get('stv_fpdc_ms', np.nan),
            'Mean_spike_mV': summary.get('spike_amplitude_mV_mean', np.nan),
            'Risk_score': ar.risk_score if ar else np.nan,
            'Classification': ar.classification if ar else '?',
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
                except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
                    st.error(f"{T('error')}: {e}")
