"""
Batch Analysis + Risk Map page.
"""

import io
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cardiac_fp_analyzer.analyze import batch_analyze
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.normalization import is_baseline
from ui.display import plot_beats, plot_signal, show_arrhythmia, show_params_table
from ui.helpers import amplitude_scale
from ui.i18n import T
from ui.reports import download_reports

# ═══════════════════════════════════════════════════════════════════════
#  Page entry point
# ═══════════════════════════════════════════════════════════════════════

def page_batch_analysis(config: AnalysisConfig):
    st.header(f"📊 {T('batch')}")
    st.caption(T('batch_desc'))

    upload_mode = st.radio(
        "Modalità upload",
        ["Seleziona cartella", "Upload file CSV", "Upload archivio ZIP"],
        horizontal=True
    )

    csv_files = []
    tmp_dir = None
    data_path = None

    if upload_mode == "Upload file CSV":
        uploaded_files = st.file_uploader(
            "Carica file CSV", type=['csv'],
            accept_multiple_files=True, key='batch_files'
        )
        if uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                dest = Path(tmp_dir) / uf.name
                dest.write_bytes(uf.read())
                csv_files.append(dest)
            st.info(f"Caricati **{len(csv_files)}** file CSV")

    elif upload_mode == "Upload archivio ZIP":
        uploaded_zip = st.file_uploader(
            "Carica un archivio ZIP contenente i CSV",
            type=['zip'], key='batch_zip'
        )
        if uploaded_zip:
            tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
                zf.extractall(tmp_dir)
            csv_files = sorted(Path(tmp_dir).rglob('*.csv'))
            st.info(f"Estratti **{len(csv_files)}** file CSV dall'archivio")

    elif upload_mode == "Seleziona cartella":
        if st.button("📂 Apri selezione cartella", type="primary", use_container_width=True):
            import platform
            import subprocess
            chosen = None
            try:
                os_name = platform.system()
                if os_name == 'Darwin':
                    proc = subprocess.run(
                        ['osascript', '-e',
                         'POSIX path of (choose folder with prompt '
                         '"Seleziona cartella dati CSV")'],
                        capture_output=True, text=True, timeout=120
                    )
                    chosen = proc.stdout.strip().rstrip('/')
                elif os_name == 'Linux':
                    for cmd in [
                        ['zenity', '--file-selection', '--directory',
                         '--title=Seleziona cartella dati CSV'],
                        ['kdialog', '--getexistingdirectory', '.'],
                    ]:
                        try:
                            proc = subprocess.run(
                                cmd, capture_output=True, text=True, timeout=120
                            )
                            if proc.returncode == 0:
                                chosen = proc.stdout.strip()
                                break
                        except FileNotFoundError:
                            continue
                elif os_name == 'Windows':
                    ps_script = (
                        'Add-Type -AssemblyName System.Windows.Forms; '
                        '$f = New-Object System.Windows.Forms.FolderBrowserDialog; '
                        '$f.Description = "Seleziona cartella dati CSV"; '
                        'if ($f.ShowDialog() -eq "OK") { $f.SelectedPath }'
                    )
                    proc = subprocess.run(
                        ['powershell', '-Command', ps_script],
                        capture_output=True, text=True, timeout=120
                    )
                    chosen = proc.stdout.strip()

                if chosen and Path(chosen).is_dir():
                    st.session_state['batch_local_path'] = chosen
                elif not chosen:
                    st.info("Nessuna cartella selezionata.")
            except (OSError, subprocess.SubprocessError, FileNotFoundError) as e:
                st.error(f"{T('error')} nell'aprire il file picker: {e}")

        # Show selected path and scan
        if 'batch_local_path' in st.session_state:
            data_path = st.session_state['batch_local_path']
            csv_files = sorted(Path(data_path).rglob('*.csv'))
            st.info(f"📁 `{data_path}` — **{len(csv_files)}** file CSV trovati")
            if st.button("❌ Cambia cartella", use_container_width=False):
                st.session_state.pop('batch_local_path', None)
                st.rerun()

    if not csv_files:
        return

    # Ground truth (optional)
    with st.expander("🏷️ Ground truth farmaci (opzionale)", expanded=False):
        st.caption("Per colorare la risk map con i dati noti. Formato: farmaco=+/−")
        gt_text = st.text_area(
            "Ground truth",
            value="terfenadine=+\nquinidine=+\ndofetilide=+\nalfuzosin=-\nmexiletine=-\nnifedipine=-\nranolazine=-",
            height=150
        )
        ground_truth = {}
        for line in gt_text.strip().split('\n'):
            if '=' in line:
                drug, val = line.split('=', 1)
                ground_truth[drug.strip().lower()] = val.strip() == '+'

    channel = st.radio(T('channel'),
                       ['auto', 'el1', 'el2', T('both_channels')],
                       horizontal=True, key='batch_ch')

    # Map translated label back to engine value
    _ch_map = {T('both_channels'): 'both'}
    channel_engine = _ch_map.get(channel, channel)

    if st.button("🚀 Avvia Analisi Batch", type="primary", use_container_width=True):
        data_dir = data_path if upload_mode == "Seleziona cartella" else tmp_dir

        progress_bar = st.progress(0, text="Avvio analisi...")
        status_text = st.empty()

        with st.spinner("Pipeline in esecuzione..."):
            try:
                results = batch_analyze(
                    data_dir, channel=channel_engine, verbose=False, config=config
                )
                st.session_state['batch_results'] = results
                st.session_state['batch_ground_truth'] = ground_truth
                progress_bar.progress(100, text="Completato!")
                status_text.success(f"Analisi completata: {len(results)} registrazioni processate.")
            except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as e:
                st.error(f"{T('error')}: {e}")
                st.code(traceback.format_exc())
                return

    # ── Display batch results ──
    results = st.session_state.get('batch_results')
    if results is None:
        return

    ground_truth = st.session_state.get('batch_ground_truth', {})

    tab_riskmap, tab_summary, tab_details = st.tabs([
        "🗺️ Risk Map", "📋 Riepilogo", "📄 Dettagli"
    ])

    with tab_riskmap:
        _show_risk_map(results, config, ground_truth)

    with tab_summary:
        _show_batch_summary(results)

    with tab_details:
        _show_batch_details(results)

    # ── Download reports ──
    st.divider()
    download_reports(results, config, data_path if upload_mode == "Seleziona cartella" else "uploaded_data")


# ═══════════════════════════════════════════════════════════════════════
#  Risk map
# ═══════════════════════════════════════════════════════════════════════

def _show_risk_map(results, config, ground_truth):
    """Generate and display interactive risk map."""
    from cardiac_fp_analyzer.risk_map import (
        RiskZoneConfig,
        aggregate_drug_metrics,
        compute_proarrhythmic_index,
    )

    metrics = aggregate_drug_metrics(results)
    if not metrics:
        st.warning("Nessun dato farmaco disponibile per la risk map.")
        return

    zone_cfg = RiskZoneConfig()

    # Build data
    rows = []
    for drug, m in sorted(metrics.items()):
        x = m.max_pct_fpdc_change if not np.isnan(m.max_pct_fpdc_change) else 0.0
        y = compute_proarrhythmic_index(m)
        gt = ground_truth.get(drug)
        label = "hERG+" if gt is True else ("hERG−" if gt is False else "?")
        if y >= zone_cfg.proarrh_mid_high:
            zone = "HIGH"
        elif y >= zone_cfg.proarrh_low_mid:
            zone = "INTERMEDIATE"
        else:
            zone = "LOW"
        rows.append({
            'drug': drug.capitalize(),
            'drug_raw': drug,
            'x': x, 'y': y,
            'gt': label, 'zone': zone,
            'spectral': m.max_spectral_change,
            'morph': m.max_morphology_instability,
            'ead_pct': m.max_ead_incidence_pct,
            'n_conc': m.n_concentrations,
            'cessation': m.has_cessation,
        })

    df = pd.DataFrame(rows)

    fig = go.Figure()

    # Zone backgrounds
    x_min = min(df['x'].min() - 10, -15)
    x_max = max(df['x'].max() + 10, 35)
    y_max = max(df['y'].max() + 10, 55)

    fig.add_shape(type="rect", x0=x_min, x1=x_max, y0=0, y1=zone_cfg.proarrh_low_mid,
                  fillcolor="rgba(40,167,69,0.12)", line_width=0)
    fig.add_shape(type="rect", x0=x_min, x1=x_max, y0=zone_cfg.proarrh_low_mid, y1=zone_cfg.proarrh_mid_high,
                  fillcolor="rgba(255,193,7,0.12)", line_width=0)
    fig.add_shape(type="rect", x0=x_min, x1=x_max, y0=zone_cfg.proarrh_mid_high, y1=y_max,
                  fillcolor="rgba(220,53,69,0.12)", line_width=0)

    # Threshold lines
    for xv in [zone_cfg.fpdc_low_mid, zone_cfg.fpdc_mid_high]:
        fig.add_vline(x=xv, line_dash="dash", line_color="gray", opacity=0.5)
    for yv in [zone_cfg.proarrh_low_mid, zone_cfg.proarrh_mid_high]:
        fig.add_hline(y=yv, line_dash="dash", line_color="gray", opacity=0.5)

    # Scatter by ground truth
    color_map = {"hERG+": "#dc3545", "hERG−": "#0d6efd", "?": "#6c757d"}
    symbol_map = {"hERG+": "diamond", "hERG−": "circle", "?": "square"}

    for gt_label in ["hERG+", "hERG−", "?"]:
        sub = df[df['gt'] == gt_label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['x'], y=sub['y'],
            mode='markers+text',
            name=gt_label,
            marker=dict(
                color=color_map[gt_label],
                size=14,
                symbol=symbol_map[gt_label],
                line=dict(color='white', width=1.5)
            ),
            text=sub['drug'],
            textposition='top right',
            textfont=dict(size=11, color='#333'),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "ΔFPDcF: %{x:.1f}%<br>"
                "Proarrhythmic Index: %{y:.1f}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        xaxis_title="Max ΔFPDcF (%)",
        yaxis_title="Indice Proaritmico (0–100)",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[-2, y_max]),
        height=550,
        margin=dict(t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Metriche per farmaco")
    display_df = df[['drug', 'gt', 'x', 'y', 'spectral', 'morph', 'ead_pct', 'zone', 'n_conc']].copy()
    display_df.columns = ['Farmaco', 'Classe', 'ΔFPDcF%', 'Indice', 'Spectral', 'Morph Inst', 'EAD%', 'Zona', 'N conc.']
    st.dataframe(
        display_df.style.format({
            'ΔFPDcF%': '{:.1f}', 'Indice': '{:.1f}',
            'Spectral': '{:.3f}', 'Morph Inst': '{:.3f}', 'EAD%': '{:.1f}'
        }),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════════════════════
#  Summary + Details
# ═══════════════════════════════════════════════════════════════════════

def _show_batch_summary(results):
    """Show summary table of all recordings."""
    rows = []
    for r in results:
        fi = r.get('file_info', {})
        s = r.get('summary', {})
        qc = r.get('qc_report')
        ar = r.get('arrhythmia_report')
        inc = r.get('inclusion', {})
        norm = r.get('normalization', {})

        rows.append({
            'File': r.get('metadata', {}).get('filename', ''),
            'Chip': fi.get('chip', ''),
            T('channel'): fi.get('analyzed_channel', ''),
            'Farmaco': fi.get('drug', ''),
            'Baseline': '✓' if is_baseline(r) else '',
            'QC': qc.grade if qc else '',
            'Inclusione': '✓' if inc.get('passed', True) else '✗',
            'BP (ms)': f"{np.mean(r.get('beat_periods', [np.nan]))*1000:.0f}",
            'FPDcF (ms)': f"{s.get('fpdc_ms_mean', np.nan):.1f}",
            'Conf FPD': f"{s.get('fpd_confidence', np.nan):.2f}",
            'ΔFPDcF%': f"{norm.get('pct_fpdc_change', np.nan):.1f}" if norm.get('has_baseline') else '—',
            'TdP Score': norm.get('tdp_score', '—'),
            'Risk': ar.risk_score if ar and not is_baseline(r) else '—',
            'Class.': ar.classification[:20] if ar and not is_baseline(r) else '—',
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=600)
    st.caption(f"Totale: {len(results)} registrazioni")


def _show_batch_details(results):
    """Show details for selected file — signal, beats, params, arrhythmia."""
    labels = []
    for i, r in enumerate(results):
        fname = r.get('metadata', {}).get('filename', f'file_{i}')
        ch = r.get('file_info', {}).get('analyzed_channel', '')
        labels.append(f"{fname} [{ch}]" if ch else fname)
    selected = st.selectbox(T('select_recording'), labels)

    idx = labels.index(selected)
    result = results[idx]

    fi = result.get('file_info', {})
    summary = result.get('summary', {})
    qc = result.get('qc_report')
    ar = result.get('arrhythmia_report')

    cols = st.columns(5)
    cols[0].metric(T('chip_channel'), f"{fi.get('chip', '?')} / {fi.get('analyzed_channel', '?')}")
    cols[1].metric(T('drug'), fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade if qc else '?')
    cols[3].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.1f}")
    cols[4].metric(T('risk_score'), f"{ar.risk_score}/100" if ar else '?')

    tab_sig, tab_params, tab_arr = st.tabs([
        f"📈 {T('signal')}", f"📋 {T('params')}", f"⚠️ {T('arrhythmia')}"
    ])

    with tab_sig:
        # Signal plot (if filtered_signal available)
        if 'filtered_signal' in result and 'time_vector' in result:
            plot_signal(result)
        else:
            st.info(T('signal_not_available'))

        # Beat overlay (using beat_template if beats_data was freed)
        tmpl = result.get('beat_template')
        if tmpl is not None and 'beats_data' not in result:
            fs = result.get('metadata', {}).get('sample_rate', 2000)
            scale, y_label = amplitude_scale(tmpl)
            fig = go.Figure()
            t_ms = np.arange(len(tmpl)) / fs * 1000
            fig.add_trace(go.Scatter(
                x=t_ms, y=tmpl * scale,
                mode='lines', name=T('template'),
                line=dict(color='#dc3545', width=2.5)
            ))
            fig.update_layout(
                xaxis_title=T('time_ms'), yaxis_title=y_label,
                height=350, margin=dict(t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        elif 'beats_data' in result:
            plot_beats(result)

    with tab_params:
        show_params_table(result)

    with tab_arr:
        if ar:
            show_arrhythmia(result)
