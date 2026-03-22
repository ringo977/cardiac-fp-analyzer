#!/usr/bin/env python3
"""
Cardiac FP Analyzer — Streamlit GUI

Launch:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile, shutil, io, zipfile, traceback, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Must be first Streamlit call ──
st.set_page_config(
    page_title="Cardiac FP Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from the analysis package ──
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.loader import load_csv, parse_filename
from cardiac_fp_analyzer.filtering import full_filter_pipeline
from cardiac_fp_analyzer.beat_detection import detect_beats, segment_beats, compute_beat_periods
from cardiac_fp_analyzer.parameters import extract_all_parameters
from cardiac_fp_analyzer.quality_control import validate_beats
from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia, _compute_template
from cardiac_fp_analyzer.analyze import analyze_single_file, batch_analyze
from cardiac_fp_analyzer.normalization import _is_baseline, _get_group_key


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Configuration
# ═══════════════════════════════════════════════════════════════════════

def build_config_from_sidebar() -> AnalysisConfig:
    """Build AnalysisConfig from sidebar widgets. Returns config object."""
    config = AnalysisConfig()

    with st.sidebar:
        st.header("⚙️ Configurazione")

        # ── Signal ──
        with st.expander("📡 Segnale", expanded=False):
            config.amplifier_gain = st.number_input(
                "Guadagno amplificatore",
                value=1e4, min_value=1.0, format="%.0e",
                help="Divisore per ottenere il voltaggio reale. µECG-Pharma Digilent = 10⁴"
            )
            config.filtering.notch_freq_hz = st.selectbox(
                "Frequenza di rete (Hz)", [50.0, 60.0], index=0
            )
            config.filtering.bandpass_low_hz = st.number_input(
                "Passa-banda basso (Hz)", value=0.5, min_value=0.1, max_value=10.0, step=0.1
            )
            config.filtering.bandpass_high_hz = st.number_input(
                "Passa-banda alto (Hz)", value=500.0, min_value=50.0, max_value=5000.0, step=50.0
            )

        # ── Beat detection ──
        with st.expander("💓 Beat Detection", expanded=False):
            config.beat_detection.method = st.selectbox(
                "Metodo", ['auto', 'prominence', 'derivative', 'peak'],
                help="'auto' sceglie il metodo migliore automaticamente"
            )
            config.beat_detection.min_distance_ms = st.number_input(
                "Distanza min tra battiti (ms)", value=400.0, min_value=100.0, max_value=2000.0
            )
            config.beat_detection.threshold_factor = st.number_input(
                "Soglia adattiva (×)", value=4.0, min_value=1.0, max_value=10.0, step=0.5
            )

        # ── FPD ──
        with st.expander("📏 FPD / Ripolarizzazione", expanded=False):
            config.repolarization.fpd_method = st.selectbox(
                "Metodo FPD",
                ['tangent', 'peak', 'max_slope', '50pct', 'baseline_return', 'consensus'],
                help="'tangent' è lo standard in letteratura"
            )
            config.repolarization.correction = st.selectbox(
                "Correzione QT", ['fridericia', 'bazett', 'none'],
                help="Fridericia raccomandata per hiPSC-CM"
            )
            config.repolarization.search_start_ms = st.number_input(
                "Inizio ricerca ripol. (ms)", value=150.0, min_value=50.0, max_value=400.0
            )
            config.repolarization.search_end_ms = st.number_input(
                "Fine ricerca ripol. (ms)", value=900.0, min_value=400.0, max_value=1500.0
            )

        # ── Arrhythmia ──
        with st.expander("⚡ Aritmie / EAD", expanded=False):
            config.arrhythmia.ead_residual_prominence = st.number_input(
                "EAD prominenza (×σ)", value=6.0, min_value=2.0, max_value=15.0, step=0.5
            )
            config.arrhythmia.ead_residual_min_amp_frac = st.number_input(
                "EAD ampiezza min (% template)", value=0.08, min_value=0.01, max_value=0.50, step=0.01
            )
            config.arrhythmia.ead_residual_min_width_ms = st.number_input(
                "EAD larghezza min (ms)", value=8.0, min_value=1.0, max_value=50.0
            )
            config.arrhythmia.ead_residual_max_width_ms = st.number_input(
                "EAD larghezza max (ms)", value=150.0, min_value=50.0, max_value=500.0
            )

        # ── Inclusion ──
        with st.expander("🔍 Criteri di inclusione", expanded=False):
            config.inclusion.max_cv_bp = st.number_input(
                "Max CV% beat period baseline", value=25.0, min_value=5.0, max_value=50.0
            )
            config.inclusion.min_fpd_confidence = st.number_input(
                "Min confidenza FPD", value=0.66, min_value=0.0, max_value=1.0, step=0.01
            )
            config.inclusion.enabled_fpdc_physiol = st.checkbox(
                "Filtro range fisiologico FPDcF", value=True
            )

        # ── Normalization ──
        with st.expander("📊 Normalizzazione", expanded=False):
            config.normalization.threshold_low = st.number_input(
                "Soglia LOW (%ΔFPDcF)", value=10.0, min_value=1.0, max_value=30.0
            )
            config.normalization.threshold_mid = st.number_input(
                "Soglia MID (%ΔFPDcF)", value=15.0, min_value=5.0, max_value=40.0
            )
            config.normalization.threshold_high = st.number_input(
                "Soglia HIGH (%ΔFPDcF)", value=20.0, min_value=10.0, max_value=50.0
            )

        # ── Config import/export ──
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            json_str = config.to_json()
            st.download_button("📥 Esporta config", json_str, "analysis_config.json",
                               mime="application/json", use_container_width=True)
        with col2:
            uploaded_cfg = st.file_uploader("📤 Importa", type=['json'], key='cfg_upload',
                                            label_visibility="collapsed")
            if uploaded_cfg is not None:
                try:
                    cfg_dict = __import__('json').loads(uploaded_cfg.read())
                    config = AnalysisConfig.from_dict(cfg_dict)
                    st.success("Config caricata!")
                except Exception as e:
                    st.error(f"Errore: {e}")

    return config


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: Single File Analysis
# ═══════════════════════════════════════════════════════════════════════

def page_single_file(config: AnalysisConfig):
    st.header("🔬 Analisi Singolo File")
    st.caption("Upload di un file CSV per visualizzare segnale, battiti, parametri e aritmie.")

    uploaded = st.file_uploader("Scegli un file CSV", type=['csv'], key='single_file')
    if uploaded is None:
        st.info("Carica un file CSV dal sistema µECG-Pharma per iniziare.")
        return

    # Save to temp
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    channel = st.radio("Canale", ['auto', 'ch1', 'ch2'], horizontal=True)

    if st.button("▶️ Analizza", type="primary", use_container_width=True):
        with st.spinner("Analisi in corso..."):
            result = analyze_single_file(tmp_path, channel=channel, verbose=False, config=config)

        if result is None:
            st.error("Errore nell'analisi. Verifica il formato del file.")
            return

        st.success(f"Analisi completata — Canale: {result['file_info'].get('analyzed_channel', '?')}")

        # Store in session
        st.session_state['single_result'] = result

    # ── Display results ──
    result = st.session_state.get('single_result')
    if result is None:
        return

    fi = result['file_info']
    summary = result['summary']
    qc = result['qc_report']
    ar = result['arrhythmia_report']

    # ── Info cards ──
    cols = st.columns(5)
    cols[0].metric("Chip/Canale", f"{fi.get('chip', '?')} / {fi.get('analyzed_channel', '?')}")
    cols[1].metric("Farmaco", fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade)
    cols[3].metric("FPDcF (ms)", f"{summary.get('fpdc_ms_mean', 0):.0f} ± {summary.get('fpdc_ms_std', 0):.0f}")
    cols[4].metric("Risk Score", f"{ar.risk_score}/100")

    # ── Tabs ──
    tab_signal, tab_beats, tab_params, tab_arrhythmia = st.tabs([
        "📈 Segnale", "💓 Battiti", "📋 Parametri", "⚡ Aritmie"
    ])

    with tab_signal:
        _plot_signal(result)

    with tab_beats:
        _plot_beats(result)

    with tab_params:
        _show_params_table(result)

    with tab_arrhythmia:
        _show_arrhythmia(result)


def _plot_signal(result):
    """Plot filtered signal with beat markers."""
    import plotly.graph_objects as go

    sig = result['filtered_signal']
    t = result['time_vector']
    bi = result['beat_indices']
    fs = result['metadata']['sample_rate']

    fig = go.Figure()
    # Downsample for display if very long
    step = max(1, len(sig) // 50000)
    fig.add_trace(go.Scatter(
        x=t[::step], y=sig[::step],
        mode='lines', name='Segnale filtrato',
        line=dict(color='#1f77b4', width=0.8)
    ))
    # Beat markers
    if len(bi) > 0:
        fig.add_trace(go.Scatter(
            x=t[bi], y=sig[bi],
            mode='markers', name='Battiti',
            marker=dict(color='red', size=6, symbol='triangle-down')
        ))
    fig.update_layout(
        xaxis_title="Tempo (s)", yaxis_title="Ampiezza",
        height=400, margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    cols = st.columns(4)
    bp = result['beat_periods']
    cols[0].metric("N. battiti (QC)", f"{len(result['beat_indices'])}")
    cols[1].metric("N. battiti (raw)", f"{len(result['beat_indices_raw'])}")
    if len(bp) > 0:
        cols[2].metric("Beat Period", f"{np.mean(bp)*1000:.0f} ms")
        cols[3].metric("BPM", f"{60/np.mean(bp):.1f}")


def _plot_beats(result):
    """Plot overlaid beat waveforms and template."""
    import plotly.graph_objects as go

    bd = result.get('beats_data', [])
    fs = result['metadata']['sample_rate']

    if not bd:
        st.warning("Nessun battito disponibile.")
        return

    fig = go.Figure()
    # Plot up to 30 beats
    n_plot = min(30, len(bd))
    indices = np.linspace(0, len(bd)-1, n_plot, dtype=int)
    for i in indices:
        t_beat = np.arange(len(bd[i])) / fs * 1000  # ms
        fig.add_trace(go.Scatter(
            x=t_beat, y=bd[i],
            mode='lines', line=dict(color='rgba(100,149,237,0.3)', width=0.8),
            showlegend=False, hoverinfo='skip'
        ))

    # Template
    tmpl = _compute_template(bd)
    if tmpl is not None:
        t_tmpl = np.arange(len(tmpl)) / fs * 1000
        fig.add_trace(go.Scatter(
            x=t_tmpl, y=tmpl,
            mode='lines', name='Template',
            line=dict(color='red', width=2.5)
        ))

    fig.update_layout(
        xaxis_title="Tempo (ms)", yaxis_title="Ampiezza",
        height=400, margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # FPD annotation
    summary = result['summary']
    cols = st.columns(4)
    cols[0].metric("FPD (ms)", f"{summary.get('fpd_ms_mean', 0):.1f} ± {summary.get('fpd_ms_std', 0):.1f}")
    cols[1].metric("FPDcF (ms)", f"{summary.get('fpdc_ms_mean', 0):.1f} ± {summary.get('fpdc_ms_std', 0):.1f}")
    cols[2].metric("Ampiezza spike", f"{summary.get('spike_amplitude_mV_mean', 0):.4f}")
    cols[3].metric("Confidenza FPD", f"{summary.get('fpd_confidence', 0):.2f}")


def _show_params_table(result):
    """Show per-beat parameter table."""
    all_p = result.get('all_params', [])
    if not all_p:
        st.warning("Nessun parametro disponibile.")
        return

    rows = []
    for p in all_p:
        rows.append({
            'Beat': p.get('beat_number', ''),
            'RR (ms)': f"{p.get('rr_interval_ms', np.nan):.1f}",
            'Spike Amp': f"{p.get('spike_amplitude_mV', np.nan):.5f}",
            'FPD (ms)': f"{p.get('fpd_ms', np.nan):.1f}",
            'FPDcF (ms)': f"{p.get('fpdc_ms', np.nan):.1f}",
            'Rise time (ms)': f"{p.get('rise_time_ms', np.nan):.2f}",
            'Max dV/dt': f"{p.get('max_dvdt', np.nan):.5f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)

    # Summary stats
    summary = result['summary']
    st.subheader("Riepilogo")
    summary_rows = []
    for key in ['spike_amplitude_mV', 'fpd_ms', 'fpdc_ms', 'rise_time_ms', 'rr_interval_ms']:
        m = summary.get(f'{key}_mean', np.nan)
        s = summary.get(f'{key}_std', np.nan)
        cv = summary.get(f'{key}_cv', np.nan)
        summary_rows.append({
            'Parametro': key.replace('_mV', ' (mV)').replace('_ms', ' (ms)'),
            'Media': f"{m:.3f}" if not np.isnan(m) else "—",
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


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: Batch Analysis + Risk Map
# ═══════════════════════════════════════════════════════════════════════

def page_batch_analysis(config: AnalysisConfig):
    st.header("📊 Analisi Batch + Risk Map")
    st.caption("Upload di più file CSV per analisi batch, normalizzazione baseline e risk map CiPA.")

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
            import subprocess, platform
            chosen = None
            try:
                os_name = platform.system()
                if os_name == 'Darwin':
                    # macOS: native AppleScript dialog (no Python icon)
                    proc = subprocess.run(
                        ['osascript', '-e',
                         'POSIX path of (choose folder with prompt '
                         '"Seleziona cartella dati CSV")'],
                        capture_output=True, text=True, timeout=120
                    )
                    chosen = proc.stdout.strip().rstrip('/')
                elif os_name == 'Linux':
                    # Linux: try zenity (GNOME), then kdialog (KDE)
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
                    # Windows: PowerShell native folder dialog
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
            except Exception as e:
                st.error(f"Errore nell'aprire il file picker: {e}")

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

    channel = st.radio("Canale", ['auto', 'ch1', 'ch2'], horizontal=True, key='batch_ch')

    if st.button("🚀 Avvia Analisi Batch", type="primary", use_container_width=True):
        data_dir = data_path if upload_mode == "Seleziona cartella" else tmp_dir

        progress_bar = st.progress(0, text="Avvio analisi...")
        status_text = st.empty()

        with st.spinner("Pipeline in esecuzione..."):
            try:
                results = batch_analyze(
                    data_dir, channel=channel, verbose=False, config=config
                )
                st.session_state['batch_results'] = results
                st.session_state['batch_ground_truth'] = ground_truth
                progress_bar.progress(100, text="Completato!")
                status_text.success(f"Analisi completata: {len(results)} registrazioni processate.")
            except Exception as e:
                st.error(f"Errore: {e}")
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
    _download_reports(results, config, data_path if upload_mode == "Seleziona cartella" else "uploaded_data")


def _show_risk_map(results, config, ground_truth):
    """Generate and display interactive risk map."""
    from cardiac_fp_analyzer.risk_map import (
        aggregate_drug_metrics, compute_proarrhythmic_index, RiskZoneConfig
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

    # Plotly scatter
    import plotly.graph_objects as go

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
            'Farmaco': fi.get('drug', ''),
            'Baseline': '✓' if _is_baseline(r) else '',
            'QC': qc.grade if qc else '',
            'Inclusione': '✓' if inc.get('passed', True) else '✗',
            'BP (ms)': f"{s.get('beat_period_ms_mean', np.nan):.0f}",
            'FPDcF (ms)': f"{s.get('fpdc_ms_mean', np.nan):.0f}",
            'Conf FPD': f"{s.get('fpd_confidence', np.nan):.2f}",
            'ΔFPDcF%': f"{norm.get('pct_fpdc_change', np.nan):.1f}" if norm.get('has_baseline') else '—',
            'TdP Score': norm.get('tdp_score', '—'),
            'Risk': ar.risk_score if ar and not _is_baseline(r) else '—',
            'Class.': ar.classification[:20] if ar and not _is_baseline(r) else '—',
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=600)
    st.caption(f"Totale: {len(results)} registrazioni")


def _show_batch_details(results):
    """Show details for selected file — signal, beats, params, arrhythmia."""
    filenames = [r.get('metadata', {}).get('filename', f'file_{i}') for i, r in enumerate(results)]
    selected = st.selectbox("Seleziona registrazione", filenames)

    idx = filenames.index(selected)
    result = results[idx]

    fi = result['file_info']
    summary = result['summary']
    qc = result['qc_report']
    ar = result['arrhythmia_report']

    cols = st.columns(5)
    cols[0].metric("Chip/Canale", f"{fi.get('chip', '?')} / {fi.get('analyzed_channel', '?')}")
    cols[1].metric("Farmaco", fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade if qc else '?')
    cols[3].metric("FPDcF (ms)", f"{summary.get('fpdc_ms_mean', 0):.0f}")
    cols[4].metric("Risk Score", f"{ar.risk_score}/100" if ar else '?')

    tab_sig, tab_params, tab_arr = st.tabs([
        "📈 Segnale", "📋 Parametri", "⚠️ Aritmie"
    ])

    with tab_sig:
        # Signal plot (if filtered_signal available)
        if 'filtered_signal' in result and 'time_vector' in result:
            _plot_signal(result)
        else:
            st.info("Segnale filtrato non disponibile in modalità batch.")

        # Beat overlay (using beat_template if beats_data was freed)
        tmpl = result.get('beat_template')
        if tmpl is not None and 'beats_data' not in result:
            import plotly.graph_objects as go
            fs = result['metadata']['sample_rate']
            fig = go.Figure()
            t_ms = np.arange(len(tmpl)) / fs * 1000
            fig.add_trace(go.Scatter(
                x=t_ms, y=tmpl,
                mode='lines', name='Template',
                line=dict(color='#dc3545', width=2.5)
            ))
            fig.update_layout(
                xaxis_title="Tempo (ms)", yaxis_title="Ampiezza",
                height=350, margin=dict(t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        elif 'beats_data' in result:
            _plot_beats(result)

    with tab_params:
        _show_params_table(result)

    with tab_arr:
        if ar:
            _show_arrhythmia(result)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: Drug Comparison
# ═══════════════════════════════════════════════════════════════════════

def page_drug_comparison(config: AnalysisConfig):
    st.header("💊 Confronto Farmaci")
    st.caption("Dashboard comparativa: dose-response, metriche aritmiche, overlay waveform.")

    results = st.session_state.get('batch_results')
    if results is None:
        st.info("Esegui prima un'analisi batch per visualizzare i confronti tra farmaci.")
        return

    from cardiac_fp_analyzer.risk_map import _canonical_drug

    # Group by drug
    drug_data = {}
    for r in results:
        if _is_baseline(r):
            continue
        fi = r.get('file_info', {})
        raw_drug = str(fi.get('drug', '')).lower()
        if not raw_drug or 'wash' in raw_drug:
            continue
        drug = _canonical_drug(raw_drug)
        if drug.startswith('ctr') or drug.startswith('ctrl'):
            continue
        drug_data.setdefault(drug, []).append(r)

    if not drug_data:
        st.warning("Nessun dato farmaco disponibile.")
        return

    drugs = sorted(drug_data.keys())
    selected_drugs = st.multiselect("Seleziona farmaci da confrontare", drugs, default=drugs[:4])

    if not selected_drugs:
        return

    tab_dose, tab_metrics, tab_waveform = st.tabs([
        "📈 Dose-Response", "📊 Metriche Aritmiche", "🔬 Waveform"
    ])

    with tab_dose:
        _plot_dose_response(drug_data, selected_drugs)

    with tab_metrics:
        _plot_arrhythmia_metrics(drug_data, selected_drugs)

    with tab_waveform:
        _plot_waveform_comparison(drug_data, selected_drugs)


def _plot_dose_response(drug_data, selected_drugs):
    """Plot dose-response curves for FPDcF change."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ['#dc3545', '#0d6efd', '#28a745', '#fd7e14', '#6f42c1', '#20c997', '#e83e8c']

    for i, drug in enumerate(selected_drugs):
        recs = drug_data.get(drug, [])
        points = []
        for r in recs:
            norm = r.get('normalization', {})
            pct = norm.get('pct_fpdc_change', np.nan)
            fi = r.get('file_info', {})
            conc = fi.get('concentration', '?')
            if not np.isnan(pct):
                points.append((conc, pct))

        if points:
            import re as _re_sort
            def _conc_sort_key(item):
                """Sort concentrations numerically when possible."""
                c = str(item[0])
                try:
                    return float(_re_sort.sub(r'[^\d.]', '', c.split()[0]))
                except (ValueError, IndexError):
                    return float('inf')
            points.sort(key=_conc_sort_key)
            labels, values = zip(*points)
            fig.add_trace(go.Scatter(
                x=list(range(len(values))), y=list(values),
                mode='markers+lines',
                name=drug.capitalize(),
                marker=dict(size=10, color=colors[i % len(colors)]),
                line=dict(color=colors[i % len(colors)]),
                text=list(labels),
                hovertemplate="<b>%{text}</b><br>ΔFPDcF: %{y:.1f}%<extra></extra>"
            ))

    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
    fig.add_hline(y=10, line_dash="dash", line_color="orange", opacity=0.5,
                  annotation_text="LOW (10%)")
    fig.add_hline(y=20, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="HIGH (20%)")

    fig.update_layout(
        xaxis_title="Concentrazione (ordine crescente)",
        yaxis_title="ΔFPDcF (%)",
        height=450, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_arrhythmia_metrics(drug_data, selected_drugs):
    """Bar chart of arrhythmia metrics per drug."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics_list = []
    for drug in selected_drugs:
        recs = drug_data.get(drug, [])
        morph_vals, ead_vals, stv_vals, spec_vals = [], [], [], []
        for r in recs:
            ar = r.get('arrhythmia_report')
            if ar and ar.residual_details:
                rd = ar.residual_details
                morph_vals.append(rd.get('morphology_instability', 0))
                ead_vals.append(rd.get('ead_incidence_pct', 0))
                stv_vals.append(rd.get('poincare_stv_fpdc_ms', np.nan))
            norm = r.get('normalization', {})
            spec = norm.get('spectral_change_score', np.nan)
            if not np.isnan(spec):
                spec_vals.append(spec)

        metrics_list.append({
            'drug': drug.capitalize(),
            'morph': np.nanmean(morph_vals) if morph_vals else 0,
            'ead': np.nanmean(ead_vals) if ead_vals else 0,
            'stv': np.nanmean(stv_vals) if stv_vals else 0,
            'spectral': np.nanmean(spec_vals) if spec_vals else 0,
        })

    df = pd.DataFrame(metrics_list)

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Morphology Instability", "EAD Incidence (%)",
        "STV FPDcF (ms)", "Spectral Change"
    ])

    colors = ['#dc3545' if d in ['terfenadine', 'quinidine', 'dofetilide']
              else '#0d6efd' for d in [m['drug'].lower() for m in metrics_list]]

    fig.add_trace(go.Bar(x=df['drug'], y=df['morph'], marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=df['drug'], y=df['ead'], marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=df['drug'], y=df['stv'], marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=df['drug'], y=df['spectral'], marker_color=colors, showlegend=False), row=2, col=2)

    fig.update_layout(height=600, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)


def _plot_waveform_comparison(drug_data, selected_drugs):
    """Overlay template waveforms from different drugs."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ['#dc3545', '#0d6efd', '#28a745', '#fd7e14', '#6f42c1', '#20c997']

    for i, drug in enumerate(selected_drugs):
        recs = drug_data.get(drug, [])
        # Pick one representative recording (highest concentration with good data)
        best_rec = None
        for r in sorted(recs, key=lambda r: r.get('summary', {}).get('fpdc_ms_mean', 0), reverse=True):
            # Accept either raw beats_data or precomputed template
            if r.get('beat_template') is not None:
                best_rec = r
                break
            bd = r.get('beats_data')
            if bd and len(bd) >= 5:
                best_rec = r
                break

        if best_rec is None:
            continue

        fs = best_rec['metadata']['sample_rate']
        # Use precomputed template if available, else compute from beats_data
        tmpl = best_rec.get('beat_template')
        if tmpl is None:
            bd = best_rec.get('beats_data')
            tmpl = _compute_template(bd) if bd else None
        if tmpl is not None:
            t_ms = np.arange(len(tmpl)) / fs * 1000
            fig.add_trace(go.Scatter(
                x=t_ms, y=tmpl,
                mode='lines', name=drug.capitalize(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    fig.update_layout(
        xaxis_title="Tempo (ms)", yaxis_title="Ampiezza",
        height=450, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  REPORT DOWNLOADS
# ═══════════════════════════════════════════════════════════════════════

def _download_reports(results, config, data_dir):
    """Provide download buttons for Excel, PDF, CDISC SEND reports."""
    st.subheader("📥 Download Report")

    cols = st.columns(4)

    with cols[0]:
        if st.button("📊 Genera Excel", use_container_width=True):
            with st.spinner("Generazione Excel..."):
                try:
                    from cardiac_fp_analyzer.report import generate_excel_report
                    buf = io.BytesIO()
                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                        generate_excel_report(results, tmp.name)
                        with open(tmp.name, 'rb') as f:
                            buf.write(f.read())
                    st.download_button(
                        "⬇️ Scarica Excel",
                        buf.getvalue(),
                        f"cardiac_fp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Errore: {e}")

    with cols[1]:
        if st.button("📄 Genera PDF", use_container_width=True):
            with st.spinner("Generazione PDF..."):
                try:
                    from cardiac_fp_analyzer.report import generate_pdf_report
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        generate_pdf_report(results, tmp.name, str(data_dir))
                        with open(tmp.name, 'rb') as f:
                            pdf_bytes = f.read()
                    st.download_button(
                        "⬇️ Scarica PDF",
                        pdf_bytes,
                        f"cardiac_fp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Errore: {e}")

    with cols[2]:
        json_str = config.to_json()
        st.download_button(
            "⚙️ Scarica Config JSON",
            json_str,
            "analysis_config.json",
            mime="application/json",
            use_container_width=True
        )

    with cols[3]:
        if st.button("🏛️ Export CDISC SEND", use_container_width=True):
            with st.spinner("Generazione pacchetto CDISC SEND (.xpt + define.xml)..."):
                try:
                    from cardiac_fp_analyzer.cdisc_export import export_send_package
                    cdisc_dir = tempfile.mkdtemp()
                    study_id = st.session_state.get('cdisc_study_id', 'CIPA001')
                    export_send_package(results, cdisc_dir, study_id=study_id)

                    # Zip all .xpt and .xml files
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fp in Path(cdisc_dir).iterdir():
                            if fp.suffix in ('.xpt', '.xml'):
                                zf.write(fp, fp.name)
                    buf.seek(0)
                    st.download_button(
                        "⬇️ Scarica SEND Package (.zip)",
                        buf.getvalue(),
                        f"cdisc_send_{study_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    st.success(f"Pacchetto CDISC SEND generato: TS, DM, EX, EG, RISK + define.xml")
                except Exception as e:
                    st.error(f"Errore export CDISC: {e}")
                    st.code(traceback.format_exc())

    # CDISC Study ID (collapsed)
    with st.expander("🏛️ Impostazioni CDISC SEND", expanded=False):
        st.session_state['cdisc_study_id'] = st.text_input(
            "Study ID", value=st.session_state.get('cdisc_study_id', 'CIPA001'),
            help="Identificativo dello studio per i file CDISC SEND"
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — Navigation
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Build config from sidebar
    config = build_config_from_sidebar()

    # Navigation
    st.sidebar.divider()
    page = st.sidebar.radio(
        "📑 Navigazione",
        ["🔬 Analisi Singolo File", "📊 Analisi Batch + Risk Map", "💊 Confronto Farmaci"],
    )

    if page == "🔬 Analisi Singolo File":
        page_single_file(config)
    elif page == "📊 Analisi Batch + Risk Map":
        page_batch_analysis(config)
    elif page == "💊 Confronto Farmaci":
        page_drug_comparison(config)


if __name__ == '__main__':
    main()
