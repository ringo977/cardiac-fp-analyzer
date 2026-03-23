"""
Drug Comparison dashboard — dose-response, arrhythmia metrics, waveform overlay.
"""

import re as _re_sort

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.arrhythmia import compute_template
from cardiac_fp_analyzer.normalization import is_baseline

from ui.i18n import T
from ui.helpers import amplitude_scale


# ═══════════════════════════════════════════════════════════════════════
#  Page entry point
# ═══════════════════════════════════════════════════════════════════════

def page_drug_comparison(config: AnalysisConfig):
    st.header(f"💊 {T('drug_comparison')}")
    st.caption(T('drug_comparison_desc'))

    results = st.session_state.get('batch_results')
    if results is None:
        st.info(T('run_batch_first'))
        return

    from cardiac_fp_analyzer.risk_map import _canonical_drug

    # Group by drug
    drug_data = {}
    for r in results:
        if is_baseline(r):
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


# ═══════════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════════

_COLORS = ['#dc3545', '#0d6efd', '#28a745', '#fd7e14', '#6f42c1', '#20c997', '#e83e8c']


def _plot_dose_response(drug_data, selected_drugs):
    """Plot dose-response curves for FPDcF change."""
    fig = go.Figure()

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
                marker=dict(size=10, color=_COLORS[i % len(_COLORS)]),
                line=dict(color=_COLORS[i % len(_COLORS)]),
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
    fig = go.Figure()

    # Collect templates first to determine scale
    templates_to_plot = []
    for i, drug in enumerate(selected_drugs):
        recs = drug_data.get(drug, [])
        best_rec = None
        for r in sorted(recs, key=lambda r: r.get('summary', {}).get('fpdc_ms_mean', 0), reverse=True):
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
        tmpl = best_rec.get('beat_template')
        if tmpl is None:
            bd = best_rec.get('beats_data')
            tmpl = compute_template(bd) if bd else None
        if tmpl is not None:
            templates_to_plot.append((drug, tmpl, fs, i))

    # Auto-scale based on first template
    if templates_to_plot:
        scale, y_label = amplitude_scale(templates_to_plot[0][1])
    else:
        scale, y_label = 1e3, T('amplitude_mV')

    for drug, tmpl, fs, i in templates_to_plot:
        t_ms = np.arange(len(tmpl)) / fs * 1000
        fig.add_trace(go.Scatter(
            x=t_ms, y=tmpl * scale,
            mode='lines', name=drug.capitalize(),
            line=dict(color=_COLORS[i % len(_COLORS)], width=2)
        ))

    fig.update_layout(
        xaxis_title=T('time_ms'), yaxis_title=y_label,
        height=450, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
