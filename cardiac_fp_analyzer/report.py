"""
report.py — Excel and PDF report generation for cardiac FP analysis.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_excel_report(results_list, output_path):
    output_path = Path(output_path)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        hdr = workbook.add_format({'bold': True, 'bg_color': '#2E86AB', 'font_color': 'white', 'border': 1, 'text_wrap': True})
        norm = workbook.add_format({'border': 1})  # noqa: F841 — available for cell formatting
        crit = workbook.add_format({'border': 1, 'bg_color': '#F8D7DA', 'font_color': '#721c24'})
        warn = workbook.add_format({'border': 1, 'bg_color': '#FFF3CD'})
        ok = workbook.add_format({'border': 1, 'bg_color': '#D4EDDA', 'font_color': '#155724'})

        # Additional formats for normalization
        pct_fmt = workbook.add_format({'border': 1, 'num_format': '0.0'})
        tdp_crit = workbook.add_format({'border': 1, 'bg_color': '#F8D7DA', 'font_color': '#721c24', 'bold': True})
        tdp_warn = workbook.add_format({'border': 1, 'bg_color': '#FFF3CD', 'font_color': '#856404', 'bold': True})
        tdp_ok = workbook.add_format({'border': 1, 'bg_color': '#D4EDDA', 'font_color': '#155724'})
        tdp_short = workbook.add_format({'border': 1, 'bg_color': '#CCE5FF', 'font_color': '#004085'})
        bool_yes = workbook.add_format({'border': 1, 'bg_color': '#F8D7DA', 'font_color': '#721c24', 'align': 'center'})
        bool_no = workbook.add_format({'border': 1, 'align': 'center'})

        # Summary sheet
        rows = []
        for r in results_list:
            m, fi, s = r.get('metadata',{}), r.get('file_info',{}), r.get('summary',{})
            ar = r.get('arrhythmia_report', None)
            qc = r.get('qc_report', None)
            nrm = r.get('normalization', {})
            row = {
                'File': m.get('filename',''), 'Experiment': fi.get('experiment',''),
                'Chip': fi.get('chip',''), 'Electrode': fi.get('analyzed_channel',''),
                'Drug': fi.get('drug',''), 'Concentration': fi.get('concentration',''),
                'QC Grade': qc.grade if qc else '',
                'Global SNR': qc.global_snr if qc else np.nan,
                'Beats Raw': qc.n_beats_input if qc else 0,
                'Beats Accepted': s.get('beat_period_ms_n',0),
                'Rejected (%)': qc.rejection_rate*100 if qc else 0,
                'Mean BP (ms)': s.get('beat_period_ms_mean',np.nan),
                'Std BP (ms)': s.get('beat_period_ms_std',np.nan),
                'CV BP (%)': s.get('beat_period_ms_cv',np.nan),
                'BPM': s.get('bpm_mean',np.nan), 'Mean Amp (mV)': s.get('spike_amplitude_mV_mean',np.nan),
                'Mean FPD (ms)': s.get('fpd_ms_mean',np.nan), 'Std FPD (ms)': s.get('fpd_ms_std',np.nan),
                'Mean FPDcF (ms)': s.get('fpdc_ms_mean',np.nan), 'Std FPDcF (ms)': s.get('fpdc_ms_std',np.nan),
                'Mean FPDcB (ms)': s.get('fpdc_bazett_ms_mean',np.nan),
                'STV (ms)': s.get('stv_ms',np.nan),
                'Classification': ar.classification if ar else '', 'Risk Score': ar.risk_score if ar else 0,
                # ─── Normalization columns ───
                'Baseline Ref': nrm.get('baseline_file', ''),
                'BL BP (ms)': nrm.get('baseline_bp_ms', np.nan),
                'BL FPDcF (ms)': nrm.get('baseline_fpdc_ms', np.nan),
                '%BP Change': nrm.get('pct_bp_change', np.nan),
                '%FPDcF Change': nrm.get('pct_fpdc_change', np.nan),
                '%AMP Change': nrm.get('pct_amp_change', np.nan),
                '>LOW (10%)': 'YES' if nrm.get('exceeds_LOW') else '',
                '>MID (15%)': 'YES' if nrm.get('exceeds_MID') else '',
                '>HIGH (20%)': 'YES' if nrm.get('exceeds_HIGH') else '',
                'TdP Score': nrm.get('tdp_score', ''),
                # ─── Inclusion criteria ───
                'Included': 'YES' if r.get('inclusion', {}).get('passed', True) else 'NO',
                'FPDcF OK': 'YES' if r.get('inclusion', {}).get('fpdc_plausible', True) else 'NO',
            }
            rows.append(row)
        df_s = pd.DataFrame(rows)
        df_s.to_excel(writer, sheet_name='Summary', index=False, startrow=1, header=False)
        ws = writer.sheets['Summary']
        for c, col in enumerate(df_s.columns):
            ws.write(0, c, col, hdr)
            ws.set_column(c, c, min(max(len(str(col)), 10), 25))

        # Conditional formatting for Risk Score
        if 'Risk Score' in df_s.columns:
            risk_col = list(df_s.columns).index('Risk Score')
            for i in range(len(df_s)):
                risk = df_s.iloc[i]['Risk Score']
                if isinstance(risk, (int, float)) and not (isinstance(risk, float) and np.isnan(risk)):
                    fmt = ok if risk < 20 else warn if risk < 50 else crit
                    ws.write(i+1, risk_col, risk, fmt)

        # Conditional formatting for TdP Score
        if 'TdP Score' in df_s.columns:
            tdp_col = list(df_s.columns).index('TdP Score')
            for i in range(len(df_s)):
                val = df_s.iloc[i]['TdP Score']
                if val == '' or (isinstance(val, float) and np.isnan(val)):
                    continue
                val = int(val) if not isinstance(val, str) else 0
                if val >= 3:
                    fmt = tdp_crit
                elif val >= 2 or val >= 1:
                    fmt = tdp_warn
                elif val <= -1:
                    fmt = tdp_short
                else:
                    fmt = tdp_ok
                ws.write(i+1, tdp_col, val, fmt)

        # Conditional formatting for threshold flags
        for col_name in ['>LOW (10%)', '>MID (15%)', '>HIGH (20%)']:
            if col_name not in df_s.columns:
                continue
            col_idx = list(df_s.columns).index(col_name)
            for i in range(len(df_s)):
                val = df_s.iloc[i][col_name]
                fmt = bool_yes if val == 'YES' else bool_no
                ws.write(i+1, col_idx, val, fmt)

        # Conditional formatting for %FPDcF Change
        if '%FPDcF Change' in df_s.columns:
            fpdc_chg_col = list(df_s.columns).index('%FPDcF Change')
            for i in range(len(df_s)):
                val = df_s.iloc[i]['%FPDcF Change']
                if isinstance(val, (int, float)) and not np.isnan(val):
                    if val >= 20:
                        fmt = tdp_crit
                    elif val >= 10:
                        fmt = tdp_warn
                    elif val <= -10:
                        fmt = tdp_short
                    else:
                        fmt = pct_fmt
                    ws.write(i+1, fpdc_chg_col, val, fmt)

        # ─── Normalization sheet (only drug recordings with baseline) ───
        nrows = []
        for r in results_list:
            nrm = r.get('normalization', {})
            if not nrm.get('has_baseline'):
                continue
            m, fi, s = r.get('metadata',{}), r.get('file_info',{}), r.get('summary',{})
            ar = r.get('arrhythmia_report', None)
            tdp = nrm.get('tdp_score', 0)
            tdp_label = {-1: 'Shortening', 0: 'No effect', 1: 'Mild prolongation',
                         2: 'Moderate prolongation', 3: 'Strong prolongation/Arrhythmia'}.get(tdp, '?')
            nrows.append({
                'Experiment': fi.get('experiment',''),
                'Chip': fi.get('chip',''),
                'Drug': fi.get('drug',''),
                'Concentration': fi.get('concentration',''),
                'Baseline File': nrm.get('baseline_file',''),
                'BL BP (ms)': nrm.get('baseline_bp_ms', np.nan),
                'Drug BP (ms)': s.get('beat_period_ms_mean', np.nan),
                '%BP Change': nrm.get('pct_bp_change', np.nan),
                'BL FPDcF (ms)': nrm.get('baseline_fpdc_ms', np.nan),
                'Drug FPDcF (ms)': s.get('fpdc_ms_mean', np.nan),
                '%FPDcF Change': nrm.get('pct_fpdc_change', np.nan),
                'BL AMP (mV)': nrm.get('baseline_amp_mV', np.nan),
                'Drug AMP (mV)': s.get('spike_amplitude_mV_mean', np.nan),
                '%AMP Change': nrm.get('pct_amp_change', np.nan),
                '>LOW': 'YES' if nrm.get('exceeds_LOW') else '',
                '>MID': 'YES' if nrm.get('exceeds_MID') else '',
                '>HIGH': 'YES' if nrm.get('exceeds_HIGH') else '',
                'TdP Score': tdp,
                'TdP Risk': tdp_label,
                'Arrhythmia': ar.classification if ar else '',
                'QC Grade': r.get('qc_report').grade if r.get('qc_report') else '',
            })
        if nrows:
            df_n = pd.DataFrame(nrows)
            df_n.to_excel(writer, sheet_name='Normalization', index=False, startrow=1, header=False)
            ws_n = writer.sheets['Normalization']
            for c, col in enumerate(df_n.columns):
                ws_n.write(0, c, col, hdr)
                ws_n.set_column(c, c, min(max(len(str(col)), 10), 22))
            # Color-code TdP Score column
            tdp_col_n = list(df_n.columns).index('TdP Score')
            for i in range(len(df_n)):
                val = df_n.iloc[i]['TdP Score']
                if val >= 3: fmt = tdp_crit
                elif val >= 1: fmt = tdp_warn
                elif val <= -1: fmt = tdp_short
                else: fmt = tdp_ok
                ws_n.write(i+1, tdp_col_n, val, fmt)
            # Color-code %FPDcF Change
            fpdc_col_n = list(df_n.columns).index('%FPDcF Change')
            for i in range(len(df_n)):
                val = df_n.iloc[i]['%FPDcF Change']
                if isinstance(val, (int, float)) and not np.isnan(val):
                    if val >= 20: fmt = tdp_crit
                    elif val >= 10: fmt = tdp_warn
                    elif val <= -10: fmt = tdp_short
                    else: fmt = pct_fmt
                    ws_n.write(i+1, fpdc_col_n, val, fmt)

        # Flags sheet
        frows = []
        for r in results_list:
            ar = r.get('arrhythmia_report')
            if ar:
                for f in ar.flags:
                    frows.append({'File': r['metadata'].get('filename',''),
                                  'Severity': f['severity'].upper(), 'Type': f['type'],
                                  'Description': f['description']})
        if frows:
            df_f = pd.DataFrame(frows)
            df_f.to_excel(writer, sheet_name='Arrhythmia Flags', index=False, startrow=1, header=False)
            ws2 = writer.sheets['Arrhythmia Flags']
            for c, col in enumerate(df_f.columns):
                ws2.write(0, c, col, hdr)

        # Per-beat sheet
        brows = []
        for r in results_list:
            for p in r.get('all_params', [])[:200]:
                row = {'File': r['metadata'].get('filename','')}
                row.update({k: v for k, v in p.items() if not isinstance(v, (np.ndarray, list))})
                brows.append(row)
        if brows:
            df_b = pd.DataFrame(brows)
            df_b.to_excel(writer, sheet_name='Per-Beat Data', index=False, startrow=1, header=False)
            ws3 = writer.sheets['Per-Beat Data']
            for c, col in enumerate(df_b.columns):
                ws3.write(0, c, col, hdr)

    print(f"  Excel saved: {output_path}")


def generate_pdf_report(results_list, output_path, data_dir=None):
    from .plotting import plot_analysis_summary, plot_beat_overlay
    output_path = Path(output_path)

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65, 'Cardiac Field Potential Analysis', ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.55, 'hiPSC-CM µECG Recordings', ha='center', fontsize=16, color='gray')
        fig.text(0.5, 0.40, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ha='center', fontsize=12)
        fig.text(0.5, 0.35, f'Files analyzed: {len(results_list)}', ha='center', fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # Per-file pages
        for r in results_list:
            meta = r.get('metadata', {})
            summary = r.get('summary', {})
            ar = r.get('arrhythmia_report')
            bi = r.get('beat_indices', np.array([]))
            filtered = r.get('filtered_signal')
            time = r.get('time_vector')

            if filtered is not None and time is not None:
                params_plot = dict(summary)
                params_plot['fs'] = meta.get('sample_rate', 2000)
                # Pass analyzed_channel into metadata for PDF title
                fi = r.get('file_info', {})
                meta_plot = dict(meta)
                meta_plot['analyzed_channel'] = fi.get('analyzed_channel', '')
                fig = plot_analysis_summary(time, filtered, bi, params_plot, meta_plot, figsize=(11, 14))
                # Footer: arrhythmia + normalization info
                footer_parts = []
                if ar:
                    footer_parts.append(f"Classification: {ar.classification} | Risk: {ar.risk_score}/100 | Flags: {len(ar.flags)}")
                nrm = r.get('normalization', {})
                if nrm.get('has_baseline'):
                    pct_fpdc = nrm.get('pct_fpdc_change', np.nan)
                    tdp = nrm.get('tdp_score', 0)
                    bl_ref = nrm.get('baseline_file', '')
                    pct_str = f"{pct_fpdc:+.1f}%" if not np.isnan(pct_fpdc) else "N/A"
                    footer_parts.append(f"vs Baseline ({bl_ref}): %FPDcF={pct_str} | TdP Score={tdp}")
                if footer_parts:
                    fig.text(0.02, 0.01, " | ".join(footer_parts),
                             fontsize=8, style='italic',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                pdf.savefig(fig); plt.close(fig)

            beats_d = r.get('beats_data', [])
            beats_t = r.get('beats_time', [])
            if len(beats_d) > 2:
                fig_ov, _ = plot_beat_overlay(beats_t, beats_d, meta, figsize=(11, 7))
                pdf.savefig(fig_ov); plt.close(fig_ov)

    print(f"  PDF saved: {output_path}")
