"""
report.py — Excel and PDF report generation for cardiac FP analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_excel_report(results_list, output_path):
    output_path = Path(output_path)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        hdr = workbook.add_format({'bold': True, 'bg_color': '#2E86AB', 'font_color': 'white', 'border': 1, 'text_wrap': True})
        norm = workbook.add_format({'border': 1})
        crit = workbook.add_format({'border': 1, 'bg_color': '#F8D7DA', 'font_color': '#721c24'})
        warn = workbook.add_format({'border': 1, 'bg_color': '#FFF3CD'})
        ok = workbook.add_format({'border': 1, 'bg_color': '#D4EDDA', 'font_color': '#155724'})

        # Summary sheet
        rows = []
        for r in results_list:
            m, fi, s = r.get('metadata',{}), r.get('file_info',{}), r.get('summary',{})
            ar = r.get('arrhythmia_report', None)
            qc = r.get('qc_report', None)
            rows.append({
                'File': m.get('filename',''), 'Experiment': fi.get('experiment',''),
                'Chip': fi.get('chip',''), 'Channel': fi.get('analyzed_channel',''),
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
                'Mean FPDc (ms)': s.get('fpdc_ms_mean',np.nan), 'STV (ms)': s.get('stv_ms',np.nan),
                'Classification': ar.classification if ar else '', 'Risk Score': ar.risk_score if ar else 0,
            })
        df_s = pd.DataFrame(rows)
        df_s.to_excel(writer, sheet_name='Summary', index=False, startrow=1, header=False)
        ws = writer.sheets['Summary']
        for c, col in enumerate(df_s.columns):
            ws.write(0, c, col, hdr)
            ws.set_column(c, c, min(max(len(str(col)), 10), 25))
        risk_col = list(df_s.columns).index('Risk Score')
        for i in range(len(df_s)):
            risk = df_s.iloc[i]['Risk Score']
            fmt = ok if risk < 20 else warn if risk < 50 else crit
            ws.write(i+2, risk_col, risk, fmt)

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
                fig = plot_analysis_summary(time, filtered, bi, params_plot, meta, figsize=(11, 14))
                if ar:
                    fig.text(0.02, 0.01,
                             f"Classification: {ar.classification} | Risk: {ar.risk_score}/100 | Flags: {len(ar.flags)}",
                             fontsize=8, style='italic',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                pdf.savefig(fig); plt.close(fig)

            beats_d = r.get('beats_data', [])
            beats_t = r.get('beats_time', [])
            if len(beats_d) > 2:
                fig_ov, _ = plot_beat_overlay(beats_t, beats_d, meta, figsize=(11, 7))
                pdf.savefig(fig_ov); plt.close(fig_ov)

    print(f"  PDF saved: {output_path}")
