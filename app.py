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
#  INTERNATIONALIZATION
# ═══════════════════════════════════════════════════════════════════════

TRANSLATIONS = {
    'it': {
        'app_title': 'Cardiac FP Analyzer',
        'config': 'Configurazione',
        'signal': 'Segnale',
        'beats': 'Battiti',
        'params': 'Parametri',
        'arrhythmia': 'Aritmie',
        'nav': 'Navigazione',
        'single_file': 'Analisi Singolo File',
        'batch': 'Analisi Batch + Risk Map',
        'drug_comparison': 'Confronto Farmaci',
        'single_file_desc': 'Upload di un file CSV per visualizzare segnale, battiti, parametri e aritmie.',
        'upload_csv': 'Scegli un file CSV',
        'upload_csv_info': 'Carica un file CSV dal sistema µECG-Pharma per iniziare.',
        'channel': 'Elettrodo',
        'analyze': 'Analizza',
        'analyzing': 'Analisi in corso...',
        'analysis_complete': 'Analisi completata',
        'analysis_error': "Errore nell'analisi. Verifica il formato del file.",
        'chip_channel': 'Chip/Elettrodo',
        'drug': 'Farmaco',
        'risk_score': 'Risk Score',
        'filtered_signal': 'Segnale filtrato',
        'raw_signal': 'Segnale grezzo',
        'show_raw_signal': 'Mostra segnale grezzo (non filtrato)',
        'beat_markers': 'Battiti',
        'time_s': 'Tempo (s)',
        'time_ms': 'Tempo (ms)',
        'amplitude_mV': 'Ampiezza (mV)',
        'amplitude_uV': 'Ampiezza (µV)',
        'template': 'Template',
        'n_beats_qc': 'N. battiti (QC)',
        'n_beats_raw': 'N. battiti (raw)',
        'beat_period': 'Beat Period',
        'fpd': 'FPD (ms)',
        'fpdc': 'FPDcF (ms)',
        'spike_amp': 'Ampiezza spike',
        'fpd_confidence': 'Confidenza FPD',
        'summary': 'Riepilogo',
        'parameter': 'Parametro',
        'mean': 'Media',
        'no_params': 'Nessun parametro disponibile.',
        'flags_detected': 'Flag rilevate',
        'no_flags': 'Nessuna flag aritmica rilevata.',
        'residual_analysis': 'Analisi residuo',
        'morph_instability': 'Morphology instability',
        'ead_incidence': 'EAD incidence',
        'baseline_relative': 'Baseline-relative',
        'events': 'Eventi',
        'batch_desc': 'Upload di più file CSV per analisi batch, normalizzazione baseline e risk map CiPA.',
        'upload_mode': 'Modalità upload',
        'select_folder': 'Seleziona cartella',
        'upload_csv_files': 'Upload file CSV',
        'upload_zip': 'Upload archivio ZIP',
        'load_csv_files': 'Carica file CSV',
        'load_zip': 'Carica un archivio ZIP contenente i CSV',
        'loaded_n_files': 'Caricati **{n}** file CSV',
        'extracted_n_files': "Estratti **{n}** file CSV dall'archivio",
        'open_folder': 'Apri selezione cartella',
        'no_folder': 'Nessuna cartella selezionata.',
        'folder_error': "Errore nell'aprire il file picker",
        'folder_info': '📁 `{path}` — **{n}** file CSV trovati',
        'change_folder': 'Cambia cartella',
        'ground_truth': 'Ground truth farmaci (opzionale)',
        'ground_truth_desc': 'Per colorare la risk map con i dati noti. Formato: farmaco=+/−',
        'start_batch': 'Avvia Analisi Batch',
        'starting': 'Avvio analisi...',
        'pipeline_running': 'Pipeline in esecuzione...',
        'batch_complete': 'Analisi completata: {n} registrazioni processate.',
        'error': 'Errore',
        'risk_map': 'Risk Map',
        'overview': 'Riepilogo',
        'details': 'Dettagli',
        'drug_metrics': 'Metriche per farmaco',
        'select_recording': 'Seleziona registrazione',
        'signal_not_available': 'Segnale filtrato non disponibile in modalità batch.',
        'no_drug_data': 'Nessun dato farmaco disponibile.',
        'no_drug_data_map': 'Nessun dato farmaco disponibile per la risk map.',
        'drug_comparison_desc': 'Dashboard comparativa: dose-response, metriche aritmiche, overlay waveform.',
        'run_batch_first': "Esegui prima un'analisi batch per visualizzare i confronti tra farmaci.",
        'select_drugs': 'Seleziona farmaci da confrontare',
        'dose_response': 'Dose-Response',
        'arrhythmia_metrics': 'Metriche Aritmiche',
        'waveform': 'Waveform',
        'concentration': 'Concentrazione (ordine crescente)',
        'proarrhythmic_index': 'Indice Proaritmico (0–100)',
        'download_reports': 'Download Report',
        'gen_excel': 'Genera Excel',
        'gen_pdf': 'Genera PDF',
        'download_excel': 'Scarica Excel',
        'download_pdf': 'Scarica PDF',
        'download_config': 'Scarica Config JSON',
        'export_cdisc': 'Export CDISC SEND',
        'generating_excel': 'Generazione Excel...',
        'generating_pdf': 'Generazione PDF...',
        'generating_cdisc': 'Generazione pacchetto CDISC SEND (.xpt + define.xml)...',
        'download_cdisc': 'Scarica SEND Package (.zip)',
        'cdisc_success': 'Pacchetto CDISC SEND generato: TS, DM, EX, EG, RISK + define.xml',
        'cdisc_error': 'Errore export CDISC',
        'cdisc_settings': 'Impostazioni CDISC SEND',
        'study_id': 'Study ID',
        'study_id_help': 'Identificativo dello studio per i file CDISC SEND',
        'export_config': 'Esporta config',
        'import_config': 'Importa',
        'config_loaded': 'Config caricata!',
        'total': 'Totale',
        'recordings': 'registrazioni',
        'cfg_signal': 'Segnale',
        'cfg_amp_gain': 'Guadagno amplificatore',
        'cfg_amp_gain_help': 'Divisore per ottenere il voltaggio reale. µECG-Pharma Digilent = 10⁴',
        'cfg_mains_freq': 'Frequenza di rete (Hz)',
        'cfg_bandpass_low': 'Passa-banda basso (Hz)',
        'cfg_bandpass_high': 'Passa-banda alto (Hz)',
        'cfg_beat_detection': 'Beat Detection',
        'cfg_method': 'Metodo',
        'cfg_method_help': "'auto' sceglie il metodo migliore automaticamente",
        'cfg_min_distance': 'Distanza min tra battiti (ms)',
        'cfg_min_distance_help': "In modalità 'auto', la correzione bimodale rileva e corregge automaticamente i falsi positivi da onda T.",
        'cfg_threshold': 'Soglia adattiva (×)',
        'cfg_qc_section': 'Controllo qualità battiti',
        'cfg_morph_threshold': 'Soglia morfologica (correlazione)',
        'cfg_morph_threshold_help': 'Correlazione minima con il template per accettare un battito. Abbassare se troppi battiti vengono scartati (segnali µECG: provare 0.2-0.3).',
        'cfg_use_morphology': 'Filtro morfologico attivo',
        'cfg_use_morphology_help': 'Disattivare se il segnale è buono ma il QC scarta troppi battiti.',
        'cfg_fpd': 'FPD / Ripolarizzazione',
        'cfg_fpd_method': 'Metodo FPD',
        'cfg_fpd_method_help': "'tangent' è lo standard in letteratura",
        'cfg_qt_correction': 'Correzione QT',
        'cfg_qt_help': 'Fridericia raccomandata per hiPSC-CM',
        'cfg_repol_start': 'Inizio ricerca ripol. (ms)',
        'cfg_repol_end': 'Fine ricerca ripol. (ms)',
        'cfg_arrhythmia': 'Aritmie / EAD',
        'cfg_ead_prominence': 'EAD prominenza (×σ)',
        'cfg_ead_min_amp': 'EAD ampiezza min (% template)',
        'cfg_ead_min_width': 'EAD larghezza min (ms)',
        'cfg_ead_max_width': 'EAD larghezza max (ms)',
        'cfg_risk_mode': 'Modalità risk score',
        'cfg_risk_mode_help': "'manual': pesi esperti (letteratura). 'data_driven': pesi da regressione logistica su dataset CiPA (sperimentale — richiede fitted_weights.json)",
        'cfg_inclusion': 'Criteri di inclusione',
        'cfg_max_cv': 'Max CV% beat period baseline',
        'cfg_min_fpd_conf': 'Min confidenza FPD',
        'cfg_physiol_filter': 'Filtro range fisiologico FPDcF',
        'cfg_normalization': 'Normalizzazione',
        'cfg_threshold_low': 'Soglia LOW (%ΔFPDcF)',
        'cfg_threshold_mid': 'Soglia MID (%ΔFPDcF)',
        'cfg_threshold_high': 'Soglia HIGH (%ΔFPDcF)',
        'folder_prompt_mac': 'Select data CSV folder',
        'folder_prompt_linux': 'Select data CSV folder',
        'folder_prompt_win': 'Select data CSV folder',
        'file': 'File',
        'chip': 'Chip',
        'baseline': 'Baseline',
        'inclusion': 'Inclusione',
        'classification': 'Class.',
        'beat_editor': 'Editor battiti',
        'beat_editor_help': 'Deseleziona la casella per escludere un battito, riselezionala per reincluderlo. Puoi anche aggiungere battiti mancanti.',
        'beat_include': 'Incluso',
        'beats_included': 'Battiti inclusi',
        'beats_excluded': 'Battiti esclusi',
        'add_beat_time': 'Aggiungi battito al tempo (s)',
        'add_beat_btn': 'Aggiungi',
        'reanalyze_btn': 'Ri-analizza con battiti modificati',
        'reanalysis_done': 'Ri-analisi completata con {n} battiti',
        'beats_modified': 'Battiti modificati: {orig} originali → {new} inclusi',
        'export_section': 'Esporta risultati',
        'export_csv': 'Parametri CSV',
        'export_excel_single': 'Report Excel',
        'export_summary': 'Riepilogo CSV',
        'both_channels': 'Entrambi',
        'channel_comparison': 'Confronto elettrodi',
        'analyzing_ch': 'Analisi {ch}...',
        'channel_failed': 'Analisi {ch} fallita',
        'no_valid_channel': 'Nessun elettrodo valido.',
        'select_channel': 'Seleziona elettrodo da visualizzare',
    },
    'en': {
        'app_title': 'Cardiac FP Analyzer',
        'config': 'Configuration',
        'signal': 'Signal',
        'beats': 'Beats',
        'params': 'Parameters',
        'arrhythmia': 'Arrhythmia',
        'nav': 'Navigation',
        'single_file': 'Single File Analysis',
        'batch': 'Batch Analysis + Risk Map',
        'drug_comparison': 'Drug Comparison',
        'single_file_desc': 'Upload a CSV file to visualize signal, beats, parameters and arrhythmias.',
        'upload_csv': 'Choose a CSV file',
        'upload_csv_info': 'Upload a CSV file from the µECG-Pharma system to begin.',
        'channel': 'Electrode',
        'analyze': 'Analyze',
        'analyzing': 'Analysis in progress...',
        'analysis_complete': 'Analysis complete',
        'analysis_error': 'Analysis error. Check the file format.',
        'chip_channel': 'Chip/Electrode',
        'drug': 'Drug',
        'risk_score': 'Risk Score',
        'filtered_signal': 'Filtered signal',
        'raw_signal': 'Raw signal',
        'show_raw_signal': 'Show raw (unfiltered) signal',
        'beat_markers': 'Beats',
        'time_s': 'Time (s)',
        'time_ms': 'Time (ms)',
        'amplitude_mV': 'Amplitude (mV)',
        'amplitude_uV': 'Amplitude (µV)',
        'template': 'Template',
        'n_beats_qc': 'N. beats (QC)',
        'n_beats_raw': 'N. beats (raw)',
        'beat_period': 'Beat Period',
        'fpd': 'FPD (ms)',
        'fpdc': 'FPDcF (ms)',
        'spike_amp': 'Spike amplitude',
        'fpd_confidence': 'FPD Confidence',
        'summary': 'Summary',
        'parameter': 'Parameter',
        'mean': 'Mean',
        'no_params': 'No parameters available.',
        'flags_detected': 'Flags detected',
        'no_flags': 'No arrhythmia flags detected.',
        'residual_analysis': 'Residual analysis',
        'morph_instability': 'Morphology instability',
        'ead_incidence': 'EAD incidence',
        'baseline_relative': 'Baseline-relative',
        'events': 'Events',
        'batch_desc': 'Upload multiple CSV files for batch analysis, baseline normalization and CiPA risk map.',
        'upload_mode': 'Upload mode',
        'select_folder': 'Select folder',
        'upload_csv_files': 'Upload CSV files',
        'upload_zip': 'Upload ZIP archive',
        'load_csv_files': 'Upload CSV files',
        'load_zip': 'Upload a ZIP archive containing CSV files',
        'loaded_n_files': 'Loaded **{n}** CSV files',
        'extracted_n_files': 'Extracted **{n}** CSV files from archive',
        'open_folder': 'Open folder selection',
        'no_folder': 'No folder selected.',
        'folder_error': 'Error opening file picker',
        'folder_info': '📁 `{path}` — **{n}** CSV files found',
        'change_folder': 'Change folder',
        'ground_truth': 'Drug ground truth (optional)',
        'ground_truth_desc': 'To color the risk map with known data. Format: drug=+/−',
        'start_batch': 'Start Batch Analysis',
        'starting': 'Starting analysis...',
        'pipeline_running': 'Pipeline running...',
        'batch_complete': 'Analysis complete: {n} recordings processed.',
        'error': 'Error',
        'risk_map': 'Risk Map',
        'overview': 'Overview',
        'details': 'Details',
        'drug_metrics': 'Drug metrics',
        'select_recording': 'Select recording',
        'signal_not_available': 'Filtered signal not available in batch mode.',
        'no_drug_data': 'No drug data available.',
        'no_drug_data_map': 'No drug data available for the risk map.',
        'drug_comparison_desc': 'Comparative dashboard: dose-response, arrhythmia metrics, waveform overlay.',
        'run_batch_first': 'Run a batch analysis first to view drug comparisons.',
        'select_drugs': 'Select drugs to compare',
        'dose_response': 'Dose-Response',
        'arrhythmia_metrics': 'Arrhythmia Metrics',
        'waveform': 'Waveform',
        'concentration': 'Concentration (ascending order)',
        'proarrhythmic_index': 'Proarrhythmic Index (0–100)',
        'download_reports': 'Download Reports',
        'gen_excel': 'Generate Excel',
        'gen_pdf': 'Generate PDF',
        'download_excel': 'Download Excel',
        'download_pdf': 'Download PDF',
        'download_config': 'Download Config JSON',
        'export_cdisc': 'Export CDISC SEND',
        'generating_excel': 'Generating Excel...',
        'generating_pdf': 'Generating PDF...',
        'generating_cdisc': 'Generating CDISC SEND package (.xpt + define.xml)...',
        'download_cdisc': 'Download SEND Package (.zip)',
        'cdisc_success': 'CDISC SEND package generated: TS, DM, EX, EG, RISK + define.xml',
        'cdisc_error': 'CDISC export error',
        'cdisc_settings': 'CDISC SEND Settings',
        'study_id': 'Study ID',
        'study_id_help': 'Study identifier for CDISC SEND files',
        'export_config': 'Export config',
        'import_config': 'Import',
        'config_loaded': 'Config loaded!',
        'total': 'Total',
        'recordings': 'recordings',
        'cfg_signal': 'Signal',
        'cfg_amp_gain': 'Amplifier gain',
        'cfg_amp_gain_help': 'Divisor to obtain real voltage. µECG-Pharma Digilent = 10⁴',
        'cfg_mains_freq': 'Mains frequency (Hz)',
        'cfg_bandpass_low': 'Bandpass low (Hz)',
        'cfg_bandpass_high': 'Bandpass high (Hz)',
        'cfg_beat_detection': 'Beat Detection',
        'cfg_method': 'Method',
        'cfg_method_help': "'auto' chooses the best method automatically",
        'cfg_min_distance': 'Min beat distance (ms)',
        'cfg_min_distance_help': "In 'auto' mode, bimodal correction automatically detects and fixes T-wave false positives.",
        'cfg_threshold': 'Adaptive threshold (×)',
        'cfg_qc_section': 'Beat quality control',
        'cfg_morph_threshold': 'Morphology threshold (correlation)',
        'cfg_morph_threshold_help': 'Minimum correlation with template to accept a beat. Lower if too many beats are rejected (µECG signals: try 0.2-0.3).',
        'cfg_use_morphology': 'Morphology filter active',
        'cfg_use_morphology_help': 'Disable if signal is good but QC rejects too many beats.',
        'cfg_fpd': 'FPD / Repolarization',
        'cfg_fpd_method': 'FPD Method',
        'cfg_fpd_method_help': "'tangent' is the literature standard",
        'cfg_qt_correction': 'QT Correction',
        'cfg_qt_help': 'Fridericia recommended for hiPSC-CM',
        'cfg_repol_start': 'Repol. search start (ms)',
        'cfg_repol_end': 'Repol. search end (ms)',
        'cfg_arrhythmia': 'Arrhythmia / EAD',
        'cfg_ead_prominence': 'EAD prominence (×σ)',
        'cfg_ead_min_amp': 'EAD min amplitude (% template)',
        'cfg_ead_min_width': 'EAD min width (ms)',
        'cfg_ead_max_width': 'EAD max width (ms)',
        'cfg_risk_mode': 'Risk score mode',
        'cfg_risk_mode_help': "'manual': expert weights (literature). 'data_driven': weights from logistic regression on CiPA dataset (experimental — requires fitted_weights.json)",
        'cfg_inclusion': 'Inclusion criteria',
        'cfg_max_cv': 'Max CV% beat period baseline',
        'cfg_min_fpd_conf': 'Min FPD confidence',
        'cfg_physiol_filter': 'Physiological FPDcF range filter',
        'cfg_normalization': 'Normalization',
        'cfg_threshold_low': 'LOW threshold (%ΔFPDcF)',
        'cfg_threshold_mid': 'MID threshold (%ΔFPDcF)',
        'cfg_threshold_high': 'HIGH threshold (%ΔFPDcF)',
        'folder_prompt_mac': 'Select data CSV folder',
        'folder_prompt_linux': 'Select data CSV folder',
        'folder_prompt_win': 'Select data CSV folder',
        'file': 'File',
        'chip': 'Chip',
        'baseline': 'Baseline',
        'inclusion': 'Inclusion',
        'classification': 'Class.',
        'beat_editor': 'Beat editor',
        'beat_editor_help': 'Uncheck the box to exclude a beat, re-check it to include it again. You can also add missing beats.',
        'beat_include': 'Included',
        'beats_included': 'Included beats',
        'beats_excluded': 'Excluded beats',
        'add_beat_time': 'Add beat at time (s)',
        'add_beat_btn': 'Add',
        'reanalyze_btn': 'Re-analyze with modified beats',
        'reanalysis_done': 'Re-analysis complete with {n} beats',
        'beats_modified': 'Beats modified: {orig} original → {new} included',
        'export_section': 'Export results',
        'export_csv': 'Parameters CSV',
        'export_excel_single': 'Excel Report',
        'export_summary': 'Summary CSV',
        'both_channels': 'Both',
        'channel_comparison': 'Electrode comparison',
        'analyzing_ch': 'Analyzing {ch}...',
        'channel_failed': 'Analysis failed for {ch}',
        'no_valid_channel': 'No valid electrode.',
        'select_channel': 'Select electrode to view',
    }
}


def T(key, **kwargs):
    """Get translated string for current language."""
    lang = st.session_state.get('lang', 'it')
    text = TRANSLATIONS.get(lang, TRANSLATIONS['it']).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def _reanalyze_with_modified_beats(result, new_beat_indices, config):
    """Re-run analysis pipeline from beat segmentation onwards using modified beat indices.

    This keeps the original filtered signal but recomputes segmentation,
    QC, parameters, and arrhythmia analysis with the user-edited beats.
    """
    filtered = result['filtered_signal']
    t = result['time_vector']
    fs = result['metadata']['sample_rate']
    rep_cfg = config.repolarization

    bi = np.sort(new_beat_indices)

    # Segment beats
    bd, btm, vi = segment_beats(filtered, t, bi, fs,
                                 pre_ms=rep_cfg.segment_pre_ms,
                                 post_ms=max(850, rep_cfg.search_end_ms + 50))

    # Use only successfully segmented beats (edge-truncated beats removed)
    bi_seg = bi[np.array(vi)] if len(vi) > 0 else bi[:0]

    # QC
    qc_report, bi_clean, bd_clean, btm_clean = validate_beats(
        filtered, bi_seg, bd, btm, fs, cfg=config.quality
    )

    # Parameters
    all_p, summary = extract_all_parameters(bd_clean, btm_clean, bi_clean, fs,
                                             cfg=rep_cfg)
    bp = compute_beat_periods(bi_clean, fs)

    # Arrhythmia
    ar = analyze_arrhythmia(bi_clean, bp, all_p, summary, fs,
                            cfg=config.arrhythmia, beats_data=bd_clean)

    # Build updated result (preserving original signal data)
    updated = dict(result)
    updated.update({
        'beat_indices': bi_clean,
        'beat_indices_raw': bi,
        'beat_periods': bp,
        'all_params': all_p,
        'summary': summary,
        'arrhythmia_report': ar,
        'beats_data': bd_clean,
        'beats_time': btm_clean,
        'qc_report': qc_report,
    })
    return updated


def _amplitude_scale(sig):
    """Choose the right display unit for a signal array.

    Returns (multiplier, y_axis_label) so that sig * multiplier is in a
    human-friendly range.  With amplifier gain = 10⁴ the filtered signal
    is in the µV range; with gain = 1 (raw) it is in the mV range.
    """
    sig_range = np.ptp(sig) if len(sig) > 0 else 0
    if sig_range < 0.001:        # < 1 mV peak-to-peak → display in µV
        return 1e6, T('amplitude_uV')
    else:                         # ≥ 1 mV → display in mV
        return 1e3, T('amplitude_mV')



# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Configuration
# ═══════════════════════════════════════════════════════════════════════

def build_config_from_sidebar() -> AnalysisConfig:
    """Build AnalysisConfig from sidebar widgets. Returns config object."""
    config = AnalysisConfig()

    with st.sidebar:
        st.header(f"⚙️ {T('config')}")

        # ── Signal ──
        with st.expander(f"📡 {T('cfg_signal')}", expanded=False):
            config.amplifier_gain = st.number_input(
                T('cfg_amp_gain'),
                value=1e4, min_value=1.0, format="%.0e",
                help=T('cfg_amp_gain_help'),
                key='cfg_amp_gain'
            )
            _notch_options = {'50 Hz': 50.0, '60 Hz': 60.0, 'Off': 0.0}
            _notch_reverse = {v: k for k, v in _notch_options.items()}
            _notch_sel = st.selectbox(
                f"{T('cfg_mains_freq')}", list(_notch_options.keys()), index=0,
                key='cfg_notch'
            )
            config.filtering.notch_freq_hz = _notch_options[_notch_sel]
            config.filtering.bandpass_low_hz = st.number_input(
                T('cfg_bandpass_low'), value=0.5, min_value=0.1, max_value=10.0, step=0.1,
                key='cfg_bp_low'
            )
            config.filtering.bandpass_high_hz = st.number_input(
                T('cfg_bandpass_high'), value=500.0, min_value=50.0, max_value=5000.0, step=50.0,
                key='cfg_bp_high'
            )

        # ── Beat detection ──
        with st.expander(f"💓 {T('cfg_beat_detection')}", expanded=False):
            config.beat_detection.method = st.selectbox(
                T('cfg_method'), ['auto', 'prominence', 'derivative', 'peak'],
                help=T('cfg_method_help'),
                key='cfg_bd_method'
            )
            config.beat_detection.min_distance_ms = st.number_input(
                T('cfg_min_distance'), value=400.0, min_value=100.0, max_value=2000.0,
                help=T('cfg_min_distance_help'),
                key='cfg_bd_min_dist'
            )
            config.beat_detection.threshold_factor = st.number_input(
                T('cfg_threshold'), value=4.0, min_value=1.0, max_value=10.0, step=0.5,
                key='cfg_bd_thresh'
            )
            st.caption(f"🔬 {T('cfg_qc_section')}")
            config.quality.morphology_threshold = st.slider(
                T('cfg_morph_threshold'), min_value=0.0, max_value=1.0,
                value=0.4, step=0.05,
                help=T('cfg_morph_threshold_help'),
                key='cfg_morph_thresh'
            )
            config.quality.use_morphology = st.checkbox(
                T('cfg_use_morphology'), value=True,
                help=T('cfg_use_morphology_help'),
                key='cfg_use_morph'
            )

        # ── FPD ──
        with st.expander(f"📏 {T('cfg_fpd')}", expanded=False):
            config.repolarization.fpd_method = st.selectbox(
                T('cfg_fpd_method'),
                ['tangent', 'peak', 'max_slope', '50pct', 'baseline_return', 'consensus'],
                help=T('cfg_fpd_method_help'),
                key='cfg_fpd_method'
            )
            config.repolarization.correction = st.selectbox(
                T('cfg_qt_correction'), ['fridericia', 'bazett', 'none'],
                help=T('cfg_qt_help'),
                key='cfg_qt_corr'
            )
            config.repolarization.search_start_ms = st.number_input(
                T('cfg_repol_start'), value=150.0, min_value=50.0, max_value=400.0,
                key='cfg_repol_start'
            )
            config.repolarization.search_end_ms = st.number_input(
                T('cfg_repol_end'), value=900.0, min_value=400.0, max_value=1500.0,
                key='cfg_repol_end'
            )

        # ── Arrhythmia ──
        with st.expander(f"⚡ {T('cfg_arrhythmia')}", expanded=False):
            config.arrhythmia.ead_residual_prominence = st.number_input(
                T('cfg_ead_prominence'), value=6.0, min_value=2.0, max_value=15.0, step=0.5,
                key='cfg_ead_prom'
            )
            config.arrhythmia.ead_residual_min_amp_frac = st.number_input(
                T('cfg_ead_min_amp'), value=0.08, min_value=0.01, max_value=0.50, step=0.01,
                key='cfg_ead_min_amp'
            )
            config.arrhythmia.ead_residual_min_width_ms = st.number_input(
                T('cfg_ead_min_width'), value=8.0, min_value=1.0, max_value=50.0,
                key='cfg_ead_min_w'
            )
            config.arrhythmia.ead_residual_max_width_ms = st.number_input(
                T('cfg_ead_max_width'), value=150.0, min_value=50.0, max_value=500.0,
                key='cfg_ead_max_w'
            )
            config.arrhythmia.risk_score_mode = st.selectbox(
                T('cfg_risk_mode'),
                ['manual', 'data_driven'],
                help=T('cfg_risk_mode_help'),
                key='cfg_risk_mode'
            )

        # ── Inclusion ──
        with st.expander(f"🔍 {T('cfg_inclusion')}", expanded=False):
            config.inclusion.max_cv_bp = st.number_input(
                T('cfg_max_cv'), value=25.0, min_value=5.0, max_value=50.0,
                key='cfg_max_cv'
            )
            config.inclusion.min_fpd_confidence = st.number_input(
                T('cfg_min_fpd_conf'), value=0.66, min_value=0.0, max_value=1.0, step=0.01,
                key='cfg_min_fpd_conf'
            )
            config.inclusion.enabled_fpdc_physiol = st.checkbox(
                T('cfg_physiol_filter'), value=True,
                key='cfg_physiol'
            )

        # ── Normalization ──
        with st.expander(f"📊 {T('cfg_normalization')}", expanded=False):
            config.normalization.threshold_low = st.number_input(
                T('cfg_threshold_low'), value=10.0, min_value=1.0, max_value=30.0,
                key='cfg_norm_low'
            )
            config.normalization.threshold_mid = st.number_input(
                T('cfg_threshold_mid'), value=15.0, min_value=5.0, max_value=40.0,
                key='cfg_norm_mid'
            )
            config.normalization.threshold_high = st.number_input(
                T('cfg_threshold_high'), value=20.0, min_value=10.0, max_value=50.0,
                key='cfg_norm_high'
            )

        # ── Config import/export ──
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            json_str = config.to_json()
            st.download_button(f"📥 {T('export_config')}", json_str, "analysis_config.json",
                               mime="application/json", use_container_width=True)
        with col2:
            uploaded_cfg = st.file_uploader(f"📤 {T('import_config')}", type=['json'], key='cfg_upload')
            if uploaded_cfg is None:
                st.session_state.pop('_cfg_imported', None)
            if uploaded_cfg is not None and not st.session_state.get('_cfg_imported', False):
                try:
                    import json as _json
                    cfg_dict = _json.loads(uploaded_cfg.read())
                    imported = AnalysisConfig.from_dict(cfg_dict)
                    # Update widget session_state keys so sidebar reflects imported values
                    _cfg_widget_map = {
                        'cfg_amp_gain': imported.amplifier_gain,
                        'cfg_bp_low': imported.filtering.bandpass_low_hz,
                        'cfg_bp_high': imported.filtering.bandpass_high_hz,
                        'cfg_bd_min_dist': imported.beat_detection.min_distance_ms,
                        'cfg_bd_thresh': imported.beat_detection.threshold_factor,
                        'cfg_morph_thresh': imported.quality.morphology_threshold,
                        'cfg_use_morph': imported.quality.use_morphology,
                        'cfg_repol_start': imported.repolarization.search_start_ms,
                        'cfg_repol_end': imported.repolarization.search_end_ms,
                        'cfg_ead_prom': imported.arrhythmia.ead_residual_prominence,
                        'cfg_ead_min_amp': imported.arrhythmia.ead_residual_min_amp_frac,
                        'cfg_ead_min_w': imported.arrhythmia.ead_residual_min_width_ms,
                        'cfg_ead_max_w': imported.arrhythmia.ead_residual_max_width_ms,
                        'cfg_max_cv': imported.inclusion.max_cv_bp,
                        'cfg_min_fpd_conf': imported.inclusion.min_fpd_confidence,
                        'cfg_physiol': imported.inclusion.enabled_fpdc_physiol,
                        'cfg_norm_low': imported.normalization.threshold_low,
                        'cfg_norm_mid': imported.normalization.threshold_mid,
                        'cfg_norm_high': imported.normalization.threshold_high,
                    }
                    for wk, wv in _cfg_widget_map.items():
                        st.session_state[wk] = wv
                    # Selectbox widgets: set by index
                    _methods = ['auto', 'prominence', 'derivative', 'peak']
                    if imported.beat_detection.method in _methods:
                        st.session_state['cfg_bd_method'] = imported.beat_detection.method
                    _fpd_methods = ['tangent', 'peak', 'max_slope', '50pct', 'baseline_return', 'consensus']
                    if imported.repolarization.fpd_method in _fpd_methods:
                        st.session_state['cfg_fpd_method'] = imported.repolarization.fpd_method
                    _qt_methods = ['fridericia', 'bazett', 'none']
                    if imported.repolarization.correction in _qt_methods:
                        st.session_state['cfg_qt_corr'] = imported.repolarization.correction
                    _risk_modes = ['manual', 'data_driven']
                    if imported.arrhythmia.risk_score_mode in _risk_modes:
                        st.session_state['cfg_risk_mode'] = imported.arrhythmia.risk_score_mode
                    # Notch filter
                    _notch_val = imported.filtering.notch_freq_hz
                    if _notch_val in _notch_reverse:
                        st.session_state['cfg_notch'] = _notch_reverse[_notch_val]
                    st.session_state['_cfg_imported'] = True
                    st.success(T('config_loaded'))
                    st.rerun()
                except Exception as e:
                    st.error(f"{T('error')}: {e}")

    return config


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: Single File Analysis
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


def _plot_signal(result, key_suffix=""):
    """Plot filtered signal with beat markers and interactive beat editor."""
    import plotly.graph_objects as go

    sig_filt = result['filtered_signal']
    scale, y_label = _amplitude_scale(sig_filt)
    sig = sig_filt * scale
    t = result['time_vector']
    fs = result['metadata']['sample_rate']
    ks = key_suffix  # shorthand for widget key suffix

    # ── Build unified beat set with inclusion state ──
    # Merge auto-detected (pre-QC) + manually added beats into one set.
    # Session state tracks which are excluded.
    bi_raw = result['beat_indices_raw']

    # Initialize beat editor state on first run or when result changes
    result_id = id(result['filtered_signal'])  # changes when result changes
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
                    updated = _reanalyze_with_modified_beats(result, final_bi, config)
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


def _plot_beats(result):
    """Plot overlaid beat waveforms and template."""
    import plotly.graph_objects as go

    bd = result.get('beats_data', [])
    fs = result['metadata']['sample_rate']

    if not bd:
        st.warning(T('no_params'))
        return

    # Detect amplitude scale from first beat
    scale, y_label = _amplitude_scale(bd[0])

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
    tmpl = _compute_template(bd)
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


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: Batch Analysis + Risk Map
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
            except Exception as e:
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
            T('channel'): fi.get('analyzed_channel', ''),
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
    labels = []
    for i, r in enumerate(results):
        fname = r.get('metadata', {}).get('filename', f'file_{i}')
        ch = r.get('file_info', {}).get('analyzed_channel', '')
        labels.append(f"{fname} [{ch}]" if ch else fname)
    selected = st.selectbox(T('select_recording'), labels)

    idx = labels.index(selected)
    result = results[idx]

    fi = result['file_info']
    summary = result['summary']
    qc = result['qc_report']
    ar = result['arrhythmia_report']

    cols = st.columns(5)
    cols[0].metric(T('chip_channel'), f"{fi.get('chip', '?')} / {fi.get('analyzed_channel', '?')}")
    cols[1].metric(T('drug'), fi.get('drug', 'N/A'))
    cols[2].metric("QC Grade", qc.grade if qc else '?')
    cols[3].metric(T('fpdc'), f"{summary.get('fpdc_ms_mean', 0):.0f}")
    cols[4].metric(T('risk_score'), f"{ar.risk_score}/100" if ar else '?')

    tab_sig, tab_params, tab_arr = st.tabs([
        f"📈 {T('signal')}", f"📋 {T('params')}", f"⚠️ {T('arrhythmia')}"
    ])

    with tab_sig:
        # Signal plot (if filtered_signal available)
        if 'filtered_signal' in result and 'time_vector' in result:
            _plot_signal(result)
        else:
            st.info(T('signal_not_available'))

        # Beat overlay (using beat_template if beats_data was freed)
        tmpl = result.get('beat_template')
        if tmpl is not None and 'beats_data' not in result:
            import plotly.graph_objects as go
            fs = result['metadata']['sample_rate']
            scale, y_label = _amplitude_scale(tmpl)
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
            tmpl = _compute_template(bd) if bd else None
        if tmpl is not None:
            templates_to_plot.append((drug, tmpl, fs, i))

    # Auto-scale based on first template
    if templates_to_plot:
        scale, y_label = _amplitude_scale(templates_to_plot[0][1])
    else:
        scale, y_label = 1e3, T('amplitude_mV')

    for drug, tmpl, fs, i in templates_to_plot:
        t_ms = np.arange(len(tmpl)) / fs * 1000
        fig.add_trace(go.Scatter(
            x=t_ms, y=tmpl * scale,
            mode='lines', name=drug.capitalize(),
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        xaxis_title=T('time_ms'), yaxis_title=y_label,
        height=450, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
#  REPORT DOWNLOADS
# ═══════════════════════════════════════════════════════════════════════

def _download_reports(results, config, data_dir):
    """Provide download buttons for Excel, PDF, CDISC SEND reports."""
    st.subheader(f"📥 {T('download_reports')}")

    cols = st.columns(4)

    with cols[0]:
        if st.button(f"📊 {T('gen_excel')}", use_container_width=True):
            with st.spinner("Generazione Excel..."):
                try:
                    from cardiac_fp_analyzer.report import generate_excel_report
                    buf = io.BytesIO()
                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                        generate_excel_report(results, tmp.name)
                        with open(tmp.name, 'rb') as f:
                            buf.write(f.read())
                    st.download_button(
                        f"⬇️ {T('download_excel')}",
                        buf.getvalue(),
                        f"cardiac_fp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"{T('error')}: {e}")

    with cols[1]:
        if st.button(f"📄 {T('gen_pdf')}", use_container_width=True):
            with st.spinner("Generazione PDF..."):
                try:
                    from cardiac_fp_analyzer.report import generate_pdf_report
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        generate_pdf_report(results, tmp.name, str(data_dir))
                        with open(tmp.name, 'rb') as f:
                            pdf_bytes = f.read()
                    st.download_button(
                        f"⬇️ {T('download_pdf')}",
                        pdf_bytes,
                        f"cardiac_fp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"{T('error')}: {e}")

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
                    st.success(f"{T('cdisc_success')} TS, DM, EX, EG, RISK + define.xml")
                except Exception as e:
                    st.error(f"{T('error')} export CDISC: {e}")
                    st.code(traceback.format_exc())

    # CDISC Study ID (collapsed)
    with st.expander(f"🏛️ {T('cdisc_settings')}", expanded=False):
        st.session_state['cdisc_study_id'] = st.text_input(
            T('study_id'), value=st.session_state.get('cdisc_study_id', 'CIPA001'),
            help=T('study_id_help')
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — Navigation
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Build config from sidebar
    config = build_config_from_sidebar()
    st.session_state['_analysis_config'] = config

    # Navigation
    st.sidebar.divider()
    page = st.sidebar.radio(
        f"📑 {T('nav')}",
        [f"🔬 {T('single_file')}", f"📊 {T('batch')}", f"💊 {T('drug_comparison')}"],
    )

    # Language selector — bottom of sidebar
    st.sidebar.divider()
    lang = st.sidebar.selectbox(
        "🌐 Language",
        ['IT', 'EN'],
        index=0 if st.session_state.get('lang', 'it') == 'it' else 1,
        key='lang_select',
    )
    st.session_state['lang'] = lang.lower()

    if page == f"🔬 {T('single_file')}":
        page_single_file(config)
    elif page == f"📊 {T('batch')}":
        page_batch_analysis(config)
    elif page == f"💊 {T('drug_comparison')}":
        page_drug_comparison(config)


if __name__ == '__main__':
    main()
