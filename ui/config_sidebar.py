"""
Sidebar configuration panel — builds AnalysisConfig from Streamlit widgets.
"""

import streamlit as st

from cardiac_fp_analyzer.config import AnalysisConfig
from ui.i18n import T


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
                    # Selectbox widgets: set by value
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
                    _notch_reverse_map = {50.0: '50 Hz', 60.0: '60 Hz', 0.0: 'Off'}
                    _notch_val = imported.filtering.notch_freq_hz
                    if _notch_val in _notch_reverse_map:
                        st.session_state['cfg_notch'] = _notch_reverse_map[_notch_val]
                    st.session_state['_cfg_imported'] = True
                    st.success(T('config_loaded'))
                    st.rerun()
                except Exception as e:
                    st.error(f"{T('error')}: {e}")

    return config
