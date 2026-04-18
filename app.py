#!/usr/bin/env python3
"""
Cardiac FP Analyzer — Streamlit GUI

Launch:
    streamlit run app.py

The UI is split into modules under the ``ui/`` package:
  i18n             — Translations and T() helper
  helpers          — Shared utility functions (reanalyze, amplitude_scale)
  config_sidebar   — Sidebar configuration builder
  single_file      — Single file analysis page
  batch            — Batch analysis + risk map page
  drug_comparison  — Drug comparison dashboard
  reports          — Report download widgets
"""

import logging
import warnings

import streamlit as st

# Suppress noisy third-party warnings in the GUI.
# Only target specific modules — no blanket category suppression.
# Pipeline warnings from cardiac_fp_analyzer are preserved and handled by logging.
warnings.filterwarnings('ignore', module='streamlit')
warnings.filterwarnings('ignore', module='matplotlib')
warnings.filterwarnings('ignore', module='plotly')

# Configure logging for the GUI — INFO level by default,
# DEBUG available via ?debug=1 query param (handled below).
logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Must be first Streamlit call ──
st.set_page_config(
    page_title="Cardiac FP Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── UI module imports ──
from ui.batch import page_batch_analysis
from ui.config_sidebar import build_config_from_sidebar
from ui.drug_comparison import page_drug_comparison
from ui.i18n import T
from ui.single_file import page_single_file


def main():
    # Build config from sidebar
    config = build_config_from_sidebar()
    st.session_state['_analysis_config'] = config

    # Language selector — must be BEFORE navigation so that T() labels
    # are already in the correct language when the radio is built.
    st.sidebar.divider()
    lang = st.sidebar.selectbox(
        "🌐 Language",
        ['IT', 'EN'],
        index=0 if st.session_state.get('lang', 'it') == 'it' else 1,
        key='lang_select',
    )
    st.session_state['lang'] = lang.lower()

    # Navigation — use a stable index so language switches don't lose
    # the selected page (label strings change, index doesn't).
    st.sidebar.divider()
    page_labels = [
        f"🔬 {T('single_file')}",
        f"📊 {T('batch')}",
        f"💊 {T('drug_comparison')}",
    ]
    page_idx = st.sidebar.radio(
        f"📑 {T('nav')}",
        range(len(page_labels)),
        format_func=lambda i: page_labels[i],
        key='nav_page',
    )

    if page_idx == 0:
        page_single_file(config)
    elif page_idx == 1:
        page_batch_analysis(config)
    elif page_idx == 2:
        page_drug_comparison(config)


if __name__ == '__main__':
    main()
