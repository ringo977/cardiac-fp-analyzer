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
import sys
from pathlib import Path

import streamlit as st

# Suppress noisy third-party warnings in the GUI (streamlit, matplotlib).
# Pipeline warnings from cardiac_fp_analyzer are handled by logging.
warnings.filterwarnings('ignore', module='streamlit')
warnings.filterwarnings('ignore', module='matplotlib')
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

# ── Ensure project root is on sys.path (for development without pip install) ──
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── UI module imports ──
from ui.i18n import T
from ui.config_sidebar import build_config_from_sidebar
from ui.single_file import page_single_file
from ui.batch import page_batch_analysis
from ui.drug_comparison import page_drug_comparison


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
