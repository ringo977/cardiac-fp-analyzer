"""
Report download widgets for batch analysis results.
"""

import io
import tempfile
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from ui.i18n import T


def download_reports(results, config, data_dir):
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
                except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
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
                except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
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
                except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
                    st.error(f"{T('error')} export CDISC: {e}")
                    st.code(traceback.format_exc())

    # CDISC Study ID (collapsed)
    with st.expander(f"🏛️ {T('cdisc_settings')}", expanded=False):
        st.session_state['cdisc_study_id'] = st.text_input(
            T('study_id'), value=st.session_state.get('cdisc_study_id', 'CIPA001'),
            help=T('study_id_help')
        )
