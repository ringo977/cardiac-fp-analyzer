"""PoC PySide6 UI for cardiac-fp-analyzer (ADR-0001, Phase 0).

Scope: validate that PySide6 + pyqtgraph can replace the Streamlit UI
for the one thing Streamlit could not do — click-to-add and
click-to-remove beats on a real signal from
:func:`cardiac_fp_analyzer.analyze.analyze_single_file`.

Run with:
    python -m pyside_app.main
"""
