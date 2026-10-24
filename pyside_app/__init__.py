"""PoC PySide6 UI for cardiac-fp-analyzer (ADR-0001, Phase 0).

Scope: validate that PySide6 + pyqtgraph can replace the Streamlit UI
for the one thing Streamlit could not do — click-to-add and
click-to-remove beats on a real signal from
:func:`cardiac_fp_analyzer.analyze.analyze_single_file`.

Run with:
    python -m pyside_app.main
"""
import os as _os

# One BLAS thread per worker: the pipeline is 1-D signal work (no large matrices) and a threaded
# BLAS would only oversubscribe the review-queue workers; it also keeps OpenBLAS away from
# ``dgetrf_parallel``. Read by OpenBLAS at import time, so it must be set before numpy is imported.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
