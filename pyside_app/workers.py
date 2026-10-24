"""Thread pools that can run the analysis pipeline.

macOS gives secondary threads a 512 KB stack. OpenBLAS's parallel LU factorisation
(``dgetrf_parallel``), reached by every ``np.linalg.solve`` — scipy's ``filtfilt`` calls one through
``lfilter_zi`` — allocates its work table on the stack and needs more than that: the process dies
with ``SIGBUS: Thread stack size exceeded`` (review queue, MacBook Pro M1, numpy 1.26, 24 Oct 2026).
CPython's own threads get 16 MB on macOS; the pools made here give Qt's threads the same.
"""
from __future__ import annotations

from PySide6.QtCore import QThreadPool

WORKER_STACK_BYTES = 16 * 1024 * 1024


def analysis_pool(parent=None, max_threads: int | None = None) -> QThreadPool:
    """A ``QThreadPool`` whose worker threads have a 16 MB stack (set before any thread is created)."""
    pool = QThreadPool(parent)
    pool.setStackSize(WORKER_STACK_BYTES)
    if max_threads:
        pool.setMaxThreadCount(max(1, int(max_threads)))
    return pool


__all__ = ["analysis_pool", "WORKER_STACK_BYTES"]
