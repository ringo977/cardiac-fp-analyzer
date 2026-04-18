"""
ui/cache.py — Streamlit-aware caching wrappers for core pipeline calls.

These are thin wrappers around pure-Python core functions (``load_csv``,
``analyze_single_file``) decorated with ``@st.cache_data`` so that
repeated Streamlit reruns on the same inputs don't re-parse the CSV or
re-run the full pipeline.

Design notes
------------

**Why wrappers instead of decorating core functions directly?**
The core package (``cardiac_fp_analyzer``) must remain usable without
Streamlit installed — ``streamlit`` lives in the ``[gui]`` optional
extra.  Decorating a core function with ``@st.cache_data`` would add a
hard streamlit dependency at import time.  The wrapper lives in
``ui/`` which is streamlit-only by design.

**Why pass ``mtime`` explicitly?**
``@st.cache_data`` hashes all positional/keyword arguments.  The
file's path alone would cache the *first* version of the file forever
— if the file is overwritten under the same path the cached result
would be stale.  Passing ``os.path.getmtime(path)`` as a separate
argument ties the cache entry to a specific filesystem snapshot.

**Why pass config as a JSON string?**
``AnalysisConfig`` is a dataclass with nested dataclasses; streamlit's
default hasher would either deep-hash it (slow) or pickle-fallback
(brittle).  Using the canonical JSON serialization is cheap, stable,
and guaranteed-hashable across reruns.

**Why ``minmax_downsample`` is NOT cached here**
The v3.3.0 assessment originally suggested also wrapping
``minmax_downsample``.  An audit of the call graph shows that helper
is only invoked from the matplotlib-based PDF report pipeline
(``cardiac_fp_analyzer.plotting.plot_raw_trace``), which is triggered
exclusively from the "Generate PDF" button — not on every Streamlit
rerun.  The actual UI signal plot in ``ui/display.py`` uses an
inline stride downsample (``sig[::step]``) which is already O(1)
memory and negligible CPU.  Caching ``minmax_downsample`` would add
complexity without user-visible benefit, so it is intentionally
skipped.
"""
from __future__ import annotations

import json

import streamlit as st

from cardiac_fp_analyzer.analyze import analyze_single_file as _analyze_single_file
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.loader import load_csv as _load_csv


@st.cache_data(show_spinner=False, max_entries=32)
def load_csv_cached(filepath: str, mtime: float):
    """Cached CSV parse.  Cache key = ``(filepath, mtime)``.

    Parameters
    ----------
    filepath : str
        Absolute path to the CSV file.
    mtime : float
        ``os.path.getmtime(filepath)`` — included so that overwriting
        the file invalidates the cache entry.

    Returns
    -------
    (metadata, df) : same tuple shape as
        :func:`cardiac_fp_analyzer.loader.load_csv`.
    """
    del mtime  # used only for cache-key purposes
    return _load_csv(filepath)


@st.cache_data(show_spinner=False, max_entries=16)
def analyze_single_file_cached(
    filepath: str,
    mtime: float,
    channel: str,
    config_json: str,
):
    """Cached single-file analysis.

    Cache key = ``(filepath, mtime, channel, config_json)``.

    Parameters
    ----------
    filepath : str
        Absolute path to the CSV file.
    mtime : float
        ``os.path.getmtime(filepath)``.
    channel : str
        ``'auto'``, ``'el1'``, or ``'el2'``.
    config_json : str
        Canonical JSON serialization of the active ``AnalysisConfig``
        (``config.to_json()``) — used both as a stable hashable cache
        key *and* to rehydrate the config inside the wrapper.

    Returns
    -------
    result : dict | None
        Same shape as
        :func:`cardiac_fp_analyzer.analyze.analyze_single_file`.
    """
    del mtime  # used only for cache-key purposes
    cfg = AnalysisConfig.from_dict(json.loads(config_json))
    return _analyze_single_file(filepath, channel=channel, verbose=False, config=cfg)
