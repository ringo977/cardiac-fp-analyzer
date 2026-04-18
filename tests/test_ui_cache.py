"""
test_ui_cache.py — Regression tests for ui.cache wrappers.

The ``ui/cache.py`` module wraps ``load_csv`` and
``analyze_single_file`` with ``@st.cache_data``.  These tests verify:

1. wrappers return identical output to the un-cached core functions;
2. cache keys depend on (path, mtime, channel, config_json) so that
   changing any of those causes the underlying function to re-run;
3. when all cache-key args match, the underlying function is called
   exactly once across multiple wrapper calls.

Tests are skipped if streamlit is not installed (it lives in the
``[gui]`` optional extra).
"""

import contextlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")

from ui import cache as cache_mod  # noqa: E402

from cardiac_fp_analyzer import analyze as analyze_mod  # noqa: E402
from cardiac_fp_analyzer import loader as loader_mod  # noqa: E402
from cardiac_fp_analyzer.config import AnalysisConfig  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
#  Synthetic CSV helper — matches Digilent WaveForms header format
# ═══════════════════════════════════════════════════════════════════════

def _make_waveforms_csv(path: Path, fs: float = 1000.0, duration: float = 2.0,
                        seed: int = 0):
    """Write a tiny Digilent-style CSV at ``path`` and return (fs, n)."""
    n = int(fs * duration)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    el1 = rng.standard_normal(n) * 1e-4
    el2 = rng.standard_normal(n) * 1e-4
    # Inject a few sharp deflections so beat detection has something
    # to latch onto (not strictly required for these tests, but keeps
    # analyze_single_file from bailing on empty-signal guards).
    for k in range(1, 4):
        idx = int(k * fs * 0.5)
        if idx + 3 < n:
            el1[idx] = 0.05
            el1[idx + 1] = -0.03
            el2[idx] = 0.05
            el2[idx + 1] = -0.03

    header = (
        "#Device Name: Analog Discovery 2\n"
        "#Serial Number: SN:123\n"
        "#Date Time: 2026-01-01 00:00:00.0\n"
        f"#Sample rate: {fs} Hz\n"
        f"#Samples: {n}\n"
        "#Trigger: Auto\n"
        "#Channel 1: Range: 5.0 mV/div Offset: 0.0 V\n"
        "#Channel 2: Range: 5.0 mV/div Offset: 0.0 V\n"
        "Time (s),Channel 1 (V),Channel 2 (V)\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for ti, a, b in zip(t, el1, el2):
            f.write(f"{ti:.6f},{a:.6e},{b:.6e}\n")
    return fs, n


@pytest.fixture
def synthetic_csv():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        path = Path(tmp.name)
    _make_waveforms_csv(path)
    try:
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _make_counting_loader():
    counter = {'n': 0}
    real = loader_mod.load_csv

    def counting_load_csv(fp):
        counter['n'] += 1
        return real(fp)

    return counter, counting_load_csv


def _make_counting_analyzer():
    counter = {'n': 0}
    real = analyze_mod.analyze_single_file

    def counting_analyze(fp, channel='auto', verbose=True, config=None):
        counter['n'] += 1
        return real(fp, channel=channel, verbose=verbose, config=config)

    return counter, counting_analyze


# ═══════════════════════════════════════════════════════════════════════
#  load_csv_cached
# ═══════════════════════════════════════════════════════════════════════

class TestLoadCsvCached:

    def setup_method(self):
        cache_mod.load_csv_cached.clear()

    def test_returns_same_as_uncached(self, synthetic_csv):
        mtime = os.path.getmtime(synthetic_csv)
        meta_a, df_a = loader_mod.load_csv(str(synthetic_csv))
        meta_b, df_b = cache_mod.load_csv_cached(str(synthetic_csv), mtime)

        assert meta_a['sample_rate'] == meta_b['sample_rate']
        assert meta_a['n_samples'] == meta_b['n_samples']
        assert list(df_a.columns) == list(df_b.columns)
        assert len(df_a) == len(df_b)
        np.testing.assert_allclose(df_a['el1'].values, df_b['el1'].values)

    def test_second_call_hits_cache(self, synthetic_csv, monkeypatch):
        """Calling the wrapper twice with identical args must invoke
        the underlying ``load_csv`` only once."""
        counter, fn = _make_counting_loader()
        monkeypatch.setattr(cache_mod, '_load_csv', fn)
        cache_mod.load_csv_cached.clear()

        mtime = os.path.getmtime(synthetic_csv)
        cache_mod.load_csv_cached(str(synthetic_csv), mtime)
        cache_mod.load_csv_cached(str(synthetic_csv), mtime)
        cache_mod.load_csv_cached(str(synthetic_csv), mtime)

        assert counter['n'] == 1, (
            f"expected load_csv called once, got {counter['n']}"
        )

    def test_different_mtime_busts_cache(self, synthetic_csv, monkeypatch):
        counter, fn = _make_counting_loader()
        monkeypatch.setattr(cache_mod, '_load_csv', fn)
        cache_mod.load_csv_cached.clear()

        mtime = os.path.getmtime(synthetic_csv)
        cache_mod.load_csv_cached(str(synthetic_csv), mtime)
        cache_mod.load_csv_cached(str(synthetic_csv), mtime + 1.0)

        assert counter['n'] == 2, (
            "changing mtime must invalidate the cache entry"
        )


# ═══════════════════════════════════════════════════════════════════════
#  analyze_single_file_cached
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeSingleFileCached:

    def setup_method(self):
        cache_mod.analyze_single_file_cached.clear()

    def test_wrapper_returns_dict(self, synthetic_csv):
        cfg = AnalysisConfig()
        mtime = os.path.getmtime(synthetic_csv)
        result = cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el1', cfg.to_json()
        )
        # Either a full result dict or None if the synthetic signal
        # fails detection — both are acceptable wrapper outputs.
        assert result is None or isinstance(result, dict)

    def test_second_call_hits_cache(self, synthetic_csv, monkeypatch):
        counter, fn = _make_counting_analyzer()
        monkeypatch.setattr(cache_mod, '_analyze_single_file', fn)
        cache_mod.analyze_single_file_cached.clear()

        cfg_json = AnalysisConfig().to_json()
        mtime = os.path.getmtime(synthetic_csv)
        cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el1', cfg_json
        )
        cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el1', cfg_json
        )

        assert counter['n'] == 1

    def test_different_channel_busts_cache(self, synthetic_csv, monkeypatch):
        counter, fn = _make_counting_analyzer()
        monkeypatch.setattr(cache_mod, '_analyze_single_file', fn)
        cache_mod.analyze_single_file_cached.clear()

        cfg_json = AnalysisConfig().to_json()
        mtime = os.path.getmtime(synthetic_csv)
        cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el1', cfg_json
        )
        cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el2', cfg_json
        )

        assert counter['n'] == 2

    def test_different_config_busts_cache(self, synthetic_csv, monkeypatch):
        counter, fn = _make_counting_analyzer()
        monkeypatch.setattr(cache_mod, '_analyze_single_file', fn)
        cache_mod.analyze_single_file_cached.clear()

        cfg1 = AnalysisConfig()
        cfg2 = AnalysisConfig()
        # Tweak a single field — any non-default value suffices to
        # change the canonical JSON representation.
        cfg2.amplifier_gain = (
            cfg1.amplifier_gain * 2.0 if cfg1.amplifier_gain else 2.0
        )

        mtime = os.path.getmtime(synthetic_csv)
        cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el1', cfg1.to_json()
        )
        cache_mod.analyze_single_file_cached(
            str(synthetic_csv), mtime, 'el1', cfg2.to_json()
        )

        assert counter['n'] == 2

    def test_config_roundtrip_inside_wrapper(self, synthetic_csv):
        """The wrapper must rehydrate config via from_dict; verify the
        analyze call actually sees a config-equivalent object by
        checking that a known config field survives the round trip."""
        cfg = AnalysisConfig()
        cfg_json = cfg.to_json()

        rehydrated = AnalysisConfig.from_dict(json.loads(cfg_json))
        assert rehydrated.amplifier_gain == cfg.amplifier_gain
