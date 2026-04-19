"""Tests for the Battiti-tab template-representativity banner (v3.4.x).

The banner fires in ``plot_beats`` when the overlaid template is unlikely
to faithfully summarise the beats — i.e. when the rhythm classifier
reports a morphology-mixing type, or when the FPD coefficient of
variation is too high (T-waves fall at wildly different offsets and
cancel in the mean/median).

We don't try to exercise Streamlit's renderer here: we stub
``streamlit.warning`` to capture whether it was called and with what
content, and call the pure helper directly.  That keeps the test fast
and deterministic while still catching regressions in the gating logic.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from ui import display as display_mod  # noqa: E402
from ui.display import _render_template_representativity_banner  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────
#  Fixture — stub st.warning so we can assert it was/wasn't called.
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def captured_warnings(monkeypatch):
    calls: list[str] = []

    def fake_warning(msg, *_, **__):
        calls.append(str(msg))

    monkeypatch.setattr(st, 'warning', fake_warning)
    return calls


# ─────────────────────────────────────────────────────────────────────────
#  Helpers for building minimal ``result`` dicts
# ─────────────────────────────────────────────────────────────────────────

def _make_result(rhythm_type: str | None = 'regular',
                 fpd_mean: float = 400.0,
                 fpd_std: float = 10.0):
    rc = {}
    if rhythm_type is not None:
        rc['rhythm_type'] = rhythm_type
    return {
        'detection_info': {'rhythm_classification': rc},
        'summary': {'fpd_ms_mean': fpd_mean, 'fpd_ms_std': fpd_std},
    }


# ─────────────────────────────────────────────────────────────────────────
#  Gating: silent on benign cases
# ─────────────────────────────────────────────────────────────────────────

class TestBannerSilentOnBenignCases:
    def test_regular_rhythm_low_cv_no_banner(self, captured_warnings):
        """Regular rhythm + low FPD CV → no banner."""
        res = _make_result(rhythm_type='regular',
                           fpd_mean=400.0, fpd_std=10.0)  # CV 2.5%
        _render_template_representativity_banner(res)
        assert captured_warnings == []

    def test_regular_with_ectopics_low_cv_no_banner(self, captured_warnings):
        """regular_with_ectopics is handled elsewhere (passthrough/dominant
        filter); template remains meaningful on dominant beats so no
        banner here."""
        res = _make_result(rhythm_type='regular_with_ectopics',
                           fpd_mean=400.0, fpd_std=40.0)  # CV 10%
        _render_template_representativity_banner(res)
        assert captured_warnings == []

    def test_empty_rhythm_and_empty_summary_no_crash(self, captured_warnings):
        """Missing keys must not crash and must not spuriously fire."""
        res = {}
        _render_template_representativity_banner(res)
        assert captured_warnings == []


# ─────────────────────────────────────────────────────────────────────────
#  Gating: fires on risky rhythm types
# ─────────────────────────────────────────────────────────────────────────

class TestBannerFiresOnRiskyRhythm:
    @pytest.mark.parametrize('rhythm', [
        'chaotic', 'ambiguous', 'alternans_2_to_1', 'trimodal'
    ])
    def test_risky_rhythm_triggers_banner(self, captured_warnings, rhythm):
        res = _make_result(rhythm_type=rhythm,
                           fpd_mean=400.0, fpd_std=10.0)  # CV 2.5%
        _render_template_representativity_banner(res)
        assert len(captured_warnings) == 1, (
            f"risky rhythm '{rhythm}' must trigger the banner")
        # The banner text should mention the rhythm name
        assert rhythm in captured_warnings[0]


# ─────────────────────────────────────────────────────────────────────────
#  Gating: fires on high FPD CV
# ─────────────────────────────────────────────────────────────────────────

class TestBannerFiresOnHighFPDCV:
    def test_high_fpd_cv_triggers_banner(self, captured_warnings):
        """FPD CV > 20 % → banner."""
        res = _make_result(rhythm_type='regular',
                           fpd_mean=400.0, fpd_std=100.0)  # CV 25%
        _render_template_representativity_banner(res)
        assert len(captured_warnings) == 1

    def test_just_at_threshold_does_not_fire(self, captured_warnings):
        """CV = 20 % exactly must NOT fire (strict `>` threshold)."""
        res = _make_result(rhythm_type='regular',
                           fpd_mean=400.0, fpd_std=80.0)  # CV = 20%
        _render_template_representativity_banner(res)
        assert captured_warnings == []

    def test_zero_fpd_mean_guards_against_div_by_zero(self, captured_warnings):
        """If FPD mean is zero/missing, CV is treated as 0 — no fire."""
        res = _make_result(rhythm_type='regular',
                           fpd_mean=0.0, fpd_std=50.0)
        _render_template_representativity_banner(res)
        assert captured_warnings == []


# ─────────────────────────────────────────────────────────────────────────
#  Gating: both triggers together — single banner with both reasons
# ─────────────────────────────────────────────────────────────────────────

class TestBannerBothReasons:
    def test_chaotic_plus_high_cv_single_banner_both_reasons(
            self, captured_warnings):
        res = _make_result(rhythm_type='chaotic',
                           fpd_mean=400.0, fpd_std=120.0)  # CV 30%
        _render_template_representativity_banner(res)
        assert len(captured_warnings) == 1
        text = captured_warnings[0]
        assert 'chaotic' in text
        # Reason body must mention the CV percentage (30%)
        assert '30' in text


# ─────────────────────────────────────────────────────────────────────────
#  Constants sanity
# ─────────────────────────────────────────────────────────────────────────

class TestBannerConstants:
    def test_risky_rhythm_set_contents(self):
        risky = display_mod._TEMPLATE_RISKY_RHYTHM_TYPES
        assert 'chaotic' in risky
        assert 'ambiguous' in risky
        assert 'alternans_2_to_1' in risky
        assert 'trimodal' in risky
        # These should NOT be in the risky set (they pass through with
        # representative templates after dominant-cluster filtering):
        assert 'regular' not in risky
        assert 'regular_with_ectopics' not in risky
        assert 'regular_with_noise' not in risky

    def test_fpd_cv_threshold_is_reasonable(self):
        cv = display_mod._FPD_CV_TEMPLATE_WARN
        assert 0.10 <= cv <= 0.30, (
            'Threshold should stay in [10 %, 30 %] — outside that '
            'range it either fires on clean signals or never fires on '
            'variable ones')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
