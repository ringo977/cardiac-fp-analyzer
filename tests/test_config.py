"""Tests for AnalysisConfig — JSON round-trip, defaults, presets."""

import json

from cardiac_fp_analyzer.config import AnalysisConfig


class TestAnalysisConfigDefaults:
    """Verify default configuration values are sensible."""

    def test_default_amplifier_gain(self):
        cfg = AnalysisConfig()
        assert cfg.amplifier_gain == 1.0

    def test_default_beat_detection_method(self):
        cfg = AnalysisConfig()
        assert cfg.beat_detection.method == 'auto'

    def test_default_correction_fridericia(self):
        cfg = AnalysisConfig()
        assert cfg.repolarization.correction == 'fridericia'

    def test_default_quality_morphology_threshold(self):
        cfg = AnalysisConfig()
        assert 0.0 < cfg.quality.morphology_threshold < 1.0


class TestAnalysisConfigJSON:
    """Verify JSON serialization round-trips correctly."""

    def test_to_json_returns_valid_json(self):
        cfg = AnalysisConfig()
        json_str = cfg.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_round_trip_preserves_values(self):
        cfg = AnalysisConfig()
        cfg.amplifier_gain = 5000
        cfg.beat_detection.min_distance_ms = 300.0
        cfg.repolarization.fpd_method = 'consensus'

        json_str = cfg.to_json()
        restored = AnalysisConfig.from_dict(json.loads(json_str))

        assert restored.amplifier_gain == 5000
        assert restored.beat_detection.min_distance_ms == 300.0
        assert restored.repolarization.fpd_method == 'consensus'

    def test_from_dict_ignores_unknown_keys(self):
        cfg_dict = json.loads(AnalysisConfig().to_json())
        cfg_dict['unknown_future_key'] = 42
        # Should not raise
        restored = AnalysisConfig.from_dict(cfg_dict)
        assert isinstance(restored, AnalysisConfig)


class TestScoringWeights:
    """Verify configurable scoring weights are present and serializable."""

    def test_beat_detection_weights_exist(self):
        cfg = AnalysisConfig()
        bd = cfg.beat_detection
        assert hasattr(bd, 'score_bp_ideal')
        assert hasattr(bd, 'score_bp_extended')
        assert hasattr(bd, 'score_cv_good')
        assert bd.score_bp_ideal > 0

    def test_channel_selection_weights_exist(self):
        cfg = AnalysisConfig()
        cs = cfg.channel_selection
        assert hasattr(cs, 'w_bp_range')
        assert hasattr(cs, 'w_corr_max')
        assert hasattr(cs, 'w_amplitude_max')
        assert cs.w_bp_range > 0

    def test_weights_survive_json_roundtrip(self):
        cfg = AnalysisConfig()
        cfg.beat_detection.score_bp_ideal = 42
        cfg.channel_selection.w_corr_max = 99

        restored = AnalysisConfig.from_dict(json.loads(cfg.to_json()))
        assert restored.beat_detection.score_bp_ideal == 42
        assert restored.channel_selection.w_corr_max == 99
