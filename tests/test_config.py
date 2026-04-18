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


class TestCvThresholdRename:
    """Verify the cv_good / cv_fair / cv_marginal rename (-> *_frac).

    The rename fixed a semantic trap where BeatDetectionConfig and
    ChannelSelectionConfig both declared identically-named fields in
    different units (fraction vs percent). The channel-selection fields
    were dead code and were removed; the beat-detection fields gained
    a `_frac` suffix to make the units obvious. A legacy migration in
    `from_dict` preserves back-compat for saved JSON configs.
    """

    def test_beat_detection_fields_are_fractions(self):
        bd = AnalysisConfig().beat_detection
        # New names exist, in fraction units (0..1)
        assert hasattr(bd, 'cv_good_frac')
        assert hasattr(bd, 'cv_fair_frac')
        assert hasattr(bd, 'cv_marginal_frac')
        assert 0.0 < bd.cv_good_frac < 1.0
        assert bd.cv_good_frac < bd.cv_fair_frac < bd.cv_marginal_frac
        # Old names are gone
        assert not hasattr(bd, 'cv_good')
        assert not hasattr(bd, 'cv_fair')
        assert not hasattr(bd, 'cv_marginal')

    def test_channel_selection_cv_fields_removed(self):
        """The dead `cv_*` percent-unit fields on ChannelSelectionConfig
        have been removed entirely."""
        cs = AnalysisConfig().channel_selection
        assert not hasattr(cs, 'cv_excellent')
        assert not hasattr(cs, 'cv_good')
        assert not hasattr(cs, 'cv_fair')

    def test_legacy_json_field_names_migrate(self):
        """A config saved by an older version (cv_good/cv_fair/cv_marginal
        under `beat_detection`) must still load into the renamed fields."""
        legacy = {
            'beat_detection': {
                'cv_good': 0.12,
                'cv_fair': 0.28,
                'cv_marginal': 0.45,
            },
        }
        cfg = AnalysisConfig.from_dict(legacy)
        assert cfg.beat_detection.cv_good_frac == 0.12
        assert cfg.beat_detection.cv_fair_frac == 0.28
        assert cfg.beat_detection.cv_marginal_frac == 0.45

    def test_legacy_channel_selection_cv_fields_silently_ignored(self):
        """Old JSON configs also had dead `cv_excellent/cv_good/cv_fair`
        under `channel_selection`. They were never read; loading them
        must not raise."""
        legacy = {
            'channel_selection': {
                'cv_excellent': 10.0,
                'cv_good': 20.0,
                'cv_fair': 35.0,
            },
        }
        # Must not raise
        cfg = AnalysisConfig.from_dict(legacy)
        assert isinstance(cfg, AnalysisConfig)

    def test_new_names_roundtrip(self):
        cfg = AnalysisConfig()
        cfg.beat_detection.cv_good_frac = 0.10
        cfg.beat_detection.cv_fair_frac = 0.25
        cfg.beat_detection.cv_marginal_frac = 0.40
        restored = AnalysisConfig.from_dict(json.loads(cfg.to_json()))
        assert restored.beat_detection.cv_good_frac == 0.10
        assert restored.beat_detection.cv_fair_frac == 0.25
        assert restored.beat_detection.cv_marginal_frac == 0.40

    def test_new_value_wins_over_legacy_if_both_present(self):
        """If a dict somehow contains BOTH old and new names, the new
        name wins (migration is skipped when the new key already exists)."""
        mixed = {
            'beat_detection': {
                'cv_good': 0.99,        # legacy (should be ignored)
                'cv_good_frac': 0.13,   # new (should win)
            },
        }
        cfg = AnalysisConfig.from_dict(mixed)
        assert cfg.beat_detection.cv_good_frac == 0.13
