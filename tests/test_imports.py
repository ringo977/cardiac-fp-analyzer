"""Smoke tests — verify all package modules import without errors."""

import pytest


class TestCoreImports:
    """Verify all core library modules are importable."""

    def test_import_config(self):
        from cardiac_fp_analyzer.config import AnalysisConfig
        assert AnalysisConfig is not None

    def test_import_loader(self):
        from cardiac_fp_analyzer.loader import load_csv, parse_filename
        assert callable(load_csv)
        assert callable(parse_filename)

    def test_import_filtering(self):
        from cardiac_fp_analyzer.filtering import full_filter_pipeline
        assert callable(full_filter_pipeline)

    def test_import_beat_detection(self):
        from cardiac_fp_analyzer.beat_detection import detect_beats, segment_beats
        assert callable(detect_beats)

    def test_import_parameters(self):
        from cardiac_fp_analyzer.parameters import extract_all_parameters
        assert callable(extract_all_parameters)

    def test_import_quality_control(self):
        from cardiac_fp_analyzer.quality_control import validate_beats
        assert callable(validate_beats)

    def test_import_arrhythmia(self):
        from cardiac_fp_analyzer.arrhythmia import analyze_arrhythmia, compute_template
        assert callable(analyze_arrhythmia)
        assert callable(compute_template)

    def test_import_normalization(self):
        from cardiac_fp_analyzer.normalization import is_baseline, get_group_key
        assert callable(is_baseline)
        assert callable(get_group_key)

    def test_import_analyze(self):
        from cardiac_fp_analyzer.analyze import analyze_single_file, batch_analyze
        assert callable(analyze_single_file)
        assert callable(batch_analyze)

    def test_import_channel_selection(self):
        from cardiac_fp_analyzer.channel_selection import select_best_channel
        assert callable(select_best_channel)

    def test_import_inclusion(self):
        from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria
        assert callable(apply_inclusion_criteria)

    def test_import_repolarization(self):
        from cardiac_fp_analyzer.repolarization import (
            find_repolarization_on_template,
            find_repolarization_per_beat,
            apply_fpd_method,
        )
        assert callable(find_repolarization_on_template)
        assert callable(find_repolarization_per_beat)
        assert callable(apply_fpd_method)


class TestBackCompatAliases:
    """Verify private-name aliases still work for backward compatibility."""

    def test_compute_template_alias(self):
        from cardiac_fp_analyzer.arrhythmia import _compute_template, compute_template
        assert _compute_template is compute_template

    def test_is_baseline_alias(self):
        from cardiac_fp_analyzer.normalization import _is_baseline, is_baseline
        assert _is_baseline is is_baseline

    def test_get_group_key_alias(self):
        from cardiac_fp_analyzer.normalization import _get_group_key, get_group_key
        assert _get_group_key is get_group_key

    def test_analyze_back_compat_aliases(self):
        from cardiac_fp_analyzer.analyze import (
            _select_best_channel, select_best_channel,
            _apply_inclusion_criteria, apply_inclusion_criteria,
        )
        # These are the same function objects re-exported as back-compat aliases
        assert callable(_select_best_channel)
        assert callable(_apply_inclusion_criteria)

    def test_repolarization_back_compat_aliases(self):
        from cardiac_fp_analyzer.repolarization import (
            _find_repolarization_on_template, find_repolarization_on_template,
            _apply_fpd_method, apply_fpd_method,
        )
        assert _find_repolarization_on_template is find_repolarization_on_template
        assert _apply_fpd_method is apply_fpd_method


class TestVersion:
    """Verify version string is set."""

    def test_version_string(self):
        import cardiac_fp_analyzer
        assert cardiac_fp_analyzer.__version__ == '3.2.1'

    def test_version_format(self):
        import cardiac_fp_analyzer
        parts = cardiac_fp_analyzer.__version__.split('.')
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
