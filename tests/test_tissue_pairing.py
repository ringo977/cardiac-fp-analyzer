"""Tissue identity, baseline pairing and drug grouping (Oct 2026).

Running the batch on the Visone 2023 data set exposed:
  1. the experiment was taken only from folders starting with upper-case
     'EXP' and the day was ignored → with 'Exp5' / 'exp1' / 'Exp 5' folders
     baselines were shared across experiments and days;
  2. classify_drug grouped raw drug strings → 'DOFE' and 'Dofetilide' were
     two drugs with opposite calls;
  3. channel='auto' chose the electrode file by file → one dose series mixed
     el1 and el2 of the same microtissue;
  4. Fridericia at RR = 40 s (tissue almost stopped) → %ΔFPDcF = +580 %.
These tests pin the fixes on synthetic results; no signal is analysed.
"""

import numpy as np
import pytest

import cardiac_fp_analyzer.analyze as analyze_mod
from cardiac_fp_analyzer.config import AnalysisConfig, NormalizationConfig
from cardiac_fp_analyzer.loader import describe_recording, parse_filename
from cardiac_fp_analyzer.normalization import (
    canonical_drug_name,
    classify_drug,
    compute_normalized_parameters,
    normalize_all_results,
    pair_with_baselines,
    recording_key,
)

# ── 1. tissue identity from the path ────────────────────────────────────


@pytest.mark.parametrize('path, tissue', [
    ('Data/Exp5/Day7/chipA/chipA_ch1_terfe_300nM.csv', 'exp5/day7/chipA_ch1'),
    ('Data/EXP 5/chipA_ch1_terfe_300nM.csv', 'exp5/-/chipA_ch1'),
    ('Data/exp1/day6/chipB_ch2_baseline.csv', 'exp1/day6/chipB_ch2'),
    ('Data/Exp_10/chipC_ch3_Dofe_1nM.csv', 'exp10/-/chipC_ch3'),
    ('Experiments-GG/Exp 5/Day8/A/Exp5_chipB_ch3_Ti01_A.csv', 'exp5/day8/chipB_ch3'),
    ('Data/Exp7/wetransfer-39e79e/EXP7/chipA_ch1_NIFE_1nM.csv', 'exp7/-/chipA_ch1'),
    # Accelera layout: chip in the folder, chamber prefix in the name
    ('Data/Exp12_Accelera/Day 7/Chip 537/Ch2_DMSO10-1_2000fs_1_3K.csv', 'exp12/day7/chip537_ch2'),
    ('Data/Exp12_Accelera/Day8/529/Ch1_DMSO10-2_2000fs_01_3K.csv', 'exp12/day8/chip529_ch1'),
    ('Data/Exp13_Accelera/Chip 569/Ch2 Bepridil/Ch2_Bepridil001_2000fs_01_3K_diff.csv', 'exp13/-/chip569_ch2'),
    ('Data/Exp11/day 8 Accelera/Chip 200/Ch1_100nMCisapride_20KHz_1Hz_3KHzfilter_diff.csv',
     'exp11/day8/chip200_ch1'),
])
def test_tissue_key_from_layout(path, tissue):
    assert describe_recording(path)['tissue'] == tissue


def test_same_chip_in_two_experiments_or_days_are_different_tissues():
    keys = {describe_recording(p)['tissue'] for p in (
        'D/Exp5/Day7/chipA_ch1_baseline.csv', 'D/Exp8/Day7/chipA_ch1_baseline.csv',
        'D/Exp8/Day6/chipA_ch1_baseline.csv')}
    assert len(keys) == 3


def test_parent_folder_named_experiments_is_not_an_experiment():
    info = describe_recording('Experiments-Toxicol.Science/chipA_ch1_terfe_1nM.csv')
    assert 'experiment' not in info and info['tissue'] == '-/-/chipA_ch1'


def test_two_tissues_in_one_file_are_flagged_not_keyed():
    info = describe_recording('GG/Exp 5/Day8/baseline/Exp5_chipC_ch1_chipA_ch1_baseline.csv')
    assert info['dual_tissue'] is True and 'tissue' not in info
    assert info['tissue_tokens'] == ['chipC_ch1', 'chipA_ch1']


@pytest.mark.parametrize('name, drug, conc, baseline', [
    ('chipA_ch1_terfe_1000nM.csv', 'terfe', '1000 nM', False),      # was '.1000 nM'
    ('chipA_ch3_DOFETILIDE_0_3_nM.csv', 'DOFETILIDE', '0.3 nM', False),
    ('chipC_ch2_CISAP_2_5nM.csv', 'CISAP', '2.5 nM', False),
    ('chipE_ch2_NIFEDIPINE_10.csv', 'NIFEDIPINE', '10', False),     # was drug 'NIFEDIPINE 10'
    ('Ch3_Alfu100nM_20000_1_3k_chan1_dif.csv', 'Alfu', '100 nM', False),
    ('Ch1_100nMCisapride_20KHz_1Hz_3KHzfilter_diff.csv', 'Cisapride', '100 nM', False),
    ('Ch3_Mexi2andhalfum_20000_1_3k_chan1_dif.csv', 'Mexi', '2.5 um', False),
    ('Ch1_bepridil 001_2000fs_1_3K.csv', 'bepridil', '001', False),
    ('Ch2_DMSO10-1_2000fs_1_3K.csv', 'DMSO', '10-1', False),
    ('Ch1_base_1h20_20000_1_3k_chan1_dif.csv', 'baseline', '0', True),
    ('Exp10_ChipF_ch2_Ti07_A.csv', 'Ti07', 'A', False),
])
def test_file_name_grammar(name, drug, conc, baseline):
    info = parse_filename(name)
    assert (info['drug'], info['concentration'], info['is_baseline']) == (drug, conc, baseline)


# ── 2. pairing ──────────────────────────────────────────────────────────


def _res(path, *, fpdc=500.0, bp=1000.0, grade='A', electrode='el1', na=False, passed=True):
    fi = describe_recording(path)
    fi['analyzed_channel'] = electrode
    stem = path.rsplit('/', 1)[-1][:-4]

    class _QC:
        pass

    qc = _QC()
    qc.grade = grade
    return {
        'metadata': {'filepath': path, 'filename': stem},
        'file_info': fi,
        'summary': {'fpdc_ms_mean': fpdc, 'beat_period_ms_mean': bp, 'beat_period_ms_median': bp,
                    'spike_amplitude_mV_mean': 1.0, 'not_analysable': na},
        'qc_report': qc,
        'inclusion': {'passed': passed, 'reason': '' if passed else 'Baseline CV=31.0% >= 25.0%'},
    }


def test_drug_is_paired_with_the_baseline_of_its_own_experiment():
    bl5 = _res('D/Exp5/Day7/chipA_ch1_baseline.csv', fpdc=400.0, grade='B')
    bl8 = _res('D/Exp8/Day7/chipA_ch1_baseline.csv', fpdc=600.0, grade='A')
    d5 = _res('D/Exp5/Day7/chipA_ch1_terfe_10nM.csv', fpdc=440.0)
    d8 = _res('D/Exp8/Day7/chipA_ch1_dofe_1nM.csv', fpdc=660.0)
    normalize_all_results([bl5, bl8, d5, d8])
    assert d5['normalization']['baseline_file'] == 'chipA_ch1_baseline'
    assert d5['normalization']['pct_fpdc_change'] == pytest.approx(10.0)   # vs Exp5, not the better-graded Exp8
    assert d8['normalization']['pct_fpdc_change'] == pytest.approx(10.0)


def test_baseline_in_the_same_folder_wins_over_better_grade():
    main = _res('D/Exp7/ChipE/chipE_ch1_baseline.csv', fpdc=500.0, grade='C')
    new = _res('D/Exp7/ChipE/new baseline/chipE_ch1_baseline.csv', fpdc=600.0, grade='A')
    dose = _res('D/Exp7/ChipE/chipE_ch1_DOFETILIDE_1nM.csv', fpdc=550.0)
    later = _res('D/Exp7/ChipE/new baseline/chipE_ch1_DOFE_2nM.csv', fpdc=660.0)
    details = {}
    m = pair_with_baselines([main, new, dose, later], details=details)
    assert m[recording_key(dose)] is main and m[recording_key(later)] is new
    assert details[recording_key(dose)]['reason'].startswith('baseline in the same folder')
    assert set(details[recording_key(dose)]['candidates']) == {'chipE_ch1_baseline'}


def test_recordings_with_the_same_name_in_two_folders_do_not_overwrite_each_other():
    a = _res('D/Exp8/Day6/chipD_ch1_baseline.csv', fpdc=400.0)
    b = _res('D/Exp8/Day7/chipD_ch1_baseline.csv', fpdc=800.0)
    da = _res('D/Exp8/Day6/chipD_ch1_DOFE_1nM.csv', fpdc=440.0)
    db = _res('D/Exp8/Day7/chipD_ch1_DOFE_1nM.csv', fpdc=880.0)
    normalize_all_results([a, b, da, db])
    assert da['normalization']['pct_fpdc_change'] == pytest.approx(10.0)
    assert db['normalization']['pct_fpdc_change'] == pytest.approx(10.0)


def test_two_tissue_files_are_not_paired_and_say_why():
    bl = _res('GG/Exp 5/Day8/baseline/Exp5_chipC_ch1_chipA_ch1_baseline.csv')
    dose = _res('GG/Exp 5/Day8/A/Exp5_chipC_ch1_Ti03_chipA_ch1_Ti04_A.csv', fpdc=700.0)
    normalize_all_results([bl, dose])
    assert dose['normalization']['has_baseline'] is False
    assert 'two tissues' in dose['normalization']['pairing']['reason']


def test_baseline_failing_inclusion_leaves_the_drug_unpaired_with_the_reason():
    bl = _res('D/Exp5/Day7/chipB_ch2_baseline.csv', passed=False)
    dose = _res('D/Exp5/Day7/chipB_ch2_terfe_10nM.csv')
    normalize_all_results([bl, dose])
    assert dose['normalization']['has_baseline'] is False
    assert 'failed inclusion' in dose['normalization']['pairing']['reason']


# ── 3. drug grouping ────────────────────────────────────────────────────


def test_spelling_variants_are_one_drug():
    assert {canonical_drug_name(x) for x in ('DOFE', 'Dofetilide', 'dofe 3')} == {'dofetilide'}
    assert canonical_drug_name('Ti07') == 'ti07'          # test-item codes keep their digits
    assert canonical_drug_name('quinine') == 'quinine'    # not folded into quinidine


def _normed(path, pct):
    r = _res(path)
    r['normalization'] = {'has_baseline': True, 'pct_fpdc_change': pct, 'tdp_score': 0}
    return r


def test_classify_drug_merges_spellings_and_ignores_washout_and_vehicle():
    rs = [_normed('D/Exp7/ChipE/chipE_ch1_DOFETILIDE_1nM.csv', 5.0),
          _normed('D/Exp8/Day7/chipE_ch2_DOFE_3nM.csv', 8.0),
          _normed('D/Exp8/Day7/chipE_ch2_DOFE_washout_40minutes.csv', 80.0),
          _normed('D/Exp13/Chip 569/Ch1 DMSO/Ch1_DMSO10-1_2000fs_01_3K_diff.csv', 40.0)]
    cls = classify_drug(rs, cfg=NormalizationConfig())
    assert set(cls) == {'dofetilide'}
    assert cls['dofetilide']['n_concentrations'] == 2
    assert cls['dofetilide']['positive'] is False   # the washout's +80 % no longer decides


# ── 4. near-cessation guard ─────────────────────────────────────────────


def test_fpdc_not_compared_when_the_tissue_has_almost_stopped():
    bl = _res('D/Exp6/chipC_ch2_baseline.csv', fpdc=362.0, bp=1500.0)
    stopped = _res('D/Exp6/chipC_ch2_CISAP_2_5nM.csv', fpdc=2461.0, bp=40614.0)
    slow_ok = _res('D/Exp6/chipC_ch2_CISAP_1nM.csv', fpdc=400.0, bp=5000.0)
    cfg = NormalizationConfig()
    n = compute_normalized_parameters(stopped, bl, cfg=cfg)
    assert np.isnan(n['pct_fpdc_change']) and 'near cessation' in n['fpdc_withheld']
    assert n['pct_bp_change'] > 2000                    # BP change still reported
    assert compute_normalized_parameters(slow_ok, bl, cfg=cfg)['pct_fpdc_change'] == pytest.approx(10.5, abs=0.1)


# ── 5. one electrode per tissue in auto mode ────────────────────────────


def test_batch_auto_mode_keeps_the_baseline_electrode_for_the_whole_tissue(tmp_path, monkeypatch):
    files = ['Exp5/Day7/chipA/chipA_ch1_baseline.csv', 'Exp5/Day7/chipA/chipA_ch1_terfe_1nM.csv',
             'Exp5/Day7/chipA/chipA_ch1_terfe_10nM.csv',
             'Exp8/Day7/chipA_ch1_baseline.csv', 'Exp8/Day7/chipA_ch1_dofe_1nM.csv']
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('')
    # what 'auto' would pick, file by file: the Exp5 baseline prefers el2,
    # its doses el1 — the old behaviour mixed them.
    auto_pick = {'chipA_ch1_baseline': 'el2', 'chipA_ch1_terfe_1nM': 'el1', 'chipA_ch1_terfe_10nM': 'el1',
                 'chipA_ch1_dofe_1nM': 'el2'}
    calls = []

    def fake_analyze(filepath, channel='auto', verbose=True, config=None):
        from pathlib import Path
        p = Path(filepath)
        el = auto_pick.get(p.stem, 'el1') if channel == 'auto' else channel
        if 'Exp8' in str(p) and p.stem.endswith('baseline') and channel == 'auto':
            el = 'el1'
        calls.append((str(p.relative_to(tmp_path)), channel, el))
        r = _res(str(p), electrode=el, fpdc=500.0 if 'baseline' in p.stem else 550.0)
        r['summary'].update({'beat_period_ms_cv': 5.0, 'fpd_confidence': 0.9, 'fpd_ms_median': 400.0})
        return r

    monkeypatch.setattr(analyze_mod, 'analyze_single_file', fake_analyze)
    monkeypatch.setattr(analyze_mod, 'generate_excel_report', lambda *a, **k: None)
    monkeypatch.setattr(analyze_mod, 'generate_pdf_report', lambda *a, **k: None)
    results = analyze_mod.batch_analyze(tmp_path, channel='auto', output_dir=tmp_path / 'out',
                                        verbose=False, config=AnalysisConfig())
    by = {r['metadata']['filename'] + '@' + r['file_info']['tissue']: r for r in results}
    exp5 = [r for k, r in by.items() if k.endswith('exp5/day7/chipA_ch1')]
    assert {r['file_info']['analyzed_channel'] for r in exp5} == {'el2'}          # baseline's electrode
    assert all(r['normalization']['has_baseline'] for r in exp5 if 'terfe' in r['metadata']['filename'])
    exp8 = [r for k, r in by.items() if k.endswith('exp8/day7/chipA_ch1')]
    assert {r['file_info']['analyzed_channel'] for r in exp8} == {'el1'}
    # baselines first with 'auto', then the doses on the tissue electrode
    first_dose = next(i for i, c in enumerate(calls) if 'baseline' not in c[0])
    assert all('baseline' in c[0] and c[1] == 'auto' for c in calls[:first_dose])
    assert all(c[1] in ('el1', 'el2') for c in calls[first_dose:])
    assert [r['metadata']['filename'] for r in results] == [f.rsplit('/', 1)[-1][:-4] for f in sorted(files)]
