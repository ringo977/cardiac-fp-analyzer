"""Drug call, pre-dose reference and high sample rates (Oct 2026, v3.8.0).

Evaluating the decision rules on all experiments of Visone et al. 2023
(12 compounds, FDA label as truth) exposed:
  1. classification_method='max' called every negative compound positive,
     the vehicle included (5/12 correct with the authors' values, 6/12 with
     the software's) → per-concentration tissue mean, ≥ 2 tissues, two
     consecutive concentrations (11/12 and 8/12);
  2. the cessation override would have made 10 of 12 compounds positive
     → reported, opt-in;
  3. the protocols before 2020 normalise to 't0', recorded right before the
     first dose, not to the earlier file named 'baseline' → t0 recognised,
     reference = last one recorded before the first dose;
  4. 20 kHz recordings: the (b, a) band-pass has a pole outside the unit
     circle and no beat is found → decimation to 2 kHz on load.
Synthetic data only; no recording of the data sets is used.
"""

import numpy as np
import pytest

import cardiac_fp_analyzer.analyze as analyze_mod
from cardiac_fp_analyzer.config import AnalysisConfig, NormalizationConfig
from cardiac_fp_analyzer.filtering import bandpass_filter
from cardiac_fp_analyzer.loader import (
    describe_recording,
    load_csv,
    parse_filename,
    recording_datetime,
)
from cardiac_fp_analyzer.normalization import (
    classify_drug,
    concentration_value,
    pair_with_baselines,
    recording_key,
)

# ── helpers ─────────────────────────────────────────────────────────────


def _res(path, *, fpdc=500.0, bp=1000.0, grade='A', electrode='el1', when=None, na=False, passed=True):
    fi = describe_recording(path)
    fi['analyzed_channel'] = electrode
    stem = path.rsplit('/', 1)[-1][:-4]

    class _QC:
        pass

    qc = _QC()
    qc.grade = grade
    return {
        'metadata': {'filepath': path, 'filename': stem,
                     'datetime': f'2019-05-16 {when}.000' if when else None},
        'file_info': fi,
        'summary': {'fpdc_ms_mean': fpdc, 'beat_period_ms_mean': bp, 'beat_period_ms_median': bp,
                    'spike_amplitude_mV_mean': 1.0, 'not_analysable': na, 'fpd_confidence': 0.9},
        'qc_report': qc,
        'inclusion': {'passed': passed, 'reason': ''},
    }


def _dose(tissue_path, drug, conc, pct, *, path_suffix=''):
    """A drug recording already normalised (no pairing needed)."""
    r = _res(f'{tissue_path}_{drug}_{conc}{path_suffix}.csv')
    r['normalization'] = {'has_baseline': True, 'pct_fpdc_change': pct, 'tdp_score': 0}
    return r


def _digilent_csv(path, fs, n, when='2020-10-21 11:44:13.041'):
    t = np.arange(n) / fs
    x = np.zeros(n)
    x[(np.arange(n) % int(fs)) == 0] = 1.0        # one spike per second
    with open(path, 'w') as f:
        f.write('#Digilent WaveForms Oscilloscope Acquisition\n')
        f.write(f'#Date Time: {when}\n')
        f.write(f'#Sample rate: {int(fs)}Hz\n#Samples: {n}\n\n')
        f.write('Time (s),Channel 1 (V),Channel 2 (V)\n')
        for ti, xi in zip(t, x):
            f.write(f'{ti:.6f},{xi:.4f},{-xi:.4f}\n')


# ── 1. per-concentration drug call ──────────────────────────────────────


def test_defaults_are_the_per_concentration_rule_and_no_cessation_override():
    cfg = NormalizationConfig()
    assert cfg.classification_method == 'concentration'
    assert (cfg.classification_min_tissues, cfg.classification_consecutive) == (2, 2)
    assert cfg.enable_cessation_override is False


def test_positive_when_the_tissue_mean_exceeds_at_two_consecutive_concentrations():
    rs = []
    for t, vals in (('D/Exp5/Day7/chipA_ch1', (2.0, 18.0, 30.0)), ('D/Exp5/Day7/chipB_ch2', (4.0, 16.0, 26.0))):
        rs += [_dose(t, 'dofe', c, v) for c, v in zip(('1nM', '2nM', '3nM'), vals)]
    c = classify_drug(rs, cfg=NormalizationConfig())['dofetilide']
    assert c['positive'] is True and c['decision'] == 'positive'
    assert c['effective_concentration'] == '2 nM'
    assert [row['n_tissues'] for row in c['per_concentration']] == [2, 2, 2]


def test_one_outlying_recording_no_longer_decides_the_call():
    rs = []
    for t, vals in (('D/Exp5/Day7/chipA_ch1', (1.0, 60.0, 3.0)), ('D/Exp5/Day7/chipB_ch2', (-2.0, 2.0, 1.0)),
                    ('D/Exp5/Day7/chipC_ch3', (0.0, -1.0, 4.0))):
        rs += [_dose(t, 'nife', c, v) for c, v in zip(('1nM', '5nM', '10nM'), vals)]
    assert classify_drug(rs, cfg=NormalizationConfig())['nifedipine']['decision'] == 'negative'
    old = NormalizationConfig()
    old.classification_method = 'max'
    assert classify_drug(rs, cfg=old)['nifedipine']['positive'] is True


def test_non_consecutive_crossings_are_negative_and_the_paper_rule_is_one_setting_away():
    rs = []
    for t in ('D/Exp5/Day7/chipA_ch1', 'D/Exp5/Day7/chipB_ch2'):
        rs += [_dose(t, 'mexi', c, v) for c, v in zip(('1uM', '10uM', '50uM'), (20.0, 5.0, 20.0))]
    assert classify_drug(rs, cfg=NormalizationConfig())['mexiletine']['decision'] == 'negative'
    paper = NormalizationConfig()
    paper.classification_min_tissues = 1
    paper.classification_consecutive = 1
    assert classify_drug(rs, cfg=paper)['mexiletine']['positive'] is True


def test_one_tissue_is_insufficient_data_not_negative():
    rs = [_dose('D/Exp5/Day7/chipA_ch1', 'terfe', c, v) for c, v in (('1nM', 30.0), ('10nM', 40.0))]
    c = classify_drug(rs, cfg=NormalizationConfig())['terfenadine']
    assert c['positive'] is False and c['decision'] == 'insufficient data'


def test_repeated_recordings_of_one_tissue_count_once_and_units_are_merged():
    rs = [_dose('D/Exp5/Day7/chipA_ch1', 'quinid', '1uM', 30.0),
          _dose('D/Exp5/Day7/chipA_ch1', 'quinid', '1uM', 34.0, path_suffix='_bis'),
          _dose('D/Exp5/Day7/chipA_ch1', 'quinid', '3uM', 40.0),
          _dose('D/Exp5/Day7/chipB_ch2', 'quinid', '1000nM', 20.0),
          _dose('D/Exp5/Day7/chipB_ch2', 'quinid', '3uM', 30.0)]
    c = classify_drug(rs, cfg=NormalizationConfig())['quinidine']
    rows = {row['value']: row for row in c['per_concentration']}
    assert rows[1.0]['n_tissues'] == 2 and rows[1.0]['mean_pct'] == pytest.approx((32.0 + 20.0) / 2)
    assert c['decision'] == 'positive'


def test_cessation_is_reported_and_changes_the_call_only_when_enabled():
    class _Cess:
        has_cessation, cessation_confidence, cessation_type = True, 0.9, 'complete'

    rs = []
    for t in ('D/Exp7/ChipA/chipA_ch1', 'D/Exp7/ChipB/chipB_ch3'):
        rs += [_dose(t, 'alfu', c, v) for c, v in zip(('1nM', '10nM'), (1.0, -3.0))]
    rs[-1]['cessation_report'] = _Cess()
    rs[-1]['summary']['fpd_confidence'] = 0.2
    c = classify_drug(rs, cfg=NormalizationConfig())['alfuzosin']
    assert c['cessation_flag'] is True and c['cessation_override'] is False and c['positive'] is False
    on = NormalizationConfig()
    on.enable_cessation_override = True
    assert classify_drug(rs, cfg=on)['alfuzosin']['positive'] is True


@pytest.mark.parametrize('label, unit, value', [
    ('300 nM', None, 0.3), ('0.3 uM', None, 0.3), ('2.5 um', None, 2.5), ('7.5', None, 7.5),
    ('001', None, 0.01), ('05', None, 0.5), ('01%', None, 0.1), ('10-1', None, 0.1), ('B', None, 2.0),
    ('10', 'nm', 0.01), ('0', None, 0.0),
])
def test_concentration_value(label, unit, value):
    assert concentration_value(label, default_unit=unit) == pytest.approx(value)


@pytest.mark.parametrize('label', ['', None, 't2', 'baseline'])
def test_concentration_value_unreadable(label):
    assert concentration_value(label) is None


# ── 2. pre-dose reference ───────────────────────────────────────────────


def test_t0_recorded_after_the_baseline_is_the_reference():
    folder = 'D/Exp3/day7/chip2inj'
    base = _res(f'{folder}/ch1_baseline_channel1_sx_channel2_dx.csv', fpdc=900.0, when='10:12:43')
    t0 = _res(f'{folder}/ch1_t0_sotalol_channel1_sx_channel2_dx.csv', fpdc=600.0, when='10:49:53')
    d1 = _res(f'{folder}/ch1_1_sotalol_channel1_sx_channel2_dx.csv', fpdc=660.0, when='11:45:11')
    assert t0['file_info']['is_baseline'] and t0['file_info']['reference_kind'] == 't0'
    details = {}
    m = pair_with_baselines([base, t0, d1], details=details)
    assert m[recording_key(d1)] is t0
    assert details[recording_key(d1)]['reason'].startswith('last reference before the first dose')


def test_second_baseline_and_references_after_the_first_dose():
    folder = 'D/exp4/day6'
    b1 = _res(f'{folder}/chip3_ch1_baseline_1.csv', when='14:54:02', grade='A')
    b2 = _res(f'{folder}/chip3_ch1_baseline_2.csv', when='15:07:16', grade='C')
    late = _res(f'{folder}/chip3_ch1_baseline_after12h.csv', when='23:30:00', grade='A')
    d = _res(f'{folder}/chip3_ch1_Quinidine0_06uM.csv', when='16:39:55')
    m = pair_with_baselines([b1, b2, late, d])
    assert m[recording_key(d)] is b2


def test_without_times_t0_is_preferred_in_the_same_folder():
    folder = 'D/exp1/day6/chipB_sotalol'
    base = _res(f'{folder}/Ch2_channel1_sx_channel2_dx_baseline.csv', grade='A')
    t0 = _res(f'{folder}/Ch2_Sotalol_t0channel1_sx_channel2_dx.csv', grade='B')
    d = _res(f'{folder}/Ch2_Sotalol_1_channel1_sx_channel2_dx.csv')
    details = {}
    assert pair_with_baselines([base, t0, d], details=details)[recording_key(d)] is t0
    assert 't0' in details[recording_key(d)]['reason']


def test_batch_takes_the_tissue_electrode_from_the_last_reference_before_the_first_dose(tmp_path, monkeypatch):
    folder = tmp_path / 'Exp3' / 'day7' / 'chip2inj'
    folder.mkdir(parents=True)
    times = {'ch1_baseline_channel1_sx_channel2_dx': '10:12:43', 'ch1_t0_sotalol_channel1_sx_channel2_dx': '10:49:53',
             'ch1_1_sotalol_channel1_sx_channel2_dx': '11:45:11', 'ch1_3_sotalol_channel1_sx_channel2_dx': '12:15:00'}
    for stem, when in times.items():
        (folder / f'{stem}.csv').write_text(f'#Digilent WaveForms\n#Date Time: 2019-05-16 {when}.000\n')
    auto_pick = {'ch1_baseline_channel1_sx_channel2_dx': 'el2', 'ch1_t0_sotalol_channel1_sx_channel2_dx': 'el1'}

    def fake_analyze(filepath, channel='auto', verbose=True, config=None):
        from pathlib import Path
        p = Path(filepath)
        el = auto_pick.get(p.stem, 'el2') if channel == 'auto' else channel
        r = _res(str(p), electrode=el, when=times[p.stem], fpdc=500.0)
        r['summary'].update({'beat_period_ms_cv': 5.0, 'fpd_ms_median': 400.0})
        return r

    monkeypatch.setattr(analyze_mod, 'analyze_single_file', fake_analyze)
    monkeypatch.setattr(analyze_mod, 'generate_excel_report', lambda *a, **k: None)
    monkeypatch.setattr(analyze_mod, 'generate_pdf_report', lambda *a, **k: None)
    results = analyze_mod.batch_analyze(tmp_path, channel='auto', output_dir=tmp_path / 'out',
                                        verbose=False, config=AnalysisConfig())
    doses = [r for r in results if 'sotalol' in r['metadata']['filename'] and 't0' not in r['metadata']['filename']]
    assert {r['file_info']['analyzed_channel'] for r in doses} == {'el1'}           # t0's electrode
    assert {r['file_info']['tissue_electrode_from'] for r in doses} == {'ch1_t0_sotalol_channel1_sx_channel2_dx.csv'}
    assert all(r['normalization']['baseline_file'] == 'ch1_t0_sotalol_channel1_sx_channel2_dx' for r in doses)


# ── 3. file names of the protocols before 2020 ──────────────────────────


@pytest.mark.parametrize('name, drug, conc, baseline', [
    ('Ch2_Sotalol_3,25_channel1_sx_channel2_dx.csv', 'Sotalol', '3.25', False),
    ('Ch2_Sotalol_t0channel1_sx_channel2_dx.csv', 'baseline', '0', True),
    ('Ch3_ASPIRINE T02_channel1_sx_channel2_dx.csv', 'baseline', '0', True),
    ('ch1_t0_sx_channel2_dx.csv', 'baseline', '0', True),
    ('Ch2_channel1_sx_channel2_dx_baseline.csv', 'baseline', '0', True),
    ('ch1_t4_channel1_sx_channel2_dx.csv', 'ctrl', 't4', False),
    ('Ch3_t2_Ctrl_channel1_sx_channel2_dx.csv', 'ctrl', 't2', False),
    ('ch3_1000Terfe_channel1_sx_channel2_dx.csv', 'Terfe', '1000', False),
    ('Ch1_Terfe_1000_channel1_sx_channel2_dx.csv', 'Terfe', '1000', False),
    ('ch1_01%_DMSO_channel1_sx_channel2_dx.csv', 'DMSO', '01', False),
    ('ch1_30_sotalol_channel1_sx_channel2_dx_bis.csv', 'sotalol', '30', False),
    ('chip_A_ch1_1000_VERAP_e_channel1_sx_channel2_dx.csv', 'VERAP', '1000', False),
    # unchanged by the new rules
    ('Ch3_5min_washout_Torem_20000_1_3k.csv', '5min washout Torem', None, False),
    ('Ch1_bepridil 001_2000fs_1_3K.csv', 'bepridil', '001', False),
])
def test_old_two_electrode_names(name, drug, conc, baseline):
    info = parse_filename(name)
    assert (info['drug'], info['concentration'], info['is_baseline']) == (drug, conc, baseline)


@pytest.mark.parametrize('path, tissue', [
    ('D/exp1/day6/chipB_sotalol/Ch2_Sotalol_15_channel1_sx_channel2_dx.csv', 'exp1/day6/chipB_ch2'),
    ('D/exp1/day7/ChipC_Verapamil_Terfenadine/Ch3_t2_Ctrl_channel1_sx_channel2_dx.csv', 'exp1/day7/chipC_ch3'),
    ('D/Exp2/inj1/ch3_1000Terfe_channel1_sx_channel2_dx.csv', 'exp2/-/chipINJ1_ch3'),
    ('D/Exp2/inj2/Ch1_3/ch3_10_Aspirine_channel1_sx_channel2_dx.csv', 'exp2/-/chipINJ2_ch3'),
    ('D/Exp3/day6/inj1/ch1_T0_channel1_sx_channel2_dx.csv', 'exp3/day6/chipINJ1_ch1'),
])
def test_tissue_of_old_layouts(path, tissue):
    assert describe_recording(path)['tissue'] == tissue


def test_no_experiment_folder_no_folder_chip():
    assert 'tissue' not in describe_recording('D/x/ch1_t0_channel1_sx_channel2_dx.csv')


# ── 4. sample rate ──────────────────────────────────────────────────────


def test_20khz_recordings_are_decimated_on_load(tmp_path):
    p = tmp_path / 'Ch1_base_1h_20000_1_3k_chan1_dif.csv'
    _digilent_csv(p, 20000.0, 40000)
    md, df = load_csv(p)
    assert md['sample_rate'] == 2000.0 and md['original_sample_rate'] == 20000.0
    assert md['decimation_factor'] == 10 and len(df) == 4000 == md['n_samples']
    assert np.diff(df['time'].to_numpy()[:3]) == pytest.approx([0.0005, 0.0005])
    md_raw, df_raw = load_csv(p, max_sample_rate=None)
    assert md_raw['sample_rate'] == 20000.0 and len(df_raw) == 40000
    assert recording_datetime(p).strftime('%H:%M:%S') == '11:44:13'


def test_bandpass_stays_finite_when_the_ba_design_is_unstable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60000)
    y = bandpass_filter(x, 20000.0, lowcut=0.5, highcut=500.0, order=4)
    assert np.all(np.isfinite(y))


def test_bandpass_at_2khz_is_unchanged():
    from scipy import signal
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    b, a = signal.butter(4, [0.5 / 1000.0, 500.0 / 1000.0], btype='band')
    assert np.array_equal(bandpass_filter(x, 2000.0, lowcut=0.5, highcut=500.0, order=4), signal.filtfilt(b, a, x))


# ── 5. inclusion with several references, names, unit warnings ──────────


def test_a_failing_earlier_baseline_no_longer_removes_the_tissue():
    from cardiac_fp_analyzer.inclusion import apply_inclusion_criteria
    from cardiac_fp_analyzer.normalization import normalize_all_results
    folder = 'D/Exp3/day7/chip2inj'
    base = _res(f'{folder}/ch1_baseline_channel1_sx_channel2_dx.csv', fpdc=900.0, when='10:12:43')
    t0 = _res(f'{folder}/ch1_t0_sotalol_channel1_sx_channel2_dx.csv', fpdc=600.0, when='10:49:53')
    d1 = _res(f'{folder}/ch1_1_sotalol_channel1_sx_channel2_dx.csv', fpdc=660.0, when='11:45:11')
    for r, cv in ((base, 46.0), (t0, 9.5), (d1, 8.0)):
        r['summary'].update({'beat_period_ms_cv': cv, 'fpd_ms_median': 400.0})
        r.pop('inclusion')
    rs = apply_inclusion_criteria([base, t0, d1], verbose=False)
    assert base['inclusion']['passed'] is False and d1['inclusion']['passed'] is True
    normalize_all_results(rs)
    assert d1['normalization']['baseline_file'] == 'ch1_t0_sotalol_channel1_sx_channel2_dx'
    assert d1['normalization']['pct_fpdc_change'] == pytest.approx(10.0)


def test_sot_is_sotalol():
    from cardiac_fp_analyzer.normalization import canonical_drug_name
    assert canonical_drug_name('SOT') == 'sotalol'


def test_a_tissue_with_its_own_concentrations_is_reported(caplog):
    rs = [_dose('D/Exp8/Day6/chipD_ch2', 'alfu', c, v) for c, v in (('10uM', 1.0), ('100uM', 2.0))]
    rs += [_dose('D/Exp7/ChipC/chipC_ch1', 'alfu', c, v) for c, v in (('10nM', 1.0), ('100nM', 2.0))]
    rs += [_dose('D/Exp7/ChipD/chipD_ch2', 'alfu', c, v) for c, v in (('10nM', 3.0), ('100nM', 1.0))]
    with caplog.at_level('WARNING'):
        classify_drug(rs, cfg=NormalizationConfig())
    assert any('exp8/day6/chipD_ch2 shares no concentration' in m.getMessage() for m in caplog.records)
