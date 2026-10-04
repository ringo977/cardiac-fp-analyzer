"""
loader.py — Recording loaders and file-name grammar.

Two file formats are read (``load_recording`` dispatches on the suffix):

* Digilent WaveForms CSV (µECG, two oscilloscope inputs): the ``#`` header
  block gives device, date/time, sampling rate and channel ranges; the
  DataFrame has columns 'time', 'el1', 'el2'.
* Multi Channel Systems HDF5 raw data (``.h5``, MCS RawData protocol — the
  µHeart / MEA2100 recordings, up to 64 electrodes at 20 kHz): columns
  'time' plus one per electrode label ('E1' … 'E64'); see mcs_hdf5.py.

Recordings sampled above MAX_SAMPLE_RATE are decimated on load.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Sample rate ───────────────────────────────────────────────────────
# The pipeline is tuned for ~2 kHz recordings: window lengths are given in
# samples (Savitzky–Golay, detector) and the 0.5–500 Hz band-pass is designed
# as (b, a) coefficients. At 20 kHz (the Accelera chips of Exp11 in the Visone
# 2023 data) that band-pass has a pole outside the unit circle, the filtered
# signal overflows and no beat is found ('only 0 depolarisation(s) detected'
# on all 37 files). Since Oct 2026 recordings above MAX_SAMPLE_RATE are
# decimated on load to about TARGET_SAMPLE_RATE (FIR anti-alias filter, zero
# phase); metadata keeps the original rate.
TARGET_SAMPLE_RATE = 2000.0
MAX_SAMPLE_RATE = 3000.0


def _decimate(df, fs):
    """Decimate the electrode columns by round(fs / TARGET_SAMPLE_RATE)."""
    q = int(round(fs / TARGET_SAMPLE_RATE))
    if q < 2:
        return df, 1
    from scipy.signal import decimate
    cols = {c: decimate(df[c].to_numpy(dtype=float), q, ftype='fir', zero_phase=True)
            for c in ('el1', 'el2')}
    n = len(cols['el1'])
    out = pd.DataFrame({'time': df['time'].to_numpy()[::q][:n], **cols})
    return out, q


def parse_header_datetime(text):
    """'2019-10-21 14:54:02.102' → datetime, or None."""
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(str(text).strip(), fmt)
        except (TypeError, ValueError):
            continue
    return None


def recording_datetime(filepath):
    """Start of acquisition from the '#Date Time:' header line, or None.

    Reads only the header, so it is cheap enough to call on every file of a
    batch before analysis (used to pick the pre-dose reference). For MCS
    HDF5 files the start comes from the file attributes.
    """
    if Path(filepath).suffix.lower() in ('.h5', '.hdf5', '.hdf'):
        from .mcs_hdf5 import recording_datetime as _h5_datetime
        return _h5_datetime(filepath)
    try:
        with open(filepath, errors='replace') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if line.startswith('#Date Time:'):
                    return parse_header_datetime(line.split(':', 1)[1])
    except OSError:
        return None
    return None


RECORDING_SUFFIXES = ('.csv', '.h5', '.hdf5', '.hdf')


def is_recording_file(filepath):
    """A CSV, or an HDF5 file in the MCS raw-data layout."""
    p = Path(filepath)
    suf = p.suffix.lower()
    if suf == '.csv':
        return True
    if suf in ('.h5', '.hdf5', '.hdf'):
        from .mcs_hdf5 import is_mcs_hdf5
        return is_mcs_hdf5(p)
    return False


def recording_channels(filepath):
    """Channel labels of a recording without loading the data: the electrode
    labels of the raw stream of an MCS file, ['el1', 'el2'] (or ['el1']) for
    a CSV. [] when they cannot be read."""
    p = Path(filepath)
    if p.suffix.lower() in ('.h5', '.hdf5', '.hdf'):
        try:
            from .mcs_hdf5 import inspect
            rec = inspect(p)['recordings'][0]
            el = [a for a in rec['analog'] if a['subtype'] == 'Electrode']
            raw = [a for a in el if 'raw' in a['label'].lower()]
            return list((raw or el or rec['analog'])[0]['channels'])
        except (OSError, KeyError, IndexError, ImportError, ValueError):
            return []
    return [f'el{c}' for c in input_columns(p)]


def electrode_columns(df):
    """Signal columns of a loaded recording, in file order ('el1', 'el2' or
    the electrode labels of an MCS file)."""
    return [c for c in df.columns if c != 'time']


def load_recording(filepath, max_sample_rate=MAX_SAMPLE_RATE, **kwargs):
    """Load a recording of either format (see module docstring).

    Extra keyword arguments go to ``load_mcs_h5`` (``stream``, ``channels``,
    ``recording``). Returns (metadata, df) like ``load_csv``; metadata has
    'format' ('csv' or 'mcs_hdf5').
    """
    p = Path(filepath)
    if p.suffix.lower() in ('.h5', '.hdf5', '.hdf'):
        from .mcs_hdf5 import load_mcs_h5
        return load_mcs_h5(p, max_sample_rate=max_sample_rate, **kwargs)
    metadata, df = load_csv(p, max_sample_rate=max_sample_rate)
    metadata.setdefault('format', 'csv')
    return metadata, df


def load_csv(filepath, max_sample_rate=MAX_SAMPLE_RATE):
    """
    Load a Digilent WaveForms CSV file.

    Recordings sampled above ``max_sample_rate`` are decimated to about
    TARGET_SAMPLE_RATE (``None`` keeps the original rate); metadata then
    carries 'original_sample_rate' and 'decimation_factor'.

    Returns
    -------
    metadata : dict
    df : pd.DataFrame with columns: 'time', 'el1', 'el2'
    """
    filepath = Path(filepath)
    metadata = {
        'filepath': str(filepath), 'filename': filepath.stem,
        'sample_rate': None, 'n_samples': None, 'device': None,
        'serial': None, 'datetime': None, 'trigger_info': None,
        'ch1_range': None, 'ch1_offset': None,
        'ch2_range': None, 'ch2_offset': None,
    }

    header_lines = 0
    with open(filepath) as f:
        for line in f:
            if not line.startswith('#'):
                break
            header_lines += 1
            line = line.strip('#').strip()

            if line.startswith('Device Name:'):
                metadata['device'] = line.split(':', 1)[1].strip()
            elif line.startswith('Serial Number:'):
                metadata['serial'] = line.split(':', 1)[1].strip()
            elif line.startswith('Date Time:'):
                metadata['datetime'] = line.split(':', 1)[1].strip()
            elif line.startswith('Sample rate:'):
                m = re.search(r'([\d.]+)\s*Hz', line)
                if m: metadata['sample_rate'] = float(m.group(1))
            elif line.startswith('Samples:'):
                m = re.search(r'(\d+)', line)
                if m: metadata['n_samples'] = int(m.group(1))
            elif line.startswith('Trigger:'):
                metadata['trigger_info'] = line.split(':', 1)[1].strip()
            elif line.startswith('Channel 1:'):
                m_r = re.search(r'Range:\s*([\d.]+)\s*mV/div', line)
                m_o = re.search(r'Offset:\s*([-\d.]+)\s*(?:m)?V', line)
                if m_r: metadata['ch1_range'] = float(m_r.group(1))
                if m_o: metadata['ch1_offset'] = float(m_o.group(1))
            elif line.startswith('Channel 2:'):
                m_r = re.search(r'Range:\s*([\d.]+)\s*mV/div', line)
                m_o = re.search(r'Offset:\s*([-\d.]+)\s*(?:m)?V', line)
                if m_r: metadata['ch2_range'] = float(m_r.group(1))
                if m_o: metadata['ch2_offset'] = float(m_o.group(1))

    df = pd.read_csv(filepath, skiprows=header_lines, header=0)
    if len(df.columns) == 3:
        df.columns = ['time', 'el1', 'el2']
    elif len(df.columns) == 2:
        df.columns = ['time', 'el1']
        df['el2'] = df['el1']
    else:
        df = df.iloc[:, :3]
        df.columns = ['time', 'el1', 'el2']

    if metadata['sample_rate'] is None and len(df) > 1:
        dt = df['time'].iloc[1] - df['time'].iloc[0]
        if dt > 0:
            metadata['sample_rate'] = round(1.0 / dt, 1)

    fs = metadata['sample_rate']
    if max_sample_rate is not None and fs and fs > max_sample_rate:
        df, q = _decimate(df, fs)
        if q > 1:
            metadata['original_sample_rate'] = fs
            metadata['decimation_factor'] = q
            metadata['sample_rate'] = fs / q
            metadata['n_samples'] = len(df)

    return metadata, df


# ── File-name grammar ─────────────────────────────────────────────────
# Two layouts are understood:
#   * 'chipA_ch1_terfe_300nM'          tissue token first (EXP-style)
#   * 'Ch3_Alfu100nM_20000_1_3k_…'     chamber prefix, chip in the folder
#                                      ('Chip 537/…', Accelera-style)
# A number followed by a unit is the concentration; '2andhalf' is 2.5.
_CONC = re.compile(r'(?<![\d.,])(\d+andhalf|\d+(?:[._,]\d+)?)[\s_]*(nM|uM|µM|μM|mM)(?![A-Za-z]*\d)',
                   re.IGNORECASE)
_CHAMBER_PREFIX = re.compile(r'^ch[\s_-]*(\d+)[\s_-]+', re.IGNORECASE)
# Acquisition settings that follow the drug token in Accelera names
# ('20000_1_3k', '2000fs', '20KHz', 'chan1', 'dif', '_1h' incubation, 'conf1' …).
_ACQ_TOKEN = re.compile(r'^(?:\d+fs|\d+k?hz\w*|\d{4,}|\d+k|chan\d+|dif+|diff\w*|conf\d+.*|\d+h(?:our)?\d*|our\w*)$',
                        re.IGNORECASE)
_BASELINE_WORD = re.compile(r'(?:baseline|basline|(?:^|[_\s])base(?:$|[_\s]))', re.IGNORECASE)
# Two-electrode layout used before 2020 ('Ch2_Sotalol_15_channel1_sx_channel2_dx',
# 'ch1_t0_sx_channel2_dx', '…_channel1_sx_channel2_dx_bis'): the suffix and any
# note after it ('_bis', '_sfter3', '_AFTERAAL') carry no drug information.
_ELECTRODE_SUFFIX = re.compile(r'[_\s]*(?:channel\d_)?sx_channel\d_dx.*$', re.IGNORECASE)
# 't0' / 'T0' / 'T02': the pre-dose reference recorded right before the first
# dose in those protocols; the authors of the Visone 2023 paper normalised to
# it, not to the earlier file named 'baseline' (13 of 15 tissues).
_T0_TOKEN = re.compile(r'(?:^|[_\s-])t0\d?(?=$|[_\s-])', re.IGNORECASE)
# 't1'…'t7' with no drug, or 'Ctrl': time-matched control recordings.
_TIME_CONTROL = re.compile(r'(?:t(\d{1,2}))?[\s_-]*(ctrl|ctr|control)?', re.IGNORECASE)
_TISSUE_IN_NAME = re.compile(r'chip[\s_-]*[A-Za-z0-9]+?[\s_-]*ch[\s_-]*\d+[\s_-]*', re.IGNORECASE)
# Multi Channel Systems recordings, one file per chip (plate) and condition:
# '2026-01-27T12-00-41McsRecording_PM01001_baseline_Recording-0_(…)'. The
# condition is 'baseline' or the dose letter; a suffix after the letter is a
# note ('D_heater'). The chambers of the chip come from the layout
# (chambers.py) or from samples.csv, not from the name.
_MCS_NAME = re.compile(r'McsRecording_(?P<plate>[A-Za-z0-9]+)_(?P<cond>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)_Recording-?(?P<rec>\d+)',
                       re.IGNORECASE)


def _conc_value(num):
    num = num.lower()
    if num.endswith('andhalf'):
        return str(int(num[:-7] or 0) + 0.5)
    return num.replace('_', '.').replace(',', '.')


def parse_filename(filename):
    """
    Parse experiment info from filename.

    Examples
    --------
    chipA_ch1_terfe_300nM          -> chip=A, channel=1, drug=terfe, conc='300 nM'
    chipA_ch3_DOFETILIDE_0_3_nM    -> drug=DOFETILIDE, conc='0.3 nM'
    chipE_ch2_NIFEDIPINE_10        -> drug=NIFEDIPINE, conc='10' (no unit given)
    Ch3_Alfu100nM_20000_1_3k_chan1 -> channel=3, drug=Alfu, conc='100 nM'
    Ch1_100nMCisapride_20KHz_…     -> drug=Cisapride, conc='100 nM'
    Ch1_base_1h20_…                -> baseline

    Before Oct 2026 the concentration kept the separator in front of it
    ('.300 nM') and a number without unit stayed inside the drug name
    ('NIFEDIPINE 10'), which split one drug into several in classify_drug.
    """
    info = {'chip': None, 'channel': None, 'drug': None,
            'concentration': None, 'is_baseline': False}
    stem = Path(filename).stem
    mcs = _MCS_NAME.search(stem)
    if mcs:
        cond = mcs.group('cond')
        info.update({'chip': mcs.group('plate'), 'mcs_recording': int(mcs.group('rec')), 'format': 'mcs_hdf5'})
        if _BASELINE_WORD.search(cond) or _T0_TOKEN.search(cond):
            info.update({'is_baseline': True, 'drug': 'baseline', 'concentration': '0',
                         'reference_kind': 't0' if _T0_TOKEN.search(cond) else 'baseline'})
        else:
            info['concentration'] = cond            # dose letter ('A') or letter with a note ('D_heater')
        return info
    old_layout = bool(_ELECTRODE_SUFFIX.search(stem))
    name = _ELECTRODE_SUFFIX.sub('', stem) or stem

    m = re.search(r'chip([A-Za-z])', name, re.IGNORECASE)
    if m: info['chip'] = m.group(1).upper()

    m = re.search(r'ch(\d+)', name, re.IGNORECASE)
    if m: info['channel'] = int(m.group(1))

    t0 = bool(_T0_TOKEN.search(name))
    if _BASELINE_WORD.search(stem) or t0:
        info['is_baseline'] = True
        info['drug'] = 'baseline'
        info['concentration'] = '0'
        info['reference_kind'] = 't0' if t0 else 'baseline'
        return info

    remainder = _TISSUE_IN_NAME.sub('', name).strip('_')
    # 'Exp10_ChipF_ch2_Ti07_A(_v2|_only)': experiment prefix, then a test-item
    # code and the dose condition letter (the folder A/B/C holds the files).
    remainder = re.sub(r'^exp[\s_-]*\d+[\s_]*', '', remainder, flags=re.IGNORECASE)
    cond = re.search(r'_([A-Z])(?:_(?:v\d+|only|bis))*$', remainder)
    if cond:
        info['drug'] = remainder[:cond.start()].replace('_', ' ').strip() or None
        info['concentration'] = cond.group(1)
        return info
    accelera = _CHAMBER_PREFIX.match(remainder)
    if accelera:
        tokens = remainder[accelera.end():].split('_')
        kept = []
        for tok in tokens:
            # the old two-electrode layout has no acquisition tokens, and its
            # '1000' is a concentration ('Ch1_Terfe_1000'), not a sample count
            if not old_layout and _ACQ_TOKEN.match(tok.strip()):
                break
            kept.append(tok)
        remainder = '_'.join(kept).strip('_ ')

    tc = _TIME_CONTROL.fullmatch(remainder)
    if tc and (tc.group(1) or tc.group(2)):
        info['drug'] = 'ctrl'
        info['concentration'] = f't{int(tc.group(1))}' if tc.group(1) else ''
        return info

    m_conc = _CONC.search(remainder)
    if m_conc:
        info['concentration'] = f"{_conc_value(m_conc.group(1))} {m_conc.group(2)}"
        drug_part = remainder[:m_conc.start()].strip('_ -')
        if not drug_part:   # concentration written first: '100nMCisapride'
            after = re.match(r'[\s_-]*([A-Za-z]+)', remainder[m_conc.end():])
            drug_part = after.group(1) if after else ''
        if drug_part:
            info['drug'] = drug_part.replace('_', ' ').strip()
        return info

    # No unit: split a leading or trailing bare number off the drug name
    # ('NIFEDIPINE_10', 'bepridil 001', 'DMSO10-1', '01DMSO', 'Sotalol_3,25',
    # '01%_DMSO', and number-first names followed by a short note: '10_VERAP_e').
    m = (re.fullmatch(r'([A-Za-z]+)[\s_-]*(\d[\d.,_-]*)%?', remainder)
         or re.fullmatch(r'(\d[\d.,_-]*)%?[\s_-]*([A-Za-z]+)(?:[\s_-]+[A-Za-z0-9]{1,3})*', remainder))
    if m:
        drug, conc = (m.group(1), m.group(2)) if m.group(1)[0].isalpha() else (m.group(2), m.group(1))
        info['drug'] = drug
        info['concentration'] = conc.strip('_-').replace(',', '.')
    elif remainder:
        info['drug'] = remainder.replace('_', ' ').strip()
    return info


# ── Recording identity from the folder layout ─────────────────────────
# A tissue (one microtissue, i.e. one chamber of one chip) is identified by
# experiment + day + chip + chamber.  Until Oct 2026 the experiment was taken
# only from a folder starting with upper-case 'EXP' and the day was ignored,
# so with folders named 'Exp5', 'exp1' or 'Exp 5' every chip letter was shared
# across experiments and days: on the Visone 2023 data set 29 of 31 groups
# mixed experiments and drug recordings were normalised against another
# experiment's baseline.
_EXP_DIR = re.compile(r'^exp[\s_-]*(\d+)', re.IGNORECASE)
_DAY_DIR = re.compile(r'^day[\s_-]*(\d+)', re.IGNORECASE)
# chip id = first token of the folder name: 'Chip 537' → 537, 'chipB_sotalol'
# → B ('chipA_dmso_aspirin', 'ChipC_Verapamil_Terfenadine' in exp1)
_CHIP_DIR = re.compile(r'^chip[\s_-]*([A-Za-z0-9]+)', re.IGNORECASE)
_CHAMBER_DIR = re.compile(r'^ch[\s_-]*(\d+)(?:\b|_)', re.IGNORECASE)
_TISSUE_TOKEN = re.compile(r'chip[\s_-]*([A-Za-z0-9]+?)[\s_-]*ch[\s_-]*(\d+)', re.IGNORECASE)


def describe_recording(filepath):
    """parse_filename() plus the identity of the tissue recorded.

    Adds to the parse_filename() dict:
      experiment  folder name of the experiment ('Exp5', 'EXP 5', 'exp1',
                  'Exp_10', 'Exp12_Accelera'), deepest matching folder
      day         folder name of the day ('Day7', 'day 7', 'day 8 Accelera')
      tissue      'exp5/day7/chipA_ch1' — the normalisation key; absent when
                  chip or chamber cannot be established
      dual_tissue True when the name carries two tissue tokens
                  ('Exp5_chipC_ch1_chipA_ch1_baseline'): each electrode is a
                  different tissue, so the file is not paired automatically

    Chip and chamber come from the file name ('chipA_ch1_…'); for names
    that start with the chamber ('Ch2_…') the chip is the first token of
    the deepest folder named 'Chip <id>' ('Chip 537', 'chipB_sotalol'),
    failing that a parent folder that is a bare number ('Day8/529/Ch1_…'),
    failing that the deepest folder below the experiment that is not a day
    or chamber folder ('Exp2/inj1/ch3_…' → chip 'INJ1'); the chamber can
    also be a 'Ch2 …' folder.
    """
    p = Path(filepath)
    info = parse_filename(p.name)
    dirs = p.parent.parts
    # 'startswith EXP' keeps folders the old rule accepted ('EXP_A').
    exp = next((d for d in reversed(dirs) if _EXP_DIR.match(d) or d.startswith('EXP')), None)
    day = next((d for d in reversed(dirs) if _DAY_DIR.match(d)), None)
    if exp:
        info['experiment'] = exp
    if day:
        info['day'] = day

    if info.get('format') == 'mcs_hdf5':
        return info                      # chip = plate; chambers from the layout or samples.csv
    tokens = _TISSUE_TOKEN.findall(p.stem)
    chip = chamber = None
    if tokens:
        chip, chamber = tokens[0][0], int(tokens[0][1])
        if len(tokens) > 1:
            info['dual_tissue'] = True
            info['tissue_tokens'] = [f'chip{c.upper()}_ch{n}' for c, n in tokens]
    else:
        m = _CHAMBER_PREFIX.match(p.stem)
        if m:
            chamber = int(m.group(1))
        else:
            cdir = next((d for d in reversed(dirs) if _CHAMBER_DIR.match(d)), None)
            chamber = int(_CHAMBER_DIR.match(cdir).group(1)) if cdir else None
        chip_dir = next((d for d in reversed(dirs) if _CHIP_DIR.match(d)), None)
        if chip_dir:
            chip = _CHIP_DIR.match(chip_dir).group(1)
        elif dirs and dirs[-1].isdigit():
            chip = dirs[-1]
        elif exp is not None and chamber is not None:
            # 'Exp2/inj1/ch3_…', 'Exp3/day6/inj2/ch1_…': the folder below the
            # experiment (day and chamber folders skipped) is the chip
            i = len(dirs) - 1 - list(reversed(dirs)).index(exp)
            below = [d for d in dirs[i + 1:] if not (_DAY_DIR.match(d) or _CHAMBER_DIR.match(d))]
            if below:
                chip = re.sub(r'\W+', '', below[-1]) or None
    if chamber is not None:
        info['channel'] = chamber
    if chip is not None and chamber is not None and not info.get('dual_tissue'):
        info['tissue'] = tissue_key(exp, day, chip, chamber)
    return info


def tissue_key(experiment, day, chip, chamber):
    """Normalisation key of one tissue: 'exp5/day7/chipB_ch1' (or
    '-/-/chipPM01001_chA' for a chamber of a multi-chamber chip).

    ``experiment`` and ``day`` are folder names ('Exp 5', 'Day7', None);
    the key reads 'exp<N>' and 'day<N>' from them, '-' when absent.
    """
    if experiment is None or not str(experiment).strip():
        ek = '-'
    elif _EXP_DIR.match(str(experiment).strip()):
        ek = f"exp{int(_EXP_DIR.match(str(experiment).strip()).group(1))}"
    else:
        ek = re.sub(r'\W+', '', str(experiment).lower()) or '-'
    dm = _DAY_DIR.match(str(day).strip()) if day else None
    dk = f"day{int(dm.group(1))}" if dm else '-'
    ch = str(chamber).strip()
    ch = str(int(ch)) if ch.isdigit() else ch.upper()      # '1' or a chamber letter ('A')
    return f"{ek}/{dk}/chip{str(chip).upper()}_ch{ch}"


# ── Two tissues in one file ───────────────────────────────────────────
# GG layout (2026): each oscilloscope input records a different tissue and
# the name lists the tissues in input order, each followed by its test item:
# 'Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A' = input 1 chip P chamber 1 (K01),
# input 2 chip Q chamber 2 (K02), dose A. An input without a tissue is
# written 'na' after the chip, after the chamber at the end of the name, or
# alone before the first tissue: 'Exp1_ChipP_na_ChipQ_ch2_K02_A', 'Exp1_ChipP_ch1_K01_A_ChipQ_ch1_na',
# 'Exp1_na_ChipQ_ch2_K02_A'. On the GG recordings the analyst's beat marks
# match one input clearly for 135 of the 182 named tissues, always the input
# the name order gives (the other 47: too few beats found to tell).
_INPUT_TOKEN = re.compile(
    r'chip[\s_-]*(?P<chip>[A-Za-z0-9]+?)[\s_-]*ch[\s_-]*(?P<ch>\d+)(?P<na>[\s_-]+na$)?'
    r'|(?P<chipna>chip[\s_-]*(?P<chipna_id>[A-Za-z0-9]+?)[\s_-]+na)(?=$|[\s_-])'
    r'|(?:(?<=[\s_-])|^)(?P<bare>na)(?=$|[\s_-])',
    re.IGNORECASE)
_CODE_WORD = re.compile(r'^([A-Za-z]{1,3}\d+)')          # 'K01bis' → 'K01', 'E4031'
_NOISE_WORD = re.compile(r'^(?:v\d+|\d+|only|bis|sx|dx|baseline|basline|base)$', re.IGNORECASE)


def _item_and_conc(tail):
    """Test item and concentration written after a tissue token."""
    conc = None
    m = _CONC.search(tail)
    if m:
        conc = f"{_conc_value(m.group(1))} {m.group(2)}"
        before = tail[:m.start()]
        tail = before if re.search(r'[A-Za-z]', before) else tail[m.end():]
    words = [w for w in re.split(r'[\s_-]+', tail) if w]
    item = None
    for j, w in enumerate(words):
        split_code = (re.fullmatch(r'[A-Za-z]{1,3}', w) and j + 1 < len(words)
                      and words[j + 1].isdigit())
        if re.fullmatch(r'[A-Z]', w) and not split_code:   # dose condition letter (A, B, C)
            conc = conc or w
            continue
        if item is not None or _NOISE_WORD.match(w):
            continue
        if split_code:
            item = w + words[j + 1]               # 'K_03' → 'K03'
            continue
        code = _CODE_WORD.match(w)
        item = code.group(1) if code else w
    return item, conc


def parse_inputs(name):
    """Tissues named in a file name, in input order.

    Returns one dict per input: {'chip', 'chamber', 'item', 'concentration'}
    for a tissue, {'empty': True} for an input marked 'na' (chip and chamber
    added when the name gives them). 'item' is the word after the tissue
    ('K07', 'vehicle', 'terfe'); 'concentration' a unit concentration next
    to it ('300 nM') or a dose letter ('A'), else None. A bare 'na' counts
    only before the first tissue; after it, it is part of the item text.
    """
    stem = Path(str(name)).stem if str(name).lower().endswith(RECORDING_SUFFIXES) else str(name)
    ms = [m for m in _INPUT_TOKEN.finditer(stem)]
    out = []
    for i, m in enumerate(ms):
        if m.group('bare') is not None:
            if out:
                continue
            out.append({'empty': True})
        elif m.group('chipna') is not None:
            out.append({'empty': True, 'chip': m.group('chipna_id').upper()})
        elif m.group('na'):
            out.append({'empty': True, 'chip': m.group('chip').upper(), 'chamber': int(m.group('ch'))})
        else:
            nxt = next((n.start() for n in ms[i + 1:] if n.group('bare') is None), len(stem))
            item, conc = _item_and_conc(stem[m.end():nxt])
            out.append({'chip': m.group('chip').upper(), 'chamber': int(m.group('ch')),
                        'item': item, 'concentration': conc})
    return out


def input_columns(filepath):
    """Oscilloscope inputs recorded in a file, from its column header:
    ['1', '2'], ['1'] or ['2'] ('Channel N' columns; without names, the
    number of data columns). [] when the header cannot be read or the file
    is not a CSV (MCS HDF5 files have electrode labels, not inputs). Reads
    only the header lines."""
    if Path(filepath).suffix.lower() != '.csv':
        return []
    try:
        with open(filepath, errors='replace') as f:
            for k, line in enumerate(f):
                if k > 400:
                    break
                if line.startswith('#') or not line.strip():
                    continue
                named = re.findall(r'channel\s*(\d)', line, re.IGNORECASE)
                if named:
                    return named
                n = len(line.split(',')) - 1
                return [str(c) for c in range(1, n + 1)][:2] if n > 0 else []
    except OSError:
        return []
    return []
