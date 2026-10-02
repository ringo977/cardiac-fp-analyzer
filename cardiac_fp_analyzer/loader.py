"""
loader.py — CSV loader for Digilent WaveForms µECG recordings.

Parses the header block (#-prefixed lines) to extract metadata
(device, date/time, sampling rate, number of samples, channel ranges)
and returns a clean DataFrame plus a metadata dict.
"""

import re
from pathlib import Path

import pandas as pd


def load_csv(filepath):
    """
    Load a Digilent WaveForms CSV file.

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

    return metadata, df


# ── File-name grammar ─────────────────────────────────────────────────
# Two layouts are understood:
#   * 'chipA_ch1_terfe_300nM'          tissue token first (EXP-style)
#   * 'Ch3_Alfu100nM_20000_1_3k_…'     chamber prefix, chip in the folder
#                                      ('Chip 537/…', Accelera-style)
# A number followed by a unit is the concentration; '2andhalf' is 2.5.
_CONC = re.compile(r'(?<![\d.])(\d+andhalf|\d+(?:[._]\d+)?)[\s_]*(nM|uM|µM|μM|mM)(?![A-Za-z]*\d)',
                   re.IGNORECASE)
_CHAMBER_PREFIX = re.compile(r'^ch[\s_-]*(\d+)[\s_-]+', re.IGNORECASE)
# Acquisition settings that follow the drug token in Accelera names
# ('20000_1_3k', '2000fs', '20KHz', 'chan1', 'dif', '_1h' incubation, 'conf1' …).
_ACQ_TOKEN = re.compile(r'^(?:\d+fs|\d+k?hz\w*|\d{4,}|\d+k|chan\d+|dif+|diff\w*|conf\d+.*|\d+h(?:our)?\d*|our\w*)$',
                        re.IGNORECASE)
_BASELINE_WORD = re.compile(r'(?:baseline|basline|(?:^|[_\s])base(?:$|[_\s]))', re.IGNORECASE)


def _conc_value(num):
    num = num.lower()
    if num.endswith('andhalf'):
        return str(int(num[:-7] or 0) + 0.5)
    return num.replace('_', '.')


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
    name = Path(filename).stem

    m = re.search(r'chip([A-Za-z])', name, re.IGNORECASE)
    if m: info['chip'] = m.group(1).upper()

    m = re.search(r'ch(\d+)', name, re.IGNORECASE)
    if m: info['channel'] = int(m.group(1))

    if _BASELINE_WORD.search(name):
        info['is_baseline'] = True
        info['drug'] = 'baseline'
        info['concentration'] = '0'
        return info

    remainder = re.sub(r'chip[A-Za-z0-9]+?_?ch\d+_?', '', name, flags=re.IGNORECASE).strip('_')
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
            if _ACQ_TOKEN.match(tok.strip()):
                break
            kept.append(tok)
        remainder = '_'.join(kept).strip('_ ')

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
    # ('NIFEDIPINE_10', 'bepridil 001', 'DMSO10-1', '01DMSO').
    m = (re.fullmatch(r'([A-Za-z]+)[\s_-]*(\d[\d._-]*)', remainder)
         or re.fullmatch(r'(\d[\d._-]*)[\s_-]*([A-Za-z]+)', remainder))
    if m:
        drug, conc = (m.group(1), m.group(2)) if m.group(1)[0].isalpha() else (m.group(2), m.group(1))
        info['drug'] = drug
        info['concentration'] = conc.strip('_-')
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
_CHIP_DIR = re.compile(r'^chip[\s_-]*(\S.*?)\s*$', re.IGNORECASE)
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
    that start with the chamber ('Ch2_…') the chip is the deepest folder
    named 'Chip <id>' or, failing that, a parent folder that is a bare
    number ('Day8/529/Ch1_…'); the chamber can also be a 'Ch2 …' folder.
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
    if chamber is not None:
        info['channel'] = chamber
    if chip is not None and chamber is not None and not info.get('dual_tissue'):
        if exp is None:
            ek = '-'
        elif _EXP_DIR.match(exp):
            ek = f"exp{int(_EXP_DIR.match(exp).group(1))}"
        else:
            ek = re.sub(r'\W+', '', exp.lower()) or '-'
        dk = f"day{int(_DAY_DIR.match(day).group(1))}" if day else '-'
        info['tissue'] = f"{ek}/{dk}/chip{str(chip).upper()}_ch{chamber}"
    return info
