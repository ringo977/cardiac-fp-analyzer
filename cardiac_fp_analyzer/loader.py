"""
loader.py — CSV loader for Digilent WaveForms µECG recordings.

Parses the header block (#-prefixed lines) to extract metadata
(device, date/time, sampling rate, number of samples, channel ranges)
and returns a clean DataFrame plus a metadata dict.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


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
    with open(filepath, 'r') as f:
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


def parse_filename(filename):
    """
    Parse experiment info from filename.
    Example: chipA_ch1_terfe_300nM -> chip=A, channel=1, drug=terfe, conc=300nM
    """
    info = {'chip': None, 'channel': None, 'drug': None,
            'concentration': None, 'is_baseline': False}
    name = Path(filename).stem

    m = re.search(r'chip([A-Za-z])', name, re.IGNORECASE)
    if m: info['chip'] = m.group(1).upper()

    m = re.search(r'ch(\d+)', name, re.IGNORECASE)
    if m: info['channel'] = int(m.group(1))

    if 'baseline' in name.lower() or 'basline' in name.lower():
        info['is_baseline'] = True
        info['drug'] = 'baseline'
        info['concentration'] = '0'
        return info

    remainder = re.sub(r'chip[A-Za-z]_ch\d+_?', '', name, flags=re.IGNORECASE).strip('_')
    m_conc = re.search(r'([\d._]+)\s*(nM|uM|µM|mM)', remainder, re.IGNORECASE)
    if m_conc:
        conc_str = m_conc.group(1).replace('_', '.')
        unit = m_conc.group(2)
        info['concentration'] = f"{conc_str} {unit}"
        drug_part = remainder[:m_conc.start()].strip('_').strip()
        if drug_part: info['drug'] = drug_part.replace('_', ' ').strip()
    else:
        info['drug'] = remainder.replace('_', ' ').strip()

    return info
