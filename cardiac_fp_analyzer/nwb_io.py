"""NWB 2 (Neurodata Without Borders) files: the WaveForms CSV → ``.nwb`` converter and the reader.

Writing (``csv_to_nwb``) uses pynwb, so the file follows the NWB schema; reading (``read_nwb``) uses h5py
only, so the analysis software does not depend on pynwb. What goes into the file:

* ``NWBFile``: ``session_description`` and ``session_id`` (experiment / day / tissues / condition),
  ``identifier`` (deterministic UUID of the source file), ``session_start_time`` (the CSV ``#Date Time``
  with the lab's time zone), ``notes`` (the complete WaveForms header), ``experiment_description``,
  ``lab`` / ``institution`` / ``experimenter`` when given, ``source_script`` (this converter);
* ``Device``: the oscilloscope (``#Device Name``, serial number in the description);
* one ``ElectrodeGroup`` per tissue (chip + chamber) and an ``electrodes`` table with one row per recorded
  input (1 or 2) and the columns ``input``, ``label`` (el1 / el2), ``chip``, ``chamber``, ``item``,
  ``condition``, ``tissue``, ``experiment``, ``day``;
* ``ElectricalSeries`` ``field_potential`` in ``acquisition``: float32 samples × channels in volts at the
  oscilloscope input, ``conversion = 1 / gain`` (volts at the electrode), ``rate``, ``starting_time`` (the
  first time stamp of the CSV, e.g. −30 s before the trigger), ``filtering`` (amplifier gain and the
  WaveForms sample mode), ``comments`` (channel ranges, offsets, trigger).

``read_nwb`` returns ``(metadata, df)`` like ``loader.load_csv`` — columns ``time``, ``el1``, ``el2`` — with the
samples already in volts at the electrode (``metadata['format'] = 'nwb'``: the pipeline applies no gain).
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

logger = logging.getLogger(__name__)

NWB_SUFFIX = '.nwb'
SERIES_NAME = 'field_potential'
ELECTRODE_COLUMNS = ('input', 'label', 'chip', 'chamber', 'item', 'condition', 'tissue', 'experiment', 'day')
_COLUMN_DESCRIPTIONS = {
    'input': 'oscilloscope input the electrode was recorded on (1 or 2)',
    'label': 'channel label used by the analysis software (el1, el2)',
    'chip': 'chip (microtissue carrier) the electrode belongs to',
    'chamber': 'chamber of the chip (one microtissue per chamber)',
    'item': 'test item / compound of the recording (empty at baseline)',
    'condition': 'condition of the recording: baseline or the dose / concentration',
    'tissue': 'tissue key experiment/day/chip_chamber',
    'experiment': 'experiment the tissue belongs to',
    'day': 'day of the experiment',
}


def _s(v):
    """h5py string / bytes / array scalar → str."""
    if isinstance(v, bytes):
        return v.decode('utf-8', 'replace')
    if isinstance(v, np.ndarray):
        return _s(v[()]) if v.shape == () else _s(v[0]) if len(v) else ''
    if isinstance(v, (np.bytes_, np.str_)):
        return str(v.astype(str)) if isinstance(v, np.bytes_) else str(v)
    return '' if v is None else str(v)


# ──────────────────────────────────────────────────────────────────────
#   Reading the WaveForms header (all of it)
# ──────────────────────────────────────────────────────────────────────

def read_header(csv_path):
    """The ``#`` lines of a WaveForms CSV, as a list and as parsed fields (device, serial, datetime,
    sample_rate, samples, trigger, channels: {n: {'range', 'offset', 'sample_mode', 'text'}})."""
    lines, fields = [], {'channels': {}}
    with open(csv_path, errors='replace') as f:
        for line in f:
            if not line.startswith('#'):
                break
            text = line.strip('#').strip()
            lines.append(text)
            if text.startswith('Device Name:'):
                fields['device'] = text.split(':', 1)[1].strip()
            elif text.startswith('Serial Number:'):
                fields['serial'] = text.split(':', 1)[1].strip()
            elif text.startswith('Date Time:'):
                fields['datetime'] = text.split(':', 1)[1].strip()
            elif text.startswith('Sample rate:'):
                m = re.search(r'([\d.]+)\s*Hz', text)
                if m:
                    fields['sample_rate'] = float(m.group(1))
            elif text.startswith('Samples:'):
                m = re.search(r'(\d+)', text)
                if m:
                    fields['samples'] = int(m.group(1))
            elif text.startswith('Trigger:'):
                fields['trigger'] = text.split(':', 1)[1].strip()
            else:
                m = re.match(r'Channel\s*(\d+):\s*(.*)', text)
                if m:
                    ch = {'text': m.group(2)}
                    mr = re.search(r'Range:\s*([\d.]+)\s*mV/div', m.group(2))
                    mo = re.search(r'Offset:\s*([-\d.]+)\s*(?:m)?V', m.group(2))
                    ms = re.search(r'Sample Mode:\s*(\w+)', m.group(2))
                    if mr:
                        ch['range_mV_div'] = float(mr.group(1))
                    if mo:
                        ch['offset_V'] = float(mo.group(1))
                    if ms:
                        ch['sample_mode'] = ms.group(1)
                    fields['channels'][m.group(1)] = ch
    return lines, fields


def _start_time(fields, csv_path, tz):
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    raw = fields.get('datetime')
    if raw:
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S'):
            try:
                return datetime.strptime(raw.strip(), fmt).replace(tzinfo=zone), True
            except ValueError:
                continue
    return datetime.fromtimestamp(Path(csv_path).stat().st_mtime, tz=zone), False


# ──────────────────────────────────────────────────────────────────────
#   Identity of the inputs
# ──────────────────────────────────────────────────────────────────────

def _tissue_key(exp, day, chip, chamber):
    from .loader import tissue_key
    return tissue_key(exp, day, chip, chamber) if chip not in (None, '') and chamber not in (None, '') else ''


def inputs_of_csv(csv_path, data_dir=None, config=None):
    """Identity of every recorded input of a CSV, one dict per input in input order, from the sample sheet
    of the experiment when there is one, else from the file name: ``input``, ``label``, ``chip``,
    ``chamber``, ``item``, ``condition``, ``tissue``, ``experiment``, ``day`` (strings, '' when unknown).
    An input that holds no tissue ('na') keeps its data and an empty identity."""
    from .loader import describe_recording, input_columns
    from .results_export import condition_label, plan_recordings, sheet_root
    csv_path = Path(csv_path)
    inputs = input_columns(csv_path) or ['1']
    info = describe_recording(str(csv_path))
    root = Path(data_dir) if data_dir else sheet_root(csv_path.parent)
    units, _plan = plan_recordings([csv_path], root, config)
    by_label = {u['electrode']: (u.get('update') or {}) for u in units}
    out = []
    for n in inputs:
        label = f'el{n}'
        upd = by_label.get(label)
        if upd is None and 'auto' in by_label:          # one tissue on both inputs
            upd = {k: info.get(k) for k in ('chip', 'channel', 'drug', 'concentration', 'is_baseline', 'experiment',
                                            'day', 'tissue')}
            upd['item'] = None if info.get('is_baseline') else info.get('drug')
            upd['chamber'] = info.get('channel')
        ident = upd or {}
        exp, day = ident.get('experiment') or info.get('experiment'), ident.get('day') or info.get('day')
        chip, chamber = ident.get('chip'), ident.get('chamber', ident.get('channel'))
        out.append({'input': str(n), 'label': label, 'chip': _s(chip) if chip is not None else '',
                    'chamber': _s(chamber) if chamber is not None else '', 'item': _s(ident.get('item') or ''),
                    'condition': condition_label(ident) if ident else '', 'tissue': ident.get('tissue') or _tissue_key(exp, day, chip, chamber),
                    'experiment': _s(exp or ''), 'day': _s(day or '')})
    return out


# ──────────────────────────────────────────────────────────────────────
#   Writing
# ──────────────────────────────────────────────────────────────────────

DEFAULT_KEYWORDS = ('cardiac', 'field potential', 'microelectrode array', 'hiPSC-CM', 'microtissue')


def csv_to_nwb(csv_path, out_path=None, *, inputs=None, gain=10000.0, tz='Europe/Rome', lab='', institution='',
               experimenter=(), notes='', session_description=None, keywords=DEFAULT_KEYWORDS, overwrite=False,
               data_dir=None, config=None, species='Homo sapiens', subject_description='hiPSC-derived cardiac microtissue'):
    """Convert one WaveForms CSV to NWB (see the module docstring). ``inputs``: identity dicts per input
    (``inputs_of_csv``); computed when omitted. ``gain``: amplifier gain between electrode and oscilloscope
    (samples ÷ gain = volts at the electrode). The series starts at t = 0 (NWB best practice): when the CSV
    time axis starts before the trigger (−30 s) the session start is moved back by as much and the trigger
    time is written in the comments. ``experimenter`` names in the form 'LastName, FirstName' satisfy the
    DANDI check. Returns the path written."""
    from datetime import timedelta

    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject

    from .loader import input_columns, load_csv
    csv_path = Path(csv_path)
    out = Path(out_path) if out_path else csv_path.with_suffix(NWB_SUFFIX)
    if out.exists() and not overwrite:
        raise FileExistsError(f'{out} exists (overwrite=True to replace it)')
    lines, fields = read_header(csv_path)
    metadata, df = load_csv(csv_path, max_sample_rate=None)            # never decimated: raw fidelity
    n_inputs = len(input_columns(csv_path) or ['1'])
    cols = ['el1', 'el2'][:n_inputs]
    data = np.column_stack([df[c].to_numpy(dtype=np.float32) for c in cols])
    fs = float(metadata['sample_rate'])
    t0 = float(df['time'].iloc[0])
    inputs = inputs if inputs is not None else inputs_of_csv(csv_path, data_dir=data_dir, config=config)
    if len(inputs) != n_inputs:
        raise ValueError(f'{len(inputs)} identities for {n_inputs} recorded input(s)')
    start, start_from_header = _start_time(fields, csv_path, tz)
    shift_note = ''
    if t0 != 0.0:
        start = start + timedelta(seconds=t0)
        shift_note = (f'WaveForms time axis started at {t0:.3f} s (trigger at t = {-t0:.3f} s of this series); '
                      f'session_start_time moved by {t0:+.3f} s so that the series starts at 0')

    tissues = [i for i in inputs if i.get('tissue')]
    conds = sorted({i['condition'] for i in tissues if i.get('condition')})
    where = ' + '.join(f"chip {i['chip']} ch{i['chamber']}" + (f" ({i['item']})" if i.get('item') else '')
                       for i in tissues) or csv_path.stem
    desc = session_description or (f"µECG field potential, {where}, " + ('/'.join(conds) if conds else 'condition unknown'))
    exp = next((i['experiment'] for i in inputs if i.get('experiment')), '')
    day = next((i['day'] for i in inputs if i.get('day')), '')
    session_id = '_'.join(x for x in (exp.replace(' ', ''), day.replace(' ', ''),
                                      '+'.join(f"chip{i['chip']}-ch{i['chamber']}" for i in tissues),
                                      '+'.join(conds)) if x) or csv_path.stem
    session_id = re.sub(r'[\\/]', '-', session_id)
    ident = uuid.uuid5(uuid.NAMESPACE_URL, f'cardiac-fp-analyzer:{csv_path.resolve()}:{csv_path.stat().st_size}')
    header_text = '\n'.join(lines)
    nwb = NWBFile(
        session_description=desc, identifier=str(ident), session_start_time=start,
        session_id=session_id, experiment_description=f'Converted from {csv_path.name}'
        + ('' if start_from_header else ' (start time from the file modification time: no #Date Time header)'),
        notes=(notes + '\n' if notes else '') + 'WaveForms header:\n' + header_text,
        lab=lab or None, institution=institution or None, experimenter=list(experimenter) or None,
        keywords=list(keywords) or None, source_script=f'cardiac_fp_analyzer {__version__} nwb_io.csv_to_nwb',
        source_script_file_name='nwb_io.py')
    subject_id = '+'.join(i['tissue'] for i in tissues if i.get('tissue')) or csv_path.stem
    nwb.subject = Subject(subject_id=subject_id.replace('/', '-'), species=species, description=subject_description,
                          sex='U', age='P0D')
    device = nwb.create_device(name=fields.get('device') or 'oscilloscope',
                               description=f"Digilent WaveForms acquisition; serial {fields.get('serial', '?')}",
                               manufacturer='Digilent')
    for col in ELECTRODE_COLUMNS:
        nwb.add_electrode_column(name=col, description=_COLUMN_DESCRIPTIONS[col])
    groups = {}
    for k, ident_in in enumerate(inputs):
        key = (ident_in.get('chip') or '', ident_in.get('chamber') or '')
        gname = f"chip{key[0]}_ch{key[1]}" if key[0] and key[1] else f"input{ident_in['input']}"
        if gname not in groups:
            groups[gname] = nwb.create_electrode_group(
                name=gname, device=device, location=f'chamber {key[1]}' if key[1] else 'unknown',
                description=f'µECG electrode of chip {key[0]} chamber {key[1]}' if key[0] else 'electrode on an oscilloscope input')
        nwb.add_electrode(id=k, location=f'chamber {key[1]}' if key[1] else 'unknown', group=groups[gname],
                          **{c: _s(ident_in.get(c, '')) for c in ELECTRODE_COLUMNS})
    region = nwb.create_electrode_table_region(list(range(len(inputs))), 'recorded electrodes, in input order')
    modes = sorted({c.get('sample_mode') for c in fields['channels'].values() if c.get('sample_mode')})
    filtering = f'amplifier gain {gain:g} (conversion = 1/gain gives volts at the electrode)'
    if modes:
        filtering += '; WaveForms sample mode ' + '/'.join(modes)
    comments = '; '.join([f"Channel {n}: {c['text']}" for n, c in fields['channels'].items()]
                         + ([f"Trigger: {fields['trigger']}"] if fields.get('trigger') else [])
                         + ([shift_note] if shift_note else []))
    series = ElectricalSeries(name=SERIES_NAME, data=data, electrodes=region, starting_time=0.0, rate=fs,
                              conversion=1.0 / float(gain), filtering=filtering, comments=comments or 'no header',
                              description='raw field potential, volts at the oscilloscope input (samples × channels)')
    nwb.add_acquisition(series)
    with NWBHDF5IO(str(out), 'w') as io:
        io.write(nwb)
    return out


def validate(nwb_path):
    """Run nwbinspector on the file; returns the list of messages (empty = clean). None without nwbinspector."""
    try:
        from nwbinspector import inspect_nwbfile
    except ImportError:
        return None
    return [f'{m.importance.name}: {m.message} ({m.location})' for m in inspect_nwbfile(nwbfile_path=str(nwb_path))]


# ──────────────────────────────────────────────────────────────────────
#   Reading (h5py only)
# ──────────────────────────────────────────────────────────────────────

def is_nwb_file(path):
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            return 'nwb_version' in f.attrs and 'acquisition' in f
    except (OSError, ImportError):
        return False


def _find_series(f, name=None):
    acq = f['acquisition']
    if name and name in acq:
        return acq[name]
    for key in acq:
        if _s(acq[key].attrs.get('neurodata_type', '')) == 'ElectricalSeries':
            return acq[key]
    raise ValueError('no ElectricalSeries in /acquisition')


def _electrode_rows(f, series):
    """One dict per channel of ``series`` from the electrodes table (custom columns when present)."""
    idx = np.asarray(series['electrodes'][()], dtype=int)
    try:
        table = f['general/extracellular_ephys/electrodes']
    except KeyError:
        return [{'label': f'el{k + 1}', 'input': str(k + 1)} for k in range(len(idx))]
    cols = [c for c in (list(_s(x) for x in table.attrs.get('colnames', [])) or list(table.keys())) if c in table]
    rows = []
    for k, i in enumerate(idx):
        row = {}
        for c in cols:
            ds = table[c]
            if ds.dtype.kind in ('O', 'S', 'U'):
                row[c] = _s(ds[i])
            elif ds.dtype.kind in ('i', 'u', 'f'):
                row[c] = ds[i].item()
        row.setdefault('label', f'el{k + 1}')
        row.setdefault('input', str(k + 1))
        if not row.get('label'):
            row['label'] = f'el{k + 1}'
        rows.append(row)
    return rows


def read_nwb(path, max_sample_rate=None, series=None):
    """Load an NWB file as ``(metadata, df)``: ``df`` has ``time``, ``el1``, ``el2`` (a single channel is
    duplicated into ``el2`` like ``load_csv`` does) in volts at the electrode. ``metadata['nwb']`` carries
    identifier, session, notes and the identity of every input (``inputs``)."""
    import h5py

    from .loader import MAX_SAMPLE_RATE, _decimate
    path = Path(path)
    with h5py.File(path, 'r') as f:
        es = _find_series(f, series)
        data = np.asarray(es['data'][()], dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        conv = float(es['data'].attrs.get('conversion', 1.0))
        offset = float(es['data'].attrs.get('offset', 0.0))
        data = data * conv + offset
        if 'starting_time' in es:
            t0 = float(es['starting_time'][()])
            fs = float(es['starting_time'].attrs.get('rate'))
            t = t0 + np.arange(data.shape[0]) / fs
        else:
            t = np.asarray(es['timestamps'][()], dtype=np.float64)
            fs = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else 0.0
        rows = _electrode_rows(f, es)
        devices = []
        if 'general/devices' in f:
            for name in f['general/devices']:
                g = f['general/devices'][name]
                devices.append({'name': name, 'description': _s(g.attrs.get('description', '')),
                                'manufacturer': _s(g.attrs.get('manufacturer', ''))})
        def _ds(key):
            return _s(f[key][()]) if key in f else ''
        md_nwb = {'identifier': _ds('identifier'), 'session_description': _ds('session_description'),
                  'session_start_time': _ds('session_start_time'), 'session_id': _ds('general/session_id'),
                  'notes': _ds('general/notes'), 'experiment_description': _ds('general/experiment_description'),
                  'lab': _ds('general/lab'), 'institution': _ds('general/institution'),
                  'series': es.name.split('/')[-1], 'description': _s(es.attrs.get('description', '')),
                  'comments': _s(es.attrs.get('comments', '')), 'filtering': _s(es.attrs.get('filtering', '')),
                  'conversion': conv, 'unit': _s(es['data'].attrs.get('unit', 'volts')), 'inputs': rows,
                  'devices': devices, 'nwb_version': _s(f.attrs.get('nwb_version', ''))}
    n_inputs = data.shape[1]
    df = pd.DataFrame({'time': t})
    for k in range(min(2, n_inputs)):
        df[f'el{k + 1}'] = data[:, k]
    if n_inputs == 1:
        df['el2'] = df['el1']
    metadata = {'filepath': str(path), 'filename': path.stem, 'format': 'nwb', 'sample_rate': fs,
                'n_samples': len(df), 'n_inputs': n_inputs, 'device': (devices[0]['name'] if devices else None),
                'serial': None, 'datetime': md_nwb['session_start_time'] or None, 'trigger_info': None,
                'gain': (1.0 / conv) if conv not in (0.0, 1.0) else 1.0, 'nwb': md_nwb}
    m = re.search(r'[Ss]ample [Mm]ode[:\s]+(\w+)', md_nwb['filtering'] + ' ' + md_nwb['comments'])
    if m:
        metadata['sample_mode'] = m.group(1)
    if md_nwb['filtering']:
        metadata['device_filters'] = md_nwb['filtering']
    limit = MAX_SAMPLE_RATE if max_sample_rate is None else max_sample_rate
    if limit is not None and fs > limit:
        df, q = _decimate(df, fs)
        if q > 1:
            metadata.update(original_sample_rate=fs, decimation_factor=q, sample_rate=fs / q, n_samples=len(df))
    return metadata, df


def recording_datetime(path):
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            s = _s(f['session_start_time'][()]) if 'session_start_time' in f else ''
        if not s:
            return None
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d
    except (OSError, ValueError, ImportError, KeyError):
        return None


def recording_inputs(path):
    """The identity rows of the electrodes table (``read_nwb``'s ``inputs``) without loading the samples."""
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            return _electrode_rows(f, _find_series(f))
    except (OSError, ValueError, ImportError, KeyError):
        return []


def describe_nwb(path):
    """Identity in the style of ``loader.parse_filename`` from the electrodes table: chip, channel, drug,
    concentration, is_baseline, tissue (one tissue) or ``dual_tissue`` with ``tissue_tokens`` (two), plus
    ``nwb_inputs`` (the rows) and ``format = 'nwb'``."""
    rows = recording_inputs(path)
    info = {'chip': None, 'channel': None, 'drug': None, 'concentration': None, 'is_baseline': False,
            'format': 'nwb', 'nwb_inputs': rows}
    tissues = [(r.get('chip') or '', str(r.get('chamber') or '')) for r in rows if r.get('chip')]
    distinct = list(dict.fromkeys(tissues))
    if rows:
        first = next((r for r in rows if r.get('chip')), rows[0])
        info['experiment'] = first.get('experiment') or None
        info['day'] = first.get('day') or None
        cond = first.get('condition') or ''
        info['is_baseline'] = cond == 'baseline'
        info['drug'] = 'baseline' if info['is_baseline'] else (first.get('item') or None)
        info['concentration'] = '0' if info['is_baseline'] else (cond or None)
        if info['is_baseline']:
            info['reference_kind'] = 'baseline'
    if len(distinct) == 1:
        chip, chamber = distinct[0]
        info['chip'] = chip
        try:
            info['channel'] = int(chamber)
        except ValueError:
            info['channel'] = chamber or None
        info['tissue'] = next((r.get('tissue') for r in rows if r.get('tissue')), None) or _tissue_key(
            info.get('experiment'), info.get('day'), chip, chamber) or None
    elif len(distinct) > 1:
        info['dual_tissue'] = True
        info['tissue_tokens'] = [f'chip{c.upper()}_ch{n}' for c, n in distinct]
        info['chip'], info['channel'] = distinct[0][0], (int(distinct[0][1]) if distinct[0][1].isdigit() else distinct[0][1])
    return info


def plan_units(path, rel, info, report):
    """Units for ``sample_sheet.plan_batch``: one per input when the inputs hold different tissues (identity
    from the electrodes table), else one ``auto`` unit."""
    from .results_export import condition_label
    rows = info.get('nwb_inputs') or recording_inputs(path)
    tissues = {(r.get('chip') or '', str(r.get('chamber') or '')) for r in rows if r.get('chip')}
    if len(rows) < 2 or len(tissues) < 2:
        return [{'file': path, 'electrode': 'auto', 'update': None, 'source': 'nwb', 'multi': False}]
    out = []
    for r in rows:
        if not r.get('chip'):
            report['excluded'].append((rel, r.get('label', '?'), 'no tissue on this input'))
            continue
        chamber = int(r['chamber']) if str(r.get('chamber', '')).isdigit() else r.get('chamber')
        ident = {'is_baseline': r.get('condition') == 'baseline', 'concentration': r.get('condition'), 'item': r.get('item')}
        upd = {'tissue': r.get('tissue') or _tissue_key(r.get('experiment') or info.get('experiment'),
                                                       r.get('day') or info.get('day'), r['chip'], chamber),
               'chip': r['chip'], 'channel': chamber, 'chamber': chamber, 'item': r.get('item') or None,
               'role': 'baseline' if ident['is_baseline'] else 'dose', 'is_baseline': ident['is_baseline'],
               'dual_tissue': False, 'two_tissue_file': True, 'source': 'nwb',
               'drug': 'baseline' if ident['is_baseline'] else (r.get('item') or None),
               'concentration': '0' if ident['is_baseline'] else (condition_label(ident) if r.get('condition') else None)}
        if ident['is_baseline']:
            upd['reference_kind'] = 'baseline'
        if r.get('experiment'):
            upd['experiment'] = r['experiment']
        out.append({'file': path, 'electrode': r.get('label') or 'auto', 'update': upd, 'source': 'nwb', 'multi': True})
    report['two_tissue_files'] += 1
    return out


# ──────────────────────────────────────────────────────────────────────
#   Command line
# ──────────────────────────────────────────────────────────────────────

def is_waveforms_csv(path):
    """True for a Digilent WaveForms recording (a ``#`` header with the sample rate); analysis tables and other
    CSV files are not."""
    try:
        return 'sample_rate' in read_header(path)[1]
    except OSError:
        return False


def _find_with_roots(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [(f, p) for f in sorted(p.rglob('*.csv')) if f.name.lower() != 'samples.csv']
        else:
            out.append((p, None))
    return out


def find_csvs(paths):
    """CSV files in ``paths`` (files as they are, folders searched recursively; ``samples.csv`` skipped)."""
    return [f for f, _ in _find_with_roots(paths)]


def convert_many(paths, *, out_dir=None, check=False, progress=None, **options):
    """Convert every WaveForms CSV in ``paths`` (see ``find_csvs``) with ``csv_to_nwb(**options)``.

    ``out_dir``: folder for the .nwb files — for a folder input the sub-folder structure is kept, so equal
    names in different sub-folders do not collide (default: next to each CSV); ``check``: run nwbinspector on
    each file written; ``progress(i, n, csv, written, message)`` is called after each file (``written`` None
    when not converted). Returns ``[{'csv', 'nwb', 'ok', 'skipped', 'message', 'issues'}]``: a CSV that is not
    a WaveForms recording is ``skipped`` (not an error); a failing file never stops the others.
    """
    from .results_export import sheet_root
    items = _find_with_roots(paths)
    results = []
    for i, (f, root) in enumerate(items, 1):
        rec = {'csv': str(f), 'nwb': None, 'ok': False, 'skipped': False, 'message': '', 'issues': None}
        if not is_waveforms_csv(f):
            rec.update(skipped=True, message='not a WaveForms recording')
        else:
            try:
                out = None
                if out_dir:
                    sub = f.parent.relative_to(root) if root is not None else Path()
                    out = Path(out_dir) / sub / f.with_suffix(NWB_SUFFIX).name
                    out.parent.mkdir(parents=True, exist_ok=True)
                written = csv_to_nwb(f, out, data_dir=sheet_root(f.parent), **options)
                rec.update(nwb=str(written), ok=True)
                if check:
                    rec['issues'] = validate(written)
            except Exception as exc:     # noqa: BLE001 — report and go on with the next file
                rec['message'] = f'{type(exc).__name__}: {exc}'
        results.append(rec)
        if progress is not None:
            progress(i, len(items), rec['csv'], rec['nwb'], rec['message'])
    return results


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Convert Digilent WaveForms CSV recordings to NWB 2 files')
    ap.add_argument('paths', nargs='+', help='CSV files or folders (recursive)')
    ap.add_argument('-o', '--out-dir', help='folder for the .nwb files (default: next to each CSV)')
    ap.add_argument('--gain', type=float, default=10000.0, help='amplifier gain electrode → oscilloscope (default 10000)')
    ap.add_argument('--tz', default='Europe/Rome', help='time zone of the acquisition clock (default Europe/Rome)')
    ap.add_argument('--lab', default='')
    ap.add_argument('--institution', default='')
    ap.add_argument('--experimenter', action='append', default=[])
    ap.add_argument('--notes', default='')
    ap.add_argument('--species', default='Homo sapiens', help="subject species (default 'Homo sapiens': hiPSC-CM)")
    ap.add_argument('--subject', default='hiPSC-derived cardiac microtissue', help='subject description')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--validate', action='store_true', help='run nwbinspector on each file written')
    a = ap.parse_args(argv)

    def show(i, n, csv, written, message):
        name = Path(csv).name
        print(f'  {name} → {written}' if written else f'  {name}: {message}')

    results = convert_many(a.paths, out_dir=a.out_dir, check=a.validate, progress=show, gain=a.gain, tz=a.tz,
                           lab=a.lab, institution=a.institution, experimenter=a.experimenter, notes=a.notes,
                           overwrite=a.overwrite, species=a.species, subject_description=a.subject)
    for r in results:
        if a.validate and r['ok']:
            issues = r['issues']
            print(f"  {Path(r['nwb']).name}: " + ('nwbinspector not installed' if issues is None
                                                   else (f'{len(issues)} message(s)' if issues else 'clean')))
            for line in issues or []:
                print('      ' + line)
    n_ok = sum(r['ok'] for r in results)
    n_skip = sum(r['skipped'] for r in results)
    print(f'{n_ok}/{len(results) - n_skip} converted' + (f', {n_skip} skipped (not WaveForms recordings)' if n_skip else ''))
    return 0 if n_ok == len(results) - n_skip else 1


if __name__ == '__main__':      # pragma: no cover
    raise SystemExit(main())
