"""
sample_sheet.py — which tissue each input of each recording file holds.

The batch reads the tissue (experiment, day, chip, chamber), the test item
and the dose from the folder layout and the file name
(``loader.describe_recording``). Two situations need more than that:

* **Two tissues in one file.** In the GG layout (2026) each oscilloscope
  input records a different tissue and the name lists them in input order
  ('Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A'). Each input becomes a recording
  of its own tissue (``plan_batch``). An input written 'na' holds no tissue.
* **Multi-chamber chips** (Multi Channel Systems files, one file per chip
  and condition): each chamber of the layout (chambers.py) becomes a
  recording of its own tissue; the sheet says which test item and dose each
  chamber received (one row per file and chamber, chamber letter in the
  'electrode' column or in 'chamber').
* **Names that are wrong or ambiguous**: a chamber number typed wrong, the
  same chip letter used for two chips, repeated recordings, a file with two
  inputs and one tissue named. A ``samples.csv`` in the analysed folder (or
  in a sub-folder) says what each input holds; for the files it lists it
  takes precedence over the names.

``draft_sample_sheet(folder)`` writes ``samples_draft.csv``: one row per file
and input as the software reads the names, with a note on the rows to check.
Review it and save it as ``samples.csv``.

Columns (English or Italian headers, comma or semicolon separated):

  file        path relative to the folder of the sheet, or the bare file
              name when it is unique there
  electrode   el1 / el2 (input 1 / 2); empty or 'auto' = the software picks,
              as for any single-tissue file. For a multi-chamber MCS file:
              the chamber letter ('A') — the software picks the electrode
              within the chamber — or an electrode label ('E18')
  experiment  optional: overrides the experiment folder (a chip recorded in
              the folder of another experiment)
  chip, chamber
  item        test item or drug
  dose        'baseline' (or 't0') for references, 'washout', otherwise the
              concentration or dose condition ('A', '300 nM')
  exclude     any text leaves the input out (the text is the reason)
  note        free text; the draft writes its checks here
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .loader import describe_recording, input_columns, parse_inputs, tissue_key

logger = logging.getLogger(__name__)

SHEET_NAME = 'samples.csv'
DRAFT_NAME = 'samples_draft.csv'
COLUMNS = ('file', 'electrode', 'experiment', 'chip', 'chamber', 'item', 'dose', 'exclude', 'note')
CHAMBERS_PER_CHIP = 3

_HEADER_ALIASES = {
    'file': ('file', 'filename', 'file name', 'nome file', 'nome_file'),
    'electrode': ('electrode', 'elettrodo', 'input', 'ingresso'),
    'experiment': ('experiment', 'esperimento'),
    'chip': ('chip',),
    'chamber': ('chamber', 'camera'),
    'item': ('item', 'test item', 'test_item', 'drug', 'farmaco', 'compound', 'composto'),
    'dose': ('dose', 'concentration', 'concentrazione', 'condition', 'condizione'),
    'exclude': ('exclude', 'escludi', 'excluded', 'escluso'),
    'note': ('note', 'notes'),
}
_REFERENCE_DOSES = {'baseline': 'baseline', 'basale': 'baseline', 't0': 't0'}
_WASHOUT_DOSES = {'washout', 'wash', 'lavaggio'}


def is_sheet_file(path) -> bool:
    """True for samples.csv / samples_draft.csv (not recordings)."""
    return Path(path).name.lower() in (SHEET_NAME, DRAFT_NAME)


def _norm_electrode(value):
    s = str(value or '').strip().lower()
    if s in ('', 'auto'):
        return 'auto'
    m = re.fullmatch(r'(?:el|input|ingresso|channel|canale)?\s*([12])', s)
    if m:
        return f'el{m.group(1)}'
    if re.fullmatch(r'[a-z]', s):                       # chamber letter of a multi-chamber chip
        return s.upper()
    if re.fullmatch(r'e\d{1,3}', s):                    # electrode label of an MCS file
        return s.upper()
    return None


@dataclass
class SheetRow:
    file: str
    electrode: str = 'auto'
    experiment: str = ''
    chip: str = ''
    chamber: int | str | None = None            # 1, 2 … or a chamber letter ('A')
    item: str = ''
    dose: str = ''
    exclude: str = ''
    note: str = ''
    line: int = 0

    @property
    def role(self) -> str:
        d = self.dose.strip().lower()
        if d in _REFERENCE_DOSES:
            return 'baseline'
        return 'washout' if d in _WASHOUT_DOSES else 'dose'


# ──────────────────────────────────────────────────────────────────────
#   Reading
# ──────────────────────────────────────────────────────────────────────

def read_sample_sheet(path):
    """(rows, problems) of one sample sheet; rows that cannot be used are
    left out and described in ``problems``."""
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], ['empty file']
    delim = ';' if lines[0].count(';') > lines[0].count(',') else ','
    reader = csv.reader(lines, delimiter=delim)
    header = [h.strip().lower() for h in next(reader)]
    col = {}
    for key, names in _HEADER_ALIASES.items():
        for i, h in enumerate(header):
            if h in names:
                col[key] = i
                break
    if 'file' not in col:
        return [], ["no 'file' column"]

    def get(rec, key):
        i = col.get(key)
        return rec[i].strip() if i is not None and i < len(rec) else ''

    rows, problems = [], []
    for n, rec in enumerate(reader, start=2):
        if not any(c.strip() for c in rec):
            continue
        name = get(rec, 'file')
        if not name:
            problems.append(f'line {n}: no file')
            continue
        el = _norm_electrode(get(rec, 'electrode'))
        if el is None:
            problems.append(f"line {n}: electrode '{get(rec, 'electrode')}' not understood (el1, el2, a chamber letter, an electrode label or empty)")
            continue
        chamber = None
        ch_txt = get(rec, 'chamber')
        if ch_txt:
            m = re.fullmatch(r'(?:ch|camera|chamber)?\s*(\d+|[A-Za-z])', ch_txt.strip())
            if m is None:
                problems.append(f"line {n}: chamber '{ch_txt}' not understood")
                continue
            chamber = int(m.group(1)) if m.group(1).isdigit() else m.group(1).upper()
        elif re.fullmatch(r'[A-Z]', el):
            chamber = el                                   # chamber letter given as the electrode
        row = SheetRow(file=name.replace('\\', '/'), electrode=el, experiment=get(rec, 'experiment'),
                       chip=re.sub(r'^chip[\s_-]*', '', get(rec, 'chip'), flags=re.IGNORECASE).upper(),
                       chamber=chamber, item=get(rec, 'item'), dose=get(rec, 'dose'),
                       exclude=get(rec, 'exclude'), note=get(rec, 'note'), line=n)
        if not row.exclude and (not row.chip or row.chamber is None):
            problems.append(f'line {n}: chip and chamber are needed (or a reason under exclude)')
            continue
        rows.append(row)
    return rows, problems


def find_sample_sheets(data_dir):
    """samples.csv in ``data_dir`` and its sub-folders, outermost first."""
    data_dir = Path(data_dir)
    found = [p for p in data_dir.rglob('*') if p.is_file() and p.name.lower() == SHEET_NAME]
    return sorted(found, key=lambda p: (len(p.relative_to(data_dir).parts), str(p)))


# ──────────────────────────────────────────────────────────────────────
#   Planning the batch
# ──────────────────────────────────────────────────────────────────────

def _is_under(path, folder):
    try:
        Path(path).relative_to(folder)
        return True
    except ValueError:
        return False


def _rel(path, root):
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _same_tissue(a, b):
    return (not a.get('empty') and not b.get('empty')
            and (a['chip'], a['chamber']) == (b['chip'], b['chamber']))


def _shared_concentration(entries, path, info):
    """Dose written once for the whole file ('…_K02_A'), else the
    concentration of the name, else a dose-letter folder ('A/')."""
    for e in entries:
        if e.get('concentration'):
            return e['concentration']
    if info.get('concentration') not in (None, '', '0'):
        return info['concentration']
    parent = Path(path).parent.name
    return parent if re.fullmatch(r'[A-Z]', parent) else None


def _name_update(entry, entries, path, info, two_tissues):
    role = 'baseline' if info.get('is_baseline') else 'dose'
    upd = {'tissue': tissue_key(info.get('experiment'), info.get('day'), entry['chip'], entry['chamber']),
           'chip': entry['chip'], 'channel': entry['chamber'], 'chamber': entry['chamber'],
           'item': entry.get('item'), 'role': role, 'is_baseline': role == 'baseline',
           'dual_tissue': False, 'two_tissue_file': two_tissues, 'source': 'file name'}
    if role == 'baseline':
        upd.update({'drug': 'baseline', 'concentration': '0',
                    'reference_kind': info.get('reference_kind', 'baseline')})
    else:
        upd.update({'drug': entry.get('item') or info.get('drug'),
                    'concentration': entry.get('concentration') or _shared_concentration(entries, path, info)})
    return upd


def _chamber_fields(path, chamber, layout=None):
    """Electrodes of a chamber of a multi-chamber file (or {} when the file
    has no layout or the chamber is not in it)."""
    from .chambers import layout_for
    from .loader import recording_channels
    if not isinstance(chamber, str):
        return {}
    lay = layout_for(recording_channels(path), layout or 'auto')
    if lay is None or chamber not in lay.names:
        return {}
    c = lay[chamber]
    return {'electrodes': list(c.electrodes), 'stimulation_electrodes': list(c.stimulation),
            'chamber_layout': lay.name, 'two_tissue_file': True}


def _sheet_update(row, info, two_tissues):
    role = row.role
    upd = {'tissue': tissue_key(row.experiment or info.get('experiment'), info.get('day'), row.chip, row.chamber),
           'chip': row.chip, 'channel': row.chamber, 'chamber': row.chamber, 'item': row.item or None,
           'role': role, 'is_baseline': role == 'baseline', 'dual_tissue': False,
           'two_tissue_file': two_tissues, 'source': SHEET_NAME}
    if row.experiment:
        upd['experiment'] = row.experiment
    if role == 'baseline':
        upd.update({'drug': 'baseline', 'concentration': '0',
                    'reference_kind': 't0' if row.dose.strip().lower() == 't0' else 'baseline'})
    else:
        upd.update({'drug': row.item or None, 'concentration': row.dose or None})
    return upd


def _units_from_sheet(path, rel, info, rows, report, layout=None):
    active = [r for r in rows if not r.exclude]
    for r in rows:
        if r.exclude:
            report['excluded'].append((rel, r.electrode, r.exclude))
    if any(isinstance(r.chamber, str) for r in active):
        # multi-chamber file: one recording per chamber row, electrode chosen
        # within the chamber (or the electrode label the row gives)
        out = []
        seen = set()
        for r in active:
            if not isinstance(r.chamber, str):
                report['issues'].append(f'{rel} line {r.line}: numbered chamber in a multi-chamber file; skipped')
                continue
            if r.chamber in seen:
                report['issues'].append(f'{rel} line {r.line}: chamber {r.chamber} listed twice; only the first is used')
                continue
            seen.add(r.chamber)
            fields = _chamber_fields(path, r.chamber, layout)
            if not fields:
                report['issues'].append(f'{rel} line {r.line}: chamber {r.chamber} not in the layout of the file; skipped')
                continue
            el = r.electrode if r.electrode.startswith('E') else 'auto'
            if el != 'auto' and el not in fields['electrodes']:
                report['issues'].append(f'{rel} line {r.line}: electrode {el} is not in chamber {r.chamber}; the software picks')
                el = 'auto'
            upd = _sheet_update(r, info, True)
            upd.update(fields)
            out.append({'file': path, 'electrode': el, 'update': upd, 'source': SHEET_NAME, 'multi': True})
        report['multi_chamber_files'] += 1
        report['from_sheet'] += len(out)
        return out
    els = [r.electrode for r in active]
    if len(active) > 1 and ('auto' in els or len(set(els)) < len(els)):
        report['issues'].append(f'{rel}: {len(active)} rows need different electrodes (el1, el2); '
                                f'only the first is used')
        active = active[:1]
    two = len({(r.experiment, r.chip, r.chamber) for r in active}) > 1
    report['two_tissue_files'] += int(two)
    report['from_sheet'] += len(active)
    return [{'file': path, 'electrode': r.electrode, 'update': _sheet_update(r, info, two),
             'source': SHEET_NAME, 'multi': len(active) > 1} for r in active]


def _units_from_layout(path, rel, info, report, layout=None):
    """Multi-chamber file without a sheet: one recording per chamber of the
    layout; chip = plate from the name, item unknown, dose from the name."""
    from .chambers import layout_for
    from .loader import recording_channels
    if info.get('format') != 'mcs_hdf5':
        return None
    lay = layout_for(recording_channels(path), layout or 'auto')
    if lay is None:
        return None
    role = 'baseline' if info.get('is_baseline') else 'dose'
    out = []
    for c in lay.chambers:
        upd = {'tissue': tissue_key(info.get('experiment'), info.get('day'), info.get('chip') or path.stem, c.name),
               'chip': info.get('chip') or path.stem, 'channel': c.name, 'chamber': c.name, 'item': None,
               'role': role, 'is_baseline': role == 'baseline', 'dual_tissue': False, 'two_tissue_file': True,
               'source': f'layout {lay.name}', 'electrodes': list(c.electrodes),
               'stimulation_electrodes': list(c.stimulation), 'chamber_layout': lay.name}
        if role == 'baseline':
            upd.update({'drug': 'baseline', 'concentration': '0', 'reference_kind': info.get('reference_kind', 'baseline')})
        else:
            upd.update({'drug': info.get('drug'), 'concentration': info.get('concentration')})
        out.append({'file': path, 'electrode': 'auto', 'update': upd, 'source': f'layout {lay.name}', 'multi': True})
    report['multi_chamber_files'] += 1
    report['from_layout'] += len(out)
    return out


def _units_from_name(path, rel, info, report):
    entries = parse_inputs(path.name)
    if len(entries) >= 2 and not _same_tissue(entries[0], entries[1]) and input_columns(path) == ['1', '2']:
        if len(entries) > 2:
            report['issues'].append(f'{rel}: {len(entries)} inputs named, the file has 2; the first two are used')
        pair = entries[:2]
        n_tissues = sum(1 for e in pair if not e.get('empty'))
        out = []
        for k, e in enumerate(pair, start=1):
            if e.get('empty'):
                report['excluded'].append((rel, f'el{k}', 'no tissue on this input (na)'))
                continue
            out.append({'file': path, 'electrode': f'el{k}',
                        'update': _name_update(e, entries, path, info, n_tissues == 2),
                        'source': 'file name', 'multi': n_tissues == 2})
        report['two_tissue_files'] += int(n_tissues == 2)
        report['from_name'] += len(out)
        return out
    # one tissue (or none) named: as before, the software picks the electrode
    return [{'file': path, 'electrode': 'auto', 'update': None, 'source': '', 'multi': False}]


def plan_batch(csv_files, data_dir, infos=None, layout=None):
    """Recordings to analyse: one per file, or one per input.

    Returns (units, report). Each unit: {'file', 'electrode' ('auto', 'el1',
    'el2', or an electrode label), 'update' (file_info fields that replace
    those read from the name, or None; for a chamber of a multi-chamber chip
    it carries 'electrodes' and 'stimulation_electrodes'), 'source', 'multi'
    (the file yields several recordings), 'uid'}. ``layout``: chamber layout
    choice ('auto', 'none' or a name; AnalysisConfig.chamber_layout). ``report`` lists the sheets used, excluded inputs, files under a
    sheet but not listed in it, and problems found.
    """
    data_dir = Path(data_dir)
    infos = infos or {}
    sheets = find_sample_sheets(data_dir)
    report = {'sheets': [_rel(s, data_dir) for s in sheets], 'issues': [], 'excluded': [],
              'not_in_sheet': [], 'two_tissue_files': 0, 'multi_chamber_files': 0, 'from_sheet': 0,
              'from_name': 0, 'from_layout': 0}
    by_file = defaultdict(list)
    for s in sheets:
        rows, problems = read_sample_sheet(s)
        rel_s = _rel(s, data_dir)
        report['issues'] += [f'{rel_s}: {m}' for m in problems]
        under = [f for f in csv_files if _is_under(f, s.parent)]
        rel_index = {Path(f).relative_to(s.parent).as_posix().lower(): f for f in under}
        name_index = defaultdict(list)
        for f in under:
            name_index[Path(f).name.lower()].append(f)
        for r in rows:
            key = r.file.lower()
            while key.startswith('./'):
                key = key[2:]
            f = rel_index.get(key) or rel_index.get(key + '.csv') or rel_index.get(key + '.h5') or rel_index.get(key + '.npz')
            if f is None:
                cands = (name_index.get(Path(key).name) or name_index.get(Path(key).name + '.csv')
                         or name_index.get(Path(key).name + '.h5') or name_index.get(Path(key).name + '.npz') or [])
                if len(cands) > 1:
                    report['issues'].append(f"{rel_s} line {r.line}: '{r.file}' matches {len(cands)} files; "
                                            f"write the path")
                    continue
                f = cands[0] if cands else None
            if f is None:
                report['issues'].append(f"{rel_s} line {r.line}: '{r.file}' not found")
                continue
            by_file[f].append(r)
    covered = [s.parent for s in sheets]
    units = []
    for f in csv_files:
        f = Path(f)
        info = infos.get(f) or describe_recording(f)
        rel = _rel(f, data_dir)
        if f in by_file:
            units += _units_from_sheet(f, rel, info, by_file[f], report, layout)
            continue
        if any(_is_under(f, d) for d in covered):
            report['not_in_sheet'].append(rel)
        from_layout = _units_from_layout(f, rel, info, report, layout)
        units += from_layout if from_layout is not None else _units_from_name(f, rel, info, report)
    for i, u in enumerate(units):
        u['uid'] = i
    for msg in report['issues']:
        logger.warning('sample sheet: %s', msg)
    if report['not_in_sheet']:
        logger.warning('%d file(s) under a samples.csv but not listed in it: read from their names (e.g. %s)',
                       len(report['not_in_sheet']), report['not_in_sheet'][0])
    return units, report


def describe_plan(report) -> list[str]:
    """Short human-readable lines about a batch plan (for logs and the UI)."""
    lines = []
    if report.get('sheets'):
        lines.append(f"samples.csv used: {', '.join(report['sheets'])} "
                     f"({report.get('from_sheet', 0)} recordings defined there)")
    if report.get('two_tissue_files'):
        lines.append(f"{report['two_tissue_files']} file(s) with two tissues, one recording per input")
    if report.get('multi_chamber_files'):
        lines.append(f"{report['multi_chamber_files']} multi-chamber file(s), one recording per chamber"
                     + (f" ({report['from_layout']} from the chip layout)" if report.get('from_layout') else ''))
    if report.get('excluded'):
        lines.append(f"{len(report['excluded'])} input(s) left out (no tissue or excluded in samples.csv)")
    if report.get('not_in_sheet'):
        lines.append(f"{len(report['not_in_sheet'])} file(s) not listed in samples.csv, read from their names")
    if report.get('issues'):
        lines.append(f"{len(report['issues'])} problem(s) in samples.csv: {report['issues'][0]}"
                     + (' …' if len(report['issues']) > 1 else ''))
    return lines


# ──────────────────────────────────────────────────────────────────────
#   Draft
# ──────────────────────────────────────────────────────────────────────

def _draft_rows_for_file(path, folder, layout=None):
    info = describe_recording(path)
    rel = _rel(path, folder)
    if info.get('format') == 'mcs_hdf5':
        from .chambers import layout_for
        from .loader import recording_channels
        lay = layout_for(recording_channels(path), layout or 'auto')
        is_ref = bool(info.get('is_baseline'))
        dose = ('t0' if info.get('reference_kind') == 't0' else 'baseline') if is_ref else (info.get('concentration') or '')
        if lay is None:
            return [{'experiment': '', 'exclude': '', 'file': rel, 'electrode': '', 'chip': info.get('chip') or '',
                     'chamber': '', 'item': '', 'dose': dose,
                     'note': ['multi-electrode file without a known chamber layout: the software picks one electrode']}], info
        rows = []
        for c in lay.chambers:
            rows.append({'experiment': '', 'exclude': '', 'file': rel, 'electrode': c.name, 'chip': info.get('chip') or '',
                         'chamber': c.name, 'item': '', 'dose': dose,
                         'note': [] if is_ref else ['write the test item of this chamber']})
        return rows, info
    entries = parse_inputs(path.name)
    inputs = input_columns(path)
    named = [e for e in entries if not e.get('empty')]
    is_ref = bool(info.get('is_baseline'))
    text = path.stem.lower()
    washout = 'wash' in text or 'recovery' in text

    def dose_of(e):
        if is_ref:
            return 't0' if info.get('reference_kind') == 't0' else 'baseline'
        if washout:
            return 'washout'
        return (e or {}).get('concentration') or _shared_concentration(entries, path, info) or ''

    def item_of(e):
        if is_ref:
            return ''
        return (e or {}).get('item') or (info.get('drug') or '')

    base = {'experiment': '', 'exclude': ''}
    rows = []
    notes = []
    n_chip_words = len(re.findall(r'chip', path.stem, re.IGNORECASE))
    n_chip_tokens = sum(1 for e in entries if e.get('chip'))
    if n_chip_words > n_chip_tokens:
        notes.append(f"the name has {n_chip_words} 'chip' but {n_chip_tokens} tissue(s) were recognised")
    if len(inputs) == 2 and len(entries) >= 2 and not _same_tissue(entries[0], entries[1]):
        if len(entries) > 2:
            notes.append(f'{len(entries)} inputs named, the file has 2')
        for k, e in enumerate(entries[:2], start=1):
            r = {**base, 'file': rel, 'electrode': f'el{k}', 'chip': e.get('chip', ''),
                 'chamber': e.get('chamber', ''), 'item': '', 'dose': '', 'note': list(notes)}
            if e.get('empty'):
                r['exclude'] = 'no tissue on this input (na)'
            else:
                r['item'], r['dose'] = item_of(e), dose_of(e)
            rows.append(r)
        return rows, info
    e = None
    if named:
        e = named[0]
        if len(inputs) == 1 and inputs[0] == '2' and len(entries) >= 2 and not entries[1].get('empty'):
            e = entries[1]
    chip = e['chip'] if e else (info.get('chip') or '')
    chamber = e['chamber'] if e else (info.get('channel') or '')
    if len(inputs) == 2 and named:
        if len(named) >= 2 and _same_tissue(named[0], named[1]):
            notes.append('both inputs on this chamber (two electrodes): the software picks one')
        else:
            notes.append('two inputs, one tissue named: write el1 or el2 if only one input holds it')
    if not chip or chamber in ('', None):
        notes.append('chip or chamber not found in the name or folders')
    rows.append({**base, 'file': rel, 'electrode': '', 'chip': chip, 'chamber': chamber,
                 'item': item_of(e), 'dose': dose_of(e), 'note': notes})
    return rows, info


def draft_sample_sheet(folder, write=True, chambers_per_chip=CHAMBERS_PER_CHIP, layout=None):
    """Rows of a sample sheet as the software reads the names, with checks.

    One row per file, or per input for files with two tissues; the 'note'
    column says what to check: a chamber number above ``chambers_per_chip``,
    a chamber whose test item changes between doses, repeated recordings of
    one chamber at one dose, chambers without a reference or missing a dose
    their item has elsewhere, a second tissue not recognised. With ``write``
    the rows go to ``samples_draft.csv`` in ``folder`` (semicolon separated,
    UTF-8). Returns (rows, path or None).
    """
    from .normalization import canonical_drug_name

    folder = Path(folder)
    from .analyze import find_recordings
    files = find_recordings(folder, is_sheet_file)
    rows = []
    for f in files:
        frows, info = _draft_rows_for_file(f, folder, layout)
        for r in frows:
            ok = r['chip'] != '' and r['chamber'] not in ('', None)
            r['_tissue'] = tissue_key(info.get('experiment'), info.get('day'), r['chip'], r['chamber']) if ok else None
            r['_ref'] = r['dose'] in ('baseline', 't0')
        rows += frows

    live = [r for r in rows if not r['exclude'] and r['_tissue']]
    by_tissue = defaultdict(list)
    for r in live:
        by_tissue[r['_tissue']].append(r)
    for r in live:
        if isinstance(r['chamber'], int) and r['chamber'] > chambers_per_chip:
            r['note'].append(f'chamber {r["chamber"]}: chips have {chambers_per_chip} chambers')
    item_doses = defaultdict(set)
    for t, rs in by_tissue.items():
        doses = [r for r in rs if not r['_ref'] and r['dose'] != 'washout']
        items = defaultdict(list)
        for r in doses:
            items[canonical_drug_name(r['item']) or '?'].append(r['dose'] or '?')
        if len(items) > 1:
            msg = 'test item changes in this chamber: ' + '; '.join(
                f"{k} ({', '.join(sorted(set(v)))})" for k, v in sorted(items.items()))
            for r in doses:
                r['note'].append(msg)
        if doses and not any(r['_ref'] for r in rs):
            for r in doses:
                r['note'].append('no reference for this chamber')
        if len(items) == 1:
            item_doses[next(iter(items))].update(r['dose'] for r in doses if r['dose'])
        counts = defaultdict(list)
        for r in rs:
            counts['reference' if r['_ref'] else (r['dose'] or '?')].append(r)
        for d, same in counts.items():
            if len(same) > 1:
                what = ('the last one before the first dose is used' if d == 'reference'
                        else 'they are averaged')
                for r in same:
                    r['note'].append(f'{len(same)} recordings of this chamber at {d}: '
                                     f'exclude the ones not to use ({what} otherwise)')
    for t, rs in by_tissue.items():
        doses = [r for r in rs if not r['_ref'] and r['dose'] != 'washout']
        items = {canonical_drug_name(r['item']) for r in doses}
        if len(items) != 1 or not doses:
            continue
        missing = sorted(item_doses[next(iter(items))] - {r['dose'] for r in doses})
        if missing:
            for r in doses:
                r['note'].append(f"no recording at {', '.join(missing)} (other chambers with "
                                 f"{doses[0]['item']} have it)")
    for r in rows:
        if not r['exclude'] and not r['_ref'] and r['dose'] != 'washout':
            if not r['dose']:
                r['note'].append('dose not found in the name')
            if not r['item']:
                r['note'].append('test item not found in the name')

    out_rows = [{**{k: r[k] for k in COLUMNS if k != 'note'}, 'note': ' | '.join(dict.fromkeys(r['note']))}
                for r in rows]
    path = None
    if write:
        path = folder / DRAFT_NAME
        with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=list(COLUMNS), delimiter=';')
            w.writeheader()
            w.writerows(out_rows)
    return out_rows, path


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write samples_draft.csv for a folder of recordings.')
    ap.add_argument('folder')
    ap.add_argument('--chambers', type=int, default=CHAMBERS_PER_CHIP, help='chambers per chip (default 3)')
    args = ap.parse_args(argv)
    rows, path = draft_sample_sheet(args.folder, chambers_per_chip=args.chambers)
    flagged = sum(1 for r in rows if r['note'])
    print(f'{path}: {len(rows)} rows, {flagged} with a note to check. '
          f'Review it and save it as {SHEET_NAME} in the same folder.')


if __name__ == '__main__':
    main()
