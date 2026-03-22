"""
cdisc_export.py — CDISC SEND export for regulatory submission.

Generates SAS Transport v5 (.xpt) files following the SEND
Implementation Guide (SENDIG v3.1.1) for nonclinical electrophysiology data.

Domains generated:
  - TS  (Trial Summary)        : study-level metadata
  - DM  (Demographics)         : subject-level (one row per microtissue)
  - EX  (Exposure)             : drug treatments and concentrations
  - EG  (ECG Test Results)     : FPD, FPDcF, BP, amplitude measurements

Also generates:
  - define.xml        : Dataset-level metadata (Define-XML 2.1)
  - define2-1-0.xsl   : Pinnacle 21 official stylesheet for define.xml rendering

Reference standards:
  - SENDIG v3.1.1 (CDISC / FDA)
  - NCI Controlled Terminology (2025-09-26)
  - CiPA framework (Blinova et al. 2017)
  - FDA Technical Conformance Guide

Usage:
    from cardiac_fp_analyzer.cdisc_export import export_send_package
    export_send_package(results, output_dir, study_id='STUDY001')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import io, warnings, shutil, re

warnings.filterwarnings('ignore')


# ── SEND Controlled Terminology (NCI CT 2025-09-26) ─────────────────

_UNITS = {
    'ms': 'ms',
    'mV': 'mV',
    'uV': 'uV',
    'Hz': 'Hz',
    'bpm': 'BEATS/MIN',
    's': 's',
    'pct': '%',
}

# SEND-compliant EGTESTCD codes (max 8 chars, uppercase)
# Format: EGTESTCD -> (category, EGTEST label, unit)
_TEST_CODES = {
    'FPD':      ('EGTEST', 'FPD',                                'ms'),
    'FPDCF':    ('EGTEST', 'FPDcF',                              'ms'),
    'QTCF':     ('EGTEST', 'QTcF Interval',                      'ms'),
    'BP':       ('EGTEST', 'Beat Period',                         'ms'),
    'SPIKEAM':  ('EGTEST', 'Spike Amplitude',                    'uV'),
    'RISETM':   ('EGTEST', 'Rise Time 10-90',                    'ms'),
    'MAXDVDT':  ('EGTEST', 'Maximum dV/dt',                      'mV'),
    'INTVL':    ('EGTEST', 'Heart Rate',                         'BEATS/MIN'),
    'FPDCFPC':  ('EGTEST', 'FPDcF Pct Change from Baseline',    '%'),
    'BPPCT':    ('EGTEST', 'Beat Period Pct Change from BL',     '%'),
    'AMPPCT':   ('EGTEST', 'Amplitude Pct Change from BL',      '%'),
    'TDPSCR':   ('EGTEST', 'TdP Risk Score',                    None),
    'QCGRADE':  ('EGTEST', 'QC Grade',                          None),
    'FPDCONF':  ('EGTEST', 'FPD Confidence',                    None),
    'RSKSCR':   ('EGTEST', 'Arrhythmia Risk Score',             None),
    'MORPHIN':  ('EGTEST', 'Morphology Instability Index',      None),
    'EADPCT':   ('EGTEST', 'EAD Incidence Pct',                 '%'),
    'STVFPDC':  ('EGTEST', 'STV of FPDcF',                      'ms'),
    'SPECCHG':  ('EGTEST', 'Spectral Change Index',             None),
    'PROAIDX':  ('EGTEST', 'Proarrhythmic Index',               None),
}

# Drug name normalization for SEND
_DRUG_SEND_NAMES = {
    'terfenadine': 'TERFENADINE',
    'quinidine':   'QUINIDINE',
    'dofetilide':  'DOFETILIDE',
    'alfuzosin':   'ALFUZOSIN',
    'mexiletine':  'MEXILETINE',
    'nifedipine':  'NIFEDIPINE',
    'ranolazine':  'RANOLAZINE',
}

# Variable labels for SEND (used in .xpt metadata)
_VAR_LABELS = {
    # DM
    'STUDYID': 'Study Identifier',
    'DOMAIN':  'Domain Abbreviation',
    'USUBJID': 'Unique Subject Identifier',
    'SUBJID':  'Subject Identifier for the Study',
    'SPECIES': 'Species',
    'STRAIN':  'Strain/Substrain',
    'SEX':     'Sex',
    'ARMCD':   'Planned Arm Code',
    'ARM':     'Description of Planned Arm',
    'SETCD':   'Set Code',
    'RFSTDTC': 'Subject Reference Start Date/Time',
    'RFENDTC': 'Subject Reference End Date/Time',
    # EX
    'EXSEQ':    'Sequence Number',
    'EXTRT':    'Name of Treatment',
    'EXDOSE':   'Dose',
    'EXDOSU':   'Dose Units',
    'EXDOSFRQ': 'Dosing Frequency per Interval',
    'EXROUTE':  'Route of Administration',
    'EXDOSFRM': 'Dose Form',
    'EPOCH':    'Epoch',
    'EXSTDTC':  'Start Date/Time of Treatment',
    'EXSTDY':   'Study Day of Start of Treatment',
    # EG
    'EGSEQ':    'Sequence Number',
    'EGTESTCD': 'ECG Test Short Name',
    'EGTEST':   'ECG Test Name',
    'EGORRES':  'Result or Finding in Original Units',
    'EGORRESU': 'Original Units',
    'EGSTRESC': 'Character Result/Finding in Std Format',
    'EGSTRESN': 'Numeric Result/Finding in Standard Units',
    'EGSTRESU': 'Standard Units',
    'EGSTAT':   'Completion Status',
    'EGMETHOD': 'Method of Test or Examination',
    'EGBLFL':   'Baseline Flag',
    'EGTPTREF': 'Time Point Reference',
    'EGDTC':    'Date/Time of Collection',
    'EGDY':     'Study Day of Collection',
    'VISITDY':  'Planned Study Day of Visit',
    'EGEVAL':   'Evaluator',
    'EGREFID':  'Reference ID',
    'EGLEAD':   'Lead Identified to Collect Measurements',
    # TS
    'TSSEQ':    'Sequence Number',
    'TSPARMCD': 'Trial Summary Parameter Short Name',
    'TSVAL':    'Parameter Value',
    'TSVALNF':  'Parameter Null Flavor',
    'TSPARM':   'Trial Summary Parameter',
}


def _canonical_drug_send(raw: str) -> str:
    """Map raw drug name to SEND-compatible name (uppercase)."""
    from .risk_map import _canonical_drug
    canon = _canonical_drug(raw)
    return _DRUG_SEND_NAMES.get(canon, canon.upper())


def _make_usubjid(study_id: str, result: dict) -> str:
    """Generate USUBJID from study + chip + electrode."""
    fi = result.get('file_info', {})
    chip = fi.get('chip', 'X')
    el = fi.get('analyzed_channel', fi.get('channel_label', 'el0'))
    return f"{study_id}-{chip}-{el}".upper()


def _make_seq(counter: dict, domain: str) -> int:
    """Auto-incrementing sequence number per domain."""
    counter.setdefault(domain, 0)
    counter[domain] += 1
    return counter[domain]


def _parse_concentration(conc_str) -> tuple:
    """Parse concentration string like '100 nM' into (numeric_value, unit).

    Returns (float_value, unit_string) or (None, None) if unparseable.
    Units are normalized to standard case: nM, uM, mM, M.
    """
    # Standard unit normalization (case-insensitive -> canonical)
    _UNIT_MAP = {
        'nm': 'nM', 'um': 'uM', 'mm': 'mM', 'm': 'M',
        'ug/ml': 'ug/mL', 'mg/ml': 'mg/mL', 'ng/ml': 'ng/mL',
    }

    if conc_str is None or conc_str == '':
        return None, None
    s = str(conc_str).strip()
    m = re.match(r'^([\d.]+)\s*(.*)$', s)
    if m:
        try:
            val = float(m.group(1))
            raw_unit = m.group(2).strip() if m.group(2).strip() else None
            if raw_unit:
                raw_unit = _UNIT_MAP.get(raw_unit.lower(), raw_unit)
            return val, raw_unit
        except ValueError:
            pass
    try:
        return float(s), None
    except ValueError:
        return None, None


# ═══════════════════════════════════════════════════════════════════════
#  TS — Trial Summary
# ═══════════════════════════════════════════════════════════════════════

def _build_ts(study_id: str, study_title: str, n_results: int,
              sponsor: str = '', start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
    """Build Trial Summary (TS) domain with all required SENDIG 3.1.1 parameters."""
    now = start_date or datetime.now().strftime('%Y-%m-%d')
    end = end_date or now

    # All required / expected TS parameters per SENDIG 3.1.1 + FDA
    # TSPARMCD -> (TSPARM [matching CDISC CT exactly], TSVAL, TSVALNF)
    params = [
        # Study identification
        ('STUDYID',  'Study Identifier',                          study_id,                   ''),
        ('SSTDTC',   'Study Start Date',                          now,                        ''),
        ('SENDTC',   'Study End Date',                            end,                        ''),
        ('STITLE',   'Study Title',                               study_title,                ''),
        ('SDESIGN',  'Study Design',                              'PARALLEL',                 ''),
        ('SSPONSOR', 'Study Sponsor',                             sponsor or 'NOT PROVIDED',  ''),
        ('STYPE',    'Study Type',                                'NONCLINICAL',              ''),

        # Species / Strain
        ('SPECIES',  'Species',                                   'HUMAN',                    ''),
        ('STRAIN',   'Strain/Substrain',                          'HIPSC-CM',                 ''),
        ('SSTYP',    'Study Subtype',                             'IN VITRO',                 ''),

        # Route and treatment
        ('ROUTE',    'Route of Administration',                   'TOPICAL',                  ''),
        ('TRT',      'Treatment',                                 'MULTIPLE DRUGS',           ''),
        ('TRTV',     'Treatment Vehicle',                         'CULTURE MEDIUM',           ''),
        ('TRTCAS',   'CAS Number of Treatment',                   '',                         'NA'),
        ('TRTUNII',  'UNII of Treatment',                         '',                         'NA'),
        ('PCLASS',   'Pharmacological Class of Treatment',        'ION CHANNEL MODULATORS',   ''),

        # Subject / design
        ('NSUBJ',    'Number of Subjects',                        str(n_results),             ''),
        ('SEXPOP',   'Sex of Participants',                       'UNKNOWN',                  ''),
        ('STCAT',    'Study Category',                            'SAFETY PHARMACOLOGY',      ''),
        ('STDIR',    'Study Director',                            sponsor or 'NOT PROVIDED',  ''),

        # Timing
        ('DOSDUR',   'Duration of Dosing',                        'ACUTE',                    ''),
        ('DOSSTDTC', 'Start Date of Dosing',                      now,                        ''),
        ('DOSENDTC', 'End Date of Dosing',                        end,                        ''),
        ('PDOSFRQ',  'Planned Dosing Frequency',                  'ONCE',                     ''),
        ('EXPSTDTC', 'Start Date of Experimental Phase',          now,                        ''),
        ('EXPENDTC', 'End Date of Experimental Phase',            end,                        ''),
        ('STSTDTC',  'Start Date of Treatment',                   now,                        ''),

        # Regulatory / compliance
        ('GLPFL',    'GLP Study Flag',                            'N',                        ''),
        ('GLPTYP',   'GLP Study Type',                            '',                         'NA'),
        ('SNDIGVER', 'SENDIG Version',                            '3.1.1',                    ''),
        ('SNDCTVER', 'SEND Controlled Terminology Version',       '2025-09-26',               ''),
        ('REGID',    'Regulatory Identifier',                     'CIPA',                     ''),

        # Test facility
        ('TSTFLOC',  'Test Facility Location',                    '',                         'NA'),
        ('TSTFNAM',  'Test Facility Name',                        '',                         'NA'),
        ('SPLRNAM',  'Supplier Name',                             '',                         'NA'),
        ('SPREFID',  'Supplier Reference Identifier',             '',                         'NA'),
        ('SPLANSUB', 'Planned Number of Animals per Subset',      '',                         'NA'),
        ('STRPSTAT', 'Strain Production Status',                  '',                         'NA'),
        ('TRMSAC',   'Reason for Sacrifice',                      '',                         'NA'),

        # Age (NA for cell lines)
        ('AGE',      'Subject Age at Start of Study',             '',                         'NA'),
        ('AGEU',     'Age Unit',                                  '',                         'NA'),
    ]

    rows = []
    for i, (parmcd, parm, val, valnf) in enumerate(params, 1):
        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'TS',
            'TSSEQ':    i,
            'TSPARMCD': parmcd,
            'TSPARM':   parm,
            'TSVAL':    val,
            'TSVALNF':  valnf,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  DM — Demographics (one row per microtissue)
# ═══════════════════════════════════════════════════════════════════════

def _build_dm(results: list, study_id: str) -> pd.DataFrame:
    """Build Demographics (DM) domain — one row per unique microtissue."""
    from .normalization import _is_baseline, _get_group_key

    seen = {}
    rows = []
    now = datetime.now().strftime('%Y-%m-%d')

    for r in results:
        usubjid = _make_usubjid(study_id, r)
        if usubjid in seen:
            continue
        seen[usubjid] = True

        fi = r.get('file_info', {})
        exp = fi.get('experiment', '')

        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'DM',
            'USUBJID':  usubjid,
            'SUBJID':   usubjid.split('-', 1)[-1] if '-' in usubjid else usubjid,
            'RFSTDTC':  now,
            'RFENDTC':  now,
            'SPECIES':  'HUMAN',
            'STRAIN':   'HIPSC-CM',
            'SEX':      'U',
            'ARMCD':    'TRT',
            'ARM':      'Treatment',
            'SETCD':    exp.replace(' ', '') if exp else 'SET1',
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  EX — Exposure (drug treatments)
# ═══════════════════════════════════════════════════════════════════════

def _build_ex(results: list, study_id: str) -> pd.DataFrame:
    """Build Exposure (EX) domain — one row per drug exposure event."""
    from .normalization import _is_baseline, _is_control

    rows = []
    seq_counter = {}
    seen_keys = set()
    now = datetime.now().strftime('%Y-%m-%d')

    for r in results:
        if _is_baseline(r) or _is_control(r):
            continue
        fi = r.get('file_info', {})
        drug_raw = str(fi.get('drug', ''))
        if not drug_raw:
            continue

        drug = _canonical_drug_send(drug_raw)
        conc_raw = fi.get('concentration', '')
        usubjid = _make_usubjid(study_id, r)

        # Parse concentration into numeric + unit
        dose_num, dose_unit = _parse_concentration(conc_raw)

        # De-duplicate: same subject + drug + concentration
        dedup_key = (usubjid, drug, dose_num)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'EX',
            'USUBJID':  usubjid,
            'EXSEQ':    _make_seq(seq_counter, usubjid),
            'EXTRT':    drug,
            'EXDOSE':   dose_num if dose_num is not None else 0.0,
            'EXDOSU':   dose_unit or 'nM',
            'EXDOSFRQ': 'ONCE',
            'EXROUTE':  'TOPICAL',
            'EXDOSFRM': 'SOLUTION',
            'EPOCH':    'TREATMENT',
            'EXSTDTC':  now,
            'EXSTDY':   1,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  EG — ECG Test Results (the main data domain)
# ═══════════════════════════════════════════════════════════════════════

def _build_eg(results: list, study_id: str) -> pd.DataFrame:
    """
    Build ECG Test Results (EG) domain.

    Each row = one measurement (test) for one recording.
    Multiple rows per file: FPD, FPDcF, BP, amplitude, etc.
    """
    from .normalization import _is_baseline, _is_control

    rows = []
    seq_counter = {}
    seen_keys = set()
    now = datetime.now().strftime('%Y-%m-%d')

    for r in results:
        fi = r.get('file_info', {})
        s = r.get('summary', {})
        norm = r.get('normalization', {})
        ar = r.get('arrhythmia_report')
        qc = r.get('qc_report')

        usubjid = _make_usubjid(study_id, r)
        is_bl = _is_baseline(r) or _is_control(r)
        drug_raw = str(fi.get('drug', ''))
        drug = _canonical_drug_send(drug_raw) if drug_raw else 'BASELINE'
        conc = fi.get('concentration', '')
        fname = r.get('metadata', {}).get('filename', '')

        tptref = 'BASELINE' if is_bl else f"{drug} {conc}".strip()
        epoch = 'BASELINE' if is_bl else 'TREATMENT'
        visitdy = 1

        def _add_row(testcd, value, unit_override=None):
            """Helper to add one EG row."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return
            code_info = _TEST_CODES.get(testcd, ('EGTEST', testcd, unit_override))
            unit = unit_override or code_info[2] or ''

            # Round EGSTRESN to match EGSTRESC (fix SD1212)
            if isinstance(value, float):
                stresc = f"{value:.4f}"
                stresn = round(value, 4)
            else:
                stresc = str(value)
                stresn = float(value) if isinstance(value, (int, float)) else None

            # De-duplicate key
            dk = (usubjid, testcd, epoch, tptref)
            if dk in seen_keys:
                return
            seen_keys.add(dk)

            rows.append({
                'STUDYID':  study_id,
                'DOMAIN':   'EG',
                'USUBJID':  usubjid,
                'EGSEQ':    _make_seq(seq_counter, usubjid),
                'EGTESTCD': testcd,
                'EGTEST':   code_info[1],
                'EGORRES':  stresc,
                'EGORRESU':  unit,
                'EGSTRESC': stresc,
                'EGSTRESN': stresn,
                'EGSTRESU': unit,
                'EGSTAT':   '',
                'EGMETHOD': 'ALGORITHMIC ANALYSIS',
                'EGBLFL':   'Y' if is_bl else '',
                'EGLEAD':   'VIRTUAL LEAD',
                'EGTPTREF': tptref,
                'EPOCH':    epoch,
                'EGDTC':    now,
                'EGDY':     1,
                'VISITDY':  visitdy,
                'EGEVAL':   'ALGORITHM',
                'EGREFID':  fname,
            })

        # ── Core electrophysiology parameters ──
        _add_row('FPD',      s.get('fpd_ms_mean'))
        _add_row('FPDCF',    s.get('fpdc_ms_mean'))
        _add_row('BP',       s.get('beat_period_ms_mean'))
        _add_row('SPIKEAM',  s.get('spike_amplitude_mV_mean'))
        _add_row('RISETM',   s.get('rise_time_ms_mean'))
        _add_row('MAXDVDT',  s.get('max_dvdt_mean'))
        _add_row('FPDCONF',  s.get('fpd_confidence'))

        # ── Heart rate ──
        bpm = s.get('bpm_mean')
        if bpm:
            _add_row('INTVL', bpm)

        # ── Quality control ──
        if qc:
            _add_row('QCGRADE', qc.grade)

        # ── Normalized parameters (drug recordings only) ──
        if norm.get('has_baseline'):
            _add_row('FPDCFPC', norm.get('pct_fpdc_change'))
            _add_row('BPPCT',   norm.get('pct_bp_change'))
            _add_row('AMPPCT',  norm.get('pct_amp_change'))
            _add_row('TDPSCR',  norm.get('tdp_score'))
            _add_row('SPECCHG', norm.get('spectral_change_score'))

        # ── Arrhythmia metrics ──
        if ar:
            _add_row('RSKSCR', ar.risk_score)
            rd = ar.residual_details or {}
            _add_row('MORPHIN', rd.get('morphology_instability'))
            _add_row('EADPCT',  rd.get('ead_incidence_pct'))
            _add_row('STVFPDC', rd.get('poincare_stv_fpdc_ms'))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Define-XML 2.1
# ═══════════════════════════════════════════════════════════════════════

def _generate_define_xml(study_id: str, datasets: dict, output_dir: Path):
    """Generate Define-XML 2.1 metadata file."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<?xml-stylesheet type="text/xsl" href="define2-1-0.xsl"?>',
        '<!-- Define-XML v2.1 for SEND submission -->',
        f'<!-- Study: {study_id} -->',
        f'<!-- Generated: {datetime.now().isoformat()} -->',
        f'<!-- Generator: Cardiac FP Analyzer v3.6 -->',
        '',
        '<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"',
        '     xmlns:def="http://www.cdisc.org/ns/def/v2.1"',
        '     xmlns:xlink="http://www.w3.org/1999/xlink"',
        f'     FileOID="DEFINE.{study_id}"',
        '     FileType="Snapshot"',
        f'     CreationDateTime="{datetime.now().isoformat()}"',
        f'     ODMVersion="1.3.2">',
        f'  <Study OID="{study_id}">',
        f'    <GlobalVariables>',
        f'      <StudyName>{study_id}</StudyName>',
        f'      <StudyDescription>CiPA In Vitro Cardiac Safety Study - hiPSC-CM Field Potential Analysis</StudyDescription>',
        f'      <ProtocolName>{study_id}</ProtocolName>',
        f'    </GlobalVariables>',
        f'    <MetaDataVersion OID="MDV.{study_id}"',
        f'                     Name="SEND {study_id}"',
        f'                     def:StandardName="SENDIG"',
        f'                     def:StandardVersion="3.1">',
    ]

    domain_info = {
        'TS':   ('Trial Summary',   'Trial summary parameters and study metadata'),
        'DM':   ('Demographics',     'One record per microtissue (chip+channel)'),
        'EX':   ('Exposure',         'Drug exposure events with concentrations'),
        'EG':   ('ECG Test Results', 'Electrophysiology measurements'),
    }

    for domain, df in datasets.items():
        info = domain_info.get(domain, (domain, ''))
        n_rows = len(df)
        n_cols = len(df.columns)
        struct = 'One record per parameter' if domain == 'TS' else 'One record per test per subject'
        lines.append(f'      <def:ItemGroupDef OID="IG.{domain}"')
        lines.append(f'                        Name="{domain}"')
        lines.append(f'                        SASDatasetName="{domain}"')
        lines.append(f'                        def:Label="{info[0]}"')
        lines.append(f'                        def:Structure="{struct}"')
        lines.append(f'                        Purpose="Tabulation"')
        lines.append(f'                        def:StandardOID="STD.SENDIG.3.1">')
        lines.append(f'        <!-- {info[1]} ({n_rows} records, {n_cols} variables) -->')

        for col in df.columns:
            lines.append(f'        <ItemRef ItemOID="IT.{domain}.{col}" Mandatory="No"/>')

        lines.append(f'      </def:ItemGroupDef>')
        lines.append('')

    lines.append('      <!-- Variable Definitions -->')
    for domain, df in datasets.items():
        for col in df.columns:
            dtype = df[col].dtype
            sas_type = 'Char' if dtype == object else 'Num'
            length = int(df[col].astype(str).str.len().max()) if len(df) > 0 else 8
            length = max(length, 1)
            label = _VAR_LABELS.get(col, col.replace('_', ' ').title())
            lines.append(f'      <ItemDef OID="IT.{domain}.{col}"')
            lines.append(f'              Name="{col}"')
            lines.append(f'              SASFieldName="{col[:8]}"')
            lines.append(f'              DataType="{sas_type}"')
            lines.append(f'              Length="{min(length, 200)}"')
            lines.append(f'              def:Label="{label}"/>')

    lines.extend([
        '    </MetaDataVersion>',
        '  </Study>',
        '</ODM>',
    ])

    out_path = output_dir / 'define.xml'
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    return out_path


# ═══════════════════════════════════════════════════════════════════════
#  XPT file writer with variable labels
# ═══════════════════════════════════════════════════════════════════════

def _write_xpt(df: pd.DataFrame, filepath: Path, dataset_name: str,
               dataset_label: str = ''):
    """Write a DataFrame to SAS Transport v5 (.xpt) format.

    Attempts pyreadstat first (supports column labels), falls back to
    xport library, otherwise writes CSV with .xpt extension.
    """
    col_map = {}
    used_names = set()
    for col in df.columns:
        short = col[:8].upper()
        counter = 0
        while short in used_names:
            short = col[:6].upper() + str(counter)
            counter += 1
        col_map[col] = short
        used_names.add(short)

    df_sas = df.rename(columns=col_map).copy()

    for col in df_sas.columns:
        if df_sas[col].dtype == object:
            df_sas[col] = (df_sas[col].fillna('').astype(str).str[:200]
                           .str.encode('latin-1', errors='replace')
                           .str.decode('latin-1'))
        else:
            df_sas[col] = pd.to_numeric(df_sas[col], errors='coerce')

    # Build column labels mapping (original col -> label)
    col_labels = {}
    for orig_col, sas_col in col_map.items():
        col_labels[sas_col] = _VAR_LABELS.get(orig_col, orig_col)

    # Try pyreadstat first (best: supports labels)
    try:
        import pyreadstat
        pyreadstat.write_xport(
            df_sas, filepath,
            table_name=dataset_name[:8],
            column_labels=[col_labels.get(c, c) for c in df_sas.columns],
        )
        return
    except (ImportError, TypeError):
        # TypeError if column_labels not supported in this version
        try:
            import pyreadstat
            pyreadstat.write_xport(df_sas, filepath, table_name=dataset_name[:8])
            return
        except ImportError:
            pass

    # Fallback: xport library
    try:
        import xport
        with open(filepath, 'wb') as f:
            xport.from_dataframe(df_sas, f)
        return
    except ImportError:
        pass

    # Final fallback: CSV with warning
    warnings.warn(
        f"xport and pyreadstat not available. Writing {filepath.name} as CSV. "
        "Install pyreadstat: pip install pyreadstat",
        UserWarning
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# WARNING: This is a CSV fallback (xport/pyreadstat not installed)\n")
        f.write(f"# Dataset: {dataset_name} - {dataset_label}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("# pip install pyreadstat to generate proper SAS Transport v5\n\n")
        df_sas.to_csv(f, index=False)


# ═══════════════════════════════════════════════════════════════════════
#  Main export function
# ═══════════════════════════════════════════════════════════════════════

def export_send_package(
    results: list,
    output_dir,
    study_id: str = 'CIPA001',
    study_title: str = 'CiPA In Vitro Cardiac Safety - hiPSC-CM Field Potential',
    sponsor: str = '',
) -> dict:
    """
    Export analysis results as a CDISC SEND submission package.

    Parameters
    ----------
    results : list
        Output of batch_analyze().
    output_dir : str or Path
        Directory for output files.
    study_id : str
        Study identifier (e.g., 'CIPA001').
    study_title : str
        Full study title.
    sponsor : str
        Sponsor name (optional).

    Returns
    -------
    dict with keys:
        'files' : list of Path — generated files
        'datasets' : dict of DataFrames
        'summary' : str — human-readable summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build SEND datasets ──
    ts_df = _build_ts(study_id, study_title, len(results), sponsor=sponsor)
    dm_df = _build_dm(results, study_id)
    ex_df = _build_ex(results, study_id)
    eg_df = _build_eg(results, study_id)

    datasets = {
        'TS': ts_df,
        'DM': dm_df,
        'EX': ex_df,
        'EG': eg_df,
    }

    # ── Write XPT files ──
    files = []
    labels = {
        'TS': 'Trial Summary',
        'DM': 'Demographics',
        'EX': 'Exposure',
        'EG': 'ECG Test Results',
    }

    for domain, df in datasets.items():
        if df.empty:
            continue
        xpt_path = output_dir / f"{domain.lower()}.xpt"
        _write_xpt(df, xpt_path, domain, labels.get(domain, domain))
        files.append(xpt_path)

    # ── Write Define-XML ──
    define_path = _generate_define_xml(study_id, datasets, output_dir)
    files.append(define_path)

    # ── Copy XSL stylesheet for Define-XML rendering ──
    _xsl_src = Path(__file__).parent / 'define2-1-0.xsl'
    if _xsl_src.exists():
        _xsl_dst = output_dir / 'define2-1-0.xsl'
        shutil.copy2(_xsl_src, _xsl_dst)
        files.append(_xsl_dst)

    # ── Write CSV copies (for human review) ──
    csv_dir = output_dir / 'csv_review'
    csv_dir.mkdir(exist_ok=True)
    for domain, df in datasets.items():
        if not df.empty:
            csv_path = csv_dir / f"{domain.lower()}.csv"
            df.to_csv(csv_path, index=False)
            files.append(csv_path)

    # ── Summary ──
    summary_lines = [
        f"CDISC SEND Export Package (SENDIG 3.1.1)",
        f"Study: {study_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Datasets:",
    ]
    for domain, df in datasets.items():
        summary_lines.append(
            f"  {domain:6s} ({labels.get(domain, ''):30s}): "
            f"{len(df):4d} records, {len(df.columns)} variables"
        )
    summary_lines.extend([
        f"",
        f"Files generated: {len(files)}",
        f"Output directory: {output_dir}",
        f"",
        f"Next steps:",
        f"  1. Validate with Pinnacle 21 Community (SEND-IG 3.1.1 FDA config)",
        f"  2. Review CSV copies in {csv_dir}",
        f"  3. Submit .xpt files + define.xml to regulatory agency",
    ])
    summary = '\n'.join(summary_lines)

    (output_dir / 'README.txt').write_text(summary, encoding='utf-8')
    files.append(output_dir / 'README.txt')

    return {
        'files': files,
        'datasets': datasets,
        'summary': summary,
    }
