"""
cdisc_export.py — CDISC SEND export for regulatory submission.

Generates SAS Transport v5 (.xpt) files following the SEND
Implementation Guide (SENDIG v3.1.1) for nonclinical electrophysiology data.

Domains generated:
  - TS  (Trial Summary)        : study-level metadata
  - DM  (Demographics)         : subject-level (one row per microtissue)
  - EX  (Exposure)             : drug treatments and concentrations
  - EG  (ECG Test Results)     : FPD, FPDcF, BP, amplitude measurements
  - TX  (Trial Sets)           : treatment group definitions
  - DS  (Disposition)          : subject completion status

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
# Variable labels MUST match SENDIG 3.1 specification exactly.

# EG domain variable labels (SENDIG v3.1.1 section 6.3.17 exact)
_EG_VAR_LABELS = {
    'STUDYID':  'Study Identifier',
    'DOMAIN':   'Domain Abbreviation',
    'USUBJID':  'Unique Subject Identifier',
    'EGSEQ':    'Sequence Number',
    'EGREFID':  'ECG Reference Identifier',
    'EGTESTCD': 'ECG Test Short Name',
    'EGTEST':   'ECG Test Name',
    'EGORRES':  'Result or Finding in Original Units',
    'EGORRESU': 'Unit of the Original Result',
    'EGSTRESC': 'Character Result/Finding in Std Format',
    'EGSTRESN': 'Numeric Result/Finding in Standard Units',
    'EGSTRESU': 'Unit of the Standardized Result',
    'EGPOS':    'ECG Position of Subject',
    'EGLEAD':   'Lead Used for Measurement',
    'EGCSTATE': 'Consciousness State',
    'EGNOMDY':  'Nominal Study Day for Tabulations',
    'EGSTAT':   'Completion Status',
    'EGMETHOD': 'Method of ECG Test',
    'EGBLFL':   'Baseline Flag',
    'EGEVAL':   'Evaluator',
    'VISITDY':  'Planned Study Day of Collection',
    'EGDTC':    'Date/Time of ECG Collection',
    'EGDY':     'Study Day of ECG Collection',
    'EGTPTREF': 'Time Point Reference',
    'EPOCH':    'Epoch',
}

# DM domain variable labels (SENDIG v3.1.1 section 5.1.1 exact)
_DM_VAR_LABELS = {
    'STUDYID':  'Study Identifier',
    'DOMAIN':   'Domain Abbreviation',
    'USUBJID':  'Unique Subject Identifier',
    'SUBJID':   'Subject Identifier for the Study',
    'RFSTDTC':  'Subject Reference Start Date/Time',
    'RFENDTC':  'Subject Reference End Date/Time',
    'SEX':      'Sex',
    'SPECIES':  'Species',
    'STRAIN':   'Strain/Substrain',
    'ARMCD':    'Planned Arm Code',
    'ARM':      'Description of Planned Arm',
    'SETCD':    'Set Code',
    'AGE':      'Age',
    'AGETXT':   'Age Range',
    'AGEU':     'Age Unit',
}

# EX domain variable labels (SENDIG v3.1.1 section 6.1.1 exact)
_EX_VAR_LABELS = {
    'STUDYID':  'Study Identifier',
    'DOMAIN':   'Domain Abbreviation',
    'USUBJID':  'Unique Subject Identifier',
    'EXSEQ':    'Sequence Number',
    'EXTRT':    'Name of Actual Treatment',
    'EXDOSE':   'Dose per Administration',
    'EXDOSU':   'Dose Units',
    'EXDOSFRM': 'Dose Form',
    'EXDOSFRQ': 'Dosing Frequency per Interval',
    'EXROUTE':  'Route of Administration',
    'EXLOT':    'Lot Number',
    'EXTRTV':   'Treatment Vehicle',
    'EXSTDTC':  'Start Date/Time of Treatment',
    'EXSTDY':   'Study Day of Start of Treatment',
    'EXTPT':    'Planned Time Point Name',
    'EXTPTNUM': 'Planned Time Point Number',
    'EPOCH':    'Epoch',
}

# TS domain variable labels
_TS_VAR_LABELS = {
    'STUDYID':  'Study Identifier',
    'DOMAIN':   'Domain Abbreviation',
    'TSSEQ':    'Sequence Number',
    'TSGRPID':  'Group Identifier',
    'TSPARMCD': 'Trial Summary Parameter Short Name',
    'TSPARM':   'Trial Summary Parameter',
    'TSVAL':    'Parameter Value',
    'TSVALNF':  'Parameter Null Flavor',
}

# TX domain variable labels
_TX_VAR_LABELS = {
    'STUDYID':  'Study Identifier',
    'DOMAIN':   'Domain Abbreviation',
    'SETCD':    'Set Code',
    'SET':      'Set Description',
    'TXSEQ':    'Sequence Number',
    'TXPARMCD': 'Trial Set Parameter Short Name',
    'TXPARM':   'Trial Set Parameter',
    'TXVAL':    'Trial Set Parameter Value',
}

# DS domain variable labels (SENDIG v3.1.1 section 6.2.1)
_DS_VAR_LABELS = {
    'STUDYID':  'Study Identifier',
    'DOMAIN':   'Domain Abbreviation',
    'USUBJID':  'Unique Subject Identifier',
    'DSSEQ':    'Sequence Number',
    'DSTERM':   'Reported Term for the Disposition Event',
    'DSDECOD':  'Standardized Disposition Term',
    'DSUSCHFL': 'Unscheduled Flag',
    'DSSTDTC':  'Date/Time of Disposition',
    'DSSTDY':   'Study Day of Disposition',
    'DSNOMDY':  'Nominal Study Day for Tabulations',
}

# All labels indexed by domain
_DOMAIN_VAR_LABELS = {
    'TS': _TS_VAR_LABELS,
    'DM': _DM_VAR_LABELS,
    'EX': _EX_VAR_LABELS,
    'EG': _EG_VAR_LABELS,
    'TX': _TX_VAR_LABELS,
    'DS': _DS_VAR_LABELS,
}

# SEND-compliant EGTESTCD codes (max 8 chars, uppercase)
# These are custom non-standard test codes for hiPSC-CM electrophysiology.
# Format: EGTESTCD -> (category, EGTEST label, unit, is_score)
_TEST_CODES = {
    'FPD':      ('EGTEST', 'FPD',                                'ms',    False),
    'FPDCF':    ('EGTEST', 'FPDcF',                              'ms',    False),
    'BP':       ('EGTEST', 'Beat Period',                         'ms',    False),
    'SPIKEAM':  ('EGTEST', 'Spike Amplitude',                    'uV',    False),
    'RISETM':   ('EGTEST', 'Rise Time',                          'ms',    False),
    'MAXDVDT':  ('EGTEST', 'Maximum dV/dt',                      'mV/ms', False),
    'INTVL':    ('EGTEST', 'Heart Rate',                         'beats/min', False),
    'FPDCFPC':  ('EGTEST', 'FPDcF Pct Change',                  '%',     False),
    'BPPCT':    ('EGTEST', 'Beat Period Pct Change',             '%',     False),
    'AMPPCT':   ('EGTEST', 'Amplitude Pct Change',              '%',     False),
    'TDPSCR':   ('EGTEST', 'TdP Risk Score',                    'SCORE', True),
    'QCGRADE':  ('EGTEST', 'QC Grade',                          None,    True),
    'FPDCONF':  ('EGTEST', 'FPD Confidence',                    'SCORE', True),
    'RSKSCR':   ('EGTEST', 'Arrhythmia Risk Score',             'SCORE', True),
    'MORPHIN':  ('EGTEST', 'Morphology Instability',            'SCORE', True),
    'EADPCT':   ('EGTEST', 'EAD Incidence',                     '%',     False),
    'STVFPDC':  ('EGTEST', 'STV of FPDcF',                      'ms',    False),
    'SPECCHG':  ('EGTEST', 'Spectral Change',                   'SCORE', True),
    'PROAIDX':  ('EGTEST', 'Proarrhythmic Index',               'SCORE', True),
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


def _subject_drug_map(results: list, study_id: str) -> dict:
    """Map each USUBJID to its primary drug (SEND name).

    Used to align DM.SETCD with TX.SETCD (both keyed on drug name).
    """
    from .normalization import _is_baseline, _is_control
    subj_drug = {}
    for r in results:
        usubjid = _make_usubjid(study_id, r)
        if usubjid in subj_drug:
            continue
        if _is_baseline(r) or _is_control(r):
            subj_drug[usubjid] = 'CONTROL'
        else:
            fi = r.get('file_info', {})
            drug_raw = str(fi.get('drug', ''))
            if drug_raw:
                subj_drug[usubjid] = _canonical_drug_send(drug_raw)
            else:
                subj_drug[usubjid] = 'CONTROL'
    return subj_drug


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
    # CDISC CT-compliant unit mapping
    _UNIT_MAP = {
        'nm': 'nmol/L', 'um': 'umol/L', 'mm': 'mmol/L', 'm': 'mol/L',
        'nmol/l': 'nmol/L', 'umol/l': 'umol/L', 'mmol/l': 'mmol/L',
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


def _get_label(domain: str, varname: str) -> str:
    """Get the correct SENDIG label for a variable in a domain."""
    labels = _DOMAIN_VAR_LABELS.get(domain, {})
    return labels.get(varname, varname)


# ═══════════════════════════════════════════════════════════════════════
#  TS — Trial Summary (CDISC CT exact TSPARM labels)
# ═══════════════════════════════════════════════════════════════════════

def _build_ts(study_id: str, study_title: str, n_results: int,
              sponsor: str = '', start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
    """Build Trial Summary (TS) domain with required SENDIG 3.1 parameters.

    TSPARM labels must match CDISC Controlled Terminology EXACTLY.
    """
    now = start_date or datetime.now().strftime('%Y-%m-%d')
    # End dates include T23:59 so they are always >= any EX timestamps (SE1149)
    end = end_date or now
    end_with_time = f"{end}T23:59" if 'T' not in end else end

    # (TSPARMCD, TSPARM [exact CDISC CT label], TSVAL, TSVALNF)
    # Labels sourced from SENDIG v3.1.1 section 7.6.2 Trial Summary Codes
    # ALL SE2xxx-required params must be present even if value is NA.
    params = [
        # Study identification — Should Include = Yes
        ('STITLE',  'Study Title',                               study_title,              ''),
        ('SSTYP',   'Study Type',                                'IN VITRO',               ''),
        ('SDESIGN', 'Study Design',                              'PARALLEL',               ''),
        ('SSPONSOR','Sponsoring Organization',                   sponsor or 'NOT PROVIDED',''),
        ('STCAT',   'Study Category',                            'SAFETY PHARMACOLOGY',    ''),
        ('STDIR',   'Study Director',                            sponsor or 'NOT PROVIDED',''),

        # Species / Strain — Should Include = Yes
        ('SPECIES', 'Species',                                   'HUMAN',                  ''),
        ('STRAIN',  'Strain/Substrain',                          'HIPSC-CM',               ''),
        ('SPLRNAM', 'Test Subject Supplier',                     '',                       'NA'),

        # Route and treatment — Should Include = Yes
        ('ROUTE',   'Route of Administration',                   'TOPICAL',                ''),
        ('TRT',     'Investigational Therapy or Treatment',      'MULTIPLE DRUGS',         ''),
        ('TRTV',    'Treatment Vehicle',                         'CULTURE MEDIUM',         ''),
        ('TRTCAS',  'Primary Treatment CAS Registry Number',     '',                       'NA'),
        ('TRTUNII', 'Primary Treatment Unique Ingredient ID',    '',                       'NA'),
        ('PCLASS',  'Pharmacological Class of Invest. Therapy',  'ION CHANNEL MODULATORS', ''),

        # Subjects / design — Should Include = Yes
        ('SEXPOP',  'Sex of Participants',                       '',                       'NA'),
        ('SPLANSUB','Planned Number of Subjects',                '',                       'NA'),

        # Age (NA for cell lines — SE2201 requires AGE or AGETXT)
        ('AGETXT',  'Age Text',                                  '',                       'NA'),
        ('AGEU',    'Age Unit',                                  '',                       'NA'),

        # Timing — Should Include = Yes
        ('STSTDTC', 'Study Start Date',                          now,                      ''),
        ('STENDTC', 'Study End Date',                            end_with_time,            ''),
        ('EXPSTDTC','Experimental Start Date',                   now,                      ''),
        ('EXPENDTC','Experimental End Date',                     end_with_time,            ''),
        ('DOSDUR',  'Dosing Duration',                           'P1D',                    ''),
        ('DOSSTDTC','Start Date/Time of Dose Interval',          now,                      ''),
        ('DOSENDTC','End Date/Time of Dose Interval',            end_with_time,            ''),
        ('PDOSFRQ', 'Planned Dosing Frequency per Interval',     'ONCE',                   ''),
        ('TRMSAC',  'Time to Terminal Sacrifice',                'P1D',                    ''),

        # Regulatory / compliance — Should Include = Yes
        ('GLPFL',   'GLP Flag',                                  'N',                      ''),
        ('GLPTYP',  'Good Laboratory Practice Type',             '',                       'NA'),
        ('SNDIGVER','SEND Implementation Guide Version',         '3.1.1',                  ''),
        ('SNDCTVER','SEND Controlled Terminology Version',       'SEND Terminology 2025-09-26', ''),

        # Test facility — Should Include = Yes
        ('TFCNTRY', 'Test Facility Country',                     '',                       'NA'),
        ('TSTFLOC', 'Test Facility Location',                    '',                       'NA'),
        ('TSTFNAM', 'Test Facility Name',                        '',                       'NA'),
        ('SPREFID', "Sponsor's Reference ID",                    '',                       'NA'),

        # Material / production (SE2267 requires STRPSTAT)
        ('STRPSTAT','Test/Reference Item Production Status',     '',                       'NA'),
    ]

    rows = []
    for i, (parmcd, parm, val, valnf) in enumerate(params, 1):
        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'TS',
            'TSSEQ':    i,
            'TSGRPID':  '',
            'TSPARMCD': parmcd,
            'TSPARM':   parm,
            'TSVAL':    val,
            'TSVALNF':  valnf,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  DM — Demographics (one row per microtissue)
#  Variable order per SENDIG 3.1: STUDYID, DOMAIN, USUBJID, SUBJID,
#    RFSTDTC, RFENDTC, SEX, SPECIES, STRAIN, ARMCD, ARM, SETCD
# ═══════════════════════════════════════════════════════════════════════

def _build_dm(results: list, study_id: str) -> pd.DataFrame:
    """Build Demographics (DM) domain — one row per unique microtissue.

    SETCD is aligned with TX domain via _subject_drug_map() so that
    DM.SETCD matches TX.SETCD for Pinnacle 21 cross-domain checks.
    """
    from .normalization import _is_baseline, _is_control

    subj_drug = _subject_drug_map(results, study_id)
    seen = {}
    rows = []
    now = datetime.now().strftime('%Y-%m-%d')

    for r in results:
        usubjid = _make_usubjid(study_id, r)
        if usubjid in seen:
            continue
        seen[usubjid] = True

        drug = subj_drug.get(usubjid, 'CONTROL')
        is_ctrl = _is_baseline(r) or _is_control(r)
        armcd = 'CTRL' if is_ctrl else 'TRT'
        arm = 'Control' if is_ctrl else 'Treatment'
        setcd = 'CONTROL' if is_ctrl else drug[:8]

        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'DM',
            'USUBJID':  usubjid,
            'SUBJID':   usubjid.split('-', 1)[-1] if '-' in usubjid else usubjid,
            'RFSTDTC':  now,
            'RFENDTC':  now,
            'SEX':      'U',
            'SPECIES':  'HUMAN',
            'STRAIN':   'HIPSC-CM',
            'ARMCD':    armcd,
            'ARM':      arm,
            'SETCD':    setcd,
            'AGE':      None,
            'AGETXT':   '',
            'AGEU':     '',
        })

    df = pd.DataFrame(rows)
    # AGE must be numeric type (Num in SAS) even if all null — SE0055
    if not df.empty:
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
    return df


# ═══════════════════════════════════════════════════════════════════════
#  EX — Exposure (drug treatments)
#  Variable order per SENDIG 3.1: STUDYID, DOMAIN, USUBJID, EXSEQ,
#    EXTRT, EXDOSE, EXDOSU, EXDOSFRM, EXDOSFRQ, EXROUTE, EXLOT,
#    EXTRTV, EPOCH, EXSTDTC, EXSTDY
# ═══════════════════════════════════════════════════════════════════════

def _build_ex(results: list, study_id: str) -> pd.DataFrame:
    """Build Exposure (EX) domain — one row per drug exposure event."""
    from .normalization import _is_baseline, _is_control

    rows = []
    seq_counter = {}
    seen_keys = set()
    now_date = datetime.now().strftime('%Y-%m-%d')

    # Track dose sequence per (subject, drug) to generate unique timestamps
    subj_drug_dose_seq = {}

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
        dose_num, dose_unit = _parse_concentration(conc_raw)

        # De-duplicate: same subject + drug + concentration
        dedup_key = (usubjid, drug, dose_num)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        # Generate unique timestamp per (subject, drug) to avoid SD1352
        sd_key = (usubjid, drug)
        subj_drug_dose_seq.setdefault(sd_key, 0)
        subj_drug_dose_seq[sd_key] += 1
        dose_idx = subj_drug_dose_seq[sd_key]
        # Offset each dose by 1 hour: T08:00, T09:00, T10:00, ...
        hour = 7 + dose_idx
        exstdtc = f"{now_date}T{hour:02d}:00"

        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'EX',
            'USUBJID':  usubjid,
            'EXSEQ':    _make_seq(seq_counter, usubjid),
            'EXTRT':    drug,
            'EXDOSE':   dose_num if dose_num is not None else 0.0,
            'EXDOSU':   dose_unit or 'nmol/L',
            'EXDOSFRM': 'SOLUTION',
            'EXDOSFRQ': 'ONCE',
            'EXROUTE':  'TOPICAL',
            'EXLOT':    '',              # SE0057: expected variable
            'EXTRTV':   'CULTURE MEDIUM',# SE0057: expected variable
            'EXSTDTC':  exstdtc,
            'EXSTDY':   1,
            'EXTPT':    f'DOSE {dose_idx}',
            'EXTPTNUM': dose_idx,
            'EPOCH':    'TREATMENT',
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  EG — ECG Test Results (the main data domain)
#  Variable order per SENDIG 3.1: STUDYID, DOMAIN, USUBJID, EGSEQ,
#    EGTESTCD, EGTEST, EGORRES, EGORRESU, EGSTRESC, EGSTRESN, EGSTRESU,
#    EGSTAT, EGMETHOD, EGCSTATE, EGLEAD, EGPOS, EGBLFL, VISITDY,
#    EGNOMDY, EGDY, EGDTC, EGTPTREF, EPOCH, EGEVAL, EGREFID
# ═══════════════════════════════════════════════════════════════════════

def _build_eg(results: list, study_id: str) -> pd.DataFrame:
    """Build ECG Test Results (EG) domain."""
    from .normalization import _is_baseline, _is_control

    rows = []
    seq_counter = {}
    seen_keys = set()
    now = datetime.now().strftime('%Y-%m-%d')

    # Track concentration index per subject for VISITDY
    subj_conc_counter = {}

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

        # Assign unique VISITDY per subject-concentration to avoid duplicates
        conc_key = (usubjid, tptref)
        if conc_key not in subj_conc_counter:
            subj_conc_counter.setdefault(usubjid, 0)
            subj_conc_counter[usubjid] += 1
            subj_conc_counter[conc_key] = subj_conc_counter[usubjid]
        visitdy = subj_conc_counter[conc_key]

        def _add_row(testcd, value, unit_override=None):
            """Helper to add one EG row."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return
            code_info = _TEST_CODES.get(testcd, ('EGTEST', testcd, unit_override, False))
            unit = unit_override or code_info[2] or ''
            is_score = code_info[3] if len(code_info) > 3 else False

            # For QCGRADE (character-only), handle specially
            if testcd == 'QCGRADE':
                stresc = str(value)
                stresn = None
                unit = ''
            elif isinstance(value, float):
                stresc = f"{value:.4f}"
                stresn = round(value, 4)
            elif isinstance(value, int):
                stresc = str(value)
                stresn = float(value)
            else:
                stresc = str(value)
                stresn = None

            # De-duplicate: unique per subject + test + visit
            dk = (usubjid, testcd, visitdy)
            if dk in seen_keys:
                return
            seen_keys.add(dk)

            # Column order matches SENDIG v3.1.1 section 6.3.17
            # EGDY = 1 for all records (same-day in vitro study, EGDTC == RFSTDTC)
            # SE0057: EGPOS, EGLEAD, EGCSTATE, EGNOMDY must exist (even if empty)
            rows.append({
                'STUDYID':  study_id,
                'DOMAIN':   'EG',
                'USUBJID':  usubjid,
                'EGSEQ':    _make_seq(seq_counter, usubjid),
                'EGREFID':  fname,
                'EGTESTCD': testcd,
                'EGTEST':   code_info[1],
                'EGPOS':    '',              # SE0057: expected variable
                'EGORRES':  stresc,
                'EGORRESU': unit,
                'EGSTRESC': stresc,
                'EGSTRESN': stresn,
                'EGSTRESU': unit,
                'EGSTAT':   '',
                'EGMETHOD': 'DERIVED',
                'EGLEAD':   '',              # SE0057: expected variable
                'EGCSTATE': '',              # SE0057: expected variable
                'EGBLFL':   'Y' if is_bl else '',
                'EGEVAL':   'ALGORITHM',
                'VISITDY':  visitdy,
                'EGDTC':    now,
                'EGDY':     1,
                'EGNOMDY':  visitdy,         # SE0057: expected variable
                'EGTPTREF': tptref,
                'EPOCH':    epoch,
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
#  TX — Trial Sets (treatment group definitions)
# ═══════════════════════════════════════════════════════════════════════

def _build_tx(results: list, study_id: str, dm_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build Trial Sets (TX) domain — one row per treatment parameter per set.

    Each drug set includes required parameters:
      TRT, TCNTRL, SPGRPCD, GRPLBL, ARMCD, TRTDOS, TRTDOSU, PLANMSUB, PLANFSUB
    Plus a CONTROL set for baseline/vehicle subjects.

    TX SETCDs are restricted to only those present in DM to avoid SE2347.
    """
    # Get SETCDs actually used in DM (for cross-domain consistency)
    if dm_df is not None and not dm_df.empty:
        dm_setcds = set(dm_df['SETCD'].unique())
    else:
        dm_setcds = None

    has_control = dm_setcds is not None and 'CONTROL' in dm_setcds

    # Collect all drugs, but only include those that have subjects in DM
    from .normalization import _is_baseline, _is_control
    drugs_seen = set()
    for r in results:
        if _is_baseline(r) or _is_control(r):
            continue
        fi = r.get('file_info', {})
        drug_raw = str(fi.get('drug', ''))
        if drug_raw:
            drug = _canonical_drug_send(drug_raw)
            if dm_setcds is None or drug[:8] in dm_setcds:
                drugs_seen.add(drug)

    rows = []
    seq = 0

    def _add_tx_param(setcd, set_desc, parmcd, parm, val):
        nonlocal seq
        seq += 1
        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'TX',
            'SETCD':    setcd,
            'SET':      set_desc,
            'TXSEQ':    seq,
            'TXPARMCD': parmcd,
            'TXPARM':   parm,
            'TXVAL':    val,
        })

    # CONTROL set
    if has_control:
        sc, sd = 'CONTROL', 'Control Group'
        _add_tx_param(sc, sd, 'TRT',      'Investigational Therapy or Treatment', 'VEHICLE')
        _add_tx_param(sc, sd, 'TCNTRL',   'Control Type',                         'VEHICLE')
        _add_tx_param(sc, sd, 'SPGRPCD',  'Sponsor-Defined Group Code',           'CTRL')
        _add_tx_param(sc, sd, 'GRPLBL',   'Group Label',                          'Vehicle Control')
        _add_tx_param(sc, sd, 'ARMCD',    'Arm Code',                     'CTRL')
        _add_tx_param(sc, sd, 'TRTDOS',   'Dose Level',              '0')
        _add_tx_param(sc, sd, 'TRTDOSU',  'Dose Units',                           'nmol/L')
        _add_tx_param(sc, sd, 'PLANMSUB', 'Planned Number of Male Subjects',      '0')
        _add_tx_param(sc, sd, 'PLANFSUB', 'Planned Number of Female Subjects',    '0')

    # Drug treatment sets
    for drug in sorted(drugs_seen):
        sc = drug[:8]
        sd = f'{drug} Treatment Group'
        _add_tx_param(sc, sd, 'TRT',      'Investigational Therapy or Treatment', drug)
        # No TCNTRL for treatment groups (SD0002: TXVAL must not be null)
        _add_tx_param(sc, sd, 'SPGRPCD',  'Sponsor-Defined Group Code',           sc)
        _add_tx_param(sc, sd, 'GRPLBL',   'Group Label',                          f'{drug} Treatment')
        _add_tx_param(sc, sd, 'ARMCD',    'Arm Code',                     'TRT')
        _add_tx_param(sc, sd, 'TRTDOS',   'Dose Level',              'SEE PROTOCOL')
        _add_tx_param(sc, sd, 'TRTDOSU',  'Dose Units',                           'SEE PROTOCOL')
        _add_tx_param(sc, sd, 'PLANMSUB', 'Planned Number of Male Subjects',      '0')
        _add_tx_param(sc, sd, 'PLANFSUB', 'Planned Number of Female Subjects',    '0')

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  DS — Disposition
# ═══════════════════════════════════════════════════════════════════════

def _build_ds(results: list, study_id: str) -> pd.DataFrame:
    """Build Disposition (DS) domain — one row per unique subject.

    DSTERM (verbatim term) and DSDECOD (standardized term) are both required.
    DSDECOD must be from CT codelist C66727 — 'COMPLETED'.
    """
    seen = {}
    rows = []
    seq_counter = {}
    now = datetime.now().strftime('%Y-%m-%d')

    for r in results:
        usubjid = _make_usubjid(study_id, r)
        if usubjid in seen:
            continue
        seen[usubjid] = True

        rows.append({
            'STUDYID':  study_id,
            'DOMAIN':   'DS',
            'USUBJID':  usubjid,
            'DSSEQ':    _make_seq(seq_counter, 'DS'),
            'DSTERM':   'TERMINAL SACRIFICE',
            'DSDECOD':  'TERMINAL SACRIFICE',
            'DSUSCHFL': '',              # SE0057: expected variable
            'DSSTDTC':  now,
            'DSSTDY':   1,
            'DSNOMDY':  1,               # SE0057: expected variable
        })

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
        f'                     def:StandardVersion="3.1.1">',
    ]

    domain_info = {
        'TS':   ('Trial Summary',   'Trial summary parameters and study metadata'),
        'DM':   ('Demographics',     'One record per microtissue'),
        'EX':   ('Exposure',         'Drug exposure events with concentrations'),
        'EG':   ('ECG Test Results', 'Electrophysiology measurements'),
        'TX':   ('Trial Sets',       'Treatment group definitions'),
        'DS':   ('Disposition',      'Subject disposition'),
    }

    for domain, df in datasets.items():
        info = domain_info.get(domain, (domain, ''))
        n_rows = len(df)
        n_cols = len(df.columns)
        struct = 'One record per parameter' if domain in ('TS', 'TX') else 'One record per test per subject'
        lines.append(f'      <def:ItemGroupDef OID="IG.{domain}"')
        lines.append(f'                        Name="{domain}"')
        lines.append(f'                        SASDatasetName="{domain}"')
        lines.append(f'                        def:Label="{info[0]}"')
        lines.append(f'                        def:Structure="{struct}"')
        lines.append(f'                        Purpose="Tabulation"')
        lines.append(f'                        def:StandardOID="STD.SENDIG.3.1.1">')
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
            label = _get_label(domain, col)
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
    """Write a DataFrame to SAS Transport v5 (.xpt) format."""
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

    # Build column labels for the domain
    domain_labels = _DOMAIN_VAR_LABELS.get(dataset_name, {})
    col_labels = []
    for orig_col, sas_col in col_map.items():
        col_labels.append(domain_labels.get(orig_col, orig_col))

    # Try pyreadstat first (best: supports labels)
    try:
        import pyreadstat
        try:
            pyreadstat.write_xport(
                df_sas, filepath,
                table_name=dataset_name[:8],
                column_labels=col_labels,
            )
        except TypeError:
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

    # Final fallback: CSV
    warnings.warn(
        f"xport and pyreadstat not available. Writing {filepath.name} as CSV. "
        "Install pyreadstat: pip install pyreadstat",
        UserWarning
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# WARNING: CSV fallback. pip install pyreadstat\n")
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
    tx_df = _build_tx(results, study_id, dm_df=dm_df)
    ds_df = _build_ds(results, study_id)

    # ── Ensure every DM subject has at least one EGBLFL='Y' row (SE2319) ──
    if not eg_df.empty and not dm_df.empty:
        dm_subjects = set(dm_df['USUBJID'].unique())
        eg_bl_subjects = set(
            eg_df.loc[eg_df['EGBLFL'] == 'Y', 'USUBJID'].unique()
        )
        missing_bl = dm_subjects - eg_bl_subjects
        for subj in missing_bl:
            subj_rows = eg_df[eg_df['USUBJID'] == subj]
            if not subj_rows.empty:
                first_idx = subj_rows.index[0]
                eg_df.loc[first_idx, 'EGBLFL'] = 'Y'

    # ── Enforce SENDIG v3.1.1 column order (SD1079) ──
    # Variable order from SENDIG v3.1.1 domain model tables (authoritative).
    # Only variables we use are listed; order matches spec exactly.
    # Non-spec variables (EPOCH) are appended at end.
    _SENDIG_COL_ORDER = {
        # TS: Section 7.6.1 (p.210)
        'TS': ['STUDYID','DOMAIN','TSSEQ','TSGRPID','TSPARMCD','TSPARM',
               'TSVAL','TSVALNF'],
        # DM: Section 5.1.1 (p.45-46)
        'DM': ['STUDYID','DOMAIN','USUBJID','SUBJID','RFSTDTC','RFENDTC',
               'SITEID','BRTHDTC','AGE','AGETXT','AGEU','SEX','SPECIES',
               'STRAIN','SBSTRAIN','ARMCD','ARM','SETCD'],
        # EX: Section 6.1.1 (p.57-59) — EPOCH not in spec, appended
        'EX': ['STUDYID','DOMAIN','USUBJID','POOLID','FOCID','EXSEQ',
               'EXTRT','EXDOSE','EXDOSTXT','EXDOSU','EXDOSFRM','EXDOSFRQ',
               'EXROUTE','EXLOT','EXLOC','EXMETHOD','EXTRTV',
               'EXVAMT','EXVAMTU','EXADJ',
               'EXSTDTC','EXENDTC','EXSTDY','EXENDY','EXDUR',
               'EXTPT','EXTPTNUM','EXELTM','EXTPTREF','EXRFTDTC',
               'EPOCH'],
        # EG: Section 6.3.17 (p.160-162) — EPOCH not in spec, appended
        'EG': ['STUDYID','DOMAIN','USUBJID','EGSEQ',
               'EGGRPID','EGREFID','EGSPID',
               'EGTESTCD','EGTEST','EGCAT','EGPOS',
               'EGORRES','EGORRESU','EGSTRESC','EGSTRESN','EGSTRESU',
               'EGSTAT','EGREASND','EGXFN','EGNAM','EGMETHOD',
               'EGLEAD','EGCSTATE','EGBLFL','EGDRVFL','EGEVAL',
               'EGEXCLFL','EGREASEX','EGUSCHFL',
               'VISITDY','EGDTC','EGENDTC','EGDY','EGENDY',
               'EGNOMDY','EGNOMLBL',
               'EGTPT','EGTPTNUM','EGELTM','EGTPTREF','EGRFTDTC',
               'EGEVLINT','EGSTINT','EGENINT',
               'EPOCH'],
        # TX: Section 7.4.1 (p.185)
        'TX': ['STUDYID','DOMAIN','SETCD','SET','TXSEQ','TXPARMCD',
               'TXPARM','TXVAL'],
        # DS: Section 6.2.1 (p.64-65) — no DSCAT or EPOCH in spec
        'DS': ['STUDYID','DOMAIN','USUBJID','DSSEQ','DSTERM','DSDECOD',
               'DSUSCHFL','VISITDY','DSSTDTC','DSSTDY','DSNOMDY','DSNOMLBL'],
    }

    def _order_columns(df, domain):
        """Reorder columns to match SENDIG 3.1 spec. Drops cols not in spec."""
        order = _SENDIG_COL_ORDER.get(domain)
        if order is None:
            return df
        # Keep only columns that exist in df, in spec order
        cols = [c for c in order if c in df.columns]
        # Append any extra columns not in spec (shouldn't happen)
        extras = [c for c in df.columns if c not in cols]
        return df[cols + extras]

    dm_df = _order_columns(dm_df, 'DM')
    ex_df = _order_columns(ex_df, 'EX')
    eg_df = _order_columns(eg_df, 'EG')
    ts_df = _order_columns(ts_df, 'TS')
    tx_df = _order_columns(tx_df, 'TX')
    ds_df = _order_columns(ds_df, 'DS')

    datasets = {
        'TS': ts_df,
        'DM': dm_df,
        'TX': tx_df,
        'DS': ds_df,
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
        'TX': 'Trial Sets',
        'DS': 'Disposition',
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

    # ── Copy XSL stylesheet ──
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
    ])
    summary = '\n'.join(summary_lines)

    (output_dir / 'README.txt').write_text(summary, encoding='utf-8')
    files.append(output_dir / 'README.txt')

    return {
        'files': files,
        'datasets': datasets,
        'summary': summary,
    }
