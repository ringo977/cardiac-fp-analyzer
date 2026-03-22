"""
cdisc_export.py — CDISC SEND export for regulatory submission.

Generates SAS Transport v5 (.xpt) files following the SEND
Implementation Guide (SENDIG) for nonclinical electrophysiology data.

Domains generated:
  - TS  (Trial Summary)        : study-level metadata
  - DM  (Demographics)         : subject-level (one row per microtissue)
  - EX  (Exposure)             : drug treatments and concentrations
  - EG  (ECG Test Results)     : FPD, FPDcF, BP, amplitude measurements
  - MI  (Custom — Arrhythmia)  : arrhythmia classification and risk scores

Also generates:
  - define.xml        : Dataset-level metadata (Define-XML 2.1)
  - define2-1-0.xsl   : Pinnacle 21 official stylesheet for define.xml rendering

Reference standards:
  - SENDIG v3.1 (CDISC)
  - NCI Controlled Terminology
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
import io, warnings, shutil

warnings.filterwarnings('ignore')


# ── SEND Controlled Terminology (NCI CT subset) ─────────────────────

_UNITS = {
    'ms': 'ms',
    'mV': 'mV',
    'uV': 'uV',
    'Hz': 'Hz',
    'bpm': 'BEATS/MIN',
    's': 's',
    'pct': '%',
}

_TEST_CODES = {
    'FPD':      ('EGTEST', 'Field Potential Duration',          'ms'),
    'FPDCF':    ('EGTEST', 'Field Potential Duration Corrected', 'ms'),
    'BP':       ('EGTEST', 'Beat Period',                        'ms'),
    'SPIKEAMP': ('EGTEST', 'Spike Amplitude',                    'uV'),
    'RISETIME': ('EGTEST', 'Rise Time 10-90%',                   'ms'),
    'MAXDVDT':  ('EGTEST', 'Maximum dV/dt',                      'mV/ms'),
    'BPM':      ('EGTEST', 'Heart Rate',                         'BEATS/MIN'),
    'FPDCFPCT': ('EGTEST', 'FPDcF Change from Baseline',         '%'),
    'BPPCT':    ('EGTEST', 'Beat Period Change from Baseline',    '%'),
    'AMPPCT':   ('EGTEST', 'Amplitude Change from Baseline',      '%'),
    'TDPSCORE': ('EGTEST', 'TdP Risk Score',                     None),
    'QCGRADE':  ('EGTEST', 'Quality Control Grade',              None),
    'FPDCONF':  ('EGTEST', 'FPD Confidence Score',               None),
    'RSKSCR':   ('EGTEST', 'Arrhythmia Risk Score',              None),
    'MORPHINST':('EGTEST', 'Morphology Instability',             None),
    'EADPCT':   ('EGTEST', 'EAD Incidence',                      '%'),
    'STVFPDC':  ('EGTEST', 'STV of FPDcF',                       'ms'),
    'SPECCHG':  ('EGTEST', 'Spectral Change Score',              None),
    'PROARIDX': ('EGTEST', 'Proarrhythmic Index',                None),
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


def _make_seq(counter: dict, domain: str) -> int:
    """Auto-incrementing sequence number per domain."""
    counter.setdefault(domain, 0)
    counter[domain] += 1
    return counter[domain]


# ═══════════════════════════════════════════════════════════════════════
#  TS — Trial Summary
# ═══════════════════════════════════════════════════════════════════════

def _build_ts(study_id: str, study_title: str, n_results: int) -> pd.DataFrame:
    """Build Trial Summary (TS) domain."""
    now = datetime.now().strftime('%Y-%m-%d')
    rows = [
        ('STUDYID',  study_id,                       'Study Identifier'),
        ('SSTDTC',   now,                            'Study Start Date'),
        ('SSESSION', 'IN VITRO',                     'Study Session'),
        ('SDESIGN',  'NON-RANDOMIZED',               'Study Design'),
        ('SSPONSOR', '',                              'Sponsor'),
        ('STITLE',   study_title,                     'Study Title'),
        ('SSPECIES', 'HUMAN IPSC-DERIVED',            'Species'),
        ('SSTRAIN',  'HIPSC-CM MICROTISSUE',          'Strain'),
        ('ROUTE',    'IN VITRO EXPOSURE',             'Route of Administration'),
        ('SENDTC',   now,                            'Study End Date'),
        ('NSUBJ',    str(n_results),                  'Number of Subjects'),
        ('REGID',    'CIPA',                          'Regulatory Identifier'),
    ]
    df = pd.DataFrame(rows, columns=['TSPARMCD', 'TSVAL', 'TSPARM'])
    df.insert(0, 'STUDYID', study_id)
    df.insert(1, 'DOMAIN', 'TS')
    df['TSSEQ'] = range(1, len(df) + 1)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  DM — Demographics (one row per microtissue)
# ═══════════════════════════════════════════════════════════════════════

def _build_dm(results: list, study_id: str) -> pd.DataFrame:
    """Build Demographics (DM) domain — one row per unique microtissue."""
    from .normalization import _is_baseline, _get_group_key

    seen = {}
    rows = []
    for r in results:
        usubjid = _make_usubjid(study_id, r)
        if usubjid in seen:
            continue
        seen[usubjid] = True

        fi = r.get('file_info', {})
        exp = fi.get('experiment', '')
        qc = r.get('qc_report')

        rows.append({
            'STUDYID': study_id,
            'DOMAIN': 'DM',
            'USUBJID': usubjid,
            'SUBJID': usubjid.split('-', 1)[-1] if '-' in usubjid else usubjid,
            'SPECIES': 'HUMAN IPSC-DERIVED',
            'STRAIN': 'HIPSC-CM',
            'SEX': 'U',                    # Unknown (cell line)
            'ARMCD': 'TREATMENT',
            'ARM': 'DRUG TREATMENT',
            'SETCD': exp.replace(' ', '') if exp else 'SET1',
            'DMDTC': datetime.now().strftime('%Y-%m-%d'),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  EX — Exposure (drug treatments)
# ═══════════════════════════════════════════════════════════════════════

def _build_ex(results: list, study_id: str) -> pd.DataFrame:
    """Build Exposure (EX) domain — one row per drug exposure event."""
    from .normalization import _is_baseline

    rows = []
    seq_counter = {}
    for r in results:
        if _is_baseline(r):
            continue
        fi = r.get('file_info', {})
        drug_raw = str(fi.get('drug', ''))
        if not drug_raw:
            continue

        drug = _canonical_drug_send(drug_raw)
        conc = fi.get('concentration', '')
        usubjid = _make_usubjid(study_id, r)

        rows.append({
            'STUDYID': study_id,
            'DOMAIN': 'EX',
            'USUBJID': usubjid,
            'EXSEQ': _make_seq(seq_counter, usubjid),
            'EXTRT': drug,
            'EXDOSE': conc if conc else '',
            'EXDOSU': '',  # Units would need parsing from concentration
            'EXROUTE': 'IN VITRO',
            'EXDOSFRM': 'SOLUTION',
            'EPOCH': 'TREATMENT',
            'EXSTDTC': datetime.now().strftime('%Y-%m-%d'),
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
    from .normalization import _is_baseline

    rows = []
    seq_counter = {}

    for r in results:
        fi = r.get('file_info', {})
        s = r.get('summary', {})
        norm = r.get('normalization', {})
        ar = r.get('arrhythmia_report')
        qc = r.get('qc_report')
        inc = r.get('inclusion', {})

        usubjid = _make_usubjid(study_id, r)
        is_bl = _is_baseline(r)
        drug_raw = str(fi.get('drug', ''))
        drug = _canonical_drug_send(drug_raw) if drug_raw else 'BASELINE'
        conc = fi.get('concentration', '')
        fname = r.get('metadata', {}).get('filename', '')

        # Time point reference
        tptref = 'BASELINE' if is_bl else f"{drug} {conc}".strip()
        epoch = 'BASELINE' if is_bl else 'TREATMENT'

        # Inclusion status
        inc_passed = 'Y' if inc.get('passed', True) else 'N'

        def _add_row(testcd, value, unit=None, stat='MEAN'):
            """Helper to add one EG row."""
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return
            code_info = _TEST_CODES.get(testcd, ('EGTEST', testcd, unit))
            rows.append({
                'STUDYID': study_id,
                'DOMAIN': 'EG',
                'USUBJID': usubjid,
                'EGSEQ': _make_seq(seq_counter, usubjid),
                'EGTESTCD': testcd,
                'EGTEST': code_info[1],
                'EGORRES': f"{value:.4f}" if isinstance(value, float) else str(value),
                'EGORRESU': code_info[2] or '',
                'EGSTRESC': f"{value:.4f}" if isinstance(value, float) else str(value),
                'EGSTRESN': float(value) if isinstance(value, (int, float)) else None,
                'EGSTRESU': code_info[2] or '',
                'EGSTAT': '',
                'EGMETHOD': s.get('fpd_method', 'TANGENT').upper(),
                'EGBLFL': 'Y' if is_bl else '',
                'EGTPTREF': tptref,
                'EPOCH': epoch,
                'EGDTC': datetime.now().strftime('%Y-%m-%d'),
                # Supplemental qualifiers (non-standard but useful)
                'EGEVAL': 'ALGORITHM',
                'EGINCLFL': inc_passed,
                'EGREFID': fname,
            })

        # ── Core electrophysiology parameters ──
        _add_row('FPD',      s.get('fpd_ms_mean'))
        _add_row('FPDCF',    s.get('fpdc_ms_mean'))
        _add_row('BP',       s.get('beat_period_ms_mean'))
        _add_row('SPIKEAMP', s.get('spike_amplitude_mV_mean'))
        _add_row('RISETIME', s.get('rise_time_ms_mean'))
        _add_row('MAXDVDT',  s.get('max_dvdt_mean'))
        _add_row('FPDCONF',  s.get('fpd_confidence'))

        # ── Heart rate ──
        bpm = s.get('bpm_mean')
        if bpm:
            _add_row('BPM', bpm)

        # ── Quality control ──
        if qc:
            _add_row('QCGRADE', qc.grade)

        # ── Normalized parameters (drug recordings only) ──
        if norm.get('has_baseline'):
            _add_row('FPDCFPCT', norm.get('pct_fpdc_change'))
            _add_row('BPPCT',    norm.get('pct_bp_change'))
            _add_row('AMPPCT',   norm.get('pct_amp_change'))
            _add_row('TDPSCORE', norm.get('tdp_score'))
            _add_row('SPECCHG',  norm.get('spectral_change_score'))

        # ── Arrhythmia metrics ──
        if ar:
            _add_row('RSKSCR', ar.risk_score)
            rd = ar.residual_details or {}
            _add_row('MORPHINST', rd.get('morphology_instability'))
            _add_row('EADPCT',    rd.get('ead_incidence_pct'))
            _add_row('STVFPDC',   rd.get('poincare_stv_fpdc_ms'))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Drug-level risk metrics (custom supplemental domain)
# ═══════════════════════════════════════════════════════════════════════

def _build_risk(results: list, study_id: str) -> pd.DataFrame:
    """Build drug-level risk classification dataset."""
    from .risk_map import aggregate_drug_metrics, compute_proarrhythmic_index

    metrics = aggregate_drug_metrics(results)
    rows = []
    for drug, m in sorted(metrics.items()):
        x = m.max_pct_fpdc_change if not np.isnan(m.max_pct_fpdc_change) else 0.0
        y = compute_proarrhythmic_index(m)

        rows.append({
            'STUDYID': study_id,
            'DOMAIN': 'RISK',
            'EXTRT': _DRUG_SEND_NAMES.get(drug, drug.upper()),
            'RISKSEQ': len(rows) + 1,
            'MAXFPDCF': x,
            'PROARIDX': y,
            'MAXSPEC': m.max_spectral_change if not np.isnan(m.max_spectral_change) else 0.0,
            'MAXMORPH': m.max_morphology_instability if not np.isnan(m.max_morphology_instability) else 0.0,
            'MAXEAD': m.max_ead_incidence_pct if not np.isnan(m.max_ead_incidence_pct) else 0.0,
            'HASCESS': 'Y' if m.has_cessation else 'N',
            'NCONC': m.n_concentrations,
            'RISKZONE': 'HIGH' if y >= 40 else ('INTERMEDIATE' if y >= 20 else 'LOW'),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Define-XML (simplified)
# ═══════════════════════════════════════════════════════════════════════

def _generate_define_xml(study_id: str, datasets: dict, output_dir: Path):
    """Generate a simplified Define-XML 2.0 metadata file."""
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

    # Document each dataset
    domain_info = {
        'TS':   ('Trial Summary',         'Trial summary parameters and study metadata'),
        'DM':   ('Demographics',           'One record per microtissue (chip+channel)'),
        'EX':   ('Exposure',               'Drug exposure events with concentrations'),
        'EG':   ('ECG Test Results',        'Electrophysiology measurements: FPD, FPDcF, BP, amplitude, arrhythmia metrics'),
        'RISK': ('Drug Risk Classification', 'CiPA risk map coordinates and classification per drug'),
    }

    for domain, df in datasets.items():
        info = domain_info.get(domain, (domain, ''))
        n_rows = len(df)
        n_cols = len(df.columns)
        lines.append(f'      <!-- Domain: {domain} ({info[0]}) -->')
        lines.append(f'      <def:ItemGroupDef OID="IG.{domain}"')
        lines.append(f'                        Name="{domain}"')
        lines.append(f'                        SASDatasetName="{domain}"')
        lines.append(f'                        def:Label="{info[0]}"')
        lines.append(f'                        def:Structure="{"One record per parameter" if domain == "TS" else "One record per test per subject"}"')
        lines.append(f'                        Purpose="Tabulation"')
        lines.append(f'                        def:StandardOID="STD.SENDIG.3.1">')
        lines.append(f'        <!-- {info[1]} -->')
        lines.append(f'        <!-- Records: {n_rows}, Variables: {n_cols} -->')

        for col in df.columns:
            dtype = 'text' if df[col].dtype == object else 'float'
            lines.append(f'        <ItemRef ItemOID="IT.{domain}.{col}" Mandatory="No"/>')

        lines.append(f'      </def:ItemGroupDef>')
        lines.append('')

    # Variable definitions
    lines.append('      <!-- Variable Definitions -->')
    for domain, df in datasets.items():
        for col in df.columns:
            dtype = df[col].dtype
            sas_type = 'Char' if dtype == object else 'Num'
            length = int(df[col].astype(str).str.len().max()) if len(df) > 0 else 8
            length = max(length, 1)
            label = _TEST_CODES.get(col, ('', col.replace('_', ' ').title(), ''))[1] if col in _TEST_CODES else col
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
#  XPT file writer
# ═══════════════════════════════════════════════════════════════════════

def _write_xpt(df: pd.DataFrame, filepath: Path, dataset_name: str,
               dataset_label: str = ''):
    """Write a DataFrame to SAS Transport v5 (.xpt) format.

    Attempts to use xport library first, falls back to pyreadstat if available,
    otherwise writes as CSV with .xpt extension and a warning.
    """
    # SAS Transport v5 constraints:
    # - Column names max 8 chars (we truncate)
    # - String values max 200 chars
    # - Dataset name max 8 chars

    # Truncate column names to 8 chars (SAS requirement)
    col_map = {}
    used_names = set()
    for col in df.columns:
        short = col[:8].upper()
        # Handle duplicates with O(1) lookup
        counter = 0
        while short in used_names:
            short = col[:6].upper() + str(counter)
            counter += 1
        col_map[col] = short
        used_names.add(short)

    df_sas = df.rename(columns=col_map).copy()

    # Ensure string columns are proper strings, truncate to 200
    # SAS Transport v5 only supports latin-1 encoding
    for col in df_sas.columns:
        if df_sas[col].dtype == object:
            df_sas[col] = (df_sas[col].fillna('').astype(str).str[:200]
                           .str.encode('latin-1', errors='replace')
                           .str.decode('latin-1'))
        else:
            df_sas[col] = pd.to_numeric(df_sas[col], errors='coerce')

    # Try to write using xport library
    try:
        import xport
        with open(filepath, 'wb') as f:
            xport.from_dataframe(df_sas, f)
        return
    except ImportError:
        pass

    # Fallback: try pyreadstat
    try:
        import pyreadstat
        pyreadstat.write_xport(df_sas, filepath, table_name=dataset_name[:8])
        return
    except ImportError:
        pass

    # Final fallback: write as CSV with .xpt extension and warning
    warnings.warn(
        f"xport and pyreadstat not available. Writing {filepath.name} as CSV. "
        "This file will need manual conversion to proper SAS Transport format.",
        UserWarning
    )

    # Write as CSV with a comment header explaining the fallback
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# WARNING: This is a CSV fallback (xport/pyreadstat not installed)\n")
        f.write(f"# Dataset: {dataset_name} - {dataset_label}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("# For proper CDISC SEND submission, convert to SAS Transport v5 format\n\n")
        df_sas.to_csv(f, index=False)


# ═══════════════════════════════════════════════════════════════════════
#  Main export function
# ═══════════════════════════════════════════════════════════════════════

def export_send_package(
    results: list,
    output_dir,
    study_id: str = 'CIPA001',
    study_title: str = 'CiPA In Vitro Cardiac Safety - hiPSC-CM Field Potential',
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
    ts_df = _build_ts(study_id, study_title, len(results))
    dm_df = _build_dm(results, study_id)
    ex_df = _build_ex(results, study_id)
    eg_df = _build_eg(results, study_id)
    risk_df = _build_risk(results, study_id)

    datasets = {
        'TS': ts_df,
        'DM': dm_df,
        'EX': ex_df,
        'EG': eg_df,
        'RISK': risk_df,
    }

    # ── Write XPT files ──
    files = []
    labels = {
        'TS': 'Trial Summary',
        'DM': 'Demographics',
        'EX': 'Exposure',
        'EG': 'ECG Test Results',
        'RISK': 'Drug Risk Classification',
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
        f"CDISC SEND Export Package",
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
        f"  1. Validate with Pinnacle 21 Community",
        f"  2. Review CSV copies in {csv_dir}",
        f"  3. Submit .xpt files + define.xml to regulatory agency",
    ])
    summary = '\n'.join(summary_lines)

    # Write summary
    (output_dir / 'README.txt').write_text(summary, encoding='utf-8')
    files.append(output_dir / 'README.txt')

    return {
        'files': files,
        'datasets': datasets,
        'summary': summary,
    }
