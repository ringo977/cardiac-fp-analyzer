# Cardiac FP Analyzer

**Versione**: 3.2.0
**Python**: ≥ 3.9

Analisi automatizzata di **field potential (FP)** per registrazioni µECG da **microtessuti cardiaci hiPSC-CM**, acquisite con oscilloscopio **Digilent WaveForms** (CSV: tempo + 2 canali).

## Funzionalità

- **Caricamento smart**: parsing header WaveForms, gestione file lunghi (fino a 360k campioni), downsampling min-max per plot
- **Filtraggio adattivo**: notch 50 Hz (+ armoniche), bandpass 0.5–500 Hz, smoothing Savitzky-Golay
- **Beat detection multi-metodo**: prominenza, derivata, ampiezza — con auto-selezione del metodo migliore e **scoring configurabile via JSON**
- **Selezione automatica canale** (el1/el2): scoring basato su regolarità, SNR e range fisiologico, con **pesi configurabili**
- **Parametri elettrofisiologici**: Beat Period, ampiezza, rise time, FPD (≈ QT), correzione Fridericia e Bazett, max dV/dt, STV
- **Quality Control**: stima SNR globale, validazione per-beat (ampiezza + correlazione morfologica), grading A–F
- **Aritmie**: classificazione automatica (tachy/bradicardia, battiti prematuri, cessazione, EAD, fibrillazione-like)
- **Normalizzazione baseline**: ΔFPDcF%, TdP scoring, classificazione farmaco
- **Risk map CiPA**: mappa 2D interattiva (Plotly) con zone LOW/INTERMEDIATE/HIGH
- **Report**: Excel multi-foglio (Summary + Arrhythmia + Per-Beat) e PDF con grafici
- **Export CDISC SEND**: pacchetto regolatorio conforme SENDIG v3.1 (TS, DM, EX, EG, RISK + define.xml)
- **GUI Streamlit**: interfaccia web con analisi singolo file, batch + risk map, confronto farmaci
- **Logging strutturato**: logging per modulo con `NullHandler`, nessuna soppressione blanket di warning
- **Configurazione completa**: tutti i parametri e pesi di scoring esportabili/importabili via JSON

## Struttura

```
cardiac_fp_analyzer/        # Libreria core di analisi
├── __init__.py             # Package init + versione + NullHandler
├── config.py               # Configurazione centralizzata (AnalysisConfig, dataclasses JSON)
├── analyze.py              # Pipeline principale + CLI + batch
├── loader.py               # Parser CSV WaveForms
├── filtering.py            # Pipeline di filtraggio
├── beat_detection.py       # Rilevamento battiti multi-metodo
├── parameters.py           # Estrazione parametri elettrofisiologici
├── quality_control.py      # QC: SNR, ampiezza, morfologia, grading A–F
├── arrhythmia.py           # Analisi e classificazione aritmie
├── normalization.py        # Normalizzazione baseline + TdP scoring
├── cessation.py            # Rilevamento cessazione battito (5 sub-detector)
├── spectral.py             # Analisi spettrale (PSD, entropia, armoniche)
├── risk_map.py             # Risk map CiPA 2D
├── cdisc_export.py         # Export CDISC SEND (xpt + define.xml)
├── plotting.py             # Visualizzazione (downsampling, overlay)
└── report.py               # Generazione Excel + PDF

ui/                         # Moduli GUI Streamlit
├── __init__.py             # Package marker
├── i18n.py                 # Traduzioni IT/EN + helper T()
├── helpers.py              # Funzioni condivise (reanalyze, amplitude_scale)
├── config_sidebar.py       # Sidebar configurazione con import/export JSON
├── single_file.py          # Pagina analisi singolo file + editor battiti
├── batch.py                # Pagina analisi batch + risk map
├── drug_comparison.py      # Dashboard confronto farmaci
└── reports.py              # Widget download report (Excel, PDF, CDISC)

app.py                      # Entry point Streamlit (~90 righe, router)
pyproject.toml              # Packaging e dipendenze
requirements.txt            # Dipendenze (legacy, per pip install -r)
```

## Installazione

```bash
# Con pyproject.toml (raccomandato)
pip install .                        # Solo core
pip install ".[gui]"                 # Core + Streamlit GUI
pip install ".[all]"                 # Core + GUI + report + CDISC
pip install ".[dev]"                 # Tutto + pytest + ruff

# Oppure con requirements.txt (legacy)
pip install -r requirements.txt
```

## Uso

### Riga di comando

```bash
# Analisi batch di tutti i CSV in una cartella (ricorsivo)
cardiac-fp /percorso/cartella/dati --channel auto -o /percorso/output

# Oppure senza installazione
python -m cardiac_fp_analyzer.analyze /percorso/cartella/dati --channel auto

# Con file di configurazione JSON
cardiac-fp /percorso/cartella/dati --config my_config.json
```

### GUI Streamlit

```bash
pip install ".[gui]"
streamlit run app.py
```

### Da Python

```python
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.analyze import batch_analyze

config = AnalysisConfig()
config.amplifier_gain = 1e4  # µECG-Pharma Digilent

results = batch_analyze('/path/to/data/', config=config)
```

## Quality Control

Il modulo QC valida ogni battito rilevato:

- **SNR globale**: rapporto segnale/rumore dell'intera registrazione
- **Validazione ampiezza**: rigetta battiti con ampiezza < 25% del riferimento (probabile rumore)
- **Correlazione morfologica**: rigetta battiti con forma anomala rispetto al template mediano
- **Grading**: A (eccellente) → F (non analizzabile)

## Changelog

### v3.2.0 (Marzo 2026)
- **UI modulare**: `app.py` (1968→90 righe) spezzato in 7 moduli sotto `ui/`
- **Beat alignment fix**: corretto bug di allineamento indici/segmenti nella pipeline
- **Packaging**: aggiunto `pyproject.toml` con gruppi di dipendenze opzionali
- **Logging strutturato**: `NullHandler` a livello package, logger per modulo, eccezioni specifiche
- **Pesi configurabili**: scoring weights di beat detection e channel selection in dataclass JSON-serializable
- **API pubblica**: `compute_template`, `is_baseline`, `get_group_key` ora pubbliche (alias back-compat mantenuti)
- **Plotly** aggiunto alle dipendenze GUI

## Licenza

Uso interno / da definire.
