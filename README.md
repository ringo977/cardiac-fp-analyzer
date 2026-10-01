# Cardiac FP Analyzer

[![CI](https://github.com/ringo977/cardiac-fp-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ringo977/cardiac-fp-analyzer/actions/workflows/ci.yml)

**Versione**: 3.4.0
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
- **GUI desktop (PySide6 + PyQtGraph)**: viewer segnale con zoom, tab Battiti con template medio ed editor interattivo (override salvati in sidecar `.overrides.json`), pannello **Studi** (Studio → Gruppo → File) con batch in background, metriche per gruppo, curve dose-risposta ed export CDISC per studio, dialog impostazioni, tema chiaro/scuro
- **GUI web Streamlit** (legacy, solo manutenzione): analisi singolo file, batch + risk map, confronto farmaci
- **Corpus di regressione su segnali reali**: 8 baseline del dataset Visone et al. 2023 con RR/FPD pubblicati come riferimento, eseguite ad ogni run di test
- **Logging strutturato**: logging per modulo con `NullHandler`; warning soppressi solo per librerie terze nella GUI
- **Configurazione completa**: tutti i parametri e pesi di scoring esportabili/importabili via JSON

## Struttura

```
cardiac_fp_analyzer/        # Libreria core di analisi
├── __init__.py             # Package init + versione + NullHandler
├── config.py               # Configurazione centralizzata (AnalysisConfig, dataclasses JSON)
├── analyze.py              # Pipeline principale + CLI + batch
├── channel_selection.py    # Selezione automatica canale (scoring multi-criterio)
├── inclusion.py            # Criteri di inclusione batch (5 criteri gerarchici)
├── repolarization.py       # Rilevamento ripolarizzazione e metodi FPD
├── loader.py               # Parser CSV WaveForms
├── filtering.py            # Pipeline di filtraggio
├── beat_detection.py       # Rilevamento battiti multi-metodo
├── parameters.py           # Estrazione parametri elettrofisiologici
├── quality_control.py      # QC: SNR, ampiezza, morfologia, grading A–F
├── rhythm_integration.py   # Filtro ritmico, RR-outlier, topologia del ritmo
├── arrhythmia.py           # Analisi e classificazione aritmie (stat + residuale)
├── residual_analysis.py    # Analisi residua: template, EAD, Poincaré STV
├── normalization.py        # Normalizzazione baseline + TdP scoring
├── study.py                # Modello dati Studio/Gruppo/File (schema versionato)
├── overrides.py            # Sidecar .overrides.json per correzioni manuali dei battiti
├── cessation.py            # Rilevamento cessazione battito (5 sub-detector)
├── spectral.py             # Analisi spettrale (PSD, entropia, armoniche)
├── risk_map.py             # Risk map CiPA 2D
├── cdisc_export.py         # Export CDISC SEND (xpt + define.xml)
├── plotting.py             # Visualizzazione (downsampling, overlay)
└── report.py               # Generazione Excel + PDF

pyside_app/                 # GUI desktop PySide6 (UI primaria, ADR-0001)
├── main.py                 # MainWindow: tab Segnale/Battiti, Ricalcola, menu
├── signal_viewer.py        # Viewer PyQtGraph (overlay raw/filtrato, marker battiti, zoom)
├── study_panel.py          # Pannello Studi: albero, batch, metriche, dose-risposta, CDISC
├── settings_dialog.py      # Dialog AnalysisConfig
├── settings_dialog_helpers.py
└── theme.py                # Tema Scuro/Chiaro (QSettings)

ui/                         # Moduli GUI Streamlit (legacy, manutenzione)
├── __init__.py             # Package marker
├── i18n.py                 # Traduzioni IT/EN + helper T()
├── helpers.py              # Funzioni condivise (reanalyze, amplitude_scale)
├── display.py              # Componenti display condivisi (plot_signal, plot_beats, show_params_table)
├── config_sidebar.py       # Sidebar configurazione con import/export JSON
├── single_file.py          # Pagina analisi singolo file + editor battiti
├── batch.py                # Pagina analisi batch + risk map
├── drug_comparison.py      # Dashboard confronto farmaci
└── reports.py              # Widget download report (Excel, PDF, CDISC)

app.py                      # Entry point Streamlit legacy (~90 righe, router)
tests/                      # 540+ test; tests/fixtures/real_signals/ = corpus reale
tools/                      # reanalyze.py (manifest di provenienza), build_real_signal_fixtures.py, ...
docs/decisions/             # ADR (0001: Streamlit → PySide6)
pyproject.toml              # Packaging e dipendenze
requirements.txt            # Dipendenze (bundle di comodo, per pip install -r)
```

## Installazione

```bash
# Con pyproject.toml (raccomandato)
pip install .                        # Solo core (CLI)
pip install ".[gui]"                 # Core + GUI desktop (PySide6 + PyQtGraph)
pip install ".[gui,reports]"         # + report Excel
pip install ".[streamlit]"           # GUI web legacy
pip install ".[all]"                 # Tutto, incluso CDISC .xpt (pyreadstat)
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

### GUI desktop (PySide6)

```bash
pip install ".[gui,reports]"
cardiac-fp-gui                       # oppure: python -m pyside_app.main
```

File → Apri CSV per un singolo file; il pannello **Studi** gestisce cartelle intere
organizzate in Studio → Gruppo (farmaco) → File (concentrazione), con analisi batch,
metriche per gruppo, curve dose-risposta ed export CDISC SEND.

### GUI web Streamlit (legacy)

```bash
pip install ".[streamlit]"
streamlit run app.py
```

### Da Python

```python
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.analyze import batch_analyze

config = AnalysisConfig()          # amplifier_gain = 1e4 di default (µECG-Pharma Digilent)

results = batch_analyze('/path/to/data/', config=config)
```

## Stato della validazione

Il confronto con i valori pubblicati (Visone et al. 2023) è documentato in
`VALIDAZIONE_vs_paper_Visone2023.md` e nei file `ASSESSMENT_*.md`. In sintesi, ad ottobre 2026:

- Su 8 baseline con elettrodo degli autori forzato, **RR e FPD entro ±12 %** del pubblicato
  (6 su 8 entro ±5 %); è il corpus di `tests/test_real_signal_regression.py`, eseguito ad ogni run.
- Il raddoppio dei battiti su Exp8 (156 rilevati contro ~75) è stato risolto con un gate ancorato
  al rumore della registrazione (`noise_floor_gate`, vedi `config.py`). Le soglie di inclusione
  calibrate negli `SPRINT*.md` erano costruite su misure affette da quel difetto e **vanno ritarate**.
- Restano aperti: la frazione di FPD "non misurabile" (gli autori lasciano in bianco ~41 % dei punti,
  il software quasi nulla — non dipendeva dal fallback argmax, ora rimosso); la direzione della
  nifedipina (allungamento nel software, accorciamento nel paper); l'assenza di correzione per il
  drift del veicolo e di un fit dose-risposta.

## Quality Control

Il modulo QC valida ogni battito rilevato:

- **SNR globale**: rapporto segnale/rumore dell'intera registrazione
- **Validazione ampiezza**: rigetta battiti con ampiezza < 25% del riferimento (probabile rumore)
- **Correlazione morfologica**: rigetta battiti con forma anomala rispetto al template mediano
- **Grading**: A (eccellente) → F (non analizzabile)

## Changelog

### v3.4.0 (Ottobre 2026)
- **Beat detection**: gate SNR ancorato al noise floor (`enable_noise_floor_gate`) — chiude il raddoppio dei battiti su Exp8 (156→81, RR −46 % → +10 % vs pubblicato, FPD invariato); zero battiti rimossi sui file puliti
- **Corpus di regressione su segnali reali**: `tests/fixtures/real_signals/` (8 baseline Visone 2023, 3.4 MB) + `tests/test_real_signal_regression.py`; `tools/build_real_signal_fixtures.py` lo rigenera dal dataset
- **Ripolarizzazione**: rimosso il fallback `argmax` — T-wave non misurabile → `None`, mai un valore fabbricato
- **Unità**: `amplifier_gain` default 1e4 (prima 1.0: ampiezze ×10⁴ da CLI/batch); CDISC `SPIKEAM` davvero in µV; etichetta Excel FPDcF/FPDcB coerente con la correzione configurata; rinominata `inclusion['fpd_reliable']` → `fpd_confidence_ok` (collisione di nome)
- **GUI PySide6 dichiarata e pacchettizzata**: extra `gui` = PySide6 + pyqtgraph, `pyside_app` nei package, entry point `cardiac-fp-gui`; Streamlit spostato nell'extra `streamlit`
- **CI**: gira anche su `feat/**` e `fix/**`, installa PySide6 e lo esercita headless; fallisce se i test GUI vengono saltati
- **ADR-0001** portata a *Accepted*; README e DOCUMENTATION descrivono la UI reale
- **Aprile–agosto 2026** (già in `main`/`feat/pyside-poc`, mai rilasciati): migrazione UI a PySide6, modello Studio/Gruppo/File, sidecar override battiti, `rhythm_integration`, RR locale per la correzione di frequenza, fix segno ripolarizzazione per battito, flag `correction` effettivo, propagazione `fpd_reliable`, criteri di inclusione FPD/RR, strumenti di ri-analisi con manifest

### v3.3.0 (Marzo 2026)
- **CI GitHub Actions**: workflow `ci.yml` con lint (ruff) + test su Python 3.9–3.12
- **Refactoring arrhythmia**: `arrhythmia.py` (708→438 righe) spezzato in `residual_analysis.py` (304 righe) — template, residui, EAD detection, Poincaré STV
- **Ruff lint clean**: risolti 194 errori ruff; configurazione in `pyproject.toml` con regole mirate
- **Test end-to-end numerici**: 17 test su segnali sintetici — beat detection, periodi, parametri, residui, classificazione aritmie, Poincaré STV
- **Suite test**: 54 test totali (smoke, config, alignment, e2e sintetici)

### v3.2.1 (Marzo 2026)
- **Refactoring core**: `analyze.py` (690→442 righe) spezzato in `channel_selection.py`, `inclusion.py`; `parameters.py` (671→293 righe) spezzato in `repolarization.py`
- **Fix denominatore re-analisi residua**: `beat_periods` ricalcolato dagli indici cleaned (non raw) nella seconda passata batch, evitando mismatch n_beats/CV
- **Hardening eccezioni**: zero `except Exception` in tutto il codebase — ogni catch è ora specifico
- **Rimosso `sys.path.insert`**: pyproject.toml gestisce tutti gli import; rimosso hack legacy da `app.py` e `analyze.py`
- **Warning mirati**: rimossa soppressione blanket `DeprecationWarning`; filtri solo per streamlit, matplotlib, plotly
- **Documentazione QC**: documentata divergenza denominatore QC vs paper nei file di confronto
- **Test**: 35 test (smoke, config, beat alignment, denominatore consistency)

### v3.2.0 (Marzo 2026)
- **UI modulare**: `app.py` (1968→90 righe) spezzato in 7 moduli sotto `ui/`
- **Beat alignment fix**: corretto bug di allineamento indici/segmenti nella pipeline
- **Packaging**: aggiunto `pyproject.toml` con gruppi di dipendenze opzionali
- **Logging strutturato**: `NullHandler` a livello package, logger per modulo, eccezioni specifiche
- **Pesi configurabili**: scoring weights di beat detection e channel selection in dataclass JSON-serializable
- **API pubblica**: `compute_template`, `is_baseline`, `get_group_key` ora pubbliche (alias back-compat mantenuti)
- **Plotly** aggiunto alle dipendenze GUI

## Licenza

MIT License — vedi `pyproject.toml` per i dettagli.
