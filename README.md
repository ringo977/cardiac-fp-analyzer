# cardiac_fp_analyzer

Analisi automatizzata di **field potential (FP)** per registrazioni µECG da **microtessuti cardiaci hiPSC-CM**, acquisite con oscilloscopio **Digilent WaveForms** (CSV: tempo + 2 canali).

## Funzionalità

- **Caricamento smart**: parsing header WaveForms, gestione file lunghi (fino a 360k campioni), downsampling min-max per plot
- **Filtraggio adattivo**: notch 50 Hz (+ armoniche), bandpass 0.5–500 Hz, smoothing Savitzky-Golay
- **Beat detection multi-metodo**: prominenza, derivata, ampiezza — con auto-selezione del metodo migliore
- **Selezione automatica canale** (ch1/ch2): scoring basato su regolarità, SNR e range fisiologico
- **Parametri elettrofisiologici**: Beat Period, ampiezza, rise time, FPD (≈ QT), correzione Fridericia e Bazett, max dV/dt, STV
- **Quality Control**: stima SNR globale, validazione per-beat (ampiezza + correlazione morfologica), grading A–F
- **Aritmie**: classificazione automatica (tachy/bradicardia, battiti prematuri, cessazione, EAD, fibrillazione-like)
- **Report**: Excel multi-foglio (Summary + Arrhythmia + Per-Beat) e PDF con grafici di analisi

## Struttura

```
cardiac_fp_analyzer/
├── __init__.py          # Package init
├── analyze.py           # Pipeline principale + CLI
├── loader.py            # Parser CSV WaveForms
├── filtering.py         # Pipeline di filtraggio
├── beat_detection.py    # Rilevamento battiti multi-metodo
├── parameters.py        # Estrazione parametri elettrofisiologici
├── quality_control.py   # QC: SNR, ampiezza, morfologia
├── arrhythmia.py        # Analisi e classificazione aritmie
├── plotting.py          # Visualizzazione (downsampling, overlay)
└── report.py            # Generazione Excel + PDF
```

## Requisiti

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# Analisi batch di tutti i CSV in una cartella (ricorsivo)
python cardiac_fp_analyzer/analyze.py /percorso/cartella/dati --channel auto -o /percorso/output

# Opzioni
#   --channel auto|ch1|ch2   Selezione canale (default: auto)
#   -o /percorso/output      Cartella output (default: analysis_results/)
#   -q                       Modalità silenziosa
```

Output: file Excel (`.xlsx`) e PDF nella cartella specificata.

## Quality Control

Il modulo QC valida ogni battito rilevato:

- **SNR globale**: rapporto segnale/rumore dell'intera registrazione
- **Validazione ampiezza**: rigetta battiti con ampiezza < 25% del riferimento (probabile rumore)
- **Correlazione morfologica**: rigetta battiti con forma anomala rispetto al template mediano
- **Grading**: A (eccellente) → F (non analizzabile)

## Licenza

Uso interno / da definire.
