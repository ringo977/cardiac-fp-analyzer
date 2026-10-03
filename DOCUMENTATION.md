# Cardiac FP Analyzer — Documentazione Completa

**Versione**: 3.9.0
**Piattaforma**: Python 3.9+
**Riferimento**: Visone, Lozano-Juan et al., *Toxicological Sciences* 191(1), 47–60, 2023
**Dataset di validazione**: 169 file CSV, 7 farmaci CiPA (3 positivi, 4 negativi)
**Accuratezza**: 6/7 sul set di validazione CiPA

---

## Indice

1. [Panoramica](#1-panoramica)
2. [Installazione e utilizzo](#2-installazione-e-utilizzo)
3. [Pipeline di analisi](#3-pipeline-di-analisi)
4. [Moduli in dettaglio](#4-moduli-in-dettaglio)
   - 4.1 [Caricamento dati (`loader.py`)](#41-caricamento-dati)
   - 4.2 [Filtraggio del segnale (`filtering.py`)](#42-filtraggio-del-segnale)
   - 4.3 [Rilevamento dei battiti (`beat_detection.py`)](#43-rilevamento-dei-battiti)
   - 4.4 [Estrazione parametri (`parameters.py`)](#44-estrazione-parametri)
   - 4.5 [Controllo qualità (`quality_control.py`)](#45-controllo-qualita)
   - 4.6 [Analisi aritmie (`arrhythmia.py`)](#46-analisi-aritmie)
   - 4.7 [Rilevamento cessazione (`cessation.py`)](#47-rilevamento-cessazione)
   - 4.8 [Analisi spettrale (`spectral.py`)](#48-analisi-spettrale)
   - 4.9 [Criteri di inclusione (batch)](#49-criteri-di-inclusione-batch)
   - 4.10 [Normalizzazione e classificazione (`normalization.py`)](#410-normalizzazione-e-classificazione)
   - 4.11 [Risk map CiPA (`risk_map.py`)](#411-risk-map-cipa)
   - 4.12 [Report (`report.py`)](#412-report)
5. [Configurazione](#5-configurazione)
6. [Razionale scientifico](#6-razionale-scientifico)
7. [Validazione](#7-validazione)
8. [Interfaccia grafica](#8-interfaccia-grafica)
9. [Export CDISC SEND](#9-export-cdisc-send)
10. [Logging e diagnostica](#10-logging-e-diagnostica)
11. [Changelog](#11-changelog)

---

## 1. Panoramica

Il Cardiac FP Analyzer è un software per l'analisi automatica di potenziali di campo (field potential, FP) registrati da microtessuti cardiaci derivati da hiPSC-CM su piattaforma µECG-Pharma. Il software implementa una pipeline completa per la valutazione del rischio proaritmico dei farmaci, seguendo il framework CiPA (Comprehensive in vitro Proarrhythmia Assay).

Il segnale FP è l'equivalente extracellulare del potenziale d'azione cardiaco. La depolarizzazione rapida produce un picco negativo (spike), seguito da una fase di ripolarizzazione che termina con una deflessione lenta. L'intervallo spike–ripolarizzazione è il Field Potential Duration (FPD), analogo all'intervallo QT dell'ECG clinico. L'allungamento del FPD è il marcatore primario del rischio di aritmie da farmaci (Torsade de Pointes, TdP).

Il software analizza ogni registrazione attraverso una pipeline a 12 stadi, raggruppa le registrazioni per chip/camera/elettrodo, normalizza rispetto al baseline, e classifica ciascun farmaco su una mappa di rischio 2D.

---

## 2. Installazione e utilizzo

### Installazione

```bash
# Con pyproject.toml (raccomandato)
pip install .                        # Solo core (numpy, pandas, scipy, matplotlib)
pip install ".[gui]"                 # Core + GUI desktop (PySide6 + PyQtGraph)
pip install ".[streamlit]"           # Core + GUI web legacy (Streamlit + Plotly)
pip install ".[reports]"             # Core + xlsxwriter
pip install ".[cdisc]"              # Core + pyreadstat
pip install ".[all]"                 # Tutto
pip install ".[dev]"                 # Tutto + pytest + ruff

# Legacy (requirements.txt)
pip install -r requirements.txt
```

### Utilizzo da linea di comando

```bash
# Dopo installazione (entry point CLI)
cardiac-fp /path/to/data/ --channel auto -o results/

# Senza installazione
python -m cardiac_fp_analyzer.analyze /path/to/data/

# Con selezione elettrodo manuale
python -m cardiac_fp_analyzer.analyze /path/to/data/ --channel el1

# Con file di configurazione JSON
python -m cardiac_fp_analyzer.analyze /path/to/data/ --config my_config.json

# Con preset
python -m cardiac_fp_analyzer.analyze /path/to/data/ --preset conservative
```

### Utilizzo da Python

```python
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.analyze import batch_analyze

# Configurazione per il sistema µECG-Pharma Digilent
config = AnalysisConfig()
config.amplifier_gain = 1e4  # Correzione guadagno ÷10⁴

# Analisi batch
results = batch_analyze('/path/to/data/', config=config)
```

### Generazione della risk map CiPA

```python
from cardiac_fp_analyzer.risk_map import generate_risk_map

ground_truth = {
    'terfenadine': True, 'quinidine': True, 'dofetilide': True,
    'alfuzosin': False, 'mexiletine': False, 'nifedipine': False,
    'ranolazine': False,
}

fig = generate_risk_map(results, config=config, ground_truth=ground_truth)
fig.savefig('risk_map.png', dpi=150)
```

### Struttura dell'output

L'analisi produce nella cartella `analysis_results/`:

- `cardiac_fp_analysis_YYYYMMDD_HHMMSS.xlsx` — Report Excel multi-sheet
- `cardiac_fp_analysis_YYYYMMDD_HHMMSS.pdf` — Report PDF con grafici per file
- `analysis_config.json` — Configurazione utilizzata (per riproducibilità)

---

## 3. Pipeline di analisi

La pipeline processa ogni file CSV in 12 stadi sequenziali, poi esegue operazioni batch (normalizzazione, baseline-relative residual analysis, classificazione).

```
CSV file
  │
  ├─ 1. Caricamento (loader.py)
  │     Parsing header Digilent WaveForms, estrazione metadati
  │
  ├─ 2. Parsing nome file (loader.py)
  │     Estrazione chip, canale, farmaco, concentrazione
  │
  ├─ 3. Selezione elettrodo (analyze.py)
  │     Scoring automatico el1 vs el2 (se channel='auto'),
  │     oppure analisi di entrambi gli elettrodi (se channel='both')
  │
  ├─ 4. Correzione guadagno (analyze.py)
  │     Segnale = segnale_raw / amplifier_gain
  │
  ├─ 5. Filtraggio (filtering.py)
  │     Notch 50Hz → Bandpass 0.5–500Hz → Savitzky-Golay
  │
  ├─ 6. Beat detection (beat_detection.py)
  │     Auto-selezione tra 3 metodi, retry se pochi battiti
  │
  ├─ 7. Segmentazione battiti (beat_detection.py)
  │     Taglio beat per beat, allineamento al picco spike
  │
  ├─ 8. Quality Control (quality_control.py)
  │     Validazione per ampiezza e morfologia, grading A–F
  │
  ├─ 9. Estrazione parametri (parameters.py)
  │     Template averaging → FPD, FPDcF, ampiezza, dV/dt
  │
  ├─ 10. Analisi aritmie (arrhythmia.py)
  │      Statistica BP + residual-based (EAD, morph instability, STV)
  │
  ├─ 11. Cessation detection (cessation.py) [opzionale]
  │      5 sub-detector per arresto del battito
  │
  └─ 12. Analisi spettrale (spectral.py) [opzionale]
         PSD Welch, entropia, armoniche, confronto vs baseline
```

**Operazioni batch** (dopo il processing di tutti i file):

```
Tutti i risultati
  │
  ├─ Criteri di inclusione (5 criteri, basati sul baseline)
  │
  ├─ Baseline-relative residual analysis (pass 2)
  │     Template dal baseline → residui dei drug recording
  │
  ├─ Normalizzazione vs baseline
  │     ΔFPDcF%, TdP score, classificazione farmaco
  │
  └─ Report (Excel + PDF)
```

---

## 4. Moduli in dettaglio

### 4.1 Caricamento dati

**Modulo**: `loader.py`

#### `load_csv(filepath)`

Legge i file CSV prodotti dal sistema Digilent WaveForms (Analog Discovery 2). Il formato include un header con metadati del dispositivo e due elettrodi di acquisizione (el1, el2).

**Output**: `(metadata, DataFrame)` dove metadata contiene sample_rate, device, serial, datetime, range e offset per ciascun elettrodo. Il DataFrame ha colonne `['time', 'el1', 'el2']`.

**Frequenze di campionamento alte (dalla v3.8)**: le registrazioni sopra 3 kHz vengono decimate al caricamento a circa 2 kHz, con filtro anti-aliasing FIR a fase zero. In metadata restano `original_sample_rate` e `decimation_factor`; `load_csv(path, max_sample_rate=None)` lascia la frequenza originale. La pipeline è tarata su 2 kHz: le finestre sono in campioni e il passa-banda 0,5–500 Hz è progettato come coefficienti (b, a). A 20 kHz quel filtro ha un polo fuori dal cerchio unitario, il segnale filtrato diverge e non viene trovato nessun battito. Era il caso di tutti i 37 file Accelera di Exp11 del dataset Visone 2023. Decimati, combaciano con gli autori: BP 838,5 contro 838,9 ms, FPDc 532 contro 527 ms.

`recording_datetime(path)` legge solo la riga `#Date Time:` dell'intestazione. Il batch la usa per scegliere il riferimento pre-dose (vedi 4.10).

**Nota sulla nomenclatura (v3.5)**: nei nomi dei file, `ch1`/`ch2`/`ch3` indicano le *camere* (chamber) del chip. Le colonne del CSV, che rappresentano i due *elettrodi* di registrazione del Digilent, sono rinominate in `el1`/`el2` per evitare ambiguità.

#### `parse_filename(filename)` e `describe_recording(path)`

`parse_filename` estrae dal nome del file farmaco, concentrazione e se si tratta di un baseline. `describe_recording` aggiunge l'identità del **tessuto** ricavata dalle cartelle: esperimento, giorno, chip e camera. La chiave `tissue` (per esempio `exp5/day7/chipA_ch1`) è quella su cui la normalizzazione abbina baseline e dosi.

| Convenzione | Esempio | Tessuto |
|---|---|---|
| tessuto nel nome | `Exp5/Day7/chipA/chipA_ch1_terfe_300nM.csv` | `exp5/day7/chipA_ch1` |
| Accelera: chip nella cartella, camera come prefisso | `Exp12_Accelera/Day 7/Chip 537/Ch2_DMSO10-1_2000fs_1_3K.csv` | `exp12/day7/chip537_ch2` |
| Accelera: chip come numero, camera in una cartella | `Day8/529/Ch1_…`, `Chip 569/Ch2 Bepridil/Ch2_…` | `…/chip529_ch1`, `…/chip569_ch2` |
| due elettrodi nel nome, prima del 2020 (dalla v3.8) | `exp1/day6/chipB_sotalol/Ch2_Sotalol_7,5_channel1_sx_channel2_dx.csv` | `exp1/day6/chipB_ch2` |
| chip come cartella sotto l'esperimento (dalla v3.8) | `Exp2/inj1/ch3_1000Terfe_channel1_sx_channel2_dx.csv` | `exp2/-/chipINJ1_ch3` |
| due tessuti nello stesso file, uno per ingresso (dalla v3.9) | `Exp1/Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv` | una registrazione per ingresso: `exp1/-/chipP_ch1` (ingresso 1), `exp1/-/chipQ_ch2` (ingresso 2) |

Le regole sono queste:

- **Esperimento:** la cartella più interna del tipo `Exp<N>`, senza distinguere maiuscole e minuscole e con o senza separatore (`Exp5`, `EXP 5`, `exp1`, `Exp_10`, `Exp12_Accelera`). Sono accettate anche le cartelle che iniziano con `EXP`, come prima.
- **Giorno:** la cartella `Day<N>` (`Day7`, `day 7`, `day 8 Accelera`).
- **Chip da cartella:** il primo token del nome (`Chip 537` → `537`, `chipB_sotalol` → `B`). Se nessuna cartella `Chip…` o numerica lo indica, si usa la cartella più interna sotto quella dell'esperimento, saltando giorno e camera (`inj1`).
- **Concentrazione:** compare senza separatore iniziale (`300 nM`; prima era `.300 nM`). Un numero senza unità non resta nel nome del farmaco (`NIFEDIPINE_10` diventa farmaco `NIFEDIPINE`, concentrazione `10`). `2andhalf` vale 2,5 e la virgola decimale vale come punto (`7,5`). I token di acquisizione dei nomi Accelera (`20000_1_3k`, `2000fs`, `chan1`, `_1h`…) vengono ignorati.
- **Nomi con due elettrodi (dalla v3.8):** il suffisso `channel1_sx_channel2_dx`, e quanto segue (`_bis`, `_sfter3`), viene ignorato. Il numero può precedere il farmaco (`ch1_30_sotalol`, `ch3_1000Terfe`, `ch1_01%_DMSO`).
- **Riferimento t0 (dalla v3.8):** `t0`, `T0`, `T02` indicano la registrazione fatta subito prima della prima dose. È un riferimento come il baseline (`is_baseline`, `reference_kind = 't0'`). Gli autori del paper Visone 2023 normalizzavano su t0, non sul file chiamato baseline registrato prima: il periodo di battito delle loro tabelle è più vicino a t0 in 13 tessuti su 15.
- **Controlli nel tempo (dalla v3.8):** `t1`…`t7` senza farmaco e `Ctrl` diventano farmaco `ctrl`, concentrazione `tN`. Sono controlli, non farmaci.

**Esempio**: `chipA_ch1_terfe_300nM.csv` → `{chip: 'A', channel: 1, drug: 'terfe', concentration: '300 nM', is_baseline: False}`. Il nome canonico del farmaco (`terfenadine`) viene assegnato in fase di classificazione (`normalization.canonical_drug_name`).

**Perché (ottobre 2026)**: fino alla v3.6 l'esperimento veniva letto solo da cartelle che iniziano con `EXP` maiuscolo, e il giorno veniva ignorato. Con cartelle `Exp5`, `exp1` o `Exp 5`, ogni lettera di chip era condivisa tra esperimenti e giorni. Sul dataset Visone 2023, 29 gruppi su 31 mescolavano esperimenti diversi: le dosi venivano normalizzate sul baseline di un altro esperimento (per esempio −27 % invece di +10 %).

#### File con due tessuti e mappa dei campioni `samples.csv` (dalla v3.9)

**Modulo**: `sample_sheet.py`; in `loader.py` `parse_inputs`, `input_columns`, `tissue_key`.

Nella convenzione GG ogni ingresso dell'oscilloscopio registra un tessuto diverso. Il nome elenca i tessuti nell'ordine degli ingressi, ciascuno seguito dal suo test item: `Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A` è chip P camera 1 (K01) sull'ingresso 1, chip Q camera 2 (K02) sull'ingresso 2, dose A. Un ingresso senza tessuto si scrive `na`: dopo il chip (`ChipP_na`), dopo la camera alla fine del nome (`…_ChipQ_ch1_na`) o da solo prima del primo tessuto (`Exp1_na_ChipQ_ch2_…`). Sui dati GG, dove i battiti marcati dall'analista indicano chiaramente un ingresso (135 tessuti su 182), è sempre quello dato dall'ordine del nome.

- **Batch (`channel='auto'`)**: un file con due tessuti diversi su due ingressi dà due registrazioni, una per ingresso. Ognuna ha tessuto, test item e dose propri, elettrodo fisso e nome `… [el1]` / `… [el2]`, e viene abbinata al riferimento dello stesso tessuto, cioè all'ingresso giusto del file di baseline. Un ingresso `na` non viene analizzato. I file con un solo tessuto restano come prima: il software sceglie l'elettrodo sul riferimento del tessuto.
- **Gruppo = tessuto**: in modalità `auto` il batch mette `file_info['group_electrode'] = ''`, così un tessuto è un solo gruppo di inclusione e di abbinamento anche quando i suoi file lo registrano su ingressi diversi.
- **Prima della v3.9** questi file non venivano abbinati: sul dataset GG 81 delle 136 registrazioni con farmaco e nessun test item riceveva una decisione.

**Mappa dei campioni.** Un file `samples.csv` nella cartella analizzata, o in una sua sottocartella, dice che cosa registra ogni ingresso. Per i file che elenca ha la precedenza sui nomi; gli altri file vengono letti dal nome e il batch li segnala.

| Colonna | Contenuto |
|---|---|
| `file` | percorso relativo alla cartella del foglio, o solo il nome se è unico |
| `electrode` | `el1` / `el2` (ingresso 1 / 2); vuoto o `auto`: sceglie il software |
| `experiment` | facoltativa: sostituisce la cartella dell'esperimento (un chip registrato nella cartella di un altro esperimento) |
| `chip`, `chamber` | chip e camera del tessuto |
| `item` | test item o farmaco |
| `dose` | `baseline` (o `t0`) per i riferimenti, `washout`, altrimenti la concentrazione o la condizione (`A`, `300 nM`) |
| `exclude` | qualunque testo esclude l'ingresso; il testo è il motivo |
| `note` | libera; la bozza ci scrive i controlli |

Le intestazioni possono essere in italiano (`elettrodo`, `esperimento`, `camera`, `farmaco`, `escludi`) e il separatore la virgola o il punto e virgola.

**Bozza.** `python -m cardiac_fp_analyzer.sample_sheet <cartella>`, oppure il pulsante "Crea bozza della mappa" nella pagina batch, scrive `samples_draft.csv`: una riga per file, o per ingresso, come il software legge i nomi. Nella colonna `note` segnala:

- camera oltre il numero di camere del chip (3);
- test item che cambia tra le dosi di una camera;
- più registrazioni della stessa camera alla stessa dose: in mancanza di esclusioni le dosi vengono mediate, e per i riferimenti si usa l'ultimo prima della prima dose;
- camere senza riferimento, o senza una dose che altre camere dello stesso test item hanno;
- file con due ingressi e un solo tessuto nel nome: quale ingresso lo registra il nome non lo dice;
- un secondo tessuto che il nome sembra contenere ma non è stato riconosciuto.

Dopo la revisione la bozza va salvata come `samples.csv`. Sui 6 esperimenti GG la bozza ha segnalato tutti gli errori trovati a mano nei nomi: camera sbagliata, chip con la stessa lettera, ripetizioni, tessuto su un ingresso non indicato.

#### Selezione automatica dell'elettrodo

**Funzione**: `analyze.py` → `_select_best_channel(df, fs, cfg)`

Il sistema µECG-Pharma registra su due elettrodi (el1, el2). In modalità `auto`, il software esegue una mini-analisi su ciascun elettrodo e assegna un punteggio **continuo** basato su 5 criteri (massimo 100 punti). Lo scoring è progettato per discriminare chiaramente un elettrodo con buon contatto (battiti riproducibili, ampiezza alta) da uno con segnale degradato.

| # | Criterio | Peso | Tipo | Dettaglio |
|---|----------|------|------|-----------|
| 1 | Beat period fisiologico | 0–15 pt | Soglia | +15 se il periodo medio è in [0.3, 4.0] s |
| 2 | Beat rate ragionevole | 0–10 pt | Soglia | +10 se il rate è in [0.3, 3.5] battiti/s |
| 3 | **Template correlation** | 0–40 pt | **Continuo** | Correlazione media di ogni battito con il template medio. Criterio **dominante**: misura la riproducibilità della forma d'onda. corr=0.9 → 36pt, corr=0.7 → 27pt, corr=0.4 → 14pt. Formula: `min(40, max(0, corr × 44 − 4))` |
| 4 | Regolarità ritmo (CV%) | 0–20 pt | Continuo | CV=0% → 20pt, CV=10% → 16pt, CV=50% → 0pt. Formula: `max(0, 20 − CV% × 0.4)` |
| 5 | Ampiezza spike (ptp) | 0–15 pt | Continuo | Mediana del peak-to-peak per battito. 0mV → 0pt, ≥500mV → 15pt. Formula: `min(15, median_ptp_mV / 500 × 15)` |

**Criterio dominante — Template correlation (40 pt)**: la correlazione del template è il criterio più pesante perché misura direttamente la qualità del segnale elettrofisiologico. Un elettrodo con buon contatto produce battiti dalla forma riproducibile (corr > 0.85), mentre un elettrodo con contatto degradato o segnale basso produce forme d'onda incoerenti (corr < 0.5). Questo metrica discrimina meglio del SNR, che può essere fuorviante quando sia il segnale sia il rumore sono bassi.

**Esempio di discriminazione**: su `chipA_ch1_baseline`, el1 ha corr=0.37 (segnale basso e incoerente) mentre el2 ha corr=0.88 (battiti chiari e riproducibili). Lo scoring risultante è el1=58.8 vs el2=92.1, con una separazione di oltre 33 punti che elimina ogni ambiguità.

L'elettrodo con lo score più alto viene selezionato. Se un elettrodo ha meno di 3 battiti rilevati, resta a score 0.

**Modalità disponibili**: `auto` (scoring continuo), `el1`, `el2`, `both` (analizza entrambi gli elettrodi separatamente, v3.5).

I parametri di scoring sono configurabili tramite `ChannelSelectionConfig`:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `bp_ideal_range_s` | (0.3, 4.0) | Range fisiologico beat period |
| `rate_range_per_s` | (0.3, 3.5) | Range beat rate accettabile |

---

### 4.2 Filtraggio del segnale

**Modulo**: `filtering.py`

Il segnale FP grezzo contiene rumore da diverse sorgenti: interferenza di rete (50 Hz), drift della baseline, rumore ad alta frequenza. La pipeline di filtraggio rimuove questi artefatti preservando la morfologia del segnale cardiaco.

#### Pipeline di filtraggio

1. **Filtro notch** a 50 Hz + armoniche (100, 150 Hz): rimuove l'interferenza di rete. Q=30 per bande strette che non distorcono il segnale cardiaco. Configurabile per 60 Hz (USA) o disattivabile completamente (opzione "Off").

2. **Filtro passa-banda Butterworth** (0.5–500 Hz, ordine 4): rimuove sia il drift DC (< 0.5 Hz) sia il rumore ad alta frequenza (> 500 Hz). La banda 0.5–500 Hz preserva interamente la morfologia del FP cardiaco. Dalla v3.8, se il progetto (b, a) è numericamente instabile (polo sul cerchio unitario o fuori, come a 20 kHz), si usano sezioni del secondo ordine. A 2 kHz il risultato resta identico a prima.

3. **Smoothing Savitzky-Golay** (finestra 7 campioni, ordine 3): smoothing finale che preserva i picchi (non li attenua come un filtro passa-basso convenzionale). Ideale per mantenere la forma dello spike di depolarizzazione.

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `notch_freq_hz` | 50.0 | Frequenza di rete (0 = disattivato) |
| `notch_harmonics` | 3 | N. armoniche da rimuovere |
| `notch_q` | 30.0 | Fattore Q (selettività) |
| `bandpass_low_hz` | 0.5 | Taglio basso passa-banda |
| `bandpass_high_hz` | 500.0 | Taglio alto passa-banda |
| `bandpass_order` | 4 | Ordine del filtro |
| `savgol_window` | 7 | Finestra Savitzky-Golay |
| `savgol_polyorder` | 3 | Ordine polinomiale |

---

### 4.3 Rilevamento dei battiti

**Modulo**: `beat_detection.py`

Il rilevamento dei battiti identifica il picco di depolarizzazione (spike) di ciascun battito cardiaco. Il software implementa tre metodi e un selettore automatico.

#### Metodi di rilevamento

**`prominence`**: Trova picchi basati sulla prominenza (altezza relativa rispetto ai minimi circostanti). Robusto per segnali con buon SNR e battiti regolari.

**`derivative`**: Identifica la massima pendenza del fronte di depolarizzazione (max dV/dt), poi raffina al picco più vicino. Più robusto per segnali rumorosi dove i picchi assoluti possono essere ambigui.

**`peak`**: Rilevamento semplice basato sull'ampiezza assoluta. Metodo di fallback.

**`auto`** (default): Esegue tutti e tre i metodi, poi seleziona quello con il punteggio di plausibilità fisiologica più alto. Il punteggio valuta: periodo medio dei battiti (ideale 0.4–3.0 s), CV del periodo (< 15% = eccellente), rate di battito (0.3–3.5 Hz), numero di battiti rilevati.

#### Correzione bimodale automatica (v3.4)

In modalità `auto`, dopo la selezione del metodo migliore, il sistema analizza la distribuzione dei beat period per rilevare un pattern bimodale. Questo pattern si verifica quando il detector conta sia lo spike di depolarizzazione Na+ sia l'onda T di ripolarizzazione come battiti separati, producendo periodi alternati corti (~400 ms) e lunghi (~800 ms).

L'algoritmo utilizza una soglia di tipo Otsu (minimizzazione della varianza intra-gruppo) per separare i due cluster. Se il rapporto tra i gruppi è nell'intervallo 1.4–3.0× e la separazione è sufficiente (gap > 2σ), il sistema aumenta automaticamente `min_distance_ms` al punto medio tra i gruppi e ripete il detection. La correzione viene applicata solo se il CV migliora.

Esempio: su un segnale con BP reale di ~800 ms, il detector iniziale trova 321 "battiti" con periodi alternati 420/780 ms. La correzione bimodale porta `min_distance` a ~600 ms, risultando in 219 battiti reali con CV=14%.

#### Retry automatico

Se il primo tentativo rileva meno di 5 battiti in una registrazione > 10 s, il software riprova con parametri rilassati (distanza minima 300 ms invece di 400, threshold ×3 invece di ×4).

#### Segmentazione

Dopo il rilevamento, ogni battito viene segmentato in una finestra che va da 50 ms prima dello spike a 850 ms dopo. Questa finestra copre l'intero ciclo depolarizzazione–ripolarizzazione anche per battiti con FPD lungo.

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `method` | 'auto' | Metodo di rilevamento |
| `min_distance_ms` | 400.0 | Distanza minima tra battiti |
| `threshold_factor` | 4.0 | Moltiplicatore soglia adattiva |
| `retry_min_distance_ms` | 300.0 | Distanza minima (retry) |
| `retry_threshold_factor` | 3.0 | Soglia adattiva (retry) |

---

### 4.4 Estrazione parametri

**Modulo**: `parameters.py`

Questo modulo estrae i parametri elettrofisiologici da ogni battito, con approccio template-guided per robustezza.

#### Template averaging

Il software costruisce un template rappresentativo del battito tipico della registrazione:

1. Seleziona fino a 60 battiti equamente distribuiti nella registrazione
2. Allinea i battiti tramite cross-correlazione nella regione di depolarizzazione (primi 100 ms)
3. Calcola la mediana robusta (non la media) per resistenza agli outlier

**Razionale**: La mediana è più robusta della media contro battiti aberranti, EAD, e artefatti. La cross-correlazione garantisce l'allineamento temporale anche quando il beat detection ha piccoli offset.

#### Misurazione FPD (Field Potential Duration)

Il FPD è l'intervallo dallo spike di depolarizzazione al punto di ripolarizzazione. La sua misurazione è il passaggio più critico e tecnicamente complesso dell'analisi.

**Metodo del picco** (default dalla v3.6.0): l'FPD termina al picco dell'onda di ripolarizzazione. È la convenzione della misura manuale di riferimento (errore mediano 0.0 % contro il gold standard, +2.5 % per la tangente).

**Metodo tangente** (default fino alla v3.5.x): trova il punto di massima pendenza discendente dopo il picco e traccia la tangente fino all'intersezione con la baseline; misura la fine dell'onda, quindi qualche punto percentuale più lungo del picco.

**Quale onda.** Prima del punto di misura conta la scelta dell'onda: con ripolarizzazioni bifasiche (lobo positivo e negativo vicini) la regola `prefer_positive` (default) prende il lobo positivo se ha almeno metà della prominenza del più grande ed è entro 400 ms; `max_prominence` è il comportamento storico.

**Metodo peak**: Identifica il picco di ripolarizzazione (deflessione positiva o negativa dopo lo spike). Più semplice ma meno preciso.

**Metodo max_slope**: Usa il punto di massima pendenza come endpoint diretto.

**Metodo 50pct**: Punto al 50% dell'ampiezza di ripolarizzazione (analogo all'APD50).

**Metodo baseline_return**: Punto in cui il segnale ritorna alla baseline post-ripolarizzazione.

**Metodo consensus**: Esegue tutti i metodi e seleziona il risultato più concordante (cluster analysis con finestra ±50 ms).

#### Correzione Fridericia

Il FPD dipende dalla frequenza cardiaca. La correzione di Fridericia normalizza per il periodo del battito:

```
FPDcF = FPD / RR^(1/3)
```

dove RR è il periodo inter-battito in secondi. Questa è la correzione standard per i FP cardiaci (preferita a Bazett che sovra-corregge a frequenze lente).

#### Confidence scoring

Ogni misura di FPD ha un punteggio di confidenza (0–1) basato su:

- **Prominenza del picco di ripolarizzazione** (60% del peso): rapporto prominenza/rumore, saturato a 3×
- **Concordanza tra metodi** (40%): spread degli endpoint tra i diversi metodi

A livello di registrazione, la confidenza FPD combina la confidenza del template (50%) e la consistenza beat-to-beat (50%, basata sul CV degli FPD individuali).

#### Parametri estratti per battito

| Parametro | Unità | Descrizione |
|-----------|-------|-------------|
| `spike_amplitude_mV` | mV (o µV con gain) | Ampiezza picco-picco dello spike |
| `rise_time_ms` | ms | Tempo di salita 10–90% |
| `fpd_ms` | ms | Field Potential Duration |
| `fpdc_ms` | ms | FPD corretto (Fridericia) |
| `repol_amplitude_mV` | mV | Ampiezza del picco di ripolarizzazione |
| `rr_interval_ms` | ms | Intervallo inter-battito |
| `max_dvdt` | mV/ms | Velocità massima di depolarizzazione |

| Config | Default | Descrizione |
|--------|---------|-------------|
| `fpd_method` | 'peak' | Metodo di misurazione FPD (era 'tangent' fino alla v3.5.x) |
| `repol_candidate_rule` | 'prefer_positive' | Scelta dell'onda di ripolarizzazione ('max_prominence' = storico) |
| `window_rr_from_template_beats` | True | Finestra di ricerca con l'RR dei battiti del template |
| `correction` | 'fridericia' | Formula di correzione |
| `max_beats_template` | 60 | N. max battiti per il template |
| `search_start_ms` | 150 | Inizio ricerca ripolarizzazione |
| `search_end_ms` | 900 | Fine ricerca ripolarizzazione |
| `tangent_max_slope_window_ms` | 300 | Finestra per max pendenza |
| `tangent_max_extension_ms` | 400 | Estensione max della tangente |

---

### 4.5 Controllo qualità

**Modulo**: `quality_control.py`

Il QC valida ogni battito individualmente e assegna un grado di qualità complessivo alla registrazione.

#### Validazione per battito

Ogni battito viene valutato su due criteri:

1. **Ampiezza**: Il battito viene rifiutato se la sua ampiezza è < 25% dell'ampiezza di riferimento (mediana del 50% superiore dei battiti). Questo elimina battiti mancati, artefatti deboli, e battiti con coupling.

2. **Morfologia**: Correlazione di Pearson con il template. Soglia nominale: 0.40 (configurabile). Questo elimina battiti con morfologia aberrante (artefatti, ectopie marcate).

#### Soglia morfologica adattiva (v3.4)

Se la soglia morfologica fissa rigetta più del 40% dei battiti, il sistema abbassa automaticamente la soglia in base alla regolarità del timing (CV dei beat period):

- **CV < 20%** (timing molto regolare): i battiti sono quasi certamente reali nonostante la bassa correlazione. Si usa il 5° percentile delle correlazioni (conserva ~95% dei battiti).
- **CV 20-35%** (timing ragionevolmente regolare): 15° percentile (~85% conservati).
- **CV > 35%** (timing irregolare): 30° percentile (~70% conservati).

La soglia non scende mai sotto 0.10 (floor di sicurezza). Questa logica è importante per i segnali µECG/MEA dove l'ampiezza dei battiti varia naturalmente tra cicli, risultando in correlazioni con il template relativamente basse (mediana ~0.3-0.4) anche quando il segnale è di buona qualità.

#### Beat period vs. parametri (v3.4)

Il beat period (BP) viene calcolato su **tutti** i battiti rilevati (post-correzione bimodale), non solo su quelli accettati dal QC morfologico. Il rationale: anche un battito con morfologia aberrante ha un timing corretto, e rimuoverlo crea un gap artificiale (BP raddoppiato). I parametri di ripolarizzazione (FPD, ampiezza spike) vengono invece estratti solo dai battiti QC-accettati, poiché richiedono una morfologia affidabile.

#### Grading della registrazione

| Grado | SNR | Condizioni |
|-------|-----|------------|
| A (Eccellente) | ≥ 8.0 | Tasso di rifiuto < 5% |
| B (Buono) | ≥ 5.0 | Tasso di rifiuto < 20% |
| C (Discreto) | ≥ 3.0 | Tasso di rifiuto < 40% |
| D (Scarso) | ≥ 2.0 | O tasso di rifiuto > 40% |
| F (Non analizzabile) | — | < 3 battiti accettati |

L'SNR globale è calcolato come rapporto tra ampiezza media dei picchi e deviazione standard delle regioni inter-battito.

**Razionale**: Il grading combina SNR e tasso di rifiuto perché un segnale può avere alto SNR ma molti battiti aberranti (effetto farmaco), o basso SNR ma battiti consistenti (segnale debole ma stabile).

---

### 4.6 Analisi aritmie

**Modulo**: `arrhythmia.py`

L'analisi delle aritmie è il cuore del sistema. Combina due approcci complementari: analisi statistica dei parametri e analisi residual-based. Il residuo viene dal paper di riferimento, che lo usava solo per individuare picchi irregolari (eventi aritmici candidati); i punteggi calcolati qui sono del software.

#### Approccio statistico

Valuta le proprietà globali del ritmo e dei parametri:

- **Tachicardia/bradicardia**: Periodo medio < 300 ms o > 2500 ms
- **Irregolarità RR**: CV del periodo > 15% (irregolare), > 30% (critico), > 40% (fibrillazione-like)
- **Battiti prematuri/ritardati**: Singoli battiti < 70% o > 150% del periodo medio
- **STV (Short-Term Variability)**: Variabilità beat-to-beat del periodo, calcolata con il diagramma di Poincaré: STV = mean|x_{i+1} - x_i| / √2
- **Prolungamento FPD**: FPD > 130% del baseline o > 500 ms assoluti
- **Instabilità d'ampiezza**: CV dell'ampiezza > 30%

#### Approccio residual-based (residuo come in Visone et al. 2023)

Questo approccio calcola il residuo tra ogni battito e un template di riferimento:

```
residuo = battito − template
```

Il residuo contiene solo le deviazioni dalla morfologia normale. In condizioni fisiologiche i residui sono piccoli (jitter termico); con farmaci proaritmici i residui crescono (cambio morfologico, EAD, instabilità).

Nel paper il residuo (segnale registrato meno il segnale ricostruito dai pattern medi) serviva a individuare picchi inattesi e irregolari, poi riportati come eventi aritmici presenti o assenti per microtessuto. Instabilità morfologica, incidenza EAD, CV d'ampiezza e indice proaritmico non sono nel paper: sono definiti in questo software.

#### Baseline-relative residual analysis (v3.3)

**Innovazione chiave della v3.3**: Il template di riferimento può provenire dal baseline (registrazione senza farmaco) anziché dalla stessa registrazione. Questo cambia radicalmente il significato del residuo:

- **Intra-recording** (v3.2): residuo = battito − template_stessa_registrazione → cattura solo il jitter beat-to-beat
- **Baseline-relative** (v3.3): residuo = battito_con_farmaco − template_baseline → cattura le deviazioni morfologiche indotte dal farmaco

L'implementazione è a due passaggi: nella prima pass si analizzano tutti i file e si memorizzano i template dei baseline per gruppo (chip+camera+elettrodo); nella seconda pass si ri-esegue l'analisi aritmia per i drug recording usando il template del baseline corrispondente.

#### Metriche dal residuo

**Morphology instability** (0–1): RMS medio del residuo diviso per l'ampiezza picco-picco del template, mappato su 0–1 (5 % → 0,12; 15 % → 0,5; 30 % → 0,88). Rispetto al template del baseline misura quanto è cambiata la forma del battito, per qualunque motivo (FPD diverso, spike più piccolo, rumore), non l'instabilità battito per battito. Il rapporto 2,2× tra positivi e negativi citato nella v3.3 veniva dal massimo su 7 composti. Sui 12 composti del paper, per concentrazione, non li distingue (AUC 0,47).

**EAD detection** (Early Afterdepolarization): Rileva depolarizzazioni secondarie nella fase di ripolarizzazione (150–500 ms post-spike). Cinque criteri simultanei:

1. **Statistico**: il picco nel residuo supera 6× la deviazione standard del rumore (σ stimata dal MAD del residuo completo)
2. **Ampiezza assoluta**: il picco supera l'8% dell'ampiezza picco-picco del template
3. **Larghezza**: la larghezza a metà altezza è tra 8 e 150 ms (esclude spike artefattuali e ondulazioni lente)
4. **Polarità**: solo picchi positivi (le EAD sono depolarizzazioni secondarie)
5. **Localizzazione**: nella finestra di ripolarizzazione (150–500 ms post-spike)

**Razionale dei 5 criteri**: Un criterio statistico da solo genera troppi falsi positivi perché il rumore di fondo ha una distribuzione non-gaussiana. I criteri di ampiezza, larghezza e localizzazione aggiungono specificità biofisica.

**Poincaré STV**: Variabilità a breve termine di FPD e FPDcF, calcolata dal diagramma di Poincaré.

#### Risk score (0–100) — incidence-based scoring

Il risk score è un punteggio composito basato su metriche normalizzate per incidenza, non su conteggi assoluti di eventi. Questo è conforme alla letteratura (Thomsen 2004, Hondeghem 2001, Blinova 2017): tutte le metriche sono rate-based o per-beat, quindi una registrazione di 30 secondi con 2 battiti prematuri su 15 (13%) riceve correttamente un punteggio più alto di una registrazione di 5 minuti con 2 prematuri su 150 (1.3%).

Le 7 componenti del risk score:

1. **Irregolarità ritmica** (0–20 punti): basata sul CV del beat period. CV < 10% = 0 punti. CV = 40% = 20 punti. Scala lineare nell'intervallo [10%, 40%].

2. **Incidenza battiti anomali** (0–10 punti): percentuale di battiti prematuri + ritardati rispetto al totale. 10% di battiti anomali = 10 punti (massimo). La metrica è intrinsecamente normalizzata per la durata della registrazione.

3. **Morphology instability** (0–20 punti): punteggio 0–1 dal residuo, normalizzato per l'ampiezza del template. Mappatura lineare: instabilità 0.5 = 10 punti, 1.0 = 20 punti. Con baseline-relative analysis (v3.3), questa metrica è discriminatoria tra farmaci positivi e negativi.

4. **EAD incidence** (0–20 punti): percentuale di battiti con eventi EAD-like. 10% di battiti con EAD = 20 punti (massimo). Non dipende dal numero assoluto di EAD ma dalla loro frequenza relativa.

5. **Amplitude instability** (0–10 punti): CV dell'ampiezza dello spike, definito in questo software (il paper riporta la variazione dell'ampiezza tra concentrazioni, non la sua variabilità). CV < 10% = 0 punti. CV = 40% = 10 punti. Cattura alterazioni della depolarizzazione indotte dal farmaco: blocco hERG (triangolazione del potenziale d'azione), blocco canali Ca²⁺ L-type (riduzione d'ampiezza, come con nifedipina), degradazione progressiva del tessuto. È una metrica statistica (CV), intrinsecamente indipendente dalla durata.

6. **Poincaré STV** (0–10 punti): variabilità a breve termine di FPDcF in ms. STV ≤ 5 ms = 0 punti. STV = 20 ms = 10 punti. La STV è per definizione una metrica beat-to-beat (mean|x_{i+1} - x_i| / √2), indipendente dalla durata.

7. **Cessazione** (0 o 10 punti): presenza di pause > 3× il periodo medio. Binario.

Il punteggio massimo è 100 (cap). Le registrazioni baseline ricevono sempre risk_score = 0 con classificazione "Baseline (reference)", poiché il risk score è definito come rischio proaritmico indotto dal farmaco e non ha significato senza trattamento.

#### Classificazione

La classificazione testuale è anch'essa basata su metriche di incidenza. La gerarchia, dalla più grave alla meno grave:

1. Fibrillation-like / Chaotic Rhythm — cessazione + CV > 40%
2. EAD with Triggered Activity — EAD incidence > 20% dei battiti
3. Proarrhythmic (EAD-prone) — EAD > 5% + CV > 15%
4. Morphologically Unstable + Irregular — instabilità > 0.6 + CV > 15%
5. Intermittent Cessation — pause rilevate
6. Highly Irregular Rhythm — CV > 30%
7. Morphologically Unstable — instabilità > 0.6
8. Frequent Premature Beats — prematuri > 10% dei battiti
9. Irregular Rhythm — CV > 15%
10. Tachycardia / Bradycardia — BP < 300 ms o > 2500 ms
11. Borderline / Mild Abnormalities — warning flag presenti
12. Normal Sinus Rhythm — nessuna anomalia

| Config | Default | Descrizione |
|--------|---------|-------------|
| `tachycardia_bp_ms` | 300 | Soglia tachicardia |
| `bradycardia_bp_ms` | 2500 | Soglia bradicardia |
| `rr_irregularity_cv` | 15.0 | CV% irregolarità |
| `rr_critical_cv` | 30.0 | CV% critico |
| `fibrillation_cv` | 40.0 | CV% fibrillazione-like |
| `ead_residual_prominence` | 6.0 | Prominenza (×σ) |
| `ead_residual_min_amp_frac` | 0.08 | Ampiezza min (% template) |
| `ead_residual_min_width_ms` | 8.0 | Larghezza min EAD |
| `ead_residual_max_width_ms` | 150.0 | Larghezza max EAD |
| `risk_score_mode` | `manual` | `manual` o `data_driven` |

#### Modalità di scoring: manual vs data-driven

Il sistema supporta due modalità di calcolo del risk score, selezionabili tramite `ArrhythmiaConfig.risk_score_mode`:

**`manual`** (default): pesi assegnati da esperto sulla base della letteratura fisiologica. Ogni componente ha un peso massimo fisso (somma = 100) con soglie e scale lineari. È la modalità raccomandata perché il risk score per-registrazione ha uno scopo diverso dalla classificazione farmacologica: quantifica quanto un singolo segnale appaia anormale, indipendentemente dal farmaco.

**`data_driven`** (sperimentale): utilizza un modello di regressione logistica addestrato sul dataset CiPA a 7 farmaci (131 registrazioni etichettate, 7 farmaci di validazione). Il modello produce una probabilità P(proaritmico) mappata a 0–100.

Risultati della calibrazione data-driven (Leave-One-Drug-Out cross-validation):

La regressione logistica per-registrazione ha un potere discriminatorio limitato (AUC-ROC < 0.5 in LODO CV). Questo è atteso perché la classificazione CiPA opera a livello di farmaco (analisi dose-risposta), non di singola registrazione: una registrazione di dofetilide a 0.3 nM è indistinguibile da un controllo negativo. Le feature più discriminatorie a livello di registrazione (Cohen's d al massimo della concentrazione) sono: CV beat period (d=+0.57), ampiezza CV (d=+0.47), e percentuale battiti anomali (d=+0.47). EAD incidence risulta paradossalmente più alta nei farmaci negativi (mexiletina d=-0.39), confermando che gli EAD a livello di registrazione non sono specifici per farmaci proaritmici.

Implicazione pratica: per la classificazione farmacologica, il sistema di normalizzazione TdP (Sezione 6) che analizza la dose-risposta di FPDcF rimane l'approccio più affidabile. Il risk score per-registrazione è utile come indicatore di qualità del segnale e per identificare registrazioni con aritmie evidenti, non per predire la classe del farmaco. I pesi data-driven sono mantenuti come opzione per futuri dataset più ricchi.

---

### 4.7 Rilevamento cessazione

**Modulo**: `cessation.py`

Rileva l'arresto dell'attività di battito, un effetto grave di alcuni farmaci. Utilizza cinque sub-detector complementari.

#### Sub-detector

1. **Energy silence** (peso 35%): Calcola l'energia RMS in finestre temporali scorrevoli (2 s con passo 0.5 s). Una regione è "silente" se l'energia < 15% dell'energia baseline. Serve una durata minima di 3 s di silenzio.

2. **Gap detection** (peso 25%): Identifica gap > 3× il periodo mediano tra battiti consecutivi, con durata minima 2 s. Cattura le pause isolate.

3. **Deterioramento progressivo** (peso 20%): Divide la registrazione in 4 segmenti e misura il trend di ampiezza. Se l'ampiezza del segmento finale è < 40% del primo, indica deterioramento.

4. **Cessazione terminale** (peso 20%): Verifica se l'ultimo 20% della registrazione è privo di battiti (silenzio > 5 s). Cattura il pattern classico di cessazione a fine recording.

5. **Waveform destruction** (bonus 15%): Basato sul QC — se un recording ha grado D/F con beat detection iniziale che trova battiti ma il QC ne rifiuta > 50%, indica distruzione del segnale.

#### Tipi di cessazione

| Tipo | Descrizione |
|------|-------------|
| `none` | Nessuna cessazione |
| `intermittent` | Pause isolate ma attività riprende |
| `terminal` | Battito si ferma nella parte finale |
| `progressive` | Ampiezza cala progressivamente |
| `full` | Cessazione completa |
| `waveform_destruction` | Il farmaco distrugge la morfologia |

**Razionale**: La cessazione è un endpoint critico nel framework CiPA ma non viene rilevata dall'analisi tradizionale del FPD (che richiede battiti per funzionare). Il paper originale non aveva un modulo dedicato; il nostro sistema lo aggiunge come contributo originale.

---

### 4.8 Analisi spettrale

**Modulo**: `spectral.py`

Analizza il contenuto in frequenza del segnale FP, fornendo metriche complementari al dominio temporale.

#### Metriche calcolate

**PSD Welch**: Power Spectral Density stimata con il metodo di Welch (segmenti di 4 s, overlap 50%, finestra Hann).

**Bande di potenza**:

| Banda | Range (Hz) | Significato |
|-------|-----------|-------------|
| Low | 0.5–5 | Drift respiratorio, artefatti lenti |
| Beat | 0.3–3.5 | Frequenza del battito e armoniche basse |
| Repol | 5–30 | Componenti della ripolarizzazione |
| High | 30–200 | Rumore, spike |

**Frequenza fondamentale**: Picco dominante nella banda 0.3–4 Hz (corrispondente a 18–240 BPM).

**Entropia spettrale** (0–1): Misura quanto il contenuto in frequenza è distribuito (0 = tono puro, 1 = rumore bianco). Un aumento indica disorganizzazione del ritmo.

**Struttura armonica**: Numero di armoniche rilevate della frequenza fondamentale e rapporto armonico. Un pattern armonico pulito indica battiti regolari con morfologia stabile.

**Centroide e bandwidth spettrale beat-level**: Calcolati dagli spettri dei singoli battiti, catturano la complessità morfologica.

**Confronto vs baseline** (calcolato nella normalizzazione): Correlazione spettrale e divergenza KL tra lo spettro del farmaco e quello del baseline. Questa è la metrica più discriminatoria: i farmaci hERG+ alterano profondamente la morfologia della ripolarizzazione (T-wave broadening, EAD, U-waves), causando uno shift spettrale misurabile.

**Razionale**: L'analisi spettrale è un contributo originale non presente nel paper MATLAB. La spectral change score è il singolo miglior discriminatore tra farmaci positivi (media 0.654) e negativi (media 0.214), con un rapporto 3×.

---

### 4.9 Criteri di inclusione (batch)

**Modulo**: `analyze.py` → `_apply_inclusion_criteria()`

In modalità batch, prima della normalizzazione viene applicata una cascata di criteri di inclusione sulle registrazioni baseline. Se una baseline fallisce, tutte le registrazioni farmaco associate allo stesso chip+camera+elettrodo vengono escluse dall'analisi (perché la normalizzazione vs baseline non è affidabile).

#### I 5 criteri (in ordine, short-circuit al primo fallimento)

1. **CV del beat period** (default: < 25%): esclude baselines con ritmo troppo irregolare. È il criterio più importante — un baseline instabile rende il ΔFPDcF inaffidabile. Corrisponde al criterio di Visone et al. 2023.

2. **Range di plausibilità FPDcF** (default: 100–1200 ms): safety net per escludere misurazioni chiaramente erronee (artefatti di detection che producono FPDcF impossibili).

3. **Confidenza FPD** (default: ≥ 0.66): esclude baselines dove l'algoritmo di misura del FPD non è affidabile. Il valore di confidenza (0–1) è calcolato dal consenso tra metodi multipli di misura della ripolarizzazione.

4. **Range fisiologico FPDcF** (opt-in, default: 350–800 ms): esclude baselines con FPDcF fuori dal range fisiologico atteso per hiPSC-CM. Più restrittivo del criterio 2, utile per dataset ben caratterizzati.

5. **Outlier nella popolazione** (opt-in, default: > 2σ dalla mediana dell'esperimento): esclude baselines con FPDcF statisticamente anomalo rispetto alle altre baselines dello stesso esperimento. Usa MAD (median absolute deviation) per robustezza.

#### Logica a cascata

Quando una baseline fallisce un criterio, il sistema:
- Marca la baseline come esclusa (con la ragione specifica)
- Identifica il gruppo chip+camera+elettrodo corrispondente
- Esclude tutte le registrazioni farmaco di quel gruppo

Le registrazioni farmaco possono anche essere segnalate individualmente per bassa confidenza FPD o FPDcF implausibile, anche se la baseline è valida.

#### Configurazione

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `max_cv_bp` | 25.0 | Max CV% del beat period per la baseline |
| `enabled_cv` | true | Attiva/disattiva criterio 1 |
| `fpdc_range_min` | 100.0 | FPDcF minimo plausibile (ms) |
| `fpdc_range_max` | 1200.0 | FPDcF massimo plausibile (ms) |
| `enabled_fpdc_range` | true | Attiva/disattiva criterio 2 |
| `min_fpd_confidence` | 0.66 | Confidenza FPD minima |
| `enabled_confidence` | true | Attiva/disattiva criterio 3 |
| `enabled_fpdc_physiol` | false | Attiva criterio 4 (opt-in) |
| `fpdc_physiol_min` | 350.0 | Range fisiologico min (ms) |
| `fpdc_physiol_max` | 800.0 | Range fisiologico max (ms) |
| `enabled_fpdc_outlier` | false | Attiva criterio 5 (opt-in) |
| `fpdc_outlier_n_sigma` | 2.0 | Soglia per outlier (×σ) |

#### Razionale

La qualità dei dati µECG è variabile: chip con cattivo contatto, microtessuti non vitali, artefatti meccanici. Piuttosto che analizzare dati inaffidabili (introducendo rumore nei risultati farmaco), è preferibile escludere a priori le registrazioni problematiche. Il criterio del CV è lo stesso usato nel paper di riferimento (Visone et al. 2023, Supplementary Methods).

---

### 4.10 Normalizzazione e classificazione

**Modulo**: `normalization.py`

#### Pairing baseline–farmaco

Ogni registrazione di farmaco viene abbinata a un baseline **dello stesso tessuto** (esperimento + giorno + chip + camera, vedi `describe_recording`) e dello stesso elettrodo.

- **Più riferimenti nello stesso tessuto (dalla v3.8):** `baseline` e `t0`, `baseline_1` e `baseline_2`, una cartella `new baseline/`, un `_bis`. Se tutti i riferimenti e almeno una dose hanno l'orario nell'intestazione (`#Date Time:`), il tessuto usa **l'ultimo registrato prima della prima dose**. Riferimenti registrati dopo, come un baseline dopo il lavaggio, non vengono usati.
  - Sul dataset Visone 2023 questa regola riproduce la scelta degli autori in tutti i casi verificati tranne uno: t0 nei protocolli prima del 2020, `baseline_2` in exp4, `baseline_bis` in Exp7 chipD. L'eccezione è Exp7 chipE ch2, dove gli autori usano il baseline registrato prima di `new baseline/`.
- **Senza orari:** si preferisce t0, poi il riferimento nella **stessa cartella** della registrazione, poi il grado QC migliore, poi il nome del file.
- **Baseline esclusi:** quelli che il verdetto di analizzabilità rifiuta non vengono mai usati. Se il baseline scelto non supera i criteri di inclusione, la registrazione resta senza abbinamento.
- **Motivazione registrata:** la scelta e il motivo finiscono in `normalization['pairing']` (`baseline_file`, `reason`, `candidates`).
- **Chiave di abbinamento:** l'abbinamento è indicizzato per percorso completo più elettrodo (`recording_key`), non più per nome del file. Così due file omonimi in cartelle diverse, o le analisi el1 ed el2 dello stesso file in modalità `both`, non si sovrascrivono.
- **File con due tessuti** (un tessuto per ingresso, convenzione GG): dalla v3.9 il batch `auto` li analizza come una registrazione per ingresso, ognuna abbinata al proprio tessuto (vedi 4.1). Analizzati come un'unica registrazione (`el1`, `el2` o `both` espliciti) non vengono abbinati, e il motivo resta registrato.

**Un elettrodo per tessuto** (`batch_analyze`, `channel='auto'`, dalla v3.7): i baseline vengono analizzati per primi con la scelta automatica. Ogni tessuto adotta l'elettrodo scelto per il suo riferimento, con la stessa regola dell'abbinamento (dalla v3.8: l'ultimo prima della prima dose; senza orari t0, poi la cartella che contiene la maggior parte delle sue registrazioni). Su quell'elettrodo vengono poi analizzate dosi e controlli. Gli eventuali altri baseline del tessuto vengono rianalizzati sullo stesso elettrodo. `file_info['tissue_electrode']` e `['tissue_electrode_from']` riportano la scelta.

Prima, la selezione file per file faceva sì che una serie dose-risposta mescolasse el1 ed el2 dello stesso microtessuto: è successo in 39 registrazioni su 75 del dataset Visone 2023.

**Fallback sull'altro elettrodo**: se il gruppo dell'elettrodo non ha un baseline, cosa possibile con `el1`/`el2` espliciti, si usa quello dello stesso tessuto sull'altro elettrodo, e `pairing.reason` lo segnala. In modalità `auto` questo caso non si presenta più.

**Arresto (dalla v3.7)**: se il baseline o la registrazione di farmaco hanno un periodo di battito oltre `NormalizationConfig.max_beat_period_for_fpdc_ms` (6 s, cioè 10 battiti/min), la %ΔFPDcF non viene calcolata e il motivo va in `fpdc_withheld`. Con RR di decine di secondi la correzione di Fridericia non ha senso: una cisapride a RR = 40 s dava +580 %. Le variazioni di BP e di ampiezza restano, e l'override di arresto vede comunque la registrazione.

#### Parametri normalizzati

Per ogni recording con baseline, si calcolano:

- **ΔBP%**: variazione percentuale del periodo rispetto al baseline
- **ΔFPDcF%**: variazione percentuale del FPDcF corretto rispetto al baseline
- **ΔAmpiezza%**: variazione percentuale dell'ampiezza dello spike

#### TdP score (Ando et al. 2017)

| Score | Condizione |
|-------|-----------|
| −1 | Accorciamento significativo (ΔFPDcF < −10%) |
| 0 | Nessun effetto significativo |
| 1 | Prolungamento lieve (10–15%) |
| 2 | Prolungamento moderato (15–20%) |
| 3 | Prolungamento severo (≥ 20%), cessazione, o aritmia severa con prolungamento |

#### Spectral change score

Confronto spettrale farmaco vs baseline nella banda 1–50 Hz:

```
spectral_change = 1 − correlazione_spettrale
```

Valore 0 = identico al baseline, 1 = completamente diverso. I farmaci hERG+ mostrano score 0.5–0.7, i negativi < 0.35.

#### Smart cessation override

Se un farmaco mostra cessazione con bassa confidenza FPD (< 0.60), la condizione viene riportata in `cessation_flag` della classificazione. Dalla v3.8 rende positivo il farmaco solo con `enable_cessation_override = True`; il default è spento. Sul dataset Visone 2023 la condizione vale per tre composti (ranolazina, mexiletina, aspirina) e cambia una sola decisione: l'aspirina, negativa, diventerebbe positiva. Nessun composto positivo viene recuperato. Il paper riporta la cessazione a parte, nella colonna "Stop".

> **Correzione (v3.8.2)**: le note della v3.8.0 riportavano "10 composti su 12, compresi quattro negativi". Quel conto usava `has_cessation` senza la soglia di confidenza (> 0,5) che la decisione applica; con la soglia i composti sono tre.

#### Filtri QC sulla normalizzazione

Le drug recording con segnale di bassa qualità possono produrre valori FPDcF inaffidabili, causando falsi positivi nella classificazione CiPA. Due filtri opzionali escludono queste recording dalla classificazione drug-level (la normalizzazione individuale resta visibile nel report):

| Config | Default | Descrizione |
|--------|---------|-------------|
| `norm_min_qc_enabled` | `False` | Attiva filtro QC grade minimo |
| `norm_min_qc_grade` | `'D'` | Grade minimo (A > B > C > D > F). Con `'C'`, solo A/B/C contribuiscono alla classificazione |
| `norm_max_cv_enabled` | `False` | Attiva filtro CV massimo |
| `norm_max_cv_bp` | `50.0` | % — recording con CV(BP) superiore sono escluse |

**Esempio**: ranolazine mostra prolungamento apparente (+44%) a 100µM, ma QC=D con 51% dei battiti scartati. Attivando `norm_min_qc_grade='C'`, questa concentrazione viene esclusa, e le sole concentrazioni QC≥C (0.1µM, 1µM) mostrano correttamente nessun effetto (+3-4%).

#### Classificazione farmaco

Quattro metodi per passare dalle registrazioni alla decisione sul farmaco (`classification_method`):

- **concentration** (default dalla v3.8). Per ogni concentrazione si fa la media della %ΔFPDcF tra i tessuti; registrazioni ripetute dello stesso tessuto alla stessa concentrazione contano una volta sola.
  - Una concentrazione conta solo se è misurata in almeno `classification_min_tissues` tessuti (2).
  - Il farmaco è positivo quando la media raggiunge la soglia in `classification_consecutive` concentrazioni adiacenti (2).
  - Con 1 e 1 è la regola del paper Visone 2023: media tra tessuti sopra soglia a una concentrazione qualsiasi.
  - Se meno di `consecutive` concentrazioni hanno abbastanza tessuti, la decisione è `insufficient data`, non negativa.
- **max** (default fino alla v3.7): positivo se una registrazione qualsiasi supera la soglia.
- **mean**: positivo se la media di tutte le registrazioni supera la soglia. Dipende da quante concentrazioni basse sono state provate: perde i composti che agiscono solo alle dosi alte.
- **n_above**: positivo se almeno N registrazioni superano la soglia.

**Concentrazioni:** l'etichetta del nome diventa un numero (`normalization.concentration_value`).
- `300 nM` e `0.3 uM` sono la stessa concentrazione.
- Un numero senza unità prende l'unità più frequente del farmaco.
- `001` vale 0,01 e `05` vale 0,5 (convenzione Accelera); `10-1` vale 0,1.
- Le lettere delle condizioni (A, B, C) seguono l'ordine alfabetico.
- Le registrazioni con un'etichetta illeggibile restano fuori dalla decisione per concentrazione, con un avviso nel log.

La classificazione riporta:
- `decision` e `positive`;
- `per_concentration`: media, numero di tessuti e se la concentrazione è usata;
- `effective_concentration`: la prima concentrazione della coppia che decide;
- `n_tissues`, `cessation_flag` e `cessation_info`.

**Perché è cambiato il default (ottobre 2026).** La regola è stata scelta sui 12 composti del paper Visone 2023, con l'etichetta FDA come verità, soglia fissa al 15 % e nessuna taratura. Il veicolo è valutato come un composto negativo. Le regole sono state confrontate su tre misure:

1. le variazioni degli autori, prese dai loro workbook (54 tessuti). Riproducono esattamente la Tabella 2 del paper per 7 composti;
2. quelle del software con riferimenti e concentrazioni assegnati a mano (49 tessuti utilizzabili);
3. il batch v3.8 sui file così come sono.

| Regola | Valori autori | Software, assegnazione a mano | Batch v3.8 |
|---|---|---|---|
| max | 5/12 | 6/12 | 6/12 |
| mean | 10/12 | 9/12 | 8/12 |
| paper (media per concentrazione) | 10/12 | 4/12 | 6/12 |
| **concentration (≥ 2 tessuti, 2 concentrazioni consecutive)** | **11/12** | **8/12** | **7/12** |

- `max` sbaglia tutti i negativi, veicolo compreso, con tutte e tre le misure. Con 18–54 registrazioni per composto, almeno una supera il 15 % per rumore.
- La regola del paper funziona sui valori degli autori, ma sul software una sola registrazione anomala domina la media di una concentrazione con pochi tessuti.
- Le due protezioni la rendono robusta: con i valori degli autori sbaglia solo la cisapride, che nel modello accorcia l'FPDcF, come nel paper.
- Nel batch i nomi dei file pesano:
  - la dose di sotalolo da 7,5 µM è scritta `7`, `75`, `7,5` e `7.5` in esperimenti diversi, quindi non si allinea tra tessuti;
  - alcuni file di Exp8 ed Exp5 hanno l'unità sbagliata (`Alfus_1000uM` per nM, `Quinidine0_06nM` per µM): il software lo segnala nel log;
  - la terfenadina manca la soglia di 0,2 punti (14,8 % a 300 nM, 10 tessuti).
- Con 12 composti, differenze di 1–2 composti restano nel rumore. `mean` resta disponibile ed è vicino sul software, ma dipende da quante concentrazioni basse sono state provate.
- Con tre livelli di concentrazione (A, B, C) la regola richiede l'effetto sia a B sia a C.

**Nomi dei farmaci (dalla v3.7)**: le registrazioni vengono raggruppate per nome canonico (`canonical_drug_name`: `DOFE` → `dofetilide`, `Quinid` → `quinidine`, `NIFEDIPINE 10` → `nifedipine`). I codici come `Ti07` restano invariati. Prima il raggruppamento usava la stringa grezza, e lo stesso farmaco poteva risultare positivo con un'abbreviazione e negativo con un'altra: sul dataset Visone 2023, 8 farmaci diventavano 17.

**Registrazioni escluse dalla decisione** (la loro %Δ resta nella tabella di normalizzazione):

- washout e recovery, perché non sono una concentrazione;
- il veicolo (`dmso`, `vehicle`), perché non è un farmaco da classificare;
- i controlli nel tempo (`ctrl`, `t1`…`t7`).

| Config | Default | Descrizione |
|--------|---------|-------------|
| `threshold_low` | 10.0% | Soglia TdP score 1 |
| `threshold_mid` | 15.0% | Soglia TdP score 2 (ottimale, paper) |
| `threshold_high` | 20.0% | Soglia TdP score 3 |
| `classification_threshold` | `'mid'` | Soglia per classificazione positivo |
| `classification_method` | `'concentration'` | Metodo di aggregazione |
| `classification_min_tissues` | `2` | Tessuti minimi per concentrazione (metodo `concentration`) |
| `classification_consecutive` | `2` | Concentrazioni adiacenti sopra soglia (metodo `concentration`) |
| `classification_n_above` | `2` | N minimo per metodo `n_above` |
| `enable_cessation_override` | `False` | La cessazione con bassa confidenza FPD rende positivo il farmaco |

---

### 4.11 Risk map CiPA

**Modulo**: `risk_map.py`

Genera una mappa di rischio 2D nello stile CiPA, posizionando ogni farmaco su due assi:

**Asse X — ΔFPDcF della decisione per farmaco (dalla v3.8.1)**: il numero che `classify_drug` confronta con la soglia (`decision_value`). Con il metodo predefinito è il livello che la media tra tessuti mantiene su 2 concentrazioni adiacenti; con `mean` è la media, con `max` e `n_above` il massimo. La linea verticale continua è la soglia della decisione (15 %), quindi un farmaco a destra della linea è positivo.
- La decisione viene ricalcolata sui risultati passati alla mappa: unendo più batch, ogni farmaco usa tutti i suoi tessuti.
- Il veicolo viene posizionato con la stessa statistica.
- I farmaci senza decisione (meno di 2 tessuti per concentrazione) compaiono vuoti in una fascia a sinistra, "no decision".
- Fino alla v3.8.0 l'asse era la variazione massima di una singola registrazione: sul dataset Visone 2023 metteva oltre la soglia tutti i composti negativi.

**Asse Y — Indice proaritmico (0–100)**: dalla v3.8.3, per default, è il cambio spettrale della forma d'onda rispetto al baseline (0–1 × 100). Le altre due componenti restano calcolate e riportate nelle tabelle e nell'export CDISC, e si possono ripesare con `compute_proarrhythmic_index(m, w_spec, w_morph, w_ead)`:

| Componente | Peso dalla v3.8.3 | Peso fino alla v3.8.2 | Note |
|------------|------|------|------|
| Spectral change | 100% | 70% | Cambio della forma d'onda nel dominio della frequenza rispetto al baseline. Sui 12 composti è l'unica componente che ordina positivi sopra negativi (AUC 0,86 da sola). |
| Morphology instability (baseline-relative) | 0% | 25% | Punteggio del software, non del paper (§4.6): misura quanto il battito differisce dal battito medio del baseline. Sui 12 composti non distingue positivi e negativi (AUC 0,47). |
| EAD incidence | 0% | 5% | Criteri del software sul residuo (§4.6). Sui 12 composti abbassa l'ordinamento (AUC 0,83 invece di 0,86) e toglie la cisapride dalla zona alta. |

**Aggregazione (dalla v3.8.2).** L'indice si calcola su ogni registrazione utilizzabile (abbinata, inclusa, analizzabile) e si aggrega come la decisione per farmaco:
- media tra tessuti a ogni concentrazione, con almeno 2 tessuti;
- livello mantenuto su 2 concentrazioni adiacenti;
- con meno tessuti l'indice non viene calcolato e il farmaco compare vuoto in basso.

Fino alla v3.8.1 ogni componente era il massimo su tutte le registrazioni: sul dataset Visone 2023 tutti i 12 composti, veicolo compreso, stavano sopra 40 (rischio alto). Con il cambio spettrale per concentrazione:
- rischio alto: cisapride, dofetilide, chinidina, ranolazina e verapamil;
- rischio basso: aspirina;
- tutti gli altri nella zona intermedia, veicolo compreso (32).

Limiti noti:
- La separazione resta debole: il verapamil, negativo, è in zona alta.
- Ricampionando i tessuti, le assegnazioni alla zona alta tengono nel 50–96 % dei casi: la cisapride nel 96 %, il verapamil nel 50 %.
- Le soglie delle zone (20 e 40) restano quelle della v3.2.

Il simbolo ⚡ segnala una cessazione con confidenza > 0,5, la stessa soglia della decisione per farmaco. Prima compariva su tutti i composti.

**Tre zone di rischio**:

| Zona | Y | Significato |
|------|---|-------------|
| LOW (verde) | < 20 | Nessun segnale proaritmico |
| INTERMEDIATE (giallo) | 20–40 | Effetti sospetti, richiede investigazione |
| HIGH (rosso) | > 40 | Alto rischio proaritmico |

Le linee verticali tratteggiate a X = 10% e 20% separano i livelli di prolungamento FPDcF; quella continua è la soglia della decisione.

#### Drug name normalization

I nomi dei farmaci dai filename vengono normalizzati automaticamente (es. 'terfe' → 'terfenadine', 'DOFE' → 'dofetilide', 'NIFE' → 'nifedipine') per aggregare correttamente le diverse notazioni.

---

### 4.12 Report

**Modulo**: `report.py`

#### Report Excel

Workbook multi-sheet:

- **Summary**: tutti i risultati con grado QC, parametri elettrofisiologici, criteri di inclusione, normalizzazione
- **Normalization**: confronti farmaco vs baseline, TdP score, classificazione
- **Flags**: flag aritmiche e eventi per file

Formattazione condizionale con colori per risk score e TdP score.

#### Report PDF

Report con grafici per ciascun file: tracciato temporale, forme d'onda sovrapposte, eventi aritmici.

---

## 5. Configurazione

Tutta la configurazione è centralizzata in `AnalysisConfig` (file `config.py`), che contiene sotto-configurazioni per ogni modulo.

### Configurazione JSON

```python
config = AnalysisConfig()
config.amplifier_gain = 1e4
config.to_json('my_config.json')

# Ricaricamento
config = AnalysisConfig.from_json('my_config.json')
```

### Preset disponibili

| Preset | Descrizione |
|--------|-------------|
| `default` | Parametri standard (FPD al picco, Fridericia, tutti i filtri) |
| `conservative` | Soglie più strette, meno falsi positivi |
| `sensitive` | Soglie più rilassate, meno falsi negativi |
| `peak_method` | Usa il metodo peak per FPD (più semplice) |
| `no_filters` | Disabilita i filtri (per debug) |

### Pesi di scoring configurabili (v3.2)

I pesi numerici usati per lo scoring automatico nella beat detection e nella channel selection sono ora campi delle rispettive dataclass (`BeatDetectionConfig`, `ChannelSelectionConfig`) anziché costanti hard-coded nel codice. Questo permette di tunarli via JSON senza modificare il sorgente:

```python
config = AnalysisConfig()
config.beat_detection.score_bp_ideal = 30       # punti per BP nel range ideale
config.beat_detection.score_bp_extended = 15    # punti per BP nel range esteso
config.channel_selection.w_bp_range = 15        # peso per BP nel range fisiologico
config.channel_selection.w_corr_max = 30        # peso max per correlazione media
config.to_json('custom_weights.json')
```

### Parametro amplifier_gain

Il sistema µECG-Pharma Digilent registra con un guadagno di 10⁴. Per ottenere i valori reali in Volt, il segnale viene diviso per il guadagno:

```
segnale_reale = segnale_ADC / 10⁴
```

Dalla v3.4.0 il default della libreria è `1e4` (prima era `1.0`: le due GUI lo forzavano a 10⁴, ma CLI, batch e Studi riportavano ampiezze 10⁴ volte troppo grandi e il gate assoluto `min_signal_amplitude_uV` non poteva mai scattare). Impostare `1.0` solo per dati già in Volt fisici.

Con `amplifier_gain = 1e4`, le ampiezze baseline risultano ~253 ± 92 µV, coerenti con il paper (251 ± 320 µV). Questo parametro influenza solo i valori assoluti di ampiezza, non le metriche temporali (FPD, BP) né le metriche relative (ΔFPDcF%).

---

## 6. Razionale scientifico

### Perché l'analisi residual-based?

Il paper di riferimento (Visone et al. 2023) usa i residui (segnale − template) per rilevare aritmie. Il vantaggio rispetto all'analisi parametrica classica è che i residui catturano qualsiasi deviazione dalla morfologia normale, incluse anomalie sottili che non si riflettono in parametri globali come il FPD medio.

### Perché il template baseline (v3.3)?

Con il template intra-recording (versione ≤ 3.2), il residuo cattura solo la variabilità beat-to-beat (jitter). Se un farmaco altera uniformemente tutti i battiti (es. allungamento uniforme della ripolarizzazione), il template intra-recording si adatta al nuovo pattern e i residui restano piccoli. Con il template baseline, il residuo cattura le deviazioni dal pattern normale pre-farmaco, rendendo la morphology instability discriminatoria.

**Prima (v3.2, intra-recording)**: nifedipine(−) morph = 0.915 > terfenadine(+) morph = 0.262 → anti-discriminatorio.

**Dopo (v3.3, baseline-relative)**: terfenadine(+) morph = 0.741 > nifedipine(−) morph = 0.380 → discriminatorio (2.2× ratio).

### Perché l'analisi spettrale?

L'analisi spettrale è complementare a quella temporale. I farmaci che bloccano il canale hERG causano alterazioni della morfologia di ripolarizzazione (broadening dell'onda T, EAD, U-waves) che si manifestano come cambiamenti nel contenuto in frequenza del segnale. La spectral change score è la metrica singola più discriminatoria (3× ratio).

### Perché il cessation detector?

Alcuni farmaci ad alta concentrazione causano l'arresto completo del battito (cessazione). Questo è un endpoint di sicurezza critico che non viene catturato dall'analisi tradizionale del FPD (che richiede battiti per essere misurato). Il detector a 5 componenti copre diversi pattern di cessazione: pause isolate, deterioramento progressivo, arresto terminale, distruzione della forma d'onda.

### Perché la correzione Fridericia e non Bazett?

Fridericia (FPDcF = FPD / RR^(1/3)) è preferita per i microtessuti cardiaci perché Bazett (FPD / RR^(1/2)) sovra-corregge a frequenze basse (BP > 1.5 s), che sono tipiche degli hiPSC-CM.

---

## 7. Validazione

### Dataset

169 registrazioni CSV da 4 esperimenti (EXP 5, 7, 8, 9) su piattaforma µECG-Pharma, con 7 farmaci del set di validazione CiPA:

| Farmaco | Classe CiPA | Meccanismo |
|---------|-------------|-----------|
| Terfenadine | Positivo (alto rischio) | Bloccante hERG |
| Quinidine | Positivo (alto rischio) | Bloccante hERG + Na |
| Dofetilide | Positivo (alto rischio) | Bloccante hERG selettivo |
| Alfuzosin | Negativo (basso rischio) | Bloccante α1-adrenergico |
| Mexiletine | Negativo (basso rischio) | Bloccante Nav1.5 (borderline CiPA) |
| Nifedipine | Negativo (basso rischio) | Bloccante canale L-type Ca²⁺ |
| Ranolazine | Negativo (basso rischio) | Bloccante INa tardiva |

### Confronto con il paper

| Parametro | Paper (n=51) | Software (n=7) | Differenza |
|-----------|-------------|----------------|------------|
| BP | 1900 ± 700 ms | 1838 ± 645 ms | −3.3% |
| FPDcF | 560 ± 150 ms | 555 ± 51 ms | −0.9% |
| Ampiezza | 251 ± 320 µV | 253 ± 92 µV | +0.8% |

### Accuratezza classificazione

6/7 farmaci correttamente classificati sulla risk map CiPA:

| Farmaco | X (ΔFPDcF%) | Y (indice) | Zona | Corretto |
|---------|-------------|-----------|------|----------|
| Quinidine (+) | +60.4 | 73.1 | HIGH | ✓ |
| Dofetilide (+) | −6.6 | 61.0 | HIGH | ✓ |
| Terfenadine (+) | +27.5 | 60.7 | HIGH | ✓ |
| Mexiletine (−) | +16.3 | 47.9 | HIGH | ✗ |
| Alfuzosin (−) | +16.7 | 31.7 | INTERMEDIATE | ✓ |
| Nifedipine (−) | 0.0 | 1.0 | LOW | ✓ |
| Ranolazine (−) | 0.0 | 0.3 | LOW | ✓ |

Mexiletine è l'unico farmaco misclassificato. Questo è coerente con il framework CiPA dove la mexiletina è un farmaco borderline (bloccante Na con effetti multipli sui canali ionici).

---

## 8. Interfaccia grafica

Dalla primavera 2026 l'interfaccia primaria è un'applicazione **desktop PySide6 + PyQtGraph** (`pyside_app/`), decisa con l'ADR-0001 (`docs/decisions/0001-abandon-streamlit-for-pyside6.md`, *Accepted*). La GUI web Streamlit (`app.py`, `ui/`) resta disponibile in manutenzione, senza nuove funzionalità.

### 8.1 GUI desktop (PySide6) — primaria

```bash
pip install ".[gui,reports]"
cardiac-fp-gui                 # oppure: python -m pyside_app.main
```

```
pyside_app/
├── main.py                   # MainWindow: tab Segnale / Battiti, Ricalcola, menu, stato config
├── signal_viewer.py          # Viewer PyQtGraph: overlay grezzo/filtrato, marker battiti, hover, zoom
├── study_panel.py            # Pannello Studi: albero Studio→Gruppo→File, batch in QThreadPool,
│                             #   metriche per gruppo (FPDc/BPM/STV), dose-risposta, CDISC per studio
├── settings_dialog.py        # Dialog AnalysisConfig (sottoinsieme dei campi)
├── settings_dialog_helpers.py
└── theme.py                  # Tema Scuro/Chiaro persistito in QSettings
```

Funzioni principali:

- **Segnale**: segnale grezzo e filtrato sovrapposti, marker dei battiti post-QC, selettore canale (el1/el2/auto) con riflesso del canale effettivamente analizzato.
- **Battiti**: template medio con banner quando il template non è rappresentativo (ritmo a rischio o CV FPD alto), editor per aggiungere/rimuovere battiti; **Ricalcola** riesegue la pipeline a valle della detection e salva le correzioni in un sidecar `<file>.overrides.json` (`cardiac_fp_analyzer/overrides.py`), riapplicato automaticamente alle analisi successive.
- **Studi**: modello `Study → Group → FileEntry` (`cardiac_fp_analyzer/study.py`, schema versionato, percorsi relativi POSIX, `dose_uM` esplicitamente `None` per dose ignota); "Aggiungi cartella al gruppo" ricorsiva; analisi batch in background con cache invalidata da fingerprint della configurazione; metriche aggregate per gruppo; curve dose-risposta (asse log, barre d'errore); export CDISC SEND per studio.
- **Impostazioni**: dialog sui campi principali di `AnalysisConfig` (il JSON completo resta la via per i restanti).

I test della GUI desktop (`tests/test_study_panel_*.py`, `tests/test_pyside_theme.py`, ~200 test) girano headless con `QT_QPA_PLATFORM=offscreen`; la CI li esegue e fallisce se vengono saltati.

### 8.2 GUI web Streamlit — legacy

Architettura (v3.2): `app.py` (~90 righe) come entry point e router, moduli in `ui/`:

```
ui/
├── i18n.py                 # Dizionario TRANSLATIONS (IT/EN) + T()
├── helpers.py              # reanalyze_with_modified_beats(), amplitude_scale()
├── config_sidebar.py       # build_config_from_sidebar() → AnalysisConfig
├── single_file.py          # Analisi singolo file + editor battiti + export
├── batch.py                # Batch + risk map + summary
├── drug_comparison.py      # Dashboard dose-response
└── reports.py              # Download Excel/PDF/Config JSON/CDISC SEND
```

```bash
pip install ".[streamlit]"
streamlit run app.py        # browser, porta 8501
```

Nota: il "Ricalcola" Streamlit (`ui/helpers.py`) usa ancora il percorso legacy dell'RR post-QC per la correzione di frequenza; la GUI desktop usa l'RR locale sul treno pre-QC (v3.4).

### Pagine

L'interfaccia è organizzata in tre pagine, selezionabili dal menu laterale.

**1. Analisi Singolo File** — Caricamento di un singolo file CSV per analisi immediata. Quattro modalità di selezione elettrodo: `auto` (scoring automatico), `el1`, `el2`, o `Entrambi` (analisi duale, v3.5). Mostra: il tracciato del segnale filtrato con beat markers sovrapposti (con opzione di overlay del segnale grezzo non filtrato per verifica), l'overlay dei battiti segmentati, la tabella dei parametri estratti (BP, FPDcF, ampiezza, durata spike) con statistiche, il report aritmico completo (rischio, classificazione, metriche residuali, incidenza EAD). Include un editor interattivo dei battiti (v3.4) per aggiungere/rimuovere manualmente i marker e ri-analizzare in tempo reale. Sezione di export con parametri CSV, riepilogo CSV e report Excel (v3.5).

**2. Analisi Batch + Risk Map** — Tre modalità di caricamento: selezione cartella tramite dialog nativo del sistema operativo (tkinter `filedialog.askdirectory`), upload multiplo di file CSV, oppure upload di archivio ZIP. Dopo il caricamento, la pipeline `batch_analyze()` processa tutte le registrazioni. I risultati sono presentati su tre tab: la risk map CiPA interattiva (Plotly), con zone colorate LOW/INTERMEDIATE/HIGH e scatter per farmaco; il riepilogo tabellare con QC, inclusione, normalizzazione e classificazione per ogni registrazione; la vista dettagliata di ogni singola registrazione. È possibile specificare opzionalmente il ground truth dei farmaci per colorare i marker sulla risk map. Nella sezione download: report Excel, report PDF, configurazione JSON, e pacchetto CDISC SEND.

**3. Confronto Farmaci** — Dashboard comparativa disponibile dopo l'analisi batch. Permette di selezionare un sottoinsieme di farmaci e visualizzare: curve dose-response (ΔFPDcF% per concentrazione crescente), barre metriche aritmiche (morphology instability, EAD%, STV FPDcF, spectral change), overlay dei template waveform rappresentativi per confronto morfologico diretto.

### Editor interattivo dei battiti (v3.4)

Nella pagina Single File, il tab "Segnale" include un expander "Editor battiti" che permette l'editing manuale dei beat markers:

- **Tabella con checkbox**: ogni battito rilevato ha una casella "Incluso". Deselezionare per escluderlo dall'analisi (sul grafico diventa grigio). Riselezionare per reincluderlo (torna rosso).
- **Aggiungi battito**: inserire il tempo in secondi dove il detector ha mancato un battito. Il sistema trova l'indice campione più vicino.
- **Ri-analizza**: dopo le modifiche, il pulsante "Ri-analizza con battiti modificati" riesegue la pipeline dalla segmentazione in poi (QC, parametri, aritmia) senza ricaricare il file.

I risultati aggiornati sostituiscono quelli originali e tutti i tab si aggiornano di conseguenza.

### Visualizzazione segnale grezzo (v3.4)

Un checkbox "Mostra segnale grezzo (non filtrato)" permette di sovrapporre il segnale originale (arancione, semitrasparente) al segnale filtrato (blu) per verificare visivamente l'effetto dei filtri.

### Analisi duale elettrodi (v3.5)

Selezionando "Entrambi" nel selettore elettrodo, il sistema analizza el1 e el2 indipendentemente e presenta:

- **Tabella comparativa** in testa alla pagina con metriche side-by-side: QC Grade, numero battiti, BP ± SD, CV%, FPDcF ± SD, ampiezza spike, risk score. Permette di valutare a colpo d'occhio quale elettrodo ha la qualità migliore.
- **Selettore elettrodo** (radio EL1 / EL2) per passare alla vista dettagliata di ciascun elettrodo, con i 4 tab standard (Segnale, Battiti, Parametri, Aritmie) e l'export.
- **Editor battiti indipendente**: ciascun elettrodo ha il proprio stato dell'editor (inclusione/esclusione battiti, battiti aggiunti manualmente). Le modifiche su un elettrodo non influenzano l'altro.
- **Ri-analisi per elettrodo**: il pulsante "Ri-analizza" aggiorna solo l'elettrodo attualmente selezionato; la tabella comparativa riflette i risultati aggiornati.

Questa modalità è utile per verificare la coerenza dei risultati tra elettrodi e per escludere manualmente un elettrodo con artefatti, senza dover ri-analizzare il file.

### Export risultati singolo file (v3.5)

L'analisi del singolo file include ora una sezione di export con tre opzioni:

- **Parametri CSV**: tabella per-battito con RR, spike amplitude, FPD, FPDcF, rise time, dV/dt, correlazione morfologica, confidenza FPD.
- **Riepilogo CSV**: una riga con tutte le statistiche riassuntive (medie, SD, CV%), QC grade, risk score e classificazione.
- **Report Excel**: report completo nello stesso formato multi-sheet utilizzato dall'analisi batch.

### Pannello di configurazione

Il sidebar contiene tutti i parametri dell'`AnalysisConfig`, organizzati in sezioni espandibili: pre-processing (filtri con opzione di disattivazione notch, gain amplificatore), beat detection (metodo, distanza minima, soglia adattiva, soglia morfologica, toggle filtro morfologico), parametri FPD (finestre di ricerca, soglie di confidenza), aritmie (soglie EAD, modalità risk score manual/data-driven), criteri di inclusione, normalizzazione. I parametri possono essere esportati come file JSON e ri-importati: al caricamento del JSON, tutti i widget della sidebar si aggiornano automaticamente con i valori importati (v3.5).

### Internazionalizzazione (v3.4)

L'interfaccia è disponibile in italiano e inglese. Il selettore di lingua si trova in fondo alla sidebar. Tutte le stringhe dell'interfaccia (>150 chiavi) sono tradotte tramite un dizionario `TRANSLATIONS` con funzione `T(key)`.

### Normalizzazione temporale (v3.4)

Il vettore temporale viene normalizzato per partire sempre da 0 secondi. L'hardware MCS può includere un pre-trigger con tempi negativi (il trigger di acquisizione corrisponde a t=0 nel file originale), ma nella visualizzazione il tempo parte da 0 per maggiore intuitività.

### Requisiti aggiuntivi

```bash
pip install ".[gui,reports]"          # desktop
pip install ".[streamlit,reports]"    # web legacy
```

---

## 9. Export CDISC SEND

Il modulo `cardiac_fp_analyzer/cdisc_export.py` genera un pacchetto CDISC SEND conforme alle specifiche SENDIG v3.1 per la submission regolatoria (FDA, EMA) dei dati di elettrofisiologia cardiaca in vitro.

### Formato di output

Il pacchetto è composto da file SAS Transport v5 (`.xpt`), il formato richiesto dalla FDA per le submission elettroniche, più un file di metadati Define-XML 2.0. Tutti i file `.xpt` utilizzano encoding latin-1 come richiesto dalle specifiche SAS Transport v5.

### Domini SEND generati

**TS (Trial Summary)** — Metadati a livello di studio: identificativo, titolo, tipo di studio (IN VITRO), specie (HUMAN IPSC-CM), piattaforma, durata, sponsor, data di inizio. 12 record che descrivono il contesto sperimentale.

**DM (Demographics)** — Un record per ogni soggetto/microtessuto, identificato da `USUBJID` univoco. Include il chip di registrazione e l'assegnazione al gruppo di trattamento (ARM/ARMCD). Il dominio permette la tracciabilità di ogni campione biologico nel dataset.

**EX (Exposure)** — Un record per ogni trattamento farmacologico applicato, con: farmaco normalizzato (uppercase SEND-compatibile), concentrazione dose, unità, timing relativo allo studio. Copre tutte le condizioni di esposizione incluse baseline e washout.

**EG (ECG Test Results)** — Il dominio principale dei dati quantitativi. Ogni riga è una misurazione su un singolo battito: Beat Period (EGBP, ms), FPDcF corretto per frequenza (EGFPDCF, ms), ampiezza del spike (EGAMP, µV), durata del depolarizzazione (EGSPKD, ms). I codici test seguono la terminologia controllata NCI. Un esperimento tipico produce 1000–2000 record EG.

**RISK (Custom — Arrhythmia Risk)** — Dominio custom che estende SENDIG con i risultati della valutazione del rischio proaritmico. Contiene: rischio per farmaco, classificazione (Normal/Low Risk/Moderate Risk/High Risk/TdP-like), indice proaritmico composito, componenti dell'indice (spectral change, morphology instability baseline-relative, EAD incidence), variazione percentuale FPDcF. Questo dominio, essendo custom, è documentato in dettaglio nel Define-XML.

**define.xml** — Metadata file in formato Define-XML 2.0 che descrive: ogni dominio con la lista delle variabili, tipo dati, label, lunghezza, ruolo CDISC (Identifier, Topic, Result, Record Qualifier). Necessario per la validazione Pinnacle 21 e per l'interpretazione dei dati da parte del reviewer FDA.

### Utilizzo da riga di comando

```python
from cardiac_fp_analyzer.cdisc_export import export_send_package
from cardiac_fp_analyzer.analyze import batch_analyze
from cardiac_fp_analyzer.config import AnalysisConfig

config = AnalysisConfig()
results = batch_analyze('data_folder/', config=config)
export_send_package(results, 'send_output/', study_id='CIPA001')
```

Il pacchetto viene generato nella cartella specificata, contenente: `ts.xpt`, `dm.xpt`, `ex.xpt`, `eg.xpt`, `risk.xpt`, e `define.xml`.

### Utilizzo dalla GUI

Nella pagina "Analisi Batch + Risk Map", dopo aver eseguito l'analisi, la sezione download include il pulsante "Export CDISC SEND". Cliccando si genera un archivio ZIP contenente tutti i file `.xpt` e il `define.xml`. Lo Study ID è configurabile tramite il pannello "Impostazioni CDISC SEND" nella stessa pagina.

### Validazione e conformità

Il pacchetto generato è progettato per superare la validazione Pinnacle 21 Community (lo strumento standard per la verifica di conformità CDISC). Per la validazione:

1. Scaricare Pinnacle 21 Community da pinnacle21.com
2. Creare un nuovo progetto di tipo SEND
3. Caricare i file `.xpt` e il `define.xml`
4. Eseguire la validazione

I campi obbligatori CDISC (STUDYID, DOMAIN, USUBJID, --SEQ) sono sempre popolati. Le variabili seguono le naming convention SEND (prefisso dominio + suffisso semantico). Le unità di misura utilizzano la terminologia controllata NCI.

### Limitazioni note

I nomi delle variabili sono limitati a 8 caratteri (requisito SAS Transport v5). I valori stringa sono limitati a 200 caratteri e codificati in latin-1 (caratteri non-ASCII vengono sostituiti). Il dominio RISK è un'estensione custom non presente nello standard SENDIG ufficiale — necessita di una Reviewer's Guide che ne giustifichi l'inclusione.

---

## 10. Logging e diagnostica (v3.2)

Il package utilizza logging strutturato Python anziché la soppressione blanket dei warning:

```python
# A livello package (__init__.py)
import logging
logging.getLogger('cardiac_fp_analyzer').addHandler(logging.NullHandler())
```

Ogni modulo usa `logger = logging.getLogger(__name__)`. I blocchi `except` specificano i tipi di eccezione attesi (`ValueError`, `IndexError`, `RuntimeError`, `np.linalg.LinAlgError`) e loggano i dettagli a livello `DEBUG`. Per abilitare il logging diagnostico:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s: %(message)s')
```

Nella GUI Streamlit, il logging è configurato a livello `INFO` di default. Il parametro `?debug=1` nell'URL può attivare il livello `DEBUG`.

---

## 11. Changelog

### v3.9.0 (Ottobre 2026) — file con due tessuti e mappa dei campioni

1. **Due tessuti nello stesso file** (convenzione GG, uno per ingresso): il batch `auto` dà una registrazione per ingresso, con tessuto, test item e dose letti dal nome nell'ordine degli ingressi; `na` indica un ingresso vuoto (§4.1).
2. **`samples.csv`**: mappa file × ingresso → esperimento, chip, camera, test item, dose, esclusione; per i file elencati ha la precedenza sui nomi. `samples_draft.csv` ne scrive una bozza con i controlli da fare (riga di comando e pagina batch).
3. **Un gruppo per tessuto** in modalità `auto` (`group_electrode`): inclusione e abbinamento non dipendono più dall'ingresso usato da ciascun file.
4. Report Excel e tabella della pagina batch con la colonna del tessuto; il file `.overrides.json` non si applica ai file con due tessuti, perché non dice su quale ingresso è stato fatto.

### v3.8.3 (Ottobre 2026) — indice della risk map: solo il cambio spettrale

L'asse Y della risk map (§4.11) è per default il cambio spettrale rispetto al baseline, aggregato per concentrazione; instabilità morfologica ed EAD restano calcolate e ripesabili. Corrette le attribuzioni al paper delle metriche sul residuo (§4.6).

### v3.8.2 (Ottobre 2026) — indice proaritmico della risk map per concentrazione

L'indice proaritmico (asse Y, §4.11) si calcola per registrazione e si aggrega come la decisione per farmaco (media tra tessuti per concentrazione, almeno 2 tessuti, 2 concentrazioni adiacenti), invece di prendere il massimo di ogni componente. Il simbolo di cessazione richiede confidenza > 0,5. Corretta la cifra sull'override di cessazione riportata nella v3.8.0 (§4.10).

### v3.8.1 (Ottobre 2026) — risk map allineata alla decisione per farmaco

L'asse X della risk map (§4.11) è la statistica che `classify_drug` confronta con la soglia (`decision_value`), non più la variazione massima di una singola registrazione. La decisione è ricalcolata sui risultati passati alla mappa, il veicolo è posizionato con la stessa statistica, i farmaci senza decisione stanno in una fascia a sinistra e i washout non compaiono più come farmaci. Stessa logica nella pagina Streamlit.

### v3.8.0 (Ottobre 2026) — decisione sul farmaco, riferimento pre-dose, registrazioni a 20 kHz

Correzioni emerse confrontando le regole di decisione sui 12 composti del paper Visone 2023 (§4.10, §4.1, §4.2). L'interfaccia PySide non è toccata.

1. **Decisione per farmaco** (`classification_method='concentration'`, default): media tra tessuti a ogni concentrazione, almeno 2 tessuti, soglia raggiunta in 2 concentrazioni consecutive; `insufficient data` quando i tessuti non bastano. Risultati sui 12 composti nella tabella di §4.10.
2. **Override di cessazione** opzionale (`enable_cessation_override=False`), sempre riportato in `cessation_flag`.
3. **Riferimento pre-dose**: t0 riconosciuto; l'ultimo riferimento registrato prima della prima dose, per orario dell'intestazione; elettrodo scelto sullo stesso riferimento. Un gruppo viene escluso solo se nessun riferimento passa l'inclusione.
4. **20 kHz**: decimazione a 2 kHz in `load_csv`; passa-banda a sezioni del secondo ordine quando il progetto (b, a) è instabile.
5. **Nomi dei protocolli prima del 2020** (`channel1_sx_channel2_dx`, `t0`, `t1`…`t7`, virgola decimale, numero prima del farmaco, chip dalla cartella).

**Verifica**: batch completo sui 10 esperimenti del paper (601 registrazioni); la suite comprende 652 test, di cui 52 nuovi (`tests/test_drug_call_and_reference.py`).

### v3.7.0 (Ottobre 2026) — abbinamento baseline e raggruppamento dei farmaci nel batch

Correzioni emerse eseguendo `batch_analyze` sul dataset Visone 2023 (§4.10 e §3, `parse_filename` e `describe_recording`). L'interfaccia PySide non è toccata, perché usa gruppi definiti dall'utente.

1. **Identità del tessuto dal percorso** (esperimento + giorno + chip + camera), comprese le cartelle Accelera. In precedenza l'esperimento si leggeva solo da cartelle `EXP…` e il giorno veniva ignorato, così le dosi venivano normalizzate sul baseline di un altro esperimento.
2. **Un elettrodo per tessuto** in modalità `auto` (baseline prima).
3. **Scelta del baseline** per cartella, poi per grado QC, registrata in `normalization['pairing']`. L'abbinamento è indicizzato per percorso più elettrodo.
4. **Nomi canonici dei farmaci** in `classify_drug`; washout e veicolo esclusi dalla decisione.
5. **Arresto**: %ΔFPDcF non calcolata oltre 6 s di periodo (`max_beat_period_for_fpdc_ms`).

**Verifica**: su Exp5 ed Exp8 del paper le variazioni prodotte dal batch coincidono con quelle calcolate abbinando a mano ogni tessuto. La suite comprende 600 test, di cui 33 nuovi (`tests/test_tissue_pairing.py`).

**CI**: action su Node 24 e runner fissato a `ubuntu-24.04`, in vista del passaggio di `ubuntu-latest` a 26.04.

### v3.6.0 (Ottobre 2026) — selezione dell'onda di ripolarizzazione

Seconda fase del lavoro sul gold standard manuale (GG). Analisi a livello di candidati sul template (riprodotta la scelta della pipeline in 121/121 elettrodi): l'onda dell'analista era tra i candidati nel 78 % degli elettrodi contro il 53 % scelto; le ripolarizzazioni sono spesso bifasiche e l'analista segna il picco del lobo positivo; in 22 elettrodi la finestra di ricerca si chiudeva prima dell'onda perché l'RR del treno completo era dimezzato dalla sovra-rilevazione.

- `parameters._align_beats_xcorr`: corretto uno sfasamento d'indice per cui l'allineamento spostava di `alignment_max_shift_ms` (50 ms) anche i battiti già allineati e raddoppiava il jitter; ora correlazione normalizzata con lag zero al centro. Test di regressione che falliscono sul codice vecchio.
- `parameters.beat_spike_inverted`: inversione per anti-correlazione (≤ `inversion_corr_threshold`, −0.5) invece del confronto tra deflessioni; prima, col template sfasato, il ~40 % degli elettrodi aveva quasi tutti i battiti marcati invertiti.
- `RepolarizationConfig.repol_candidate_rule = 'prefer_positive'` (+ `repol_positive_min_rel_prom` 0.5, `repol_positive_max_offset_ms` 400): parametri scelti con leave-one-experiment-out dentro il DEV (stessi valori in ogni piega; 67.9 % sull'esperimento escluso contro 68.8 % interno).
- `RepolarizationConfig.window_rr_from_template_beats = True`: finestra e pavimento adattivo del template usano max(RR del treno completo, RR dei battiti del template); `repolarization.next_spike_cut_index` (guardia di forma: correlazione ≥ 0.9 e ampiezza ≥ 0.5 con lo spike principale) ferma l'estensione prima del battito successivo.
- `RepolarizationConfig.fpd_method = 'peak'` (era 'tangent').
- `per_beat_prefer_template_sign` aggiunto ma spento (valutato: nessun guadagno).
- Diagnostica nel summary: `template_repol_sign`, `repol_candidate_rule`, `repol_window_rr_ms`, `repol_window_extended`.

Risultati, FPD entro ±10 % tra gli elettrodi riportati: DEV 45 → 61 % (grado A 61 → 74 %); **test tenuto da parte 48 → 69 %** (grado A 69 → 84 %, B 41 → 59 %, C 20 → 50 %), errore mediano −1.3 → −0.1 %. BP invariato. Corpus del paper invariato (errore medio 3.4 → 3.2 %). 567 test.

### v3.5.1 (Ottobre 2026)

- `RepolarizationConfig.max_adaptive_min_fpd_ms` 350 → 600 ms (razionale nel config). Caratterizzazione completa dell'errore FPD contro il gold standard manuale in `docs/FPD_vs_gold_standard_2026-10.md`: la definizione del punto di fine non è il problema; lo è la selezione del candidato sui ritmi lenti e rumorosi. Punteggio prominenza×larghezza provato e scartato (peggiora).

### v3.5.0 (Ottobre 2026) — validazione cieca su gold standard manuale

Primo confronto con una misura manuale indipendente su un dataset interno (GG: 179 CSV da 60 s, 313 elettrodi con BP, FPD, tempi dei singoli battiti e note dell'analista). Metodo: sviluppo su Exp 5/8/10, **test cieco su Exp 6/7/9 eseguito una sola volta**; split bilanciato per qualità (13–14 % non analizzabili, 52–59 % grado C/D/F per metà). Risultati nel README (changelog v3.5.1). Tre modifiche, tutte decise sul solo set di sviluppo:

- **Verdetto di analizzabilità** (`quality_control.assess_analysability`, `QualityConfig.enable_analysability_verdict`): (a) SNR mediana dei battiti rilevati rispetto al noise floor della registrazione < 1.6; (b) meno di 16 battiti con CV RR > 40 %. In entrambi i casi: grado F, `summary['not_analysable']=True` con motivo, FPD/FPDc → NaN, `fpd_reliable=False`, classificazione aritmica "Not analysable", esclusione dalla normalizzazione come farmaco e come baseline. Il conteggio dei battiti resta per audit. Sul test cieco: 12/13 non analizzabili riconosciuti; 19 elettrodi misurati dall'analista marcati, sui quali la v3.4.1 sbagliava in 18.
- **Accettazione del matched filter** (`mf_count_ratio` (0.7,1.5) → (0.3,2.0)): sulle registrazioni rumorose il detector a derivata sovra-rileva 2–3×; nei 97 disaccordi del set di sviluppo il matched filter era più vicino all'analista in 61.
- **Popolazione minore di ampiezza** (`beat_detection._reject_minor_amplitude_population`): split di Otsu sul log-ampiezza; il cluster piccolo è scartato solo se mediane ≥ 2.5× distanti, il cluster grande da solo ha CV RR ≤ 0.8× dell'insieme, e i piccoli non sono in fase fissa 1:1 (alternans → tenuti).

**Accuratezza per fascia di SNR** (set di sviluppo, v3.4.1, elettrodi analizzabili): SNR < 1.6 → BP ±10 % 9 %, FPD 2 %; 1.6–2.5 → 35 % / 10 %; 2.5–4 → 41 % / 31 %; 4–8 → 54 % / 44 %; > 8 → 81 % / 62 %. Non c'è un gradino: la qualità del risultato segue la qualità del segnale, e il grado QC lo riflette.

**Aperto**: FPD entro ±10 % solo nel 62–69 % anche a SNR alta / grado A — problema di *misura* della ripolarizzazione (definizione del punto di fine rispetto all'analista), non di detection; copertura al 77 %; ampiezza software ≈ 60 % di quella dell'analista (definizione diversa della finestra).

### v3.4.1 (Ottobre 2026)

- **Matched filter per il regime a bassa SNR** (`BeatDetectionConfig.enable_matched_filter_refine`, `mf_*`): quando la SNR mediana dei battiti rilevati è < 3, il detector a derivata manca ~1/3 degli spike reali e il gap-filling inserisce ipotesi a posizioni "attese" (sulla baseline di laboratorio `chipA_ch1` el1: 73 mancanti e 58 spurii su ~220, verificati sull'altro elettrodo). Né l'ampiezza né la dV/dt del singolo battito li distinguono dal rumore; la *forma* sì. Template = mediana dei battiti più ripidi (±25 ms), correlazione **non normalizzata** (la NCC divide per l'energia locale, che a questa SNR è tutta rumore), picchi sopra mediana + 3.5·MAD con refrattario 0.5·RR. Risultato: 219 battiti, 3 mancanti, 2 spurii, RR 810 vs 811 ms, QC 218/218. Disattivo nel regime ad alta SNR, dove lo stesso filtro raccoglie onde T e after-potential (Exp7 chipE: 201 contro 109 battiti veri); i file puliti del corpus stanno tutti a SNR ≥ 4.4.
- **Gate noise-floor ricalibrato**: la prima versione (floor fisso 1.5) scartava 44 battiti veri su 208 nel file a bassa SNR, dove ogni battito reale sta a SNR 1.1–2.3 — lo stesso intervallo dei falsi dell'Exp8. Ciò che li distingue è l'insieme: i falsi hanno mediana 1.12 (indistinguibile da finestre di rumore) accanto a un cluster 4× più alto; i veri a bassa SNR hanno mediana 1.8 e nessun cluster superiore. Quindi floor 1.0 (solo "più silenzioso di una finestra di rumore"), cluster "rumore" solo con mediana ≤ 1.35 e separazione ≥ 3×. Exp8 156→81, Exp6 76→35 e i sei file puliti invariati; file a bassa SNR 208→205.
- **Corpus**: nona fixture `lab_lowsnr_chipA_ch1` con `reference_signal` = el2, test battito-per-battito (mancanti ≤ 5 %, spurii ≤ 3 %) e test che documenta il fallimento del solo detector a derivata.

### v3.4.0 (Ottobre 2026)

- **Beat detection — gate SNR sul noise floor** (`BeatDetectionConfig.enable_noise_floor_gate`): una detection la cui ampiezza locale non supera chiaramente il rumore della registrazione è rumore, qualunque cosa dica il ritmo. Regola a due livelli ancorata al floor (hard floor 1.5×; cluster inferiore scartato solo se interamente compatibile col rumore e separato da un salto ≥1.5× da un cluster ≥3× più alto). Chiude il raddoppio su Exp8/Day6/chipD_ch1 (156→81 battiti, RR −46 % → +10 % vs pubblicato, FPD invariato) e lo stesso difetto su finestre di Exp6; zero battiti rimossi su sei file puliti. Nessuna salvaguardia "non dimezzare il conteggio": su Exp8 la metà *è* la risposta.
- **Corpus di regressione su segnali reali**: `tests/fixtures/real_signals/` (8 baseline Visone 2023, elettrodo degli autori, RR/FPD pubblicati) + `tests/test_real_signal_regression.py` (RR, FPD, assenza di sovra-rilevazione, meccanismo del gate). Primo test della suite in cui un FPD misurato è confrontato con un riferimento indipendente su segnale reale.
- **Ripolarizzazione**: rimosso il fallback `argmax(|seg|)` (template e per-battito); T-wave non misurabile → `None`. Effetto misurato piccolo: il divario con il ~41 % di punti non misurati dagli autori **non** dipendeva da qui e resta aperto.
- **Unità**: `amplifier_gain` default 1e4; CDISC `SPIKEAM` in µV (era scritto in mV); etichette Excel FPDcF/FPDcB coerenti con `correction`; `inclusion['fpd_reliable']` → `fpd_confidence_ok`.
- **Packaging/CI/doc**: extra `gui` = PySide6 + pyqtgraph, `pyside_app` nei package, entry point `cardiac-fp-gui`; CI su `feat/**`/`fix/**` con PySide6 headless e budget di skip; ADR-0001 *Accepted*; documentazione allineata alla UI reale.
- **Consolidamento aprile–agosto 2026** (mai rilasciato prima): GUI PySide6; modello Studio/Gruppo/File; sidecar override; `rhythm_integration`; RR locale pre-QC per FPDc (`compute_local_rr`); segno ripolarizzazione tracciato per battito; flag `correction` effettivo; `fpd_reliable` propagato a normalizzazione e CDISC (`FPDREL`); criteri di inclusione FPD/RR e precisione baseline (opt-in); `tools/reanalyze.py` con manifest di provenienza.

### v3.3.0 (Marzo 2026)

- **CI GitHub Actions**: workflow `.github/workflows/ci.yml` con lint (ruff) e test su Python 3.9, 3.10, 3.11, 3.12.
- **Refactoring arrhythmia**: `arrhythmia.py` (708→438 righe) spezzato in `residual_analysis.py` (304 righe) — template computation, residual RMS, EAD detection, Poincaré STV.
- **Ruff lint clean**: 194 errori risolti; configurazione ruff in `pyproject.toml` con ignore mirate per codice scientifico.
- **Test end-to-end numerici**: 17 test su segnali sintetici con ground truth noto — copertura da beat detection a classificazione aritmie.
- **Suite test**: 54 test totali (smoke, config, alignment, denominatore, e2e sintetici).

### v3.2.1 (Marzo 2026)

- **Refactoring core**: `analyze.py` (690→442 righe) spezzato in `channel_selection.py`, `inclusion.py`; `parameters.py` (671→293 righe) spezzato in `repolarization.py`.
- **Fix denominatore re-analisi residua**: nella seconda passata batch (baseline-relative), `beat_periods` veniva preso dal result dict (calcolato da tutti i beat raw) anziché ricalcolato dai `beat_indices` cleaned. Quando il QC scarta molti beat, questo causava mismatch tra `n_beats` e denominatori CV/incidenze in `analyze_arrhythmia()`. Fix: `bp = compute_beat_periods(bi, fs)` nella re-analysis loop.
- **Hardening eccezioni**: rimossi tutti i `except Exception` (7 occorrenze UI + 2 core) e sostituiti con tipi specifici (`OSError`, `ValueError`, `KeyError`, ecc.).
- **Rimosso `sys.path.insert`**: eliminato hack legacy da `app.py` e `analyze.py`; pyproject.toml gestisce gli import.
- **Warning mirati**: rimossa soppressione blanket `DeprecationWarning`; filtri solo per streamlit, matplotlib, plotly.
- **Documentazione**: sezione "Divergenza Denominatore QC" aggiunta a `confronto_software_vs_paper.md` e `comparison_with_paper.md`.
- **Test**: 35 test totali; aggiunti test per nuovi moduli (channel_selection, inclusion, repolarization) e regressione denominatore.

### v3.2.0 (Marzo 2026)

- **UI modulare**: `app.py` (1968 righe) spezzato in `ui/` package con 7 moduli (i18n, helpers, config_sidebar, single_file, batch, drug_comparison, reports). Il router `app.py` è ora ~90 righe.
- **Beat alignment fix**: corretto bug critico di disallineamento tra `beat_indices` e `beats_data`/`beats_time` causato dal mancato passaggio dell'array `valid` da `segment_beats()` a `validate_beats()`. Aggiunto assert di allineamento in `analyze.py`, `quality_control.py`, `parameters.py`.
- **Packaging**: aggiunto `pyproject.toml` con gruppi di dipendenze opzionali (`[gui]`, `[reports]`, `[cdisc]`, `[all]`, `[dev]`), entry point CLI `cardiac-fp`, configurazione ruff + pytest.
- **Logging strutturato**: `NullHandler` a livello package, `logger = logging.getLogger(__name__)` per modulo, blocchi `except` con tipi specifici + `logger.debug`. Rimossa soppressione blanket `warnings.filterwarnings('ignore')`.
- **Pesi configurabili**: scoring weights di beat detection (`score_bp_ideal`, `score_cv_good`, ecc.) e channel selection (`w_bp_range`, `w_corr_max`, `w_amplitude_max`, ecc.) spostati in campi delle dataclass `BeatDetectionConfig` e `ChannelSelectionConfig`, serializzabili via JSON.
- **API pubblica**: `arrhythmia.compute_template`, `normalization.is_baseline`, `normalization.get_group_key` ora pubbliche (alias `_compute_template`, `_is_baseline`, `_get_group_key` mantenuti per retrocompatibilità).
- **Plotly** aggiunto alle dipendenze `[gui]`.

### v3.1.0 (Marzo 2026)

- Configurazione centralizzata `AnalysisConfig` con preset e JSON I/O.
- Export CDISC SEND (TS, DM, EX, EG, RISK + define.xml), validato Pinnacle 21.
- Risk map CiPA interattiva con zone colorate e ground truth.
- Analisi spettrale (PSD Welch, entropia, KL divergenza).
- Cessation detector a 5 componenti.
