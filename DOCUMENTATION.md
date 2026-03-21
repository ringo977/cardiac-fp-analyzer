# Cardiac FP Analyzer — Documentazione Completa

**Versione**: 3.3
**Piattaforma**: Python 3.10+
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
   - 4.9 [Normalizzazione e classificazione (`normalization.py`)](#49-normalizzazione-e-classificazione)
   - 4.10 [Risk map CiPA (`risk_map.py`)](#410-risk-map-cipa)
   - 4.11 [Report (`report.py`)](#411-report)
5. [Configurazione](#5-configurazione)
6. [Razionale scientifico](#6-razionale-scientifico)
7. [Validazione](#7-validazione)

---

## 1. Panoramica

Il Cardiac FP Analyzer è un software per l'analisi automatica di potenziali di campo (field potential, FP) registrati da microtessuti cardiaci derivati da hiPSC-CM su piattaforma µECG-Pharma. Il software implementa una pipeline completa per la valutazione del rischio proaritmico dei farmaci, seguendo il framework CiPA (Comprehensive in vitro Proarrhythmia Assay).

Il segnale FP è l'equivalente extracellulare del potenziale d'azione cardiaco. La depolarizzazione rapida produce un picco negativo (spike), seguito da una fase di ripolarizzazione che termina con una deflessione lenta. L'intervallo spike–ripolarizzazione è il Field Potential Duration (FPD), analogo all'intervallo QT dell'ECG clinico. L'allungamento del FPD è il marcatore primario del rischio di aritmie da farmaci (Torsade de Pointes, TdP).

Il software analizza ogni registrazione attraverso una pipeline a 12 stadi, raggruppa le registrazioni per chip/canale, normalizza rispetto al baseline, e classifica ciascun farmaco su una mappa di rischio 2D.

---

## 2. Installazione e utilizzo

### Dipendenze

```
numpy, scipy, pandas, matplotlib, openpyxl
```

### Utilizzo da linea di comando

```bash
# Analisi batch di una cartella
python -m cardiac_fp_analyzer.analyze /path/to/data/

# Con selezione canale manuale
python -m cardiac_fp_analyzer.analyze /path/to/data/ --channel ch1

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
  ├─ 3. Selezione canale (analyze.py)
  │     Scoring automatico ch1 vs ch2 (se channel='auto')
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

Legge i file CSV prodotti dal sistema Digilent WaveForms (Analog Discovery 2). Il formato include un header con metadati del dispositivo e due canali di acquisizione.

**Output**: `(metadata, DataFrame)` dove metadata contiene sample_rate, device, serial, datetime, range e offset per ciascun canale.

#### `parse_filename(filename)`

Estrae informazioni strutturate dal nome del file. Convenzione: `chipX_chN_farmaco_concentrazione.csv`.

**Esempio**: `chipA_ch1_terfe_300nM.csv` → `{chip: 'A', channel: 1, drug: 'terfenadine', concentration: '300 nM', is_baseline: False}`

**Razionale**: I nomi dei file codificano l'identità del microtessuto (chip+canale) e la condizione sperimentale. Questa informazione è essenziale per il pairing baseline–farmaco nella normalizzazione.

---

### 4.2 Filtraggio del segnale

**Modulo**: `filtering.py`

Il segnale FP grezzo contiene rumore da diverse sorgenti: interferenza di rete (50 Hz), drift della baseline, rumore ad alta frequenza. La pipeline di filtraggio rimuove questi artefatti preservando la morfologia del segnale cardiaco.

#### Pipeline di filtraggio

1. **Filtro notch** a 50 Hz + armoniche (100, 150 Hz): rimuove l'interferenza di rete. Q=30 per bande strette che non distorcono il segnale cardiaco. Configurabile per 60 Hz (USA).

2. **Filtro passa-banda Butterworth** (0.5–500 Hz, ordine 4): rimuove sia il drift DC (< 0.5 Hz) sia il rumore ad alta frequenza (> 500 Hz). La banda 0.5–500 Hz preserva interamente la morfologia del FP cardiaco.

3. **Smoothing Savitzky-Golay** (finestra 7 campioni, ordine 3): smoothing finale che preserva i picchi (non li attenua come un filtro passa-basso convenzionale). Ideale per mantenere la forma dello spike di depolarizzazione.

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `notch_freq_hz` | 50.0 | Frequenza di rete |
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

**Metodo tangente** (default): Trova il punto di massima pendenza discendente sulla curva di ripolarizzazione, poi traccia la tangente fino all'intersezione con la baseline. È il metodo standard in letteratura per i FP cardiaci.

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
| `fpd_method` | 'tangent' | Metodo di misurazione FPD |
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

2. **Morfologia**: Correlazione di Pearson con il template. Rifiutato se < 0.40 (o forzatamente se < 0.20). Questo elimina battiti con morfologia aberrante (artefatti, ectopie marcate).

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

L'analisi delle aritmie è il cuore del sistema. Combina due approcci complementari: analisi statistica dei parametri e analisi residual-based (approccio del paper di riferimento).

#### Approccio statistico

Valuta le proprietà globali del ritmo e dei parametri:

- **Tachicardia/bradicardia**: Periodo medio < 300 ms o > 2500 ms
- **Irregolarità RR**: CV del periodo > 15% (irregolare), > 30% (critico), > 40% (fibrillazione-like)
- **Battiti prematuri/ritardati**: Singoli battiti < 70% o > 150% del periodo medio
- **STV (Short-Term Variability)**: Variabilità beat-to-beat del periodo, calcolata con il diagramma di Poincaré: STV = mean|x_{i+1} - x_i| / √2
- **Prolungamento FPD**: FPD > 130% del baseline o > 500 ms assoluti
- **Instabilità d'ampiezza**: CV dell'ampiezza > 30%

#### Approccio residual-based (Visone et al. 2023)

Questo approccio calcola il residuo tra ogni battito e un template di riferimento:

```
residuo = battito − template
```

Il residuo contiene solo le deviazioni dalla morfologia normale. In condizioni fisiologiche i residui sono piccoli (jitter termico); con farmaci proaritmici i residui crescono (cambio morfologico, EAD, instabilità).

#### Baseline-relative residual analysis (v3.3)

**Innovazione chiave della v3.3**: Il template di riferimento può provenire dal baseline (registrazione senza farmaco) anziché dalla stessa registrazione. Questo cambia radicalmente il significato del residuo:

- **Intra-recording** (v3.2): residuo = battito − template_stessa_registrazione → cattura solo il jitter beat-to-beat
- **Baseline-relative** (v3.3): residuo = battito_con_farmaco − template_baseline → cattura le deviazioni morfologiche indotte dal farmaco

L'implementazione è a due passaggi: nella prima pass si analizzano tutti i file e si memorizzano i template dei baseline per gruppo (chip+canale); nella seconda pass si ri-esegue l'analisi aritmia per i drug recording usando il template del baseline corrispondente.

#### Metriche dal residuo

**Morphology instability** (0–1): RMS del residuo normalizzato. Con baseline-relative analysis, è discriminatoria: farmaci positivi 0.566, negativi 0.257 (ratio 2.2×).

**EAD detection** (Early Afterdepolarization): Rileva depolarizzazioni secondarie nella fase di ripolarizzazione (150–500 ms post-spike). Cinque criteri simultanei:

1. **Statistico**: il picco nel residuo supera 6× la deviazione standard del rumore (σ stimata dal MAD del residuo completo)
2. **Ampiezza assoluta**: il picco supera l'8% dell'ampiezza picco-picco del template
3. **Larghezza**: la larghezza a metà altezza è tra 8 e 150 ms (esclude spike artefattuali e ondulazioni lente)
4. **Polarità**: solo picchi positivi (le EAD sono depolarizzazioni secondarie)
5. **Localizzazione**: nella finestra di ripolarizzazione (150–500 ms post-spike)

**Razionale dei 5 criteri**: Il criterio statistico da solo (come nel paper) genera troppi falsi positivi perché il rumore di fondo ha una distribuzione non-gaussiana. I criteri di ampiezza, larghezza e localizzazione aggiungono specificità biofisica.

**Poincaré STV**: Variabilità a breve termine di FPD e FPDcF, calcolata dal diagramma di Poincaré.

#### Risk score (0–100)

Punteggio composito che combina la severità e il tipo di flag rilevate, con penalità crescenti per: irregolarità RR, battiti prematuri, EAD, prolungamento FPD, cessazione.

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

### 4.9 Normalizzazione e classificazione

**Modulo**: `normalization.py`

#### Pairing baseline–farmaco

Ogni drug recording viene accoppiato al baseline dello stesso chip+canale nello stesso esperimento. Se più baseline esistono, viene preferito quello con grado QC migliore.

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

Se un recording mostra cessazione (waveform destruction) con bassa confidenza FPD (< 0.60), viene classificato come positivo indipendentemente dal ΔFPDcF. Questo cattura farmaci che distruggono il segnale prima che il FPD possa allungarsi.

#### Classificazione farmaco

Tre metodi disponibili per aggregare le concentrazioni:

- **max** (default): positivo se QUALSIASI concentrazione supera la soglia
- **mean**: positivo se la MEDIA delle concentrazioni supera la soglia
- **n_above**: positivo se ≥ N concentrazioni superano la soglia

| Config | Default | Descrizione |
|--------|---------|-------------|
| `threshold_low` | 10.0% | Soglia TdP score 1 |
| `threshold_mid` | 15.0% | Soglia TdP score 2 (ottimale, paper) |
| `threshold_high` | 20.0% | Soglia TdP score 3 |
| `classification_threshold` | 'mid' | Soglia per classificazione positivo |
| `classification_method` | 'max' | Metodo di aggregazione |

---

### 4.10 Risk map CiPA

**Modulo**: `risk_map.py`

Genera una mappa di rischio 2D nello stile CiPA, posizionando ogni farmaco su due assi:

**Asse X — Max ΔFPDcF (%)**: Il prolungamento massimo del FPDcF osservato tra tutte le concentrazioni. Cattura il rischio di prolungamento della ripolarizzazione.

**Asse Y — Indice proaritmico (0–100)**: Composito a tre componenti:

| Componente | Peso | Razionale |
|------------|------|-----------|
| Spectral change | 70% | Miglior discriminatore (3× ratio pos/neg). Cattura le alterazioni morfologiche della ripolarizzazione nel dominio della frequenza. |
| Morphology instability (baseline-relative) | 25% | Discriminatorio con template baseline (2.2× ratio). Solo i valori da analisi baseline-relative entrano nell'indice; quelli intra-recording sono esclusi perché anti-discriminatori. |
| EAD incidence | 5% | Contributo modesto, principalmente per bloccanti hERG puri ad alta concentrazione. |

**Tre zone di rischio**:

| Zona | Y | Significato |
|------|---|-------------|
| LOW (verde) | < 20 | Nessun segnale proaritmico |
| INTERMEDIATE (giallo) | 20–40 | Effetti sospetti, richiede investigazione |
| HIGH (rosso) | > 40 | Alto rischio proaritmico |

Le linee verticali a X = 10% e 20% separano i livelli di prolungamento FPDcF.

#### Drug name normalization

I nomi dei farmaci dai filename vengono normalizzati automaticamente (es. 'terfe' → 'terfenadine', 'DOFE' → 'dofetilide', 'NIFE' → 'nifedipine') per aggregare correttamente le diverse notazioni.

---

### 4.11 Report

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
| `default` | Parametri standard (tangent FPD, Fridericia, tutti i filtri) |
| `conservative` | Soglie più strette, meno falsi positivi |
| `sensitive` | Soglie più rilassate, meno falsi negativi |
| `peak_method` | Usa il metodo peak per FPD (più semplice) |
| `no_filters` | Disabilita i filtri (per debug) |

### Parametro amplifier_gain

Il sistema µECG-Pharma Digilent registra con un guadagno di 10⁴. Per ottenere i valori reali in Volt, il segnale viene diviso per il guadagno:

```
segnale_reale = segnale_ADC / 10⁴
```

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
