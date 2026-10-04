---
title: "Cardiac FP Analyzer — Manuale utente"
subtitle: "Versione 3.14.1 · analisi del field potential cardiaco su MEA (CSV Digilent e HDF5 Multi Channel Systems)"
author: "Marco Rasponi · github.com/ringo977/cardiac-fp-analyzer"
date: "4 ottobre 2026"
lang: it
---

# 1. Introduzione

## 1.1 Che cosa fa il software

Cardiac FP Analyzer misura, su registrazioni extracellulari di microtessuti cardiaci (hiPSC-CM) acquisite con elettrodi, i parametri del field potential: il **periodo di battito** (BP), la **durata del field potential** (FPD, dallo spike di depolarizzazione all'onda di ripolarizzazione), l'**FPD corretto per la frequenza** (FPDc, Fridericia), l'ampiezza dello spike, la regolarità del ritmo, le aritmie. Su una cartella di registrazioni (baseline e dosi crescenti di uno o più composti) abbina ogni dose al suo riferimento, calcola la variazione percentuale dell'FPDc e decide per ogni composto se **prolunga** la ripolarizzazione, secondo la regola del flusso CiPA (Visone, Lozano-Juan et al., *Toxicological Sciences* 2023).

Il software legge tre formati di registrazione:

| Formato | Origine | Elettrodi | Estensione |
|----------|--------------|------------|------|
| CSV Digilent WaveForms | oscilloscopio Analog Discovery + amplificatore µECG-Pharma | 2 (`el1`, `el2`), 1 tessuto per ingresso | `.csv` |
| HDF5 Multi Channel Systems | MCS Experimenter / DataManager, protocollo RawData v3 | fino a 64 (`E1`…`E64`); il chip µHeart MVP ne ha 4 camere × (12 registrazione + 4 stimolazione) | `.h5`, `.hdf5` |
| Copia compatta MCS | conversione lossless degli export MCS (`mcs_compact` v1) | come il file HDF5 di origine | `.npz` |

Sui chip a più camere ogni **camera** (un microtessuto) è analizzata separatamente: il software sceglie l'elettrodo migliore, ma periodo, ritmo e FPD della camera vengono dal **consenso di tutti i suoi elettrodi** (cap. 7).

## 1.2 Concetti

| Termine | Significato |
|------|------------------|
| Field potential (FP) | potenziale extracellulare registrato dall'elettrodo: uno spike rapido alla depolarizzazione, un'onda lenta alla ripolarizzazione (analogo del QRS e dell'onda T). |
| Battito | lo spike di depolarizzazione; il programma lo cerca con tre rivelatori e sceglie il migliore (§8.2). |
| Periodo di battito (BP) | intervallo tra spike consecutivi; CV = deviazione standard / media. |
| FPD | dallo spike al **picco** dell'onda di ripolarizzazione (metodo di default `peak`; alternative in §8.5). |
| FPDc | FPD / RR^(1/3) (Fridericia). È la grandezza su cui si misura l'effetto di un composto. |
| ΔFPDc % | (FPDc dose − FPDc riferimento) / FPDc riferimento × 100. |
| Riferimento | la registrazione baseline (o `t0`) dello stesso tessuto fatta subito prima della prima dose. |
| Tessuto | esperimento / giorno / chip / camera (es. `exp5/day7/chipA_ch1`, `-/-/chipPM01001_chA`): è l'unità su cui si abbinano baseline e dosi. |
| Camera | una vasca del chip con un microtessuto; sul µHeart MVP 64 sono quattro (A, B, C, D). |
| Stato del ritmo di camera | `regular`, `irregular`, `conduction_lost`, `silent`, `insufficient` (§7.3). |
| QC | voto A–F della registrazione (SNR, battiti scartati, morfologia). |
| Non analizzabile | registrazione senza FPD: troppo rumore, pochi battiti sparsi, ritmo irregolare, conduzione persa, tessuto fermo. |
| Decisione sul composto | positivo se la media per tessuto del ΔFPDc raggiunge il 15 % in due concentrazioni adiacenti misurate su almeno 2 tessuti (§8.9). |

## 1.3 Flussi di lavoro

- **Un file**: aprirlo nella GUI, controllare i battiti e l'onda scelta, leggere i parametri, correggere a mano se serve.
- **Una cartella** (un esperimento: baseline + dosi su più tessuti): riga di comando `cardiac-fp`, con la mappa dei campioni `samples.csv` per dire che cosa c'è in ogni file o camera; report Excel e PDF, decisione per composto.
- **Studio dose–risposta nella GUI**: pannello Studi (Studio → Gruppo → File), curve dose–risposta, export CDISC SEND.

## 1.4 Architettura

- `cardiac_fp_analyzer/`: la libreria (nessuna dipendenza grafica). Punto d'ingresso `analyze.analyze_single_file` e `analyze.batch_analyze`.
- `cardiac-fp`: riga di comando per il batch.
- `pyside_app/`: GUI desktop (PySide6 + pyqtgraph), comando `cardiac-fp-gui`.
- `app.py` + `ui/`: GUI web Streamlit, legacy (solo manutenzione).
- File accessori su disco: `samples.csv` (mappa dei campioni), `<file>.overrides.json` (correzioni manuali dei battiti), `.cfp-study.json` (studio della GUI), `analysis_config.json` (configurazione usata dal batch).

# 2. Installazione

## 2.1 Requisiti

- Python 3.9–3.12 (il pacchetto richiede `numpy < 2` e `pyqtgraph < 0.14`: per questo si usa un ambiente virtuale dedicato, che non tocca il Python di sistema).
- Windows 10+, macOS 12+, Linux. Il codice è portabile; la GUI è provata su macOS e in integrazione continua su Linux.
- RAM: 4 GB per i CSV; **8 GB consigliati per i file MCS a 64 canali** (un file di 5 minuti a 20 kHz occupa 650 MB su disco e si carica in circa 20 s con meno di 0,7 GB di memoria).

## 2.2 Installazione in un ambiente virtuale (consigliata)

```bash
git clone https://github.com/ringo977/cardiac-fp-analyzer.git
cd cardiac-fp-analyzer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[gui,reports,mcs]"
```

Gli extra disponibili:

| Extra | Pacchetti | Serve per |
|--------|------------|------------------|
| `gui` | PySide6 ≥ 6.5, pyqtgraph 0.13 | GUI desktop `cardiac-fp-gui` |
| `reports` | xlsxwriter | report Excel del batch |
| `mcs` | h5py | file HDF5 Multi Channel Systems (`.h5`); i `.npz` non lo richiedono |
| `cdisc` | pyreadstat | export CDISC SEND in `.xpt` |
| `streamlit` | streamlit, plotly | GUI web legacy |
| `all` | tutti i precedenti | installazione completa |
| `dev` | `all` + pytest, ruff | sviluppo e test |

Attenzione: ogni volta che si apre un nuovo terminale va riattivato l'ambiente (`source .venv/bin/activate`), altrimenti `cardiac-fp` e `cardiac-fp-gui` non si trovano ("command not found"). Non aggiungere commenti `# …` in coda ai comandi `pip`: alcune shell li passano a pip come argomento ("Invalid requirement: '#'").

## 2.3 Verifica

```bash
cardiac-fp --help        # stampa le opzioni del batch
cardiac-fp-gui           # apre la finestra principale
python -m pytest -q      # con l'extra dev: la suite di test
```

## 2.4 Aggiornamento

```bash
cd cardiac-fp-analyzer
git pull
source .venv/bin/activate
pip install -e ".[gui,reports,mcs]"
```

# 3. Dati in ingresso

## 3.1 CSV Digilent WaveForms

Intestazione con righe `#` (`Device Name`, `Serial Number`, `Date Time`, `Sample rate`, `Samples`, `Trigger`, `Channel N: Range/Offset`) seguita da `time,el1,el2`. I valori sono in volt all'uscita dell'amplificatore: il software li divide per `amplifier_gain` (default 1e4, µECG-Pharma). Le registrazioni sopra 3 kHz sono decimate a 2 kHz al caricamento; a 10 o 20 kHz il risultato è identico a quello di una registrazione a 2 kHz.

Il nome del file e le cartelle danno l'identità del tessuto e la condizione (§3.4).

## 3.2 HDF5 Multi Channel Systems

File del protocollo RawData v3 di MCS (`/Data/Recording_0/AnalogStream/Stream_s/ChannelData`, `InfoChannel`), HDF5 standard, leggibili anche da MATLAB o HDFView. Il software legge lo stream `Electrode` con "Raw" nell'etichetta (o il primo `Electrode`), converte i codici in volt con `ConversionFactor`, `Exponent` e `ADZero`, decima a 2 kHz a blocchi (senza caricare tutto il file) e produce una colonna per elettrodo con l'etichetta MCS (`E1`…`E64`). Non si applica alcun guadagno. Dai metadati vengono anche la data (`DateInTicks`), gli eventi dello stimolatore (`paced` = vero se c'è almeno un evento `StgSideband`/`DigitalPort`) e, se presenti, i tempi degli spike di MCS. **Le registrazioni stimolate (pacing) non sono ancora analizzate** come tali: il flag `paced` viene solo riportato.

Nome dei file: `McsRecording_<chip>_<condizione>_Recording-<n>_…` → chip (es. `PM01001`) e condizione (`baseline`, lettera di dose, altro) dal nome. Nota: i nomi MCS contengono `;` e `,`: nei fogli `samples.csv` vanno tra virgolette (la bozza li scrive già così).

## 3.3 Copie compatte `.npz`

Formato `mcs_compact v1`: un array di codici ADC (int16/int32) per elettrodo e un `meta` JSON (passo dell'ADC in volt, passo temporale, etichette, file di origine). Si apre esattamente come il file HDF5 di origine (stessi metadati, stessa decimazione), senza h5py; i canali si leggono uno alla volta (circa 30 s per 64 canali × 5 minuti). Lo script di conversione è `PHOENIX/_report/scripts` (fuori dal pacchetto).

## 3.4 Identità del tessuto dai nomi

Per i CSV il software legge dalle cartelle e dal nome: esperimento (`Exp5`, `exp1`, `Exp12_Accelera`), giorno (`Day7`), chip (`chipA`, `Chip 537`, cartella `inj1`), camera (`ch1`…), composto, concentrazione (`300nM`, `7,5`, `2andhalf`), `baseline`/`t0` (riferimento), `t1`…`t7` e `Ctrl` (controlli), `washout`. Esempio: `Exp5/Day7/chipA/chipA_ch1_terfe_300nM.csv` → tessuto `exp5/day7/chipA_ch1`, composto `terfe`, 300 nM.

Nei file con **due tessuti**, uno per ingresso (`Exp1_ChipP_ch1_K01_ChipQ_ch2_K02_A.csv`), il nome elenca i tessuti nell'ordine degli ingressi con il loro test item; `na` segna un ingresso vuoto. Il batch produce una registrazione per ingresso.

Per i file MCS il tessuto è `<esperimento>/-/chip<piastra>_ch<lettera camera>` (es. `-/-/chipPM01001_chA`).

## 3.5 La mappa dei campioni `samples.csv`

Un file `samples.csv` nella cartella analizzata (o in una sottocartella) dice che cosa registra ogni file, ingresso o camera, e **ha la precedenza sui nomi** per i file che elenca.

| Colonna | Contenuto |
|------|------------------|
| `file` | percorso relativo alla cartella del foglio, o il solo nome se è unico; tra virgolette se contiene `;` o `,` |
| `electrode` | CSV: `el1` / `el2` (ingresso), vuoto o `auto`; MCS: la **lettera della camera** (`A`…`D`), oppure l'etichetta di un elettrodo (`E18`) per imporlo |
| `chamber` | in alternativa a `electrode` per la lettera della camera |
| `experiment`, `chip` | facoltative: sostituiscono quanto letto dalle cartelle |
| `item` | test item o composto |
| `dose` | `baseline` (o `t0`) per i riferimenti, `washout`, altrimenti la concentrazione o la condizione (`A`, `300 nM`) |
| `exclude` | qualunque testo esclude la riga; il testo è il motivo |
| `note` | libera |

Intestazioni accettate anche in italiano (`elettrodo`, `camera`, `esperimento`, `farmaco`, `escludi`); separatore virgola o punto e virgola.

**Bozza automatica**: `python -m cardiac_fp_analyzer.sample_sheet <cartella>` scrive `samples_draft.csv` con una riga per file, ingresso o camera, come il software legge i nomi, e in `note` segnala i controlli (camera fuori intervallo, test item che cambia tra le dosi, registrazioni ripetute, camere senza riferimento, ingressi ambigui). Si rivede e si salva come `samples.csv`. Per un chip a quattro camere la bozza ha quattro righe per file: va compilato `item` di ciascuna camera (es. A verapamil, B verapamil, C aspirina, D sotalolo).

# 4. Riga di comando: `cardiac-fp`

```
cardiac-fp <cartella> [-o <uscita>] [--channel auto|el1|el2|both|E18]
           [--config cfg.json | --preset nome] [--correction fridericia|bazett|none]
           [--fpd-method peak|tangent|max_slope|50pct|baseline_return|consensus]
           [--inclusion-cv 25] [--no-fpdc-filter] [-q]
```

| Opzione | Default | Significato |
|----------|------|-------------------|
| `cartella` | — | scansione ricorsiva di `.csv`, `.h5`, `.npz` (i fogli campioni e i file che non sono registrazioni sono ignorati) |
| `-o`, `--output` | `<cartella>/analysis_results` | cartella dei report |
| `--channel` | `auto` | `auto` sceglie l'elettrodo (sul riferimento del tessuto; le dosi lo ereditano); `el1`/`el2`/`both` per i CSV; un'etichetta (`E18`) per i file MCS |
| `--config` | — | file JSON di `AnalysisConfig` (ha la precedenza su tutto il resto) |
| `--preset` | — | `default`, `conservative`, `sensitive`, `peak_method`, `no_filters` |
| `--correction` | `fridericia` | formula dell'FPDc |
| `--fpd-method` | `peak` | punto dell'onda misurato (§8.5) |
| `--inclusion-cv` | 25 | CV massimo del periodo sui baseline; 0 disattiva |
| `--no-fpdc-filter` | — | disattiva l'intervallo di plausibilità 100–1200 ms |
| `-q` | — | silenzioso |

Che cosa fa il batch, nell'ordine: legge `samples.csv` (o i nomi) e costruisce la lista delle registrazioni (una per file, per ingresso o per camera); analizza prima i **riferimenti** di ogni tessuto, poi le dosi con lo stesso elettrodo e, sui chip, con i template di camera del riferimento; applica i criteri di inclusione ai baseline; abbina ogni dose al suo riferimento e calcola ΔBP, ΔFPDc, Δampiezza; ricalcola le aritmie delle dosi con il template del baseline; decide per ogni composto; scrive i report.

Esempi:

```bash
cardiac-fp /dati/exp7 -o /dati/exp7/out                    # CSV Digilent, elettrodo automatico
cardiac-fp /dati/PM01001_h5 -o /dati/PM01001_h5/out          # chip µHeart: una registrazione per camera
cardiac-fp /dati/exp2 --preset conservative                  # segnali rumorosi
cardiac-fp /dati/exp4 --channel el1 --correction bazett
```

**Uscita** in `analysis_results/`:

- `cardiac_fp_analysis_<data>.xlsx`: fogli `Summary` (una riga per registrazione: tessuto, composto, dose, elettrodo, camera, BP, CV, FPD, FPDc, QC, stato, motivo di non analizzabilità, inclusione), `Normalization` (ΔBP, ΔFPDc, Δampiezza, TdP score, riferimento usato), `Arrhythmia Flags`, `Per-Beat Data`;
- `cardiac_fp_analysis_<data>.pdf`: una pagina per registrazione (segnale, battiti, template, onda scelta) e le pagine di sintesi;
- `analysis_config.json`: la configurazione completa usata, da conservare con i risultati.

La decisione per composto e la mappa di rischio si ottengono da Python (§4.1).

## 4.1 Da Python

```python
from cardiac_fp_analyzer.config import AnalysisConfig
from cardiac_fp_analyzer.analyze import analyze_single_file, batch_analyze
from cardiac_fp_analyzer.normalization import classify_drug
from cardiac_fp_analyzer.risk_map import generate_risk_map

cfg = AnalysisConfig()                       # default: Fridericia, metodo 'peak', consenso di camera attivo
r = analyze_single_file('exp7/chipA_ch1_baseline.csv', channel='auto', config=cfg)
print(r['summary']['fpdc_ms_mean'], r['summary']['beat_period_ms_median'], r['qc_report'].grade)

results = batch_analyze('/dati/exp7', channel='auto', config=cfg, output_dir='/dati/exp7/out')
calls = classify_drug(results, cfg)          # una decisione per composto
fig = generate_risk_map(results, config=cfg)
```

Sui file MCS `analyze_single_file(path, channel='auto')` analizza il chip come una sola camera a meno che `file_info_update` non indichi la camera (è ciò che fanno il batch e la GUI); per i chip conviene quindi passare dal batch o dalla GUI.

# 5. GUI desktop: `cardiac-fp-gui`

## 5.1 Finestra principale

Menu **File**: *Apri registrazione…* (Ctrl+O; CSV, `.h5`, `.npz`), *Impostazioni…* (Ctrl+, ; su Mac nel menu dell'app), *Esci* (Ctrl+Q). Menu **Studi**: *Mostra pannello studi* (Ctrl+Shift+P). Menu **Visualizza ▸ Tema**: scuro / chiaro. Scorciatoie: Ctrl++ / Ctrl+− zoom, Ctrl+0 vista intera.

Quattro schede: **Segnale**, **Battiti**, **Parametri**, **Aritmie**. Il titolo della finestra riporta il file, e per i chip la camera e l'elettrodo (`camera A · E24`).

## 5.2 Scheda Segnale

Barra dei comandi:

- **Camera** (solo file a più camere): la camera analizzata; cambiandola l'elettrodo torna su *Auto* e l'analisi è rifatta senza rileggere il file.
- **Canale**: *Auto*, oppure `el1`/`el2` per i CSV, oppure gli elettrodi della camera per i chip (quelli di stimolazione sono marcati *(stim)* e non vengono scelti da *Auto*). Accanto, l'elettrodo effettivamente analizzato.
- **Modalità** *Visualizza* / *Edit*; **Mostra raw** sovrappone il segnale non filtrato; *Pan* / *Box* per il tipo di zoom; **＋ − Auto**; **PNG** esporta la vista; **Ricalcola** rilancia l'analisi con i battiti correnti.

**Mappa del chip** (file MCS): una riga per camera, un riquadro per elettrodo nell'ordine fisico lungo il canale (le due coppie di stimolazione alle estremità, tratteggiate; i 12 di registrazione in mezzo). Il colore è il **punteggio rapido** dell'elettrodo (scuro = nessun battito, giallo = spike netti, ritmo regolare, ripolarizzazione visibile); il riquadro bianco è l'elettrodo in uso. Passando il mouse: numero di spike, periodo, CV, SNR, ampiezza relativa della ripolarizzazione. Un **clic** su un elettrodo lo analizza (cambiando camera se serve). L'etichetta di ogni riga riporta il periodo della camera.

Il grafico: segnale filtrato, marcatori verticali sui battiti (colore per stato: accettato, scartato dal QC, aggiunto a mano), zoom con rotella e trascinamento.

**Correzione manuale**: in modalità *Edit* un clic sinistro sul grafico apre un menu: *Aggiungi depolarizzazione (qui)* o *Rimuovi depolarizzazione* (il battito più vicino), *Aggiungi ripolarizzazione (qui)* o *Rimuovi ripolarizzazione*; il punto aggiunto si aggancia all'estremo locale del segnale. Nel file delle correzioni restano i battiti (depolarizzazioni) aggiunti e rimossi; i marcatori di ripolarizzazione servono alla vista. Dopo le modifiche **Ricalcola** (F5) rilancia la pipeline con i battiti corretti e li salva in `<file>.overrides.json` accanto alla registrazione (tempi in secondi, tolleranza 50 ms); alla prossima apertura sono riapplicati (`use_overrides`). Con correzioni manuali il treno del ritmo (§8.3) non viene applicato: il set di battiti è quello dell'utente.

## 5.3 Scheda Battiti

Tabella battito per battito (tempo, ampiezza, periodo, FPD, FPDc, correlazione con il template, stato QC, flag) con casella *Includi*; *Seleziona tutti* / *Deseleziona tutti*; a destra il template (battito medio) con l'onda di ripolarizzazione scelta. Escludere un battito e premere **Ricalcola** equivale a rimuoverlo in modalità Edit.

## 5.4 Scheda Parametri

*Parametri per battito* e *Riepilogo*: periodo (media, mediana, CV), FPD, FPDc, ampiezza, tempo di salita, `fpd_reliable`, `fpd_confidence`, voto QC; sui chip anche lo **stato della camera**, la sincronia, il CV robusto, la **provenienza dell'FPD** (`chamber consensus (10 electrodes)`, `single electrode`) e, tra parentesi, i valori del solo elettrodo. Se la registrazione non è analizzabile compare il motivo.

## 5.5 Scheda Aritmie

Classe del ritmo e punteggio di rischio 0–100, flag rilevate (tachicardia, bradicardia, irregolarità, prematuri, ritardati, pause, EAD, FPD molto lungo, e sui chip *ritmo di camera irregolare / conduzione persa / tessuto fermo*), analisi del residuo (EAD), eventi.

## 5.6 Impostazioni

File ▸ Impostazioni. In alto il **preset** (`default`, `conservative`, `sensitive`, `peak_method`, `no_filters`) da caricare nei campi; cinque pagine: **Segnale** (notch, passa-banda, Savitzky–Golay, guadagno), **Detection** (metodo, distanza minima, soglia, secondo tentativo, validazione morfologica), **Repolarizzazione** (metodo FPD, correzione, finestra di ricerca, FPD minimo, soglie di qualità), **QC** (soglie SNR del voto, frazioni di scarto), **Inclusione** (criteri sui baseline). *Applica* o *OK* rendono effettive le modifiche; il file corrente va ricalcolato. Il significato di ogni campo è nella scheda dei parametri (cap. 8 e `DOCUMENTATION.md` §11).

## 5.7 Pannello Studi

Modello **Studio → Gruppo → File**: un gruppo è una condizione (baseline, oppure composto + dose in µM) con la propria configurazione; i file sono registrazioni (percorsi relativi alla cartella dello studio). Azioni: *Nuovo studio*, *Apri studio*, *Chiudi studio*, *Aggiungi gruppo*, *Aggiungi CSV al gruppo*, *Aggiungi cartella al gruppo*, *Impostazioni gruppo*, *Rimuovi*, *Analizza gruppo*, *Analizza studio* (in background, con avanzamento e annulla), *Dose-risposta…* (FPDc, FPD, BPM o STV contro la dose, media ± SD per gruppo, asse logaritmico, baseline tratteggiato), *Esporta CDISC…*, *Elimina sidecar studio*. Ogni gruppo mostra l'aggregato (`FPDc 410±12 ms · 58±3 BPM · STV 2.1 ms · n=4/5`); un pallino ● segnala i risultati non aggiornati rispetto alla configurazione corrente. Lo studio è salvato in `.cfp-study.json` nella cartella (percorsi POSIX, portabile tra macchine); il doppio clic su un file lo apre nella finestra principale.

Il pannello lavora sui file che gli si aggiungono; la lettura di `samples.csv` e il flusso per camera dei chip sono oggi del batch da riga di comando.

## 5.8 GUI web Streamlit (legacy)

`pip install ".[streamlit]"` e `streamlit run app.py`: pagine *Analisi singolo file*, *Batch + Risk Map*, *Confronto farmaci*. In sola manutenzione: nessuna nuova funzione (le camere, i file MCS, il consenso vi arrivano solo attraverso la libreria).

# 6. Come leggere un risultato

Per ogni registrazione il risultato (`analyze_single_file` → dizionario; nel report, una riga) contiene:

| Campo | Significato |
|------------|---------------|
| `summary.n_beats` | battiti rilevati |
| `summary.beat_period_ms_mean / _median / _cv` | periodo di battito (sui chip: della camera; il valore dell'elettrodo in `*_electrode`) |
| `summary.fpd_ms_median` | **FPD della registrazione** (mediana per battito, o consenso di camera) |
| `summary.fpdc_ms_mean` | **FPDc della registrazione** (usato per ΔFPDc e per la decisione) |
| `summary.fpd_reliable`, `fpd_valid_ratio` | almeno metà dei battiti ha un FPD valido |
| `summary.fpd_confidence` | 0–1, qualità dell'onda sul template e coerenza tra battiti |
| `summary.fpd_source` | da dove viene l'FPD: `chamber consensus (n)`, `chamber same wave (n)`, `single electrode (…)` |
| `summary.chamber_status`, `chamber_status_reason`, `chamber_synchrony`, `beat_period_cv_robust_pct` | stato del ritmo della camera |
| `summary.not_analysable`, `not_analysable_reason` | perché la registrazione non ha FPD |
| `summary.spike_amplitude_mV_mean`, `rise_time_ms_mean` | spike |
| `qc_report.grade` | A–F |
| `arrhythmia_report.classification`, `.risk_score`, `.flags` | aritmie |
| `inclusion.passed`, `inclusion.fail_reason` | criteri sui baseline (batch) |
| `normalization.pct_fpdc_change`, `pct_bp_change`, `pct_amp_change`, `baseline_file`, `tdp_score` | variazioni rispetto al riferimento (batch) |
| `file_info.tissue`, `chamber`, `analyzed_channel`, `electrode_scores`, `tissue_electrode_from`, `paced` | identità e scelta dell'elettrodo |
| `chamber` | tutte le misure di camera (FPD per elettrodo, correlazioni, template) |
| `detection_info` | rilevatore scelto, polarità, topologia del ritmo, treno del ritmo, battiti recuperati |

Controlli consigliati su una registrazione prima di fidarsi del numero: (1) i marcatori sono sugli spike, senza doppi né mancanti; (2) nella scheda Battiti il template mostra l'onda di ripolarizzazione marcata sull'onda giusta (sul baseline), e sulle dosi `fpd_source` è `same wave`; (3) `fpd_reliable` vero e QC ≥ C; (4) sui chip, stato di camera `regular` e almeno 3–5 elettrodi concordi.

# 7. Chip a più camere (µHeart MVP 64)

## 7.1 Layout

Quattro moduli da 16 elettrodi; in ciascuno due coppie di stimolazione alle estremità del canale e 12 elettrodi di registrazione in fila con passo 400 µm (il dodicesimo è uno dei pad E61–E64). Lettere del laboratorio: **A** = E16–E30 + E62, **B** = E31–E45 + E63, **C** = E1–E15 + E61, **D** = E46–E60 + E64. Il layout si riconosce automaticamente da 64 canali `E1…E64` (`chamber_layout = 'auto'`); `'none'` tratta il file come un solo tessuto.

## 7.2 Una registrazione per camera

Nel batch ogni file MCS diventa quattro registrazioni, una per camera, con il tessuto `chip<piastra>_ch<lettera>`; il test item di ogni camera va in `samples.csv` (§3.5). Per ogni camera il software sceglie l'elettrodo di registrazione con il punteggio rapido più alto (spike netti, ritmo regolare, ripolarizzazione visibile; 1–2 s per file), lo usa per la pipeline completa e lo mantiene sulle dosi dello stesso tessuto; se su una dose quell'elettrodo non dà un'analisi valida riprova con il migliore degli altri (`tissue_electrode_from` lo annota). Sui ritmi veloci (periodo degli spike sotto 440 ms, regolare) la distanza minima tra battiti scende a metà del periodo, così non si perde un battito su due.

## 7.3 Misure di camera (consenso)

Con `chamber_consensus = True` (default) ogni registrazione di camera è misurata anche con **tutti i suoi elettrodi di registrazione**:

1. **Elettrodi usabili**: segnale né piatto né saturo, almeno 10 spike, periodo entro il 20 % di quello mediano. Ne servono almeno 3, altrimenti lo stato è `insufficient` e restano i valori del singolo elettrodo.
2. **Battiti comuni**: visti da almeno 3 elettrodi entro 60 ms. Il **periodo di camera** è la loro mediana; il CV robusto è 1,4826·MAD/mediana; la **sincronia** è la frazione dei battiti di ogni elettrodo che coincide con un battito comune.
3. **Stato del ritmo**: `silent` (meno di 10 battiti comuni, o molti meno elettrodi attivi che al baseline), `conduction_lost` (sincronia < 0,5: il battito non attraversa più il tessuto), `irregular` (CV robusto > 15 %), altrimenti `regular`.
4. **FPD di consenso** (riferimento): su ogni elettrodo l'FPD del battito mediano; validi quelli sotto l'80 % del periodo; il gruppo più numeroso di elettrodi concordi entro il 15 % dà l'FPD (mediana) e il numero di elettrodi (`fpd_source = chamber consensus (10 electrodes)`). L'onda di ogni elettrodo diventa il template della camera.
5. **Stessa onda** (dosi): il template del riferimento è cercato per correlazione (≥ 0,8) sul battito mediano della dose di ogni elettrodo; stessa regola di concordanza (`chamber same wave (7 electrodes)`). Così la dose misura l'onda che il riferimento misurava, non un'altra deflessione resa più prominente dal composto.
6. **FPDc** = FPD / periodo di camera^(1/3).

Nel sommario periodo, CV, FPD e FPDc di camera prendono il posto di quelli dell'elettrodo (che restano in `*_electrode`). Con stato `irregular`, `conduction_lost` o `silent` la registrazione è **non analizzabile** con quel motivo in italiano ("ritmo irregolare: CV robusto 57 %", "conduzione persa", "tessuto fermo"): niente FPD, perché un FPD su un tessuto aritmico non misura la ripolarizzazione. Il periodo resta. Aprendo un file da solo nella GUI (senza riferimento) l'FPD è il consenso della registrazione stessa.

Sulle cinque piastre PHOENIX (108 registrazioni di camera) il consenso riproduce l'analisi di riferimento con differenza mediana dello 0,1 % sul periodo e del 2,0 % sull'FPDc (88 % entro il 5 %, 96 % entro il 10 %), contro il 5,1 % (49 %, 75 %) dell'elettrodo singolo.

## 7.4 Che cosa aspettarsi dai composti

Con il flusso per camera sui dati PHOENIX: verapamil (A, B) accorcia l'FPDc in modo dose-dipendente e accelera il ritmo; aspirina (C) non lo modifica; sotalolo (D) lo allunga alle dosi basse e alle dosi alte rende il tessuto irregolare o con conduzione persa (stati `irregular`/`conduction_lost`, nessun FPD: è il comportamento atteso, non un errore del software). Il periodo di battito si accorcia del 5–10 % nei 5 minuti di ogni registrazione (deriva termica): confrontare tratti omologhi delle registrazioni.

# 8. Scheda dei parametri (come si ottiene ogni numero)

Questo capitolo riassume le regole; la versione completa, verificata riga per riga sul codice della v3.14.0, con tutti i nomi dei campi di `AnalysisConfig` e i loro default, è il §11 di `DOCUMENTATION.md`. I nomi tra parentesi sono i campi di configurazione.

## 8.1 Ordine delle operazioni

Caricamento e decimazione a 2 kHz → scelta dell'elettrodo → guadagno (solo CSV) → filtri (notch 50 Hz con 3 armoniche, passa-banda 0,5–500 Hz, Savitzky–Golay 7/3) → rilevamento dei battiti (+ secondo tentativo) → correzioni manuali → treno del ritmo → segmentazione, QC, verdetto di analizzabilità → template → FPD sul template e per battito → FPDc → aritmie, cessazione, spettro → misure di camera.

Insiemi di battiti: **tutte le rilevazioni** (verdetto, aritmie, cessazione); **treno del ritmo** se applicato, altrimenti tutte (periodo, CV, RR locale dell'FPDc); **accettati dal QC** (residui, spettro); **battiti per l'FPD** = accettati meno i gruppi minoritari e le pause (template, FPD, FPDc, ampiezza); **battiti comuni della camera** (periodo e CV di camera).

## 8.2 Battiti

Tre rivelatori sul segnale filtrato (prominenza, derivata, picco), distanza minima 400 ms (`min_distance_ms`), soglia 4 × rumore robusto (`threshold_factor`). Ciascuno riceve un punteggio (periodo medio in 0,4–3 s: 30; CV < 15/30/50 %: 30/20/10; numero di battiti plausibile: 20; bonus 15 al rivelatore a derivata) e vince il più alto. Seguono, in ordine: correzione del periodo bimodale, soglia di rumore, scarto della popolazione minore, gruppi di ampiezza, classificazione della topologia (regolare, caotico, alternanza, ectopici, rumore, trimodale), validazione morfologica (correlazione ≥ 0,7 con il template, ampiezza ≥ 25 %), recupero dei battiti mancanti nei buchi, seconda soglia di rumore, filtro adattato a basso SNR. Se restano meno di 5 battiti su più di 10 s, secondo tentativo con distanza 300 ms e soglia 3 (stessa configurazione per il resto).

## 8.3 Periodo, frequenza, CV

Intervalli tra battiti consecutivi: media, mediana, CV (std/media), `bpm_mean`, STV di Poincaré. Se il CV di tutte le rilevazioni è ≥ 25 % e ci sono ≥ 6 battiti, il **treno del ritmo** (`enable_rhythm_train`, attivo) trova il periodo dominante e tiene la sottosequenza di battiti compatibile: periodo e CV descrivono il ritmo di fondo e non le rilevazioni spurie; su un tessuto regolare non cambia nulla. Sui chip il periodo e il CV sono quelli della camera (§7.3).

## 8.4 QC e analizzabilità

Per battito: ampiezza ≥ 25 % della mediana (`amplitude_reject_fraction`), correlazione con il template ≥ 0,40 (`morphology_threshold`, con soglia adattiva fino a 0,20 sui tracciati difficili), riammissione dei battiti sul ritmo. Voto: F (< 3 battiti, o SNR < 2 e scarti > 60 %), D (SNR < 3 o scarti > 40 %), C (SNR < 5 o scarti > 20 % o r medio < 0,40), B (SNR < 8 o scarti > 5 %), A.

**Non analizzabile** se: battiti < 3; SNR mediano dei battiti < 1,6; meno di 16 battiti con CV > 40 %; sui chip, camera irregolare / conduzione persa / ferma. Effetto: FPD, FPDc e confidenza a NaN, `fpd_reliable` falso, voto F, classe "Not analysable"; periodo e ampiezza restano.

## 8.5 FPD

Prima **sul template** (battito mediano di fino a 60 battiti allineati): finestra da 150 ms dopo lo spike a max(900 ms, 70 % del RR), tagliata prima dello spike successivo; passa-basso 20 Hz e detrend; FPD minimo max(120 ms, 20 % RR, ≤ 600 ms); candidati = picchi positivi e negativi con prominenza ≥ 15 % della deviazione; **scelta dell'onda** `prefer_positive` (attiva): il picco positivo più prominente, purché abbia almeno metà della prominenza massima e stia entro 400 ms dal candidato più prominente, altrimenti il più prominente (alternativa `max_prominence`); soglia di qualità prominenza/rumore ≥ 2. **Punto misurato** (`fpd_method`): `peak` (attivo) = latenza del picco dell'onda; alternative `tangent` (intersezione della tangente di discesa con lo zero), `max_slope`, `50pct`, `baseline_return`, `consensus`. Il picco è il punto più ripetibile tra battiti ed elettrodi, ed è quello con cui è stata fatta la validazione sul report PHOENIX D10.1.

Poi **per battito**, attorno alla posizione trovata sul template (± 150 ms), con soglia 1,5 × rumore; i battiti senza onda hanno FPD NaN. `fpd_ms_median` è la mediana; `fpd_reliable` richiede ≥ 50 % di battiti validi; `fpd_confidence` = metà qualità del template, metà coerenza tra battiti.

Sui chip l'FPD è il **consenso di camera** (§7.3); il valore dell'elettrodo resta in `fpd_ms_median_electrode`.

## 8.6 FPDc

Per battito, FPD / RR^(1/3) con il RR locale (battito precedente), `correction = 'fridericia'` (alternative `bazett`, `none`; il sommario riporta sempre entrambe le formule). `fpdc_ms_mean` è la media per battito, oppure l'FPDc di camera (FPD di consenso / periodo di camera^(1/3)). Nessun ΔFPDc se il periodo supera 6 s (`max_beat_period_for_fpdc_ms`): la correzione non ha senso su un tessuto quasi fermo.

## 8.7 Ampiezza e spike

Picco-picco da 10 ms prima a 20 ms dopo lo spike, sul segnale filtrato e corretto per il guadagno (`spike_amplitude_mV_mean`); tempo di salita 10–90 %; dV/dt massimo per battito.

## 8.8 Aritmie

Su tutte le rilevazioni: tachicardia (BP < 300 ms), bradicardia (> 2500 ms), irregolarità (CV > 15 %, critica > 30 %), prematuri (< 0,7 × medio), ritardati (> 1,5 ×), pause (> 3 ×), STV alta (> 10 ms), FPD > 500 ms, EAD statistici (> 3 MAD) e da residuo (picco positivo del residuo battito − template, 100–450 ms dopo lo spike). Punteggio 0–100 a pesi fissi; classe dalla prima condizione che vale (Fibrillation-like, EAD with Triggered Activity, …, Irregular Rhythm, Normal Sinus Rhythm). La **cessazione** (silenzio, buchi, decadimento, silenzio terminale) alimenta il TdP score e la risk map; lo **spettro** (Welch, entropia, armoniche) dà l'indice proaritmico della risk map.

## 8.9 Batch: inclusione, abbinamento, decisione

**Inclusione** (solo baseline, primo criterio che fallisce): CV del periodo ≥ 25 % o NaN; FPDc fuori 100–1200 ms; confidenza < 0,66; FPD/RR > 0,80. Un tessuto è escluso solo se **tutti** i suoi baseline falliscono.

**Riferimento** di una dose: l'ultimo baseline/t0 analizzabile, e che ha passato l'inclusione, del tessuto prima della prima dose (con gli orari di acquisizione); senza orari, stessa cartella, poi t0, poi voto QC. Se nessun baseline del tessuto ha passato l'inclusione la dose resta senza riferimento, con il motivo.

**Variazioni**: ΔBP da `beat_period_ms_mean`, ΔFPDc da `fpdc_ms_mean`, Δampiezza da `spike_amplitude_mV_mean`. TdP score per registrazione: 3 (cessazione, ΔFPDc ≥ 20 %, o EAD critico con ≥ 10 %), 2 (≥ 15 %), 1 (≥ 10 %), −1 (≤ −10 %), 0.

**Decisione sul composto** (`classification_method = 'concentration'`): dosi abbinate, analizzabili, incluse (washout, veicolo e controlli esclusi); concentrazioni lette come numeri; media per tessuto, poi tra tessuti; una concentrazione conta se misurata su ≥ 2 tessuti; **positivo** se 2 concentrazioni adiacenti hanno media ≥ 15 % (`classification_threshold = 'mid'`); **dati insufficienti** se contano meno di 2 concentrazioni; **negativo** altrimenti. Non esiste una classe "accorcia": l'accorciamento compare nel TdP score −1 e nei ΔFPDc negativi delle singole registrazioni. Alternative: `n_above`, `mean`, `max` (quella usata fino alla v3.7, che sui dati Visone 2023 chiamava positivi tutti i negativi).

# 9. Configurazione

`AnalysisConfig` (`cardiac_fp_analyzer/config.py`) raccoglie le sezioni `filtering`, `beat_detection`, `repolarization`, `quality`, `inclusion`, `normalization`, `arrhythmia`, `channel_selection` e i campi di vertice `amplifier_gain` (1e4), `enable_cessation` (True), `enable_spectral` (True), `use_overrides` (True), `chamber_layout` (`'auto'`), `chamber_consensus` (True). Si esporta e importa in JSON (`to_json`, `from_json`); il batch salva sempre `analysis_config.json` nei risultati.

Preset: `default`; `conservative` (morfologia 0,8, QC 0,6, confidenza minima 0,75: segnali rumorosi); `sensitive` (morfologia 0,5, FPD minimo 80 ms: segnali deboli o bradicardici); `peak_method` (oggi identico al default); `no_filters` (disattiva i filtri di ritmo, ampiezza e topologia: per confronti con pipeline vecchie).

Esempio di JSON minimale:

```json
{
  "repolarization": {"fpd_method": "peak", "correction": "fridericia"},
  "inclusion": {"max_cv_bp": 25.0},
  "normalization": {"classification_threshold": "mid"},
  "amplifier_gain": 10000.0,
  "chamber_consensus": true
}
```

I parametri di cessazione e spettro non sono modificabili da `AnalysisConfig`. Alcuni campi che esistevano ma non venivano letti (`filtering.highpass_*`, `filtering.lowpass_*`, `topology_noise_gap_ratio`, `ead_critical_count`, `premature_count_threshold`, `tdp_require_severe_only`, `snr_good/snr_fair` della scelta del canale) sono stati rimossi nella v3.14.1; i file JSON che li contengono si leggono comunque.

# 10. Export CDISC SEND

```python
from cardiac_fp_analyzer.cdisc_export import export_send_package
export_send_package(results, 'out_send/', study_id='CIPA2026_01')
```

Domini TS, DM, EX, EG, TX, DS, SUPPEG + `define.xml` (SENDIG 3.1.1); formato `.xpt` con pyreadstat (extra `cdisc`), altrimenti `xport`, altrimenti CSV con avviso (non regolatorio). Dalla GUI: pannello Studi ▸ *Esporta CDISC…*. Validare con Pinnacle 21 prima di una submission.

# 11. Limiti noti e lavori in corso

- **Pacing**: le registrazioni stimolate sono riconosciute (`paced`) ma non analizzate (artefatto, cattura, latenza).
- **GUI**: la mappa dei campioni, il flusso per camera del batch e la decisione per composto sono solo da riga di comando/Python; il pannello Studi lavora per file. I punteggi degli elettrodi sono compositi (SNR + ripolarizzazione − CV); nella mappa del chip il colore è relativo al file.
- **Vincoli di versione**: `numpy < 2` e `pyqtgraph < 0.14` finché non saranno provati; da qui l'ambiente virtuale.
- **Validazione**: CiPA (Visone 2023) 6/7 composti; gold standard manuale GG; PHOENIX D10.1 cinque piastre (§7.3). Il consenso di camera è validato contro l'analisi di riferimento del report PHOENIX, non ancora contro misure manuali indipendenti del laboratorio.
- **Verifica della v3.14.0**: le discordanze tra commenti e codice e il difetto del filtro di ritmo (indici di gruppo non aggiornati) trovati scrivendo la scheda dei parametri sono stati corretti nella v3.14.1 (`DOCUMENTATION.md` §11.12); i valori di camera sulla piastra PHOENIX PM01001 sono invariati.

# 12. Risoluzione dei problemi

| Sintomo | Causa probabile e rimedio |
|------|------------------|
| `cardiac-fp-gui: command not found` | ambiente virtuale non attivo: `source .venv/bin/activate` |
| `Invalid requirement: '#'` da pip | commento in coda al comando: ripeterlo senza `# …` |
| `No module named h5py` aprendo un `.h5` | `pip install -e ".[mcs]"`; i `.npz` non lo richiedono |
| "Nessun battito rilevato" | guadagno sbagliato (CSV: `amplifier_gain`), tessuto fermo, elettrodo piatto: provare un altro elettrodo o il preset `sensitive` |
| FPD NaN con `fpd_reliable` falso | onda di ripolarizzazione non visibile: controllare il template; sui chip il motivo è in `chamber_status_reason` |
| "ritmo irregolare" / "conduzione persa" / "tessuto fermo" | stato della camera: è un risultato, non un errore; il periodo resta disponibile |
| Dose senza ΔFPDc | nessun riferimento analizzabile per quel tessuto, baseline escluso dall'inclusione, o periodo > 6 s: vedere `Normalization` nel report |
| Composto "dati insufficienti" | meno di 2 concentrazioni misurate su ≥ 2 tessuti: controllare `samples.csv` (item, dosi, unità) |
| File MCS lento | 64 canali × 5 min a 20 kHz: ~20 s per `.h5`, ~30 s per `.npz`, è normale; servono 8 GB di RAM |
| Risultato non aggiornato (●) nel pannello Studi | configurazione cambiata dopo l'analisi: *Analizza gruppo* |
| Il batch legge male i nomi | scrivere `samples.csv` (bozza con `python -m cardiac_fp_analyzer.sample_sheet <cartella>`) |

# Appendice A — Glossario dei campi usati nei report

`beat_period_ms_median` periodo di battito; `beat_period_ms_cv` CV %; `fpd_ms_median` FPD; `fpdc_ms_mean` FPDc; `fpd_source` provenienza dell'FPD; `fpd_reliable` / `fpd_valid_ratio` affidabilità; `fpd_confidence` confidenza 0–1; `chamber_status` stato del ritmo di camera; `chamber_synchrony` sincronia tra elettrodi; `beat_period_cv_robust_pct` CV robusto; `not_analysable_reason` motivo; `tissue` tessuto; `analyzed_channel` elettrodo; `tissue_electrode_from` perché quell'elettrodo; `pct_fpdc_change` ΔFPDc %; `tdp_score` punteggio TdP per registrazione; `inclusion.passed` baseline incluso; `QC Grade` voto nel report Excel.

# Appendice B — Riferimenti

- Visone R., Lozano-Juan F., et al. *Toxicological Sciences* 191(1):47–60, 2023 — flusso CiPA su microtessuti, correzione di Fridericia, soglie 10/15/20 %.
- Multi Channel Systems, *HDF5 MCS Raw Data Definition*, protocollo v3.
- Progetto PHOENIX, deliverable D1.2 (layout µHeart MVP 64) e D10.1 (analisi di riferimento).
- `DOCUMENTATION.md` del repository: documentazione tecnica completa, §11 scheda dei parametri, §12 changelog.
