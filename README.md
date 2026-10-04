# Cardiac FP Analyzer

[![CI](https://github.com/ringo977/cardiac-fp-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ringo977/cardiac-fp-analyzer/actions/workflows/ci.yml)

**Versione**: 3.12.1
**Python**: ≥ 3.9

Analisi automatizzata di **field potential (FP)** per registrazioni µECG da **microtessuti cardiaci hiPSC-CM**, acquisite con oscilloscopio **Digilent WaveForms** (CSV: tempo + 2 canali) o con sistemi **Multi Channel Systems** (file HDF5 del protocollo MCS RawData, fino a 64 elettrodi).

## Funzionalità

- **Caricamento smart**: parsing header WaveForms, file HDF5 Multi Channel Systems (64 elettrodi, 20 kHz, lettura a blocchi), decimazione a 2 kHz, downsampling min-max per plot
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

**Ottobre 2026 — gold standard manuale cieco (GG, 313 elettrodi).** Vedi i changelog v3.5.0 e v3.6.0. In sintesi, sul test tenuto da parte: dove il segnale è buono (grado A) il software è affidabile (BP entro ±10 % nel 91 %, FPD nell'84 %); sui gradi B/C l'FPD è entro ±10 % nel 59 %/50 %; dove il segnale non è analizzabile **lo dichiara** invece di produrre numeri. Resta aperto: la copertura (77 %), i template senza onda T visibile e la sovra-rilevazione residua sui segnali rumorosi.

Il confronto con i valori pubblicati (Visone et al. 2023) è documentato in
`VALIDAZIONE_vs_paper_Visone2023.md` e nei file `ASSESSMENT_*.md`. In sintesi, ad ottobre 2026:

- Su 8 baseline con elettrodo degli autori forzato, **RR e FPD entro ±12 %** del pubblicato
  (6 su 8 entro ±5 %); è il corpus di `tests/test_real_signal_regression.py`, eseguito ad ogni run.
  Una nona fixture (baseline di laboratorio a bassa SNR) è confrontata battito per battito con
  l'altro elettrodo della stessa registrazione.
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

### v3.12.1 (Ottobre 2026) — copie compatte `.npz` degli export MCS
- I file `.npz` del formato `mcs_compact` (copia senza perdita dell'export CSV di DataManager) si aprono come i file HDF5, in GUI (`File ▶ Apri registrazione…`), batch e riga di comando, con una registrazione per camera.

### v3.12.0 (Ottobre 2026) — chip a più camere: una registrazione per camera
- **Layout dei chip** (`chambers.py`): il µHeart MVP a 64 canali (4 camere × 12 elettrodi di registrazione + 4 di stimolazione, passo 400 µm; A = E16–E30 + E62, B = E31–E45 + E63, C = E1–E15 + E61, D = E46–E60 + E64). Un file MCS diventa una registrazione per camera, limitata agli elettrodi della camera; `AnalysisConfig.chamber_layout` (`auto`, nome, `none`).
- **`samples.csv` per camera**: una riga per file e camera (lettera della camera in `electrode` o `chamber`, o l'etichetta di un elettrodo per imporlo) con test item e dose; la bozza le scrive già così. Nomi dei file MCS letti per chip e condizione.
- **Scelta rapida dell'elettrodo** nella camera (spike, regolarità, onda di ripolarizzazione; elettrodi di stimolazione esclusi): circa 20 s per un file da 64 elettrodi invece di 4 minuti. L'elettrodo del baseline vale per le dosi; se su una dose non è analizzabile, la dose è rifatta su un altro elettrodo della camera.
- **Ritmi veloci**: la distanza minima tra battiti segue il periodo degli spike quando il treno è regolare (verapamil 5 µM, 250–330 ms).
- Sulle cinque piastre PHOENIX il batch per camera riproduce il periodo dell'analisi di riferimento (mediana 0,1 %, 95 % entro il 2 %); FPDc sui ritmi sopra 450 ms entro il 10 % nell'81 % dei casi.

### v3.11.0 (Ottobre 2026) — file HDF5 di Multi Channel Systems
- **Lettore MCS-HDF5** (`mcs_hdf5.py`, extra `pip install ".[mcs]"`): i file `.h5` di Multi Channel Experimenter / DataManager (protocollo RawData) si analizzano come i CSV, da riga di comando, batch e GUI. Stream analogici con etichette (`E1`…`E64`), unità e fattore di conversione degli elettrodi; eventi dello stimolatore e della porta digitale (`paced`, `stimulus_times_s` in `file_info`); tempi e ritagli degli spike del rivelatore MCS.
- Lettura a blocchi con decimazione a 2 kHz senza ritardo: una registrazione di 64 canali × 5 minuti a 20 kHz (650 MB) in circa 20 s e 0,7 GB di memoria.
- `--channel` accetta l'etichetta di un elettrodo (`E18`); `auto` valuta tutti gli elettrodi del file e tiene il migliore.
- Non ancora: mappa elettrodi → camere e FPD di consenso per camera, analisi delle registrazioni stimolate.

### v3.10.0 (Ottobre 2026) — treno del ritmo per periodo e CV
- **Treno del ritmo** (`enable_rhythm_train`, attivo di default). Quando il CV del treno rilevato raggiunge il 25 %, periodo di battito, CV, RR locale della correzione e finestra di ripolarizzazione vengono dalla sequenza più regolare tra i rilevamenti, con le lacune riempite dal recupero guidato dalla periodicità. Segmentazione, QC, FPD e aritmie usano ancora tutti i rilevamenti.
  - Motivo: sui dati GG (battito lento, rumore a raffiche, artefatti, onde T grandi) il rivelatore trova un terzo di eventi in più rispetto ai battiti dell'analista. Il CV saliva sopra il 25 % e l'inclusione escludeva il tessuto.
  - Registrazioni con CV ≥ 25 %: da 49 a 3 su 114 negli esperimenti GG di sviluppo (analista: 4), da 30 a 1 su 66 in quelli di verifica (analista: 2).
  - Periodo entro ±10 % dall'analista: da 74 a 86 su 114 (sviluppo), da 43 a 47 su 66 (verifica).
  - Decisioni: GG, 10 test item su 13 come l'analista (prima 9); Visone 2023, 8 composti su 12 (prima 7).
  - Prezzo: entrano registrazioni più rumorose, dove l'FPD è meno affidabile; la differenza mediana di ΔFPDc con l'analista sale di circa un punto.

### v3.9.0 (Ottobre 2026) — file con due tessuti e mappa dei campioni
- **File con due tessuti, uno per ingresso** (convenzione GG): il batch `auto` dà una registrazione per ingresso. Tessuto, test item e dose vengono letti dal nome nell'ordine degli ingressi; `na` indica un ingresso senza tessuto. Prima questi file non venivano abbinati: sul dataset GG 81 delle 136 registrazioni con farmaco, e nessun test item riceveva una decisione.
- **`samples.csv`**: per ogni file e ingresso indica esperimento, chip, camera, test item, dose ed eventuale esclusione, e per i file elencati ha la precedenza sui nomi. La bozza `samples_draft.csv` si crea con `python -m cardiac_fp_analyzer.sample_sheet <cartella>` o dalla pagina batch, e segnala le righe da controllare: camera fuori numero, test item che cambia in una camera, ripetizioni, camere senza riferimento.
- In modalità `auto` un tessuto è un solo gruppo di inclusione e di abbinamento, qualunque ingresso usino i suoi file.
- Colonna del tessuto nel report Excel e nella pagina batch.

### v3.8.3 (Ottobre 2026) — indice della risk map: solo il cambio spettrale
- **Asse Y della risk map = cambio spettrale della forma d'onda rispetto al baseline**, aggregato per concentrazione come nella v3.8.2. Prima i pesi erano 70 % spettrale, 25 % instabilità morfologica, 5 % EAD.
  - Instabilità morfologica e incidenza EAD sono punteggi definiti in questo software, non nel paper: il paper usava il residuo solo per cercare picchi irregolari.
  - Sui 12 composti l'instabilità morfologica non distingue positivi e negativi (AUC 0,47) e l'EAD abbassa l'ordinamento.
  - Il cambio spettrale da solo ordina meglio (AUC 0,86 contro 0,78). Le due componenti restano calcolate e riportate e si possono ripesare.
  - Zona alta: cisapride, dofetilide, chinidina, ranolazina e verapamil; zona bassa: aspirina; veicolo a 32. La separazione resta debole: il verapamil, negativo, è in zona alta.
- **Attribuzioni corrette** nel codice e nella documentazione: le metriche sul residuo (instabilità morfologica, incidenza EAD, CV d'ampiezza) erano indicate come metodo del paper.
- Asse Y rinominato "Waveform change vs baseline"; etichette delle zone spostate a destra per non finire sotto la legenda.

### v3.8.2 (Ottobre 2026) — indice proaritmico della risk map per concentrazione
- **Asse Y della risk map aggregato come la decisione per farmaco.**
  - L'indice (cambio spettrale 70 %, instabilità morfologica 25 %, EAD 5 %) si calcola su ogni registrazione utilizzabile.
  - Poi si fa la media tra tessuti a ogni concentrazione (almeno 2 tessuti) e si prende il livello mantenuto su 2 concentrazioni adiacenti.
  - Prima ogni componente era il massimo su tutte le registrazioni: sul dataset Visone 2023 tutti i 12 composti, veicolo compreso, finivano nella zona ad alto rischio.
  - Ora in zona alta ci sono chinidina, dofetilide e cisapride, in zona bassa l'aspirina, tutti gli altri in zona intermedia (veicolo a 30).
  - La separazione resta debole: pesi e zone non sono stati ritoccati.
- **⚡ cessazione** solo con confidenza > 0,5, la soglia della decisione per farmaco: prima compariva su tutti i composti.
- **Correzione delle note della v3.8.0.** L'override di cessazione non avrebbe reso positivi 10 composti su 12: con la soglia di confidenza scatta per tre composti e cambia una sola decisione, l'aspirina da negativa a positiva. Il default spento resta corretto.
- **Verifica**: risk map sui 10 esperimenti del paper. Aggiunti 3 test

### v3.8.1 (Ottobre 2026) — risk map allineata alla decisione per farmaco
- **Asse X della risk map = statistica della decisione per farmaco** (`decision_value` di `classify_drug`).
  - Con il metodo predefinito è il livello che la media tra tessuti mantiene su 2 concentrazioni adiacenti.
  - La linea continua è la soglia: a destra il farmaco è positivo, a sinistra negativo.
  - Prima l'asse era la variazione massima di una singola registrazione. Sul dataset Visone 2023 quinidina e cisapride arrivavano a circa 500 % e il veicolo a 234 %, quindi tutti i negativi stavano oltre la soglia.
- La decisione viene ricalcolata sui risultati passati alla mappa: unendo più batch, ogni farmaco usa tutti i suoi tessuti. Il veicolo è posizionato con la stessa statistica; i farmaci senza decisione stanno in una fascia grigia a sinistra.
- I washout non compaiono più come farmaci a sé (`wash12hours`, `washout1h`).
- Stessa logica nella risk map interattiva di Streamlit, con la decisione nella tabella.
- `classify_drug` restituisce `decision_value` e accetta `include_vehicle`.
- **Verifica**: risk map sui 10 esperimenti del paper, ogni farmaco dal lato della soglia che corrisponde alla sua decisione. Aggiunti 3 test

### v3.8.0 (Ottobre 2026) — decisione sul farmaco, riferimento pre-dose, registrazioni a 20 kHz
Correzioni emerse confrontando le regole di decisione sui 12 composti del paper Visone 2023 (etichetta FDA come verità). Riguardano `batch_analyze` e `classify_drug`; l'interfaccia PySide non è toccata. **Cambia la decisione per farmaco**: chi la usa da `classify_drug` o dagli strumenti in `tools/` deve aspettarsi chiamate diverse.
- **Nuova regola predefinita `classification_method='concentration'`.** Per ogni concentrazione si fa la media della %ΔFPDcF tra i tessuti, contando solo le concentrazioni misurate in almeno 2 tessuti. Il farmaco è positivo quando la media raggiunge il 15 % in 2 concentrazioni consecutive; con un solo tessuto la decisione è `insufficient data`.
  - Sui 12 composti: 11/12 con le variazioni degli autori (solo la cisapride sbagliata, come nel paper), 8/12 con quelle del software e 7/12 col batch sui file così come sono.
  - Con `max`, il default precedente: 5/12, 6/12 e 6/12, con tutti i negativi chiamati positivi, veicolo compreso.
  - `classification_min_tissues=1` e `classification_consecutive=1` danno la regola del paper; `max`, `mean` e `n_above` restano disponibili.
- **Override di cessazione spento per default** (`enable_cessation_override=False`). Sul dataset del paper scatta per tre composti e cambia una sola decisione, l'aspirina da negativa a positiva (corretto nella v3.8.2: la prima versione di questa nota diceva 10 composti su 12). Ora la condizione è riportata in `cessation_flag`.
- **Riferimento pre-dose.** `t0`/`T0` è riconosciuto come riferimento. Per ogni tessuto si usa l'ultimo riferimento registrato prima della prima dose, letto dall'orario nell'intestazione; senza orari si preferisce t0.
  - Gli autori normalizzavano su t0 in 13 tessuti su 15, non sul file chiamato baseline registrato prima.
  - L'elettrodo del tessuto si sceglie sullo stesso riferimento.
  - Un gruppo viene escluso dall'inclusione solo se nessuno dei suoi riferimenti passa: prima un baseline scartato e non usato faceva perdere la serie intera.
- **File a 20 kHz.** Vengono decimati a 2 kHz al caricamento. Prima il passa-banda divergeva e non veniva trovato nessun battito: 37 file su 37 di Exp11 Accelera. Il passa-banda passa inoltre a sezioni del secondo ordine quando il progetto (b, a) è instabile; a 2 kHz i risultati non cambiano.
- **Nomi dei protocolli prima del 2020.**
  - Riconosciuti il suffisso `channel1_sx_channel2_dx`, la virgola decimale (`7,5`) e il numero prima del farmaco (`ch1_30_sotalol`).
  - I controlli nel tempo (`t1`…`t7`, `Ctrl`) non sono farmaci; l'abbreviazione `SOT` vale sotalolo.
  - Il chip si legge dalla prima parola della cartella (`chipB_sotalol`) o dalla cartella sotto l'esperimento (`inj1`).
  - Un tessuto che non condivide nessuna concentrazione con gli altri dello stesso farmaco viene segnalato nel log: di solito l'unità nel nome è sbagliata.
- **Verifica**: batch completo sui 10 esperimenti del paper, 601 registrazioni, tutte con un tessuto. In 68 tessuti il riferimento è stato scelto per orario. Aggiunti 52 test

### v3.7.0 (Ottobre 2026) — abbinamento baseline e raggruppamento dei farmaci nel batch
Correzioni emerse eseguendo il batch sul dataset Visone 2023. Riguardano `batch_analyze` (CLI, interfaccia Streamlit, report Excel/PDF, export CDISC), non l'interfaccia PySide. **Cambiano i numeri normalizzati** degli studi le cui cartelle non si chiamavano `EXP…`: conviene rianalizzarli.
- **Tessuto = esperimento + giorno + chip + camera** (`loader.describe_recording`). L'esperimento viene letto dalle cartelle `Exp<N>` con qualunque combinazione di maiuscole e separatori, il giorno da `Day<N>`. Prima valevano solo le cartelle `EXP…` e il giorno veniva ignorato: sul dataset del paper 29 gruppi su 31 mescolavano esperimenti diversi, e le dosi venivano normalizzate sul baseline di un altro esperimento (per esempio −27 % invece di +10 %)
- **Cartelle Accelera** riconosciute (`Chip 537/Ch2_…`, `Day8/529/Ch1_…`, `Chip 569/Ch2 Bepridil/…`). Prima quei file non si abbinavano a nessun baseline, senza alcun avviso
- **Un elettrodo per tessuto** in modalità `auto`: prima i baseline, poi le dosi sull'elettrodo scelto per il loro baseline. Prima la serie dose-risposta mescolava el1 ed el2 in 39 registrazioni su 75
- **Scelta del baseline**: quello nella stessa cartella, poi il miglior grado QC; il motivo resta in `normalization['pairing']`. L'abbinamento è indicizzato per percorso più elettrodo: due file omonimi in cartelle diverse si sovrascrivevano a vicenda
- **Nomi dei farmaci canonici** in `classify_drug`: prima 8 farmaci diventavano 17, con esiti contraddittori. Washout e veicolo restano esclusi dalla decisione per farmaco
- **Arresto**: la %ΔFPDcF non viene calcolata se il periodo di battito supera 6 s (`max_beat_period_for_fpdc_ms`). Una cisapride a RR = 40 s dava +580 %
- **Nomi file**: concentrazione senza punto iniziale (`300 nM`) e numero senza unità separato dal nome del farmaco (`NIFEDIPINE_10`). I file con due tessuti, uno per elettrodo come in GG, restano non abbinati con il motivo esplicito
- **CI**: action su Node 24 (`checkout@v7`, `setup-python@v7`) e runner fissato a `ubuntu-24.04`
- **Verifica**: su Exp5 ed Exp8 del paper le variazioni prodotte dal batch coincidono con quelle calcolate abbinando a mano ogni tessuto (54 coppie, stesso elettrodo). Aggiunti 33 test

### v3.6.0 (Ottobre 2026) — selezione dell'onda di ripolarizzazione
Calibrata sullo split di sviluppo del gold standard manuale, verificata una volta sul test tenuto da parte (Exp 6/7/9): **FPD entro ±10 % dal 48 % al 69 %** (grado A 69 → 84 %, B 41 → 59 %, C 20 → 50 %), errore mediano −1.3 % → −0.1 %. Corpus del paper invariato (errore medio 3.4 → 3.2 %). Dettagli in `docs/FPD_vs_gold_standard_2026-10.md`.
- **Fix: allineamento dei battiti prima della mediana** (`_align_beats_xcorr`): uno sfasamento d'indice spostava di 50 ms anche i battiti già allineati e amplificava il jitter; ogni template era 50 ms in ritardo rispetto ai suoi battiti
- **Fix: inversione di polarità per battito** decisa per anti-correlazione con lo spike del template; col template sfasato il vecchio test leggeva la linea di base e in ~40 % degli elettrodi marcava invertiti quasi tutti i battiti, togliendo loro la guida del template
- **Regola `prefer_positive`** (`repol_candidate_rule`): nelle ripolarizzazioni bifasiche si prende il lobo positivo (prominenza ≥ 0.5× il massimo, entro 400 ms), come fa l'analista nell'87 % dei casi; parametri stabili in leave-one-experiment-out
- **Finestra di ricerca con l'RR dei battiti del template** quando il treno completo è sovra-rilevato (`window_rr_from_template_beats`), con guardia di forma che ferma l'estensione prima del battito successivo
- **Punto finale `peak` di default** (era `tangent`): è la convenzione dell'analista; con l'onda giusta, errore mediano 0.0 % contro +2.5 %. `tangent` resta disponibile
- `tools/compare_gold.py --set sezione.campo=valore` per le ablazioni

### v3.5.1 (Ottobre 2026)
- **Minimo FPD adattivo**: tetto `max_adaptive_min_fpd_ms` 350 → 600 ms. Su ritmi lenti il template sceglieva un after-potential a ~500 ms al posto della T a ~0.5×RR; anticipi per battito 12.5 → 9.1 %, entro ±20 % 76 → 80 % (DEV), neutro su TEST a livello di elettrodo
- **Caratterizzazione dell'errore FPD** contro il gold standard, battito per battito: il residuo sono *onde diverse* scelte da software e analista, non punti di misura diversi — `docs/FPD_vs_gold_standard_2026-10.md` (i conteggi per il grado A riportati in questa voce erano errati e sono corretti nel documento)
- Segnalati 5 blocchi del gold standard con FPD > RR (sfasamento di riga probabile)

### v3.5.0 (Ottobre 2026) — prima validazione cieca su gold standard manuale
Dataset interno GG: 179 CSV, 313 elettrodi con misura manuale (BP, FPD, tempi dei singoli battiti, note "non analizzabile"). Sviluppo su Exp 5/8/10 (217 elettrodi), **test cieco su Exp 6/7/9 (96 elettrodi), eseguito una sola volta**.

| | DEV v3.4.1 → v3.5.0 | **TEST v3.4.1 → v3.5.0** |
|---|---|---|
| "Non analizzabile" riconosciuti | 0/28 → 26/28 | **0/13 → 12/13** |
| Battiti mancanti / spurii (vs analista) | 45 % / 85 % → 10 % / 20 % | **14 % / 55 % → 14 % / 29 %** |
| Copertura (elettrodi analizzabili su cui il software risponde) | 98 % → 77 % | **95 % → 77 %** |
| BP entro ±10 % (tra i riportati) | 41 % → 58 % | **43 % → 69 %** |
| FPD entro ±10 % (tra i riportati) | 27 % → 43 % | **39 % → 48 %** |
| Errore mediano BP / FPD | −8 % / −12 % → ≈ 0 | **−5 % / −6 % → ≈ 0** |
| Grado A: BP / FPD entro ±10 % | 77 % / 58 % | **91 % / 69 %** |

Il 2 ottobre l'analista ha corretto 5 blocchi del riferimento che il confronto aveva segnalato come incoerenti (FPD > RR); con il riferimento corretto e lo stesso codice, FPD entro ±20 % sale al 58 % (DEV) e 69 % (TEST), grado A ±10 % al 61 % (DEV). Dettagli in `docs/FPD_vs_gold_standard_2026-10.md`.

Il verdetto "non analizzabile" marca anche elettrodi che l'analista ha misurato (23 % in entrambi gli split): su quelli il software v3.4.1 sbagliava in 54/55 (DEV) e 18/19 (TEST). Non è un falso allarme: è smettere di stampare numeri senza segnale dietro.

- **Verdetto di analizzabilità** (`QualityConfig.enable_analysability_verdict`): SNR mediana dei battiti sul noise floor < 1.6, oppure < 16 battiti con CV RR > 40 % → grado F, FPD/FPDc non emessi, escluso dalla normalizzazione (anche come baseline)
- **Matched filter accettato anche quando trova fino a 3× meno battiti** della derivata (`mf_count_ratio` 0.3–2.0): nei disaccordi aveva ragione lui in 61 casi su 97
- **Popolazione minore di ampiezza** (`enable_minor_population_reject`): deflessioni 2.5× più piccole degli spike, fuori ritmo, non in alternans → scartate (rumore d'incubatore, sorgente asincrona)
- **Strumenti**: `tools/blind_report.py` (report per elettrodo su una cartella), `tools/compare_gold.py` (confronto con gold standard, split dev/test, scorecard)
- Noise gate rimisurato su questi dati e lasciato invariato (senza: spurii 104 %)

### v3.4.1 (Ottobre 2026)
- **Matched filter per segnali a bassa SNR** (`enable_matched_filter_refine`): quando gli spike sono appena sopra il rumore (SNR mediana < 3) i battiti vengono ri-rilevati per correlazione con il template dei battiti più ripidi. Sulla baseline di laboratorio `chipA_ch1` (el1, spike ~15 µV): da 205 battiti con 73 mancanti e 58 spurii a 219 con 3 mancanti e 2 spurii rispetto all'elettrodo pulito; QC da 167 a 218 accettati, RR 810 vs 811 ms. Mai attivo su segnali ad alta SNR (dove raccoglierebbe onde T)
- **Gate noise-floor ricalibrato**: floor 1.5 → 1.0; un cluster è "rumore" solo se la sua mediana è ≤ 1.35× il floor e c'è un cluster ≥ 3× sopra. La prima versione scartava 44 battiti veri su 208 nel file a bassa SNR; i risultati su Exp8/Exp6 e sui sei file puliti sono invariati
- **Corpus**: aggiunta la baseline di laboratorio a bassa SNR con l'altro elettrodo come riferimento posizionale; 9 fixture, 47 test di regressione su segnali reali

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
