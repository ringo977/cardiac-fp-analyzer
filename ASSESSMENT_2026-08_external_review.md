# Cardiac FP Analyzer — Assessment esterno

**Data:** 5 agosto 2026
**Versione dichiarata:** 3.3.0 (`pyproject.toml`, `__init__.py`)
**Branch analizzato:** `feat/pyside-poc` (32 commit avanti a `main`)
**Prospettiva:** revisione "a occhi nuovi", come se non conoscessi la storia del progetto

---

## 0. Executive summary

Cardiac FP Analyzer è una pipeline Python (~11.7k LOC core + ~7.9k LOC UI PySide6 + ~10.8k LOC test) per l'analisi di field potential da cardiomiociti hiPSC su registrazioni Digilent, con export CDISC SEND e classificazione di rischio proaritmico in stile CiPA.

**Giudizio complessivo:** il progetto ha un'architettura sorprendentemente pulita e una cultura di documentazione-del-perché superiore alla media del software scientifico accademico. Ma non è nello stato che il numero di versione e la presenza di un export "regulatory-grade" suggeriscono. Ci sono **tre bug scientifici verificati** che invalidano potenzialmente numeri già prodotti, e un disallineamento strutturale grave: **tutta la documentazione, il packaging e la CI descrivono una UI abbandonata**, mentre quella reale non è né pacchettizzata né testata in CI.

| Asse | Voto | Sintesi |
|---|---|---|
| Architettura / layering | **A−** | Core disaccoppiato dalla UI, zero cicli di import, modello dati Study/Group/FileEntry esemplare |
| Gestione errori | **A−** | Whitelist esplicita di eccezioni batch-safe, 0 `except:` nudi — raro e ben fatto |
| Documentazione *inline* | **A** | I commenti spiegano il *perché*, con citazioni di letteratura |
| **Correttezza scientifica** | **D+** | 3 bug verificati sul percorso di misura primario (FPD) |
| **Validazione / test** | **D** | 580 test, ma l'endpoint primario (FPD) non è mai confrontato con un ground truth |
| **Packaging / CI** | **F** | La UI reale non è dichiarata, non è installabile, non è testata in CI |
| Documentazione *utente* | **F** | 97 KB di docs descrivono Streamlit; zero menzioni di PySide |
| Igiene repo | **D** | Git history pulita (5 MB), working tree 16 GB di detriti |

**Se dovessi dare un solo consiglio:** fermare lo sviluppo di feature per uno sprint e chiudere i tre bug scientifici + il gap di validazione su FPD. Tutto il resto è debito gestibile; quelli sono difetti che producono *numeri sbagliati con l'aria di essere giusti*.

---

## 1. Cosa fa il software

Pipeline lineare, ben identificabile:

```
CSV Digilent (2ch)
  → loader.py            parsing header # + dati
  → channel_selection.py scelta automatica EL1/EL2 (score pesato)
  → filtering.py         notch 50Hz + armoniche → Butterworth 0.5–500 Hz → Savitzky-Golay
  → beat_detection.py    detection multi-metodo + 4 passate di gap-filling
  → quality_control.py   gate ampiezza / morfologia / SNR → grade A–F
  → rhythm_integration.py filtro ritmico + RR-outlier
  → parameters.py        FPD, FPDc (Fridericia + Bazett), ampiezze, STV
  → repolarization.py    detection onda T, metodo tangente/50%/picco
  → arrhythmia.py        risk score 0–100, classificazione
  → normalization.py     %Δ vs baseline per elettrodo → TdP score
  → cdisc_export.py      SEND datasets (EG, DM, EX, TS, TX, DS) + define.xml
```

Due UI: `ui/` (Streamlit, ~2.4k LOC, in manutenzione) e `pyside_app/` (PySide6 + pyqtgraph, ~7.9k LOC, dove avviene tutto lo sviluppo attivo).

---

## 2. Punti di forza reali

Vanno detti, perché sono sostanziali e insoliti.

**2.1 — Il core non conosce la UI.** Verificato: zero import di `streamlit`, `PySide6`, `ui.*` dentro `cardiac_fp_analyzer/`. Direzione delle dipendenze strettamente `UI → core`. Nessun ciclo di import. Questa è la ragione per cui la migrazione Streamlit→PySide è stata possibile senza riscrivere la scienza.

**2.2 — La gestione errori batch è di livello industriale.** `analyze.py:60-64` definisce una whitelist esplicita:

```python
_BATCH_SAFE_EXCEPTIONS = (KeyError, ValueError, IndexError, RuntimeError,
                          AssertionError, FileNotFoundError, OSError,
                          UnicodeError, pd.errors.ParserError,
                          pd.errors.EmptyDataError)
```

con 16 righe di commento che spiegano *perché* ogni voce è lì, e la stessa tupla applicata simmetricamente al ramo seriale e a quello `ProcessPoolExecutor` così che i due abbiano semantica identica. In tutto il repo: **zero `except:` nudi**, 9 `except Exception` tutti annotati e tutti con log o modale.

**2.3 — Il modello dati Study/Group/FileEntry è il codice migliore del repo.** `cardiac_fp_analyzer/study.py`: dataclass semplici, `SCHEMA_VERSION`, `to_dict`/`from_dict` espliciti, e decisioni motivate nel punto in cui vengono prese — `csv_relpath` POSIX-relativo per portabilità, `dose_uM: float | None` con la spiegazione di perché `None` e non `-1` (così un controllo a 0 µM non viene mai confuso con "dose ignota"), `Study.folder` deliberatamente non serializzato.

**2.4 — `config.py` non è over-engineered, è sotto-esposto.** 872 righe di cui 376 di commento (43%), con citazioni di letteratura (Clements & Thomas PLOS ONE 2013 per `morphology_min_corr = 0.7`) e trappole di unità documentate. Ha persino una tabella di migrazione per rinomine di campi legacy (`cv_good` → `cv_good_frac`). Il problema opposto: dei 202 campi, la dialog PySide ne espone 40.

---

## 3. Difetti scientifici — verificati nel codice

Questi li ho controllati personalmente riga per riga, non sono impressioni.

### 3.1 🔴 Il flag `correction` non seleziona nulla — etichetta Fridericia come Bazett

`config.py:283-287` definisce `correction: 'fridericia' | 'bazett' | 'none'` e `analyze.py:847-848` la espone come `--correction`. Ma:

```python
# parameters.py:240-242
params['fpdc_ms']        = (fpd / (rr_interval ** (1/3))) * 1000   # SEMPRE Fridericia
params['fpdc_bazett_ms'] = (fpd / np.sqrt(rr_interval)) * 1000
```

e l'unico consumatore della config è:

```python
# parameters.py:488
summary['correction'] = rc.correction     # solo un'etichetta
```

**Conseguenza:** lanciare `--correction bazett` produce valori **Fridericia** etichettati `bazett` nel summary e nell'export CDISC. `--correction none` applica comunque Fridericia. Chiunque abbia usato quel flag ha dati mislabellati.

*Nota:* entrambi i valori *sono* calcolati (`fpdc_ms` e `fpdc_bazett_ms`), quindi il fix è banale — basta far scegliere alla config quale finisce in `fpdc_ms`. Ma finché non è fatto, l'etichetta mente.

### 3.2 🔴 Bug di segno nella detection per-battito della ripolarizzazione

`repolarization.py:528` cicla su entrambe le polarità per trovare il picco di ripolarizzazione:

```python
for sign in [template_repol_sign, -template_repol_sign]:
    pks, props = sig.find_peaks(sign * seg_det, ...)
    ...
    if score > best_score:
        best_score = score
        best_idx = best_pk          # ← registra QUALE picco, non QUALE segno
```

Il segno vincente non viene mai salvato. Poi:

```python
# repolarization.py:615-616
fpd_idx = apply_fpd_method(seg_det, best_idx, template_repol_sign, fs, ...)
                                              # ← sempre il segno del TEMPLATE
```

Per confronto, il percorso *template* lo fa correttamente (`repolarization.py:380` passa `best_sign`). L'asimmetria è la prova che è un bug, non una scelta.

**Conseguenza:** per ogni battito la cui onda T è invertita rispetto al template, la matematica tangente/50%/baseline-return gira sulla polarità sbagliata e produce un endpoint privo di significato, riportato silenziosamente come FPD. E `parameters.py:373-381` *inverte deliberatamente* il segno per i battiti invertiti — quindi questo percorso è esercitato su dati reali, non è teorico.

### 3.3 🔴 `fpd_reliable` è calcolato e mai applicato dove conta

`parameters.py:81-87` calcola `summary['fpd_reliable']` in base a `min_valid_fpd_ratio = 0.50`. Questo flag esiste perché avevate colpito un caso reale documentato nel codice: **1 battito valido su 7, che produceva "FPDcF 318.8 ± 0.0"**.

Grep completo dei consumatori:

```
inclusion.py:184,187      → lo SETTA (non lo legge)
ui/single_file.py:100     → lo legge (ma è la UI Streamlit abbandonata)
settings_dialog_helpers.py:296 → solo testo di help
```

**`normalization.py` e `cdisc_export.py` non lo leggono mai.** Quindi una registrazione con 1 FPD valido su 7 contribuisce comunque il suo `fpdc_ms_mean` al `%ΔFPDcF` e alla classificazione del farmaco. Nel flusso PySide (quello che usate) non c'è alcun blocco.

### 3.4 🟠 Bias di selezione strutturale su FPDc

La media FPDc è presa solo sui battiti dove l'onda T era rilevabile (`parameters.py:407-408`) — cioè, per costruzione, i battiti con l'onda T più grande. Un farmaco che appiattisce l'onda T sposta sistematicamente questa selezione, **e la direzione del bias è la direzione dell'effetto che state misurando.** Questo non è un bug risolvibile con una riga: è una proprietà del disegno che va almeno dichiarata, e idealmente gestita riportando FPD solo quando la valid-ratio supera una soglia (cioè: applicando 3.3).

### 3.5 🟠 Quattro passate di gap-filling che inseriscono battiti dove il ritmo li "aspetta"

`beat_detection.py` ha `_recover_missed_beats` più tre passate successive (pass 2/3/4) che aggiungono battiti in posizioni ritmicamente attese. È un prior forte verso la regolarità — e le metriche che ne risultano deflazionate (CV(RR), STV, conteggio di battiti prematuri) sono **esattamente quelle usate a valle per lo scoring aritmico**. Il rischio è di misurare la regolarità che l'algoritmo ha imposto, non quella del preparato.

### 3.6 🟠 Nessuna statistica inferenziale, e pseudo-replicazione

- Il `± SD` riportato ovunque è la SD **tra battiti dentro una singola registrazione** (`parameters.py:442-443`), non tra repliche biologiche. I battiti dentro un costrutto non sono campioni indipendenti dell'effetto del farmaco.
- `classify_drug` aggrega le concentrazioni con `'max'` (`normalization.py:446`): positivo se **una qualsiasi** concentrazione supera il 15%. È una statistica max-di-N senza correzione per molteplicità — il tasso di falsi positivi cresce col numero di concentrazioni testate.
- Nessun fit dose-risposta (EC50/Hill), nessun controllo di monotonicità, nessun intervallo di confidenza.

### 3.7 🟠 Soglia tarata su due file specifici

`config.py:487-493` dice testualmente che `min_fpd_confidence = 0.66` serve solo a *"separare chipE_ch1 (0.659, bad) da chipA_ch1 (0.691, good). Gap = 0.032"*. Un criterio di inclusione tarato su un gap di 0.032 tra due registrazioni specifiche, poi applicato a tutti i dati futuri — e il fallimento **esclude l'intero gruppo chip+canale** (`inclusion.py:118-121`). È validazione circolare.

### 3.8 🟠 Il gate di ampiezza è inerte ai default, e catastrofico se attivato

`parameters.py:298-317` confronta `(max−min)*1e6` (assume volt) contro `min_signal_amplitude_uV = 10`. Ma `amplifier_gain` di default è `1.0` mentre il guadagno hardware è documentato come `1e4` (`config.py:712-714`). Ai default il gate non scatta mai **e tutti gli `spike_amplitude_mV` riportati sono sbagliati di un fattore 1e4**; impostando il gain, il gate scatta e azzera l'intero FPD della registrazione.

### 3.9 🟡 Nessuna correzione veicolo / drift temporale

`normalization.py`: i controlli veicolo (`_is_control`, righe 95-99) sono usati solo come baseline sostitutiva quando manca la baseline, e altrimenti esclusi dalla classificazione (righe 353-354). **Nulla viene mai sottratto per il drift temporale.** La letteratura documenta +22.6% di beat rate e −7.7% di FPD in soli 20 minuti di equilibrazione su hiPSC-CM; un protocollo cumulativo dura molto di più. Questo è il punto che avevamo già discusso — lo riporto qui perché in un assessment esterno è una lacuna metodologica di primo livello, non un nice-to-have.

---

## 4. Validazione — il buco più grande

**580 test, 10.8k LOC. E l'endpoint primario non è mai verificato.**

`tests/golden_signals.py:86` genera segnali sintetici e calcola `'fpd_ms_approx': fpd_ms` come ground truth. Grep in tutto il repo: **una sola occorrenza — la definizione.** Nessun test lo legge mai.

Identico in `tests/test_e2e_synthetic.py`: `make_synthetic_signal` restituisce `true_fpd`, la fixture lo salva, e ogni test usa `true_bp` (beat period) invece. `TestRepolarizationGate` spacchetta `true_fpd` e lo scarta.

Quindi la suite "golden" verifica: conteggio battiti (±2), periodo (±10%), CV < 5%, stringa di classificazione, risk score. **Mai un FPD misurato contro un FPD noto.** E FPD è l'input di FPDc → ΔFPDcF% → TdP score → export CDISC.

Altri gap:

| Modulo | LOC | Test |
|---|---|---|
| `cdisc_export.py` | 1455 | ~10, del tipo "i file sono stati creati" |
| `filtering.py` | 103 | **0 funzionali** (solo import smoke) |
| `inclusion.py` | 192 | **0 funzionali** — decide quali dati entrano nel dataset |
| `channel_selection.py` | 121 | **0 funzionali** |

E la distribuzione è rovesciata: `tests/test_study_panel_helpers.py` da solo è **1975 LOC / 173 test** (30% di tutti i test) per testo di badge, tooltip e formattazione numerica.

**Nessun corpus di segnali reali.** Tutto è sintesi gaussiana. `test_bradycardia_fpd_robustness.py` è il test migliore del repo — documenta il fallimento reale su `Exp6_chipD_ch1_chipB_ch1_baseline.csv` — ma riproduce il bug sinteticamente; il CSV vero non è una fixture.

`TestDeterminism` confronta un segnale con sé stesso nello stesso processo: non è un baseline salvato, quindi un cambio d'algoritmo che sposta ogni FPD di 20 ms passa il test.

**Per un tool che produce output GLP/CDISC, alla domanda di audit "mostrami il test che dimostra che FPD è corretto" non c'è risposta.**

---

## 5. Packaging, CI, documentazione — il disallineamento strutturale

Questo è il difetto che, visto da fuori, colpisce di più.

**5.1 — La UI reale non è dichiarata da nessuna parte.** `PySide6` e `pyqtgraph` compaiono **zero volte** in `pyproject.toml`, `requirements.txt` e `.github/workflows/ci.yml`. Verificato con grep. Eppure `pyside_app/` è 7.9k LOC e gli ultimi 32 commit sono tutti PySide.

**5.2 — E non è installabile.** `[tool.setuptools.packages.find] include = ["cardiac_fp_analyzer*", "ui*"]` — `pyside_app` non è nella lista. Un `pip install .` ti dà la UI Streamlit abbandonata e nessun modo di lanciare quella vera. L'extra si chiama letteralmente `gui` e installa Streamlit.

**5.3 — La CI è strutturalmente cieca.** Ogni test PySide inizia con `pytest.importorskip("PySide6.QtGui")`. Siccome la CI non installa PySide6, **~240 test (40% della suite) vengono saltati silenziosamente ad ogni run** — e il badge resta verde. Inoltre `on: push: branches: [main]`, quindi **il branch con tutti i 32 commit attivi non triggera mai la CI**. Il badge nel README riflette `main`, fermo a marzo.

**5.4 — 97 KB di documentazione descrivono la UI sbagliata.**

| File | Dimensione | "streamlit" | "pyside" |
|---|---|---|---|
| `README.md` | 8.2 KB | 8 | **0** |
| `DOCUMENTATION.md` | 66.9 KB | 9 | **0** |
| `ASSESSMENT_v3.3.0.md` | 30.0 KB | 8 | **0** |

Il README dice ancora "GUI Streamlit: interfaccia web" e `streamlit run app.py`. L'unico documento che sa dell'esistenza di PySide è `docs/decisions/0001-abandon-streamlit-for-pyside6.md` — che però ha ancora **`Status: Proposed (awaiting proof-of-concept validation)`**. Il PoC è arrivato 32 commit e 7.9k LOC fa. L'unico documento accurato del repo è formalmente marcato "non deciso".

**5.5 — Duplicazione di policy scientifica tra le due UI.** `_TEMPLATE_RISKY_RHYTHM_TYPES` e `_FPD_CV_TEMPLATE_WARN = 0.20` esistono identici in `ui/display.py:277-289` **e** `pyside_app/main.py:64-70`, con i predicati del banner ricalcolati indipendentemente nei due posti. Cambiare la soglia in uno fa divergere silenziosamente il giudizio di qualità del segnale tra le due UI. Questa roba appartiene al core.

---

## 6. Debito architetturale

**`StudyPanel` è un God-object:** 1808 righe / 26 metodi (`study_panel.py:1168`), che gestisce CRUD di studi+gruppi+file, scan ricorsivo cartelle, orchestrazione batch con thread pool e dialog di progresso, export CDISC, plot dose-risposta, rendering albero, calcolo badge/staleness e pruning cache. Dieci responsabilità. Sotto, 33 funzioni libere a livello di modulo (righe 2982-3848) che non c'entrano l'una con l'altra.

Buona notizia: quelle funzioni sono pure e si estraggono meccanicamente. La decomposizione naturale è `study/{models,batch,export,dose_response,metrics,staleness,tree,dialogs,panel}.py`.

**`FileResult` sta nel modulo GUI** — è il modello del risultato di analisi batch, è importato da 3.4k LOC di test, e li costringe tutti a `importorskip("PySide6")`. Appartiene a `cardiac_fp_analyzer/study.py` accanto a `Study`/`Group`/`FileEntry`.

**`MainWindow`:** 768 righe, con `_on_recompute` a 187 righe che fa cinque cose distinte. Otto campi di stato ad-hoc, di cui due alias dello stesso oggetto (`_config` / `_current_config`, ammesso nei commenti alle righe 1775-1778) e un flag di reentrancy (`_suppress_pending`) che dovrebbe essere un `QSignalBlocker`.

**Type hints a due velocità:** `pyside_app/` 90% annotato, `cardiac_fp_analyzer/` 34% — e le firme non annotate sono proprio quelle pubbliche (`analyze_single_file`, `batch_analyze`, `recompute_from_beats`). Nessun `mypy`/`pyright` configurato, quindi il 90% non è comunque verificato.

**36 `print()` in `analyze.py`** accanto a un logger configurato, con `verbose=True` di default — che significa scrivere su stdout da dentro i worker di `ProcessPoolExecutor`.

**`diagnostics/pr3/repolarization.py`** (609 righe) è un fork divergente di `cardiac_fp_analyzer/repolarization.py` (635 righe). Una copia stantia di un modulo scientifico core che sta nel tree è un pericolo attivo.

---

## 7. Igiene repo

La git history è pulita: **91 file tracciati, `.git` da 5.1 MB, zero `.dat`/`.patch`/`.bundle` committati**. Il problema è tutto nel working tree, che pesa **16 GB**.

- **13 file `.dat` che sono git bundle rinominati** (`head -c 60` restituisce `# v2 git bundle`) — rinominati per aggirare la regola `*.bundle` in `.gitignore`
- 11 `.bundle` veri, incluso il ciclo `send-fixes-v8…v15c`
- 5 `.patch`, di cui `wip-click-picking.patch` duplica il branch `wip/click-picking-ux` (stesso lavoro in due meccanismi)
- **9 directory `analysis_results*` per 3.7 GB** (solo la prima è gitignorata)
- `.git-tmp-copy/` — **233 MB, una copia stantia di `.git`, 45× più grande dell'originale**
- `studio_MR/` 338 MB, `µECG-Pharma Calibration - EXCEL DATA/` **11 GB**
- `test_qc_strict_rhythm_readmission_new.py` a root, byte-identico a quello in `tests/`, mai eseguito (`testpaths = ["tests"]`)
- `docs/MANUALE_Cardiac_FP_Analyzer.docx` (30 aprile, il documento più recente del repo) — **untracked**

10 branch locali, 9 già mergiati, 8 senza upstream (esistono su una sola macchina). `main..feat/pyside-poc` = **32 commit**; il verso opposto = 0.

Aggiungere `*.dat`, `*.patch` e `analysis_results*/` a `.gitignore` elimina 20 dei 31 untracked. Cancellare `.git-tmp-copy/` recupera 233 MB.

---

## 8. Piano d'azione consigliato

Ordinato per rapporto danno-evitato / costo.

### Sprint 0 — Blocco correttezza (prima di produrre altri numeri)

1. **Fix del bug di segno** in `repolarization.py:528/616` — tracciare `best_sign` nel loop e passarlo a `apply_fpd_method`, come già fa il percorso template.
2. **Fix del flag `correction`** — far sì che `rc.correction` scelga quale formula finisce in `fpdc_ms`. Entrambi i valori sono già calcolati; è questione di un `if`.
3. **Applicare `fpd_reliable`** in `normalization.compute_normalized_parameters` e in `cdisc_export` — una registrazione con valid-ratio sotto soglia non deve contribuire al `%ΔFPDcF` senza almeno un flag esplicito.
4. **Risolvere l'ambiguità `amplifier_gain`** — decidere se il default è 1.0 o 1e4 e allineare `min_signal_amplitude_uV` di conseguenza. Oggi i `spike_amplitude_mV` sono sbagliati di 1e4 ai default.
5. **Ri-analizzare i dataset già prodotti** dopo 1–4 e diffare i risultati. Se i numeri non cambiano, ottimo, l'avete documentato. Se cambiano, meglio scoprirlo ora.

### Sprint 1 — Validazione dell'endpoint primario

6. **Far asserire `fpd_ms_approx` / `true_fpd`** nei test golden — il ground truth è già generato in due posti e mai usato. È il test più economico e più importante che manchi.
7. **Costruire un corpus di regressione su segnali reali**: 5–10 CSV rappresentativi (incluso `Exp6_chipD_ch1` che è già documentato come caso patologico), con valori attesi di FPD/FPDc/n_beats salvati come baseline numerica. Senza questo, ogni refactor della detection è un salto nel buio.
8. **Test funzionali per `filtering`, `inclusion`, `channel_selection`** — tre moduli che decidono *quali dati vengono analizzati*, oggi con zero copertura.

### Sprint 2 — Sbloccare CI e packaging (mezza giornata, alto ritorno)

9. Aggiungere l'extra `pyside = ["PySide6>=6.5,<7.0", "pyqtgraph>=0.13,<0.14"]`, mettere `"pyside_app*"` nei package, aggiungere lo script `cardiac-fp-gui`.
10. Aggiungere `pyside` all'install di CI + `QT_QPA_PLATFORM=offscreen` → 240 test smettono di essere saltati.
11. Estendere i trigger CI ai branch `feat/**`.
12. **Mergiare `feat/pyside-poc` in `main`** e rinominare mentalmente: non è più un PoC, è il prodotto.

### Sprint 3 — Metodologia (la discussione che avevamo aperto)

13. Modello `Group` con `role: 'treatment' | 'vehicle' | 'reference'` e `paired_vehicle_group` per abilitare la sottrazione del drift.
14. Vista dose-risposta con: %Δ da baseline, %Δ corretta per drift veicolo, e assoluti raw.
15. Almeno un fit dose-risposta (Hill/EC50) al posto dell'aggregazione `max`.

### Continuo — Igiene

16. `.gitignore`: `*.dat`, `*.patch`, `analysis_results*/`, `.git-tmp-copy/`, `studio_MR/`, `diagnostics/`.
17. `git branch -d` sui 7 branch mergiati; `rm -rf .git-tmp-copy/`.
18. Cancellare `diagnostics/pr3/repolarization.py` (fork divergente) e il test duplicato a root.
19. Riscrivere README + DOCUMENTATION per PySide; portare l'ADR-0001 a `Status: Accepted`.
20. Committare `docs/MANUALE_Cardiac_FP_Analyzer.docx`.
21. Spostare `_FPD_CV_TEMPLATE_WARN` e `_TEMPLATE_RISKY_RHYTHM_TYPES` nel core, letti da entrambe le UI.

---

## 9. Nota di chiusura

Il divario più interessante di questo progetto è tra la **qualità del ragionamento** e la **qualità della verifica**. I commenti nel codice mostrano che le decisioni difficili sono state pensate bene — la whitelist di eccezioni, la scelta di `None` invece di un sentinella per la dose, la tabella di migrazione della config, le citazioni di letteratura sulle soglie. Quel livello di cura è raro.

Ma quella cura non si è estesa a *dimostrare* che i numeri sono giusti. Il ground truth di FPD è generato in due file e non letto da nessuno; il flag di affidabilità è calcolato e non applicato; il flag di correzione è esposto in CLI e ignorato. Sono tutti casi della stessa forma: **il meccanismo giusto costruito e poi non collegato.**

La buona notizia è che questo è il tipo di debito più economico da ripagare — i pezzi ci sono già, mancano i collegamenti. Lo Sprint 0 e 1 qui sopra sono, realisticamente, una-due settimane di lavoro, e trasformerebbero il giudizio scientifico da D+ a B/B+.

---

*Assessment prodotto per revisione interna. Le affermazioni nelle sezioni 3, 4 e 5 sono state verificate leggendo il codice sorgente ai riferimenti file:riga indicati.*
