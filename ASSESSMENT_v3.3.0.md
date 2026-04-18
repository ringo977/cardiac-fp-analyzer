# Assessment tecnico — Cardiac FP Analyzer v3.3.0

**Data**: 18 aprile 2026 (revisione 1)
**Autore assessment**: revisione indipendente del codebase
**Stato test suite**: 138/138 passati, ruff clean
**Scope**: tutto il pacchetto (`cardiac_fp_analyzer/`, `ui/`, `tests/`, packaging)

> **Addendum revisione 1** — Dopo un'ispezione più approfondita, la sezione
> 3.2 è stata riscritta: il "template truncation bug" segnalato inizialmente
> si è rivelato un falso positivo (codice difensivo inoperante, non un bug
> reale). Al suo posto è documentato il **bug reale** osservato da Marco su
> segnali lenti — `segment_beats` con `post_ms` fisso — che è già stato
> risolto sul branch `fix/search-window-adaptive-segmentation` (merged,
> commits `04ecb23` + `275efbe`).

---

## 1. Sintesi esecutiva

Il software è **ben tenuto sul piano della forma** (linting pulito, suite di test passante, CI attiva, refactoring documentati nel changelog, separazione `core` / `ui` / `tests`). I problemi che ti fanno dire che "non funziona perfettamente" non sono dovuti a sciatteria, ma a **tre classi di debito tecnico nascosto** che la suite di test sintetici non riesce a far emergere:

1. **Configurazione fragile e ridondante**: parametri con lo stesso nome ma semantica e unità diverse in dataclass differenti.
2. **Algoritmi sensibili a casi reali (rumore, drift, beat troncati, segnali al limite)**: la logica funziona sui segnali sintetici dei test, ma collassa su pattern che si vedono solo nei dati veri.
3. **Pipeline opaca**: doppi gate di qualità in cascata, soglie hardcoded sparse, nessuna invalidazione di cache nella UI Streamlit, gestione errori asimmetrica fra ramo seriale e parallelo del batch.

L'impressione complessiva è quella di un codice **scritto velocemente da un agente molto capace, ma senza un humano che integrasse mentalmente le scelte** — molte decisioni microscopiche sono ragionevoli in isolamento, ma incoerenti tra loro.

---

## 2. Cosa funziona bene (per non gettare il bambino con l'acqua sporca)

- **Architettura modulare**: niente import circolari, separazione dei concerns chiara fra acquisizione → filtraggio → detection → parametri → QC → aritmie → report.
- **Refactoring storici riusciti**: `app.py` da 1968 a 95 righe, `analyze.py` da 690 a 463, `parameters.py` spezzato in `repolarization.py`. Il codice è effettivamente più leggibile di quanto fosse.
- **Eccezioni specifiche**: davvero zero `except Exception` (verificato). Nei limiti di ciò che cattura, la gestione errori è disciplinata.
- **i18n robusta**: meccanismo `T()` con fallback a chiave (non crasha su chiave mancante), copertura IT/EN completa.
- **Configurazione serializzabile**: presets nominati, JSON I/O, dataclass — buona base, anche se va semplificata (vedi §3).
- **Tracciabilità scientifica**: `confronto_software_vs_paper.md` è un documento serio che dichiara apertamente le divergenze metodologiche dal paper di riferimento. Pochi software ricerca lo fanno.

---

## 3. Problemi CRITICI verificati

### 3.1 Parametri duplicati con unità incompatibili in config.py
**File**: `cardiac_fp_analyzer/config.py`, righe 75–79 e 503–506

```python
# In BeatDetectionConfig:
cv_good: float = 0.15        # frazione (15%)
cv_fair: float = 0.30
cv_marginal: float = 0.50
bp_ideal_range_s: tuple = (0.4, 3.0)

# In ChannelSelectionConfig:
cv_good: float = 20.0        # numero (20%) — UNITÀ DIVERSA
cv_fair: float = 35.0
bp_ideal_range_s: tuple = (0.3, 4.0)  # range diverso
```

Stesso nome semantico, stessa entità fisica, **valori in unità diverse** (frazione vs percentuale) e **soglie diverse** (15% vs 20%). Modificare l'una non modifica l'altra; un utente che esporta/importa la config tramite JSON e ne tocca un campo si troverà comportamenti incoerenti fra `beat_detection` e `channel_selection`.

**Intervento**: spostare i parametri condivisi in una `CommonThresholds` e referenziarli da entrambe; o almeno uniformare le unità (sempre frazione 0–1).

### 3.2 Finestra di segmentazione troppo stretta per ritmi lenti ✅ FIXED
**File**: `cardiac_fp_analyzer/analyze.py:113–116` e `channel_selection.py:71`
**Stato**: risolto sul branch `fix/search-window-adaptive-segmentation`
(commit `04ecb23`, merged in `main` via `275efbe`)

**Descrizione del bug osservato su dati reali (screenshot Marco, BP ≈ 3500 ms)**:

`analyze.py` chiamava `segment_beats` con un `post_ms` **fisso**:

```python
bd, btm, vi = segment_beats(filtered, df['time'].values, bi, fs,
                             pre_ms=rep_cfg.segment_pre_ms,
                             post_ms=max(850, rep_cfg.search_end_ms + 50))
```

Con `search_end_ms=1500` → `post_ms=1550 ms`. Il template dei battiti era
quindi lungo al massimo 1550 ms dopo lo spike. A valle, in
`repolarization.find_repolarization_on_template`, la logica di ricerca
adattiva calcolava correttamente:

```python
effective_end_ms = max(fixed_end_ms, search_end_pct_rr × RR_ms)
# Per BP=3500 ms, pct_rr=0.70 → effective_end_ms = 2450 ms
```

…ma poi incontrava il troncamento silenzioso:

```python
search_end = spike_idx + int(effective_end_ms / 1000 * fs)
if search_end > len(template):
    search_end = len(template)   # <-- silent clip a 1550 ms
```

Risultato: per ritmi lenti (tipico dofetilide o blocco hERG), la T-wave
reale a ~2000 ms dallo spike cadeva **fuori dal template** e la
ripolarizzazione veniva attribuita a rumore residuo o afterpotential
precoce. La FPD riportata era sistematicamente sottostimata.

**Fix applicato**: in `analyze.py` e `channel_selection.py` il `post_ms`
di `segment_beats` ora è adattivo:

```python
post_ms = max(fixed_search_end_ms + 50,
              search_end_pct_rr × median_BP × 1000 + 50)
```

Così il template si estende almeno quanto la finestra di ricerca
adattiva. Copertura test: `tests/test_slow_rhythm_segmentation.py`
(5 test, incluso un end-to-end con BP=3.5 s/FPD=2 s).

**Validato**: rianalizzando il segnale dello screenshot i marker verdi
di ripolarizzazione cadono ora sulla T-wave corretta (~2000 ms
post-spike), non più sull'afterpotential.

---

#### Nota su un falso positivo correlato

L'analisi iniziale di questa sezione segnalava anche un pattern a
`parameters.py:111-114`:

```python
min_len = min(len(b) for b in aligned)
mat = np.array([b[:min_len] for b in aligned])
```

sostenendo che un singolo beat troncato avrebbe troncato l'intero
template. **Questa diagnosi era errata**: `segment_beats` produce
battiti di lunghezza uniforme per costruzione (ogni beat è
`data[idx−pre : idx+post]` con `pre`/`post` fissi), quindi in
`aligned` tutti i beat hanno identica lunghezza e `min_len` equivale
sempre a `len(aligned[0])`. Il pattern compare in tre punti
(`parameters.py:111`, `quality_control.py:191`,
`residual_analysis.py:28`) ed è **codice difensivo inoperante**. Non
costituisce un bug, ma è rumore visivo che potrebbe confondere chi
cerca le cause di problemi veri: va eliminato in un futuro refactor
(o coperto da un invariante `assert all(len(b) == len(aligned[0]))`
che lo documenti come precondizione).

### 3.3 Fallback CDISC SEND non regulatory-compliant
**File**: `cardiac_fp_analyzer/cdisc_export.py`, righe 1116–1133

Se `pyreadstat` e `xport` mancano, il codice scrive **CSV sotto un nome `.xpt`**, con un commento `# WARNING` come prima riga e un `UserWarning` Python. Una sottomissione CDISC con file CSV mascherati da XPORT v5 viene rifiutata da qualsiasi reviewer FDA.

Il problema è strutturale: `pyreadstat` è dichiarato dipendenza opzionale (`extras_require=["cdisc"]`), ma una funzionalità "regulatory-grade" non può essere opzionale.

**Intervento**: rendere `pyreadstat` dipendenza obbligatoria del gruppo `cdisc`; in `export_send_package()` fare un check `try: import pyreadstat` all'inizio e sollevare `RuntimeError` chiaro se manca, **prima** di scrivere qualsiasi file.

### 3.4 Nessuna cache Streamlit nella UI
**Verifica**: `grep -r "@st.cache" .` → 0 occorrenze.

Conseguenza: ogni interazione UI (cambio slider, switch lingua, click navigazione) **ricarica e rianalizza tutto da zero**. Su batch da 169 file, ogni rerun costa minuti. Anche il downsampling minmax viene rifatto a ogni redraw del grafico.

**Intervento**: avvolgere `load_csv()`, `analyze_single_file()` e `minmax_downsample()` con `@st.cache_data`, usando come chiave l'hash della config + percorso file + mtime. Da soli, cambiano l'esperienza in modo radicale.

---

## 4. Problemi IMPORTANTI

### 4.1 Gestione errori asimmetrica nel batch
**File**: `cardiac_fp_analyzer/analyze.py`, righe 280–298

Nel ramo **parallelo** (`ProcessPoolExecutor`) c'è un `try/except (KeyError, ValueError, IndexError, RuntimeError, OSError)` che cattura e logga. Nel ramo **seriale** (chiamato quando `n_workers=1` o `len(csv_files)==1`) **non c'è alcun try/except**: una `pd.errors.ParserError` o un file con encoding strano fa crashare l'intero batch.

Inoltre `pd.errors.ParserError` non è sottoclasse di nessuna delle eccezioni catturate nel ramo parallelo, quindi crasha anche lì.

**Intervento**: estrarre la gestione errori in un wrapper `_safe_analyze()` chiamato da entrambi i rami, e includere `pd.errors.ParserError`, `pd.errors.EmptyDataError`, `UnicodeError`.

### 4.2 Doppia validazione di qualità in cascata
**File**: `beat_detection.py:431` (`validate_beats_morphology`) + `analyze.py:131` (chiamata a `quality_control.validate_beats`)

Il software ha **due gate di qualità**:
1. Durante la beat detection (`morphology_min_corr=0.7`, soglia *fissa* in BeatDetectionConfig)
2. Dopo la segmentazione (QualityConfig, soglia *adattiva* basata su CV)

I due gate non sanno l'uno dell'altro. Il primo già scarta beat anomali, ma il secondo riapplica un criterio simile su un sottoinsieme già filtrato. Quando la soglia adattiva del secondo gate si attiva (perché `n_pass < 0.6 * n_totale`), i percentili usati (5°/15°/30° in funzione di CV<0.20/0.35/altro) sono **hardcoded in `quality_control.py:379–393`**, non esposti a config.

Conseguenza: chi cerca di tarare il QC modificando `morphology_threshold` non capisce perché la soglia effettiva sia diversa, e la documentazione del README su `accepted/n_input` è ambigua (vedi §6 di `confronto_software_vs_paper.md`, già documentato dall'autore).

### 4.3 Soglie magiche hardcoded sparse
Esempi che ho rilevato:
- `repolarization.py:155`: `agreement_radius_ms = 50.0` per il consensus FPD — non scala con la durata FPD del soggetto (50 ms è il 5% di 1000 ms ma il 25% di 200 ms).
- `arrhythmia.py:353`: fattore `1.4826` (conversione MAD→σ) hardcoded, assume distribuzione Gaussiana del campione FPD; nessun guard se MAD ≈ 0 (segnale piatto → falsi EAD).
- `cessation.py:163-173`: peso QC del composite score hardcoded a `0.15`, e l'invariante "weight_energy + weight_gaps + weight_deterioration + weight_terminal = 1.0" non è validato (se l'utente importa una config con somma 2.0, lo score viene quasi dimezzato in modo silente).
- `parameters.py:99`: il template è limitato a `max_beats_template` battiti — selezione `np.linspace` (uniforme nel tempo). Su recording lunghi questo butta via informazione utile.

### 4.4 Test suite tutta sintetica
**File**: `tests/golden_signals.py`, `tests/test_e2e_synthetic.py`

I segnali generati sono onde gaussiane + rumore bianco gaussiano (`std=0.0005 V`). **Mancano completamente**:
- Drift di baseline lento (deriva farmaco-indotta, comune)
- Saturazione (clipping per spike >range ADC)
- Artefatti 50/60 Hz e armoniche (la pipeline ha un notch, ma non è testata su un segnale che lo richiede davvero)
- Eventi morfologici autentici (EAD, DAD come oscillazioni sostenute, non come spike isolati)
- File con encoding/separator vari (UTF-8 con BOM, CRLF, decimali con virgola europea)
- Frequenze di campionamento anomale (250 Hz molto basso, 50 kHz molto alto)
- Recording molto brevi (< 5 beat) o senza ripolarizzazione misurabile

Diversi test usano pattern tautologici tipo `assert spike_amplitude_mV > 0` o `assert abs(n_detected - n_expected) <= 2` (tollera 13% di errore). Il `test_golden_reference.py` confronta contro snapshot della pipeline stessa — blocca regressioni ma non verifica correttezza biologica.

---

## 5. Problemi MINORI ma da segnalare

- **`config.py` è enorme (686 LOC, ~75 parametri)** in 9 dataclass. Difficilissimo da navigare. Andrebbe almeno spezzato per modulo (un file `config_beat.py`, un `config_repol.py`, ecc.) o documentato meglio quali parametri sono "user-facing" e quali "internal tuning".
- **`beat_detection.py` (1357 LOC) e `cdisc_export.py` (1359 LOC)** vanno spezzati. Il primo in `detection_core.py` + `validation.py` + `recovery.py` + `scoring.py`. Il secondo in `domains/` (un file per dominio TS/DM/EX/EG/RISK) + `xport_writer.py` + `define_xml.py`.
- **Race condition potenziale in `_recover_missed_beats`** (`beat_detection.py:178`): la lista `all_bi` viene modificata mentre si itera per check di vicinanza. Da verificare con un test mirato; potenzialmente innocuo ma fragile.
- **Filtro Butterworth con ordine fisso a 4** + `filtfilt` (ordine effettivo 8): a frequenze di Nyquist molto basse (lowcut=0.5 Hz, fs=10 kHz) può introdurre instabilità. Valutare ordine adattivo.
- **`risk_score=0` per baseline** è azzerato in `analyze.py:300` senza log: se un utente analizza una baseline aspettandosi un confronto, non capisce dove è finito il valore.
- **Filename collision** nel report: `f"...{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"` collide se due download partono nello stesso secondo (raro ma vero).
- **PDF batch in memoria**: con 169 file e grafici matplotlib full-resolution, può raggiungere diversi GB. Nessun warning all'utente.

---

## 6. Pattern problematici trasversali

1. **Soglie scientifiche hardcoded nei file algoritmici** invece che in `config.py`. Effetto: l'autore non può cambiare soglia da un'unica fonte di verità.
2. **Nessun fixture di dati reali** nei test. Effetto: ogni release può rompere il comportamento sui dati veri senza che nessun test lo segnali.
3. **`@st.cache_data` mai usato**. Effetto: la UI è inusabile per batch grandi.
4. **Assenza di `validate_invariants()` nelle config**: dataclass create senza validazione; un JSON malformato genera errori a runtime molto in profondità nello stack.
5. **Nessuna metrica di confidenza propagata fino al report finale**: il `confidence` calcolato in `repolarization.py` viene usato per il filtering ma non è esposto chiaramente all'utente nei tab di summary.

---

## 7. Roadmap di intervento consigliata

In ordine di rapporto valore/sforzo decrescente:

**Sprint 1 — risolvere i bug silenti (1–2 giorni)**
1. ✅ Fix search-window truncation in `analyze.py`/`channel_selection.py` (CRITICO 3.2 — commit `04ecb23`)
2. Uniformare parametri duplicati in `config.py` (CRITICO 3.1)
3. Rendere `pyreadstat` obbligatorio per CDISC + check upfront (CRITICO 3.3, spostato da Sprint 2)
4. Aggiungere `@st.cache_data` su `load_csv`, `analyze_single_file`, `minmax_downsample` (CRITICO 3.4)
5. Estendere try/except del batch e includere errori pandas (4.1)
6. Aggiungere `assert` di invariante o rimuovere il pattern `min_len` inoperante in `parameters.py:111`, `quality_control.py:191`, `residual_analysis.py:28` (cleanup non critico documentato in 3.2)

**Sprint 2 — robustezza algoritmica (3–5 giorni)**
7. Aggiungere fixture di dati reali nei test: 5–10 CSV reali con problemi noti (drift, saturazione, dofetilide, baseline cattiva); test che verifichino `assert FPD ∈ range_atteso`, non `> 0`.
8. Spostare le soglie hardcoded di `quality_control.py` (percentili adattivi) e `repolarization.py` (agreement_radius) in `config.py`.
9. Rendere `agreement_radius_ms` adattivo (es. `5% × FPD_mediana`).

**Sprint 3 — manutenibilità (5–10 giorni)**
10. Spezzare `beat_detection.py` e `cdisc_export.py` in moduli più piccoli.
11. Spezzare `config.py` per modulo + aggiungere `__post_init__` con `validate_invariants()` su ogni dataclass.
12. Documentare il flusso del doppio gate QC nel docstring di `analyze.py`, e idealmente unificarli.
13. Aggiungere log strutturato della pipeline (quale beat scartato in quale step e perché).

**Sprint 4 — validazione (continuo)**
14. Validazione su dataset esterni (CiPA pubblici).
15. Test statistici (Kruskal-Wallis + Dunn) come da limitazione 3 di `confronto_software_vs_paper.md`.
16. Correzione `amplifier_gain` di default (oggi a 1.0; con dato Digilent dovrebbe essere 1e4).

---

## 8. Conclusione

Il codice **non è da rifare**. È in ottimo stato strutturale per uno strumento di ricerca sviluppato in agentic mode. I problemi che incontri sono concentrati in **5–10 punti chirurgici**, non distribuiti ovunque. Ti consiglio di:

- iniziare dalle fix dello Sprint 1, che risolvono 3–4 bug verificati con poco rischio;
- aggiungere fixture di dati reali al test suite **prima** di toccare gli algoritmi (così le modifiche sono coperte);
- rivedere insieme la logica del doppio gate QC, che è probabilmente la causa principale dei comportamenti che vedi come "non perfetti" sui dati farmacologici.

L'incremental testing che già pratichi (memoria: "mai riscrivere beat detection in un colpo solo") è la strategia giusta — applicalo anche al refactoring di `config.py`.
