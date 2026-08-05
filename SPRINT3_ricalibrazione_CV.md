# Ricalibrazione del criterio CV — e perché va sostituito

**Data:** 5 agosto 2026
**Base:** 36 baseline e 201 registrazioni con RR corretto (post-`927568f`)
**Sostituisce:** `SPRINT1_soglia_CV_baseline.md`, i cui numeri erano viziati dal bug dell'RR

---

## 1. La domanda giusta

Non "quale numero mettere al posto di 25 %", ma: **il CV misura la cosa che ci serve?**

Una baseline serve a fornire un **riferimento FPDc** contro cui confrontare il farmaco. Quello che conta è quanto quel riferimento è determinato con precisione — l'errore standard relativo della media, `SD / (media · √n)` — non quanto il ritmo fosse regolare.

## 2. Il CV è un cattivo predittore

Correlazione fra CV(RR) e dispersione dell'FPDc, sulle 36 baseline:

```
r = +0.613     →  R² = 0.376
```

Il CV spiega il **38 %** della varianza di ciò che dovrebbe rappresentare.

Verifica diretta sul gate CV < 25 %:

| | conteggio |
|---|---:|
| falsi positivi (CV ≥ 25 % ma rSEM < 2 %) | **5** |
| falsi negativi (CV < 25 % ma rSEM ≥ 4 %) | **0** |

Il gate scarta riferimenti precisi e non intercetta nulla che un criterio di precisione non intercetti. Alcuni dei falsi positivi:

| File | CV(RR) | rSEM | n | QC |
|---|---:|---:|---:|:--:|
| `chip4_ch3_BASELINE` | 26.2 % | **1.14 %** | 84 | B |
| `chipD_ch3_baseline` | 29.2 % | **1.25 %** | 65 | B |
| `chipD_ch2_baseline` | 38.5 % | **1.63 %** | 17 | B |
| `chipC_ch2_baseline` | 29.3 % | **1.68 %** | 70 | C |

Sono preparazioni con battito irregolare ma ben campionate: la media assorbe l'irregolarità e il riferimento è ottimo. Scartarle costa un intero gruppo dose-risposta ciascuna.

## 3. Distribuzione del SEM relativo

Sulle 36 baseline: min 0.13 %, p25 0.79 %, **mediana 1.41 %**, p75 2.31 %, p90 3.84 %, max 6.37 %.

Anche la peggiore ha il riferimento determinato al 6.4 %, perché n è grande (16-118 battiti con FPD).

**Soglia proposta: 3 %.** La soglia di classificazione è un cambiamento del 15 % dell'FPDc; un riferimento incerto al 3 % contribuisce al massimo un quinto dell'effetto che si sta chiamando. Corrisponde a escludere circa l'ultimo decile.

## 4. Confronto end-to-end — 201 registrazioni

| scenario | inclusi | farmaci |
|---|---:|---:|
| **A** — CV < 25 % (attuale) | 134/201 | 11 |
| **B** — rSEM ≤ 3 % | **169/201** | **18** |
| **C** — entrambi | 134/201 | 11 |

**C è identico ad A.** Aggiungere il CV al criterio di precisione costa 35 file e 7 farmaci e non guadagna nulla: è strettamente dominato.

Farmaci recuperati passando da A a B: `mexil`, `nife`, `nife wash1h`, `nife washout 40minutes`, `chip1 ch1 ranolazine`, `chip1 ch1 washout1h`, `chip4 ch3 quinidine`.

Fra questi la **nifedipina** e la **mexiletina**, che avevamo perso nello Sprint 2.

## 5. Il limite del criterio — e perché resta opt-in

Delle 8 baseline che la precisione ammette e il CV escludeva, **3 hanno QC D o F**:

| File | rSEM | n | QC | conf | FPD/RR |
|---|---:|---:|:--:|---:|---:|
| `chip1_ch1_baseline` | 2.16 % | 44 | **D** | 0.80 | 66 % |
| `chipC_ch3_baseline` | 2.19 % | 67 | **F** | 0.84 | 54 % |
| `chipE_ch3_baseline` | 2.30 % | 52 | **F** | 0.82 | 48 % |

Il punto statistico: **la precisione protegge dall'errore casuale, non da quello sistematico.** Se il rumore fa sì che il detector agganci stabilmente un afterpotential invece dell'onda T, mediare 60 battiti produce una stima *precisa di una quantità sbagliata*. La precisione da sola non è quindi sufficiente.

Una nota è che il criterio dipende da `n`, quindi una registrazione breve di una preparazione sana ottiene un punteggio peggiore. È difendibile — una registrazione più corta *dà* effettivamente un riferimento meno preciso — ma conferma che si misura il riferimento, non la biologia.

Un caso di conferma nell'altro verso: `chipA_ch1_basline` (QC F, rSEM 1.34 %) entrerebbe per precisione ma viene fermata dal criterio FPD/RR (97 %, impossibile). I criteri si coprono a vicenda, ed è così che devono lavorare.

## 6. Proposta

`enabled_baseline_precision` implementato e **lasciato opt-in**, con `max_baseline_fpdc_rsem = 3.0`.

L'evidenza per farne il default è forte — +35 file, +7 farmaci, criterio dominante — ma le 3 ammissioni di grado D/F sono una domanda aperta che spetta a chi conosce le preparazioni, non ai numeri. Tre possibilità:

1. **precisione da sola** — massima resa, ammette 3 registrazioni rumorose;
2. **precisione + QC ≤ C** — le esclude, ma va verificato quanto costa altrove (con QC ≤ C avevamo riperso la dofetilide);
3. **precisione + soglia su `fpd_confidence` più alta** — le tre stanno a 0.80-0.84, quindi una soglia a 0.85 le escluderebbe senza toccare il grado QC.

La terza è la più mirata e la meno costosa; la verificherei prima di decidere.

## 7. Metodo

Questa analisi sostituisce `SPRINT1_soglia_CV_baseline.md`. Quel documento tarava una soglia su CV gonfiati da un bug scoperto dopo, e le sue conclusioni numeriche non valgono. L'argomento concettuale — un CV basso significa regolare, non buono — regge, e anzi qui trova la sua formulazione corretta: **il CV non è la grandezza da guardare affatto.**
