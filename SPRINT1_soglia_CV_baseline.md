# Soglia CV sulle baseline — analisi sui dati

**Data:** 5 agosto 2026
**Domanda:** la soglia `max_cv_bp = 25%` che esclude le baseline è troppo severa?
**Dati:** 36 baseline del dataset di calibrazione (tutte quelle riconosciute da `is_baseline`)

---

## Conclusione in una riga

**No, non è troppo severa — ma è lo strumento sbagliato.** Il CV da solo sbaglia in entrambe le direzioni: esclude baseline sane e ammette un file che l'operatore stesso ha chiamato `nosignal`.

La mia ipotesi iniziale ("alziamo la soglia") era sbagliata, e i dati la smentiscono.

---

## 1. La distribuzione

| | CV(BP) |
|---|---:|
| min | 4.1 % |
| p25 | 15.2 % |
| **mediana** | **34.0 %** |
| p75 | 52.5 % |
| p90 | 90.0 % |
| max | 237.0 % |

Con soglia 25 % passano **15 baseline su 36 (42 %)**. Cioè il 58 % dei gruppi dose-risposta viene rimosso a monte.

Sembra un massacro. Ma:

## 2. Il CV alto è quasi sempre segnale brutto, non biologia

| QC | n | CV mediano | CV min | CV max |
|:--:|--:|---:|---:|---:|
| **A** | 7 | **7.4 %** | 4.1 % | **18.8 %** |
| B | 8 | 27.8 % | 6.4 % | 40.1 % |
| C | 5 | 56.4 % | 24.8 % | 237.0 % |
| D | 11 | 50.3 % | 14.1 % | 170.6 % |
| F | 5 | 47.9 % | 23.5 % | 64.4 % |

**Tutte e 7 le baseline di grado A stanno sotto il 18.8 %** — la soglia non ne tocca nemmeno una. E i CV estremi (116 %, 127 %, 171 %, 237 %) non sono ritmo irregolare: sono registrazioni con BPM 9-15 o con il 68 % dei battiti senza ripolarizzazione, cioè preparati morenti o rumore scambiato per battiti.

Quindi la soglia sta facendo, grosso modo, il suo lavoro. Alzarla a 40 % farebbe passare 20 baseline invece di 15, ma recupererebbe soprattutto roba di grado C/D.

## 3. Dove sbaglia davvero

**Falsi negativi — esclude baseline sane.** Cinque baseline con confidence alta e QC buono cadono solo per il CV:

| File | CV | QC | conf | BPM | % senza repol |
|---|---:|:--:|---:|---:|---:|
| `chipE_ch1_baseline` | 30.8 % | B | 0.86 | 43 | 4 % |
| `chipA_ch1_baseline` | 36.2 % | B | 0.88 | 21 | 0 % |
| `chip4_ch3_BASELINE` | 40.1 % | B | 0.78 | 30 | 1 % |
| `chipA_ch3_baseline` | 39.2 % | C | 0.79 | 37 | 8 % |
| `chipC_ch2_baseline` | 56.4 % | C | 0.87 | 27 | 14 % |

Ripolarizzazione rilevata sul 96-100 % dei battiti, confidence 0.78-0.88. Sono battiti spontanei irregolari, che negli hiPSC-CM è la norma — non segnali rotti.

**Il CV non discrimina la qualità.** Il caso che lo mostra:

```
chip1_ch3_baseline_nosignal.csv    CV=24.8%   →  PASSA il criterio CV
                                   QC=C  conf=0.59  BPM=106  noRepol=33%
```

Il nome del file dice `nosignal`. BPM 106 su preparato spontaneo (implausibile), un terzo dei battiti senza ripolarizzazione. Eppure il **CV è buono**: 24.8 %, sotto soglia. Perché il rumore periodico è più regolare di un cuore vero.

> **Correzione (verificata dopo la prima stesura).** Questo file **non** entra
> nell'analisi: viene escluso dal criterio 3, `confidence = 0.591 < 0.66`.
> La pipeline completa lo intercetta. L'affermazione vale per il criterio CV
> preso da solo, non per il comportamento reale del software — la prima
> versione di questo documento era imprecisa su questo punto.

Il punto concettuale resta, ed è quello che conta: **un CV basso non è evidenza di qualità, è evidenza di regolarità** — che il rumore può avere e un preparato sano può non avere. Il CV va usato in combinazione con altri indicatori, non come gate autonomo. Nel caso specifico è la confidence a salvare la situazione, ma è una coincidenza fortunata, non un progetto.

## 4. Regola alternativa

Confronto a parità di condizioni, con **tutti** i criteri attivi come nella pipeline reale (quindi la confidence a 0.66 vale per entrambe le regole):

| Regola | baseline ammesse |
|---|--:|
| `CV < 25%` — attuale | **13 / 36** |
| `QC ≤ C AND CV < 60%` + guardrail plausibilità | **15 / 36** |

Scambio: 10 invariate, **5 entrano, 3 escono**.

**Entrano** — erano escluse *solo* per irregolarità del ritmo:

| File | CV | QC | conf |
|---|---:|:--:|---:|
| `ChipE/chipE_ch1_baseline` | 30.8 % | B | 0.86 |
| `day7/chipA_ch1_baseline` | 36.2 % | B | 0.88 |
| `Exp4/chip4_ch3_BASELINE` | 40.1 % | B | 0.78 |
| `EXP6/chipA_ch3_baseline` | 39.2 % | C | 0.79 |
| `day7/chipC_ch2_baseline` | 56.4 % | C | 0.87 |

**Escono** — CV basso ma qualità di segnale scarsa:

| File | CV | QC | conf |
|---|---:|:--:|---:|
| `ChipE/chipE_ch2_baseline` | 14.1 % | **D** | 0.92 |
| `day7/chipA_ch2_baseline` | 20.1 % | **D** | 0.86 |
| `chipA/chipA_ch1_basline` | 23.5 % | **F** | 0.93 |

Le tre che escono sono il rovescio esatto della medaglia: CV eccellente (14-23 %) e confidence alta, ma grado QC D/F. Sono i casi in cui il CV basso stava mascherando un segnale povero — esattamente ciò che il criterio da solo non può vedere.

**Bilancio netto: +2 baseline, e composizione migliore.** Non è un recupero massiccio; è uno scambio di qualità.

## 5. E la dofetilide?

```
chipD_ch1_baseline (EXP 8/day6)
  CV=31.8%   QC=D   conf=0.82   BPM=26   noRepol=3%   77 battiti
  oggi        → ESCLUSA (CV ≥ 25%)
  combinata   → ESCLUSA (QC = D)
```

Resta fuori in entrambi i casi. Ma il motivo cambia: da "il ritmo è irregolare" a "la qualità del segnale è scarsa", che è una motivazione più difendibile davanti a un revisore — e verificabile, perché il grado QC ha una sua diagnostica.

Va detto però che questa baseline ha il 97 % dei battiti con ripolarizzazione e confidence 0.82. Se dopo aver rivisto i criteri la dofetilide resta l'unico controllo positivo CiPA che non arriva mai alla classificazione, **il problema non è la soglia: è che quel gruppo va riacquisito.**

---

## 6. Cosa propongo

1. **Non alzare il CV.** Sostituire il criterio singolo con quello combinato `QC ≤ C AND CV < 60% AND conf ≥ 0.66`, dietro flag di config, default sul comportamento attuale finché non validi lo scambio dei 5 file.

2. **Rendere visibili le esclusioni.** Oggi un gruppo intero sparisce senza che nulla lo dica nel report. Serve un blocco esplicito *"gruppi rimossi e perché"* nell'output di batch e nella UI. Questa è la parte più importante delle due: un filtro sbagliato di cui ti accorgi è un problema gestibile, uno silenzioso no.

3. **Aggiungere un guardrail di plausibilità** indipendente dal CV: BPM fuori da un range fisiologico (diciamo 10-120), o percentuale di battiti senza ripolarizzazione sopra il 50 %, dovrebbero escludere a prescindere. Sarebbe bastato questo per fermare `nosignal`.

4. **Rinominare o rimuovere** `chip1_ch3_baseline_nosignal.csv` dal dataset di calibrazione — se l'operatore sapeva che non c'era segnale, non deve competere con i criteri automatici.

---

## 7. Riproducibilità

Lo script che ha prodotto questi numeri è in `/tmp/cv_survey.py` (non committato). Analizza tutte le baseline e salva `cv_bp`, `fpdc`, `conf`, `bpm`, `n_beats`, `pct_no_repol`, `qc` in JSON. Se vuoi lo porto in `tools/` insieme al comparatore.
