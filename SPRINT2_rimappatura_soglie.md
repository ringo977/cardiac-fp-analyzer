# Rimappatura delle soglie su valori corretti

**Data:** 5 agosto 2026
**Perché:** l'analisi delle soglie in `SPRINT1_soglia_CV_baseline.md` era costruita su CV gonfiati dal bug dell'RR post-QC. Rifatta sulle stesse 36 baseline con il codice corretto.
**Esito:** due conclusioni dello Sprint 1 vanno riviste, e ne emerge un criterio migliore di quello attuale.

---

## 1. Il CV cambia faccia

| | prima | dopo |
|---|---:|---:|
| p25 | 15.2 % | 10.0 % |
| **mediana** | **34.0 %** | **22.6 %** |
| p75 | 52.5 % | 29.0 % |
| p90 | 90.0 % | 36.8 % |
| max | 237.0 % | 150.9 % |

Baseline che passano la soglia:

| soglia | prima | dopo |
|---:|---:|---:|
| 25 % | 15 | **22** |
| 30 % | 16 | **30** |
| 40 % | 20 | 35 |

La distribuzione è molto più plausibile per battito spontaneo di hiPSC-CM. Con la soglia invariata al 25 %, **7 baseline rientrano** senza toccare nulla.

## 2. Prima revisione: il CV non correla con la qualità

Questa è la conclusione dello Sprint 1 che salta.

CV mediano per grado QC:

| QC | prima | **dopo** |
|:--:|---:|---:|
| A | 7.4 % | **7.4 %** |
| B | 27.8 % | **24.9 %** |
| C | 56.4 % | **22.7 %** |
| D | 50.3 % | **27.9 %** |
| F | 47.9 % | **26.1 %** |

Prima sembrava esserci un gradiente netto A→F. Adesso **solo il grado A si distingue** (7.4 %); B, C, D ed F stanno tutti fra 22 e 28 %, indistinguibili.

Il gradiente apparente era l'artefatto: QC peggiore → più scarti → CV più gonfiato. Rimosso l'artefatto, il CV separa molto meno di quanto sembrasse.

Nello Sprint 1 avevo scritto *"la soglia non è troppo severa, il CV correla con la qualità"*. La prima metà resta vera per il grado A; **la seconda metà era un artefatto**. Il che rafforza — non indebolisce — l'argomento per la regola combinata che usa il grado QC esplicitamente invece di dedurlo dal CV.

## 3. Seconda revisione: il range assoluto di FPDc è lo strumento sbagliato

FPDc delle baseline:

| | prima | dopo |
|---|---:|---:|
| p10 | 345 | 382 |
| mediana | 503 | 528 |
| p75 | 578 | 640 |
| p90 | 685 | **801** |
| max | 825 | 927 |

Il filtro `[350-800] ms` escludeva 6 baseline prima e 6 dopo, ma **non le stesse**: 3 rientrano da sotto, 3 escono da sopra. Fra queste ultime c'è `chipE_ch2_baseline`, che si porta via tutta la nifedipina.

### Il test fisiologico

La ripolarizzazione non può occupare l'intero ciclo cardiaco. Ricavando FPD e RR dai dati (`FPD = FPDc · RR^⅓`, `RR = 60/BPM`):

| File | BPM | RR | FPD | **FPD/RR** | QC |
|---|--:|--:|--:|--:|:--:|
| `chipE_ch2_baseline` | 89 | 678 ms | 740 ms | **109 %** | D |
| `chip1_ch3_baseline_nosignal` | 108 | 555 ms | 581 ms | **105 %** | C |
| `chipA_ch1_basline` | 65 | 924 ms | 903 ms | **98 %** | F |
| `chipA_ch2_baseline` | 69 | 866 ms | 818 ms | **94 %** | D |

Le prime due hanno **FPD più lungo dell'RR**: la ripolarizzazione finirebbe dopo il battito successivo. È fisicamente impossibile — sono errori di detection, con ogni probabilità un afterpotential o la depolarizzazione seguente scambiati per onda T.

Distribuzione su tutte le baseline: p25 = 32 %, mediana = **41 %**, p75 = 55 %, p90 = 83 %, max 109 %. Solo 4 su 36 superano l'80 %, e sono esattamente quelle sopra.

### Confronto fra i due criteri

| criterio | baseline escluse |
|---|--:|
| range assoluto `[350-800] ms` | 6 |
| **rapporto FPD/RR > 80 %** | **4** |
| in comune | 3 |

Le 3 escluse **solo** dal range assoluto:

| File | FPDc | FPD/RR | QC | lettura |
|---|--:|--:|:--:|---|
| `chipD_ch3_baseline` | 871 ms | **51 %** | B | bradicardica (27 BPM), FPD lungo ma proporzionato |
| `chipD_ch1_baseline` | 277 ms | 31 % | B | rapida (70 BPM), FPD corto ma proporzionato |
| `chipB_ch2_baseline` | 270 ms | 17 % | C | borderline, possibile onda T mancata |

Le prime due sono preparazioni sane a frequenze estreme. **Il range assoluto le boccia proprio dove dovrebbe essere più tollerante.**

Il motivo è che l'FPDc conserva una dipendenza residua dalla frequenza: la correzione di Fridericia non normalizza completamente, quindi una finestra fissa sull'FPDc sbaglia agli estremi di frequenza. Il rapporto FPD/RR non ha questo problema — è adimensionale e codifica direttamente il vincolo fisiologico.

### Un bonus

`chip1_ch3_baseline_nosignal.csv` — il file che l'operatore aveva marcato come privo di segnale — viene escluso dal criterio del rapporto **su basi fisiologiche** (105 %), non per la coincidenza fortunata della soglia di confidence che avevo trovato nello Sprint 1.

## 4. Proposta

1. **Sostituire** `enabled_fpdc_physiol` (range assoluto) con un criterio sul rapporto **FPD/RR ≤ 80 %**, adimensionale e valido a qualsiasi frequenza. Mantenere il range assoluto come rete di sicurezza molto larga (`fpdc_range` [100-1200] esiste già e va bene).
2. **Lasciare la soglia CV al 25 %** per ora: con i CV corretti fa passare 22/36 invece di 15/36, ed è un valore di letteratura. Ma vale come criterio *secondario* — il primario dovrebbe essere il grado QC (§2).
3. **Attivare la regola combinata** (`enabled_combined_rule`) come default, ora che sappiamo che il CV da solo non separa B/C/D/F.
4. Solo dopo 1-3, **ri-analizzare i dataset storici**.

Combinazioni misurate, per riferimento:

| CV | range FPDc | passano |
|---:|---|--:|
| 25 % | [350-800] | 17/36 |
| 30 % | [350-800] | 22/36 |
| 30 % | [350-900] | 25/36 |
| 35 % | [350-1000] | 27/36 |

Ma questi numeri riguardano il criterio *assoluto*; con il rapporto FPD/RR la scelta del range diventa in gran parte irrilevante, che è il punto.

## 5. Nota di metodo

Due conclusioni dello Sprint 1 sono cadute perché costruite su numeri viziati da un bug scoperto dopo. Vale la pena registrarlo: **prima di tarare una soglia su dati misurati, verificare che la misura sia corretta.** In entrambi i casi il segnale d'allarme c'era — una mediana di CV al 34 % su preparazioni spontanee era alta, e l'ho annotata senza indagarla.
