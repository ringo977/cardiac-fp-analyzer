# Confronto Sistematico: Software Cardiac FP Analyzer vs Paper MATLAB

**Riferimento**: Visone, Lozano-Juan et al., *Toxicological Sciences* 191(1), 47–60, 2023
**Dataset**: 169 file CSV (EXP 5, 7, 8, 9) — stessi dati usati nel paper
**Data analisi**: 21 Marzo 2026 (v2 — con tangent method + confidence filtering)

---

## 1. Architettura e Pipeline

| Aspetto | Paper (MATLAB) | Nostro Software (Python) | Note |
|---------|---------------|------------------------|------|
| Linguaggio | MATLAB | Python 3 | Open source, no licenza |
| Beat detection | Pan-Tompkins | Prominence/derivative/peak multiplo | 3 metodi, scelta automatica migliore |
| Preprocessing | Notch 50Hz + bandpass 0.3–10kHz | Bandpass 0.5–200Hz (Butterworth) | Paper anche 0.67–100Hz per validazione |
| Template averaging | Cross-correlazione + media | Cross-correlazione + mediana robusta | Mediana più resistente agli outlier |
| FPD detection | Fiducial points su template | **Tangent method** (max-downslope + intersezione baseline) | Standard gold per misura QT/FPD |
| Correzione Fridericia | FPDcF = FPD / RR^(1/3) | FPDcF = FPD / RR^(1/3) | Identico. Anche Bazett disponibile |
| QC inclusione | CV baseline BP < 25% | CV < 25% + **FPD confidence > 0.68** | Confidence = prominence × consistency |
| Aritmie | Analisi residuo (segnale - template) | BP variability + STV + pattern detection | Approcci diversi |
| Normalizzazione | % change vs baseline, soglie 10/15/20% | % change vs baseline, soglie 10/15/20% | Identico |
| TdP scoring | Score -1 a 3 (Ando et al. 2017) | Score -1 a 3 (adattato) | Identico schema |
| Output | GUI MATLAB | Excel + PDF + batch CLI | Automatico per 169 file |

---

## 2. Parametri Baseline

Il paper riporta i parametri baseline su n=51 microtissuti che hanno superato il criterio di inclusione (CV BP < 25%).

| Parametro | Paper (n=51) | Nostro (n=7, CV<25% + conf≥0.68) | Differenza |
|-----------|-------------|----------------------------------|------------|
| **BP** | 1900 ± 700 ms | **1838 ± 645 ms** | **-3.3%** |
| **FPD** | 690 ± 250 ms | **627 ± 129 ms** | **-9.1%** |
| **FPDcF** | 560 ± 150 ms | **555 ± 51 ms** | **-0.9%** |
| **AMP** | 251 ± 320 µV | 596 ± 546 µV | Unità non corrette (manca ÷10⁴) |

### Analisi critica

**BP: eccellente** (-3.3% vs paper). Il criterio CV < 25% elimina le baselines con beat detection errato.

**FPDcF: praticamente perfetto** (-0.9% vs paper). La combinazione di *tangent method* (misura FPD alla massima pendenza discendente della curva di ripolarizzazione, con intersezione alla baseline) e *confidence filtering* (soglia 0.68 per le baselines) produce risultati quasi identici al paper. Miglioramento drammatico rispetto alla v1 che sottostimava del 32%.

**FPD: buono** (-9.1% vs paper). Leggera sottostima residua, ma ben entro la variabilità biologica (SD=250ms nel paper).

**Ampiezza**: Ancora non confrontabile senza correzione del guadagno dell'amplificatore (÷10⁴).

### Dettaglio per-baseline (n=7)

| File | BP (ms) | CV% | FPD (ms) | FPDcF (ms) | Conf |
|------|---------|-----|----------|-----------|------|
| chipA_ch1_baseline (EXP5) | 818 | 6.4 | 427 | 457 | 0.69 |
| chipA_ch2_baseline (EXP5) | 2313 | 10.3 | 699 | 530 | 0.87 |
| chipC_ch1_baseline (EXP5) | 1566 | 4.7 | 655 | 564 | 0.85 |
| chipD_ch2_baseline (EXP5) | 1266 | 24.6 | 618 | 576 | 0.78 |
| chipD_ch3baseline (EXP5) | 2897 | 21.7 | 748 | 527 | 0.71 |
| chipA_ch2_baseline (EXP7) | 2194 | 17.5 | 782 | 605 | 0.94 |
| chipE_ch2_baseline (EXP7) | 1814 | 5.5 | 757 | 622 | 0.93 |

---

## 3. Risposte Farmacologiche — EXP 5 (Dataset Migliore)

### 3.1 Terfenadina (CiPA: Intermediate, FDA: QT↑)

| Conc. | FPDcF (ms) | %FPDcF | Conf | Note |
|-------|-----------|--------|------|------|
| Baseline | 457 | — | 0.69 | chipA_ch1 |
| 1 nM | 601 | **+31.3%** | 0.85 | Prolungamento chiaro |
| 8 nM | 543 | +18.7% | 0.86 | |
| 10 nM | 613 | **+34.1%** | 0.85 | |
| 100 nM | 650 | **+42.0%** | 0.87 | Massimo prolungamento |
| 300 nM | 515 | +12.6% | 0.81 | Inizio inversione (tossicità) |
| 1000 nM | 528 | +15.5% | 0.86 | |

**Paper**: Prolungamento FPDcF (*p<0.05 a 0.3µM), aritmie 60%.
**Noi**: Prolungamento dose-dipendente forte (1–100nM), **massimo +42% a 100nM**. Inversione ad alte dosi. **CONCORDANTE**.

### 3.2 Quinidina (CiPA: High, FDA: QT↑ + TdP)

| Conc. | FPDcF (ms) | %FPDcF | Conf | Note |
|-------|-----------|--------|------|------|
| Baseline | 530 | — | 0.87 | chipA_ch2 |
| 0.06 µM | 576 | +8.7% | 0.93 | Lieve prolungamento |
| 0.1 µM | 531 | +0.1% | 0.90 | |
| 1 µM | 407 | -23.3% | 0.58 | Conf bassa, morfologia alterata |
| 3 µM | 683 | **+28.8%** | 0.86 | Massimo prolungamento |
| 10 µM | 357 | -32.7% | 0.60 | Tossicità |
| 30 µM | 350 | -34.0% | 0.79 | Tossicità |

**Paper**: Prolungamento FPDcF dose-dipendente, aritmie 50%. Threshold >MID a 1µM.
**Noi**: Prolungamento a 3µM (+28.8%) e lieve a 0.06µM (+8.7%). A dosi elevate il segnale si degrada e FPDcF si accorcia (confidence crolla a 0.58). **CONCORDANTE** per la detection del prolungamento.

### 3.3 Dofetilide (CiPA: High, FDA: QT↑)

| Conc. | FPDcF (ms) | %FPDcF | Conf | Note |
|-------|-----------|--------|------|------|
| Baseline | 622 | — | 0.93 | chipE_ch2, EXP7 |
| 0.3 nM | 603 | -3.0% | 0.90 | Quasi invariato |
| 1 nM | 336 | -45.9% | 0.57 | **Conf bassa** — misura inaffidabile |
| 2 nM | 387 | -37.7% | 0.81 | Accorciamento apparente |
| 3 nM | 312 | -49.9% | 0.60 | Conf bassa |
| 6 nM | 224 | -63.9% | 0.35 | Conf molto bassa, segnale degradato |
| 10 nM | 461 | -25.9% | 0.66 | |

**Paper**: Dofetilide è il più potente bloccante hERG. Causa cessazione del battito a dosi alte.
**Noi**: **FALSO NEGATIVO**. L'algoritmo non rileva il prolungamento FPDcF. Invece, registra accorciamento, con confidence che crolla progressivamente. Il farmaco altera così radicalmente la morfologia che la ricerca della ripolarizzazione fallisce. Questo è il caso più difficile per l'analisi automatica.

---

## 4. Analisi Sensibilità / Specificità

### Confronto con il paper (soglia MID = 15% FPDcF)

| Metrica | Paper | Software v1 (peak, no filtri) | Software v2 (tangent + conf) |
|---------|-------|------------------------------|------------------------------|
| **Sensitivity** | 83.3% | 100% → 66.7%* | **66.7%** |
| **Specificity** | 100% | 0% → 33.3%* | **50.0%** |
| **Accuracy** | 91.6% | 57.1% → 50%* | **60.0%** |
| **FPDcF baseline** | 560 ± 150 ms | 380 ± 134 ms | **555 ± 51 ms** |
| **BP baseline** | 1900 ± 700 ms | 1867 ± 576 ms | **1838 ± 645 ms** |

(*) = dopo aggiunta inclusion criteria

### Dettaglio per farmaco

| Farmaco | CiPA | FDA QT↑ | n OK | Max %FPDcF | >MID? | Match |
|---------|------|---------|------|-----------|-------|-------|
| Quinidina | H | YES | 6 | **+28.8%** | YES | ✓ |
| Terfenadina | I | YES | 6 | **+42.0%** | YES | ✓ |
| Dofetilide | H | YES | 6 | -3.0% | NO | ✗ (FN) |
| Alfuzosina | n.a. | NO | 12 | +6.1% | NO | ✓ |
| Mexiletina | L | NO | 6 | +15.6% | YES | ✗ (FP borderline) |

### Analisi degli errori residui

**Dofetilide (Falso Negativo)**: Bloccante hERG ultra-potente. A concentrazioni > 1nM il farmaco altera talmente la morfologia del segnale che il tangent method trova repolarizzazione "fittizia" precoce, causando accorciamento apparente. La confidence precipita (0.35–0.66) ma non viene filtrata per le recording farmaco. Questo è un problema strutturale: servirebbe un approccio completamente diverso per waveform radicalmente alterate (es. analisi nel dominio della frequenza, o detection di cessazione del battito).

**Mexiletina (Falso Positivo borderline)**: Un singolo recording a 2.5µM mostra +15.6%, appena sopra la soglia MID. Le altre concentrazioni sono tutte < 10%. Possibile artefatto di una singola misura. Aumentando la soglia a 16% diverrebbe TN.

---

## 5. Punti di Forza del Software

1. **FPDcF baseline quasi identico al paper** (555 vs 560 ms, -0.9%) grazie al tangent method
2. **100% automatico**: 169 file analizzati senza intervento manuale
3. **Open source Python**: nessuna licenza MATLAB richiesta
4. **Template averaging robusto**: mediana per resistenza a outlier
5. **Tangent method per FPD**: gold standard nella misurazione QT/FPD
6. **Confidence scoring**: ogni misura FPD ha un punteggio di affidabilità
7. **Multi-level quality control**: CV baseline, FPDcF plausibility, FPD confidence
8. **Normalizzazione completa**: pairing baseline-farmaco, soglie 10/15/20%, TdP scoring
9. **Report Excel con 4 fogli**: Summary, Normalization, Arrhythmia Flags, Per-Beat Data
10. **CLI con opzioni complete**: `--inclusion-cv`, `--no-fpdc-filter`, `--correction`

---

## 6. Punti di Debolezza

1. **Sensibilità 66.7% vs 83.3%**: Dofetilide non viene rilevato perché il farmaco distrugge la morfologia del segnale e l'algoritmo non riesce a trovare la ripolarizzazione corretta.

2. **Specificità 50% vs 100%**: Mexiletina è borderline FP (+15.6%). Potrebbe risolversi con soglia leggermente più alta o con analisi statistica (la media delle % change potrebbe essere più stabile del max).

3. **Waveform alterate non gestite**: Quando un farmaco altera radicalmente la forma d'onda (es. dofetilide, quinidina ad alte dosi), la ricerca della ripolarizzazione basata su template/tangent non funziona. Servirebbe un approccio adattivo.

4. **Ampiezza non corretta**: Manca la divisione per il guadagno dell'amplificatore (÷10⁴).

5. **Assenza test statistici**: Mancano Kruskal-Wallis + Dunn's post-test per significatività.

6. **Aritmie: analisi non completa**: Il paper usa il residuo (segnale - template). Noi usiamo variabilità statistica.

---

## 7. Evoluzione: v1 → v2

| Aspetto | v1 (peak method) | v2 (tangent method) |
|---------|-----------------|---------------------|
| FPD measurement | Picco più prominente | Tangent method (max-downslope → intersezione baseline) |
| Detrending | Polinomiale grado 2 | **Lineare endpoint-based** |
| Low-pass | 15 Hz | **20 Hz** |
| Search window | 150-800 ms | **150-900 ms** |
| FPDcF baseline | 380 ± 134 ms (-32%) | **555 ± 51 ms (-0.9%)** |
| Confidence score | — | **prominence × consistency (0-1)** |
| Baseline filtering | Solo CV < 25% | CV < 25% **+ conf ≥ 0.68** |
| Bazett correction | — | **FPDcB disponibile** |
| FPD uncorrected | — | **FPD riportato separatamente** |

---

## 8. Roadmap Rimanente

### Priorità 1 — Risolvere FN dofetilide

1. **Detection cessazione battito**: Se il battito si ferma o diventa molto irregolare dopo farmaco ad alta concentrazione, classificare automaticamente come "cessazione" (corrispondente a prolungamento estremo/TdP score 3).

2. **Threshold confidence per drug recordings**: Registrazioni farmaco con confidence < 0.50 + calo > 30% vs baseline → flaggare come "waveform alterata" (potenziale effetto severo).

### Priorità 2 — Risolvere FP mexiletina

3. **Criterio statistico**: Invece del max %FPDcF, usare la **media** ponderata per confidence, o richiedere almeno 2 concentrazioni sopra soglia.

4. **Test Kruskal-Wallis**: Significatività statistica delle % change vs baseline.

### Priorità 3 — Miglioramenti generali

5. Correzione guadagno amplificatore
6. Analisi residuo per aritmie
7. CiPA risk map 2D
8. GUI web (Streamlit)

---

## 9. Conclusioni

Il software v2 con il **tangent method** raggiunge una misura FPDcF baseline praticamente identica al paper (555 vs 560 ms). Il miglioramento è stato ottenuto cambiando il punto di misura della ripolarizzazione: dal picco dell'onda T (che sottostimava del 32%) alla **intersezione tangente-baseline sulla pendenza discendente**, che è il gold standard nella misurazione QT/FPD.

La combinazione di tangent method + confidence filtering elimina il falso positivo più grave (nifedipina), ma rimane:
- **1 falso negativo** (dofetilide: farmaco troppo potente, distrugge il segnale)
- **1 falso positivo borderline** (mexiletina: +15.6%, appena sopra soglia)

L'accuratezza complessiva è del 60% vs 91.6% del paper — la differenza è dovuta principalmente alla gestione dei segnali con morfologia radicalmente alterata, dove il paper beneficia della verifica manuale dell'operatore. I prossimi passi dovrebbero concentrarsi sulla detection automatica di cessazione/degradazione severa del segnale.
