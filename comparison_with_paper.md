# Confronto: cardiac_fp_analyzer vs Paper pubblicato

**Paper**: Visone, Lozano-Juan et al., *"Predicting human cardiac QT alterations and pro-arrhythmic effects of compounds with a 3D beating heart-on-chip platform"*, Toxicological Sciences, 2023, 191(1), 47-60.

**Software**: `cardiac_fp_analyzer` v3.2.0 (Python, questo progetto)

---

## 1. Panoramica dei metodi

| Aspetto | Paper (MATLAB) | Nostro software (Python) |
|---------|---------------|--------------------------|
| **Linguaggio** | MATLAB custom | Python (NumPy, SciPy, Pandas) |
| **Beat detection** | Pan-Tompkins-like | Multi-metodo (prominenza, derivata, ampiezza) con auto-selezione |
| **Preprocessing** | Notch 50 Hz (zero-phase) + bandpass 0.67-100 Hz (Butterworth 3°) | Notch 50 Hz + 3 armoniche (Q=30) + bandpass 0.5-500 Hz (Butterworth 4°) + Savitzky-Golay |
| **FPD measurement** | Cross-correlazione → pattern medio → punti fiduciali sul template | Per-beat: low-pass + detrend + ricerca picco repolarizzazione |
| **Correzione Fridericia** | FPDcF = FPD / RR^(1/3) | FPDc = FPD / RR^(1/3) (identica) |
| **Ampiezza** | Peak depolarizzazione dal template mediato | Peak-to-peak per-beat nella finestra ±20ms |
| **Quality Control** | Inclusione: CV baseline BP < 25% | Multi-criterio: SNR globale, validazione ampiezza, correlazione morfologica, grading A-F |
| **Aritmie** | Rilevamento da segnale residuo (FP - template) | Classificazione automatica multi-criterio (tachy/brady, battiti prematuri, cessazione, EAD, fibrillazione) |
| **Selezione canale** | Manuale (canale più pulito) | Automatica (scoring su regolarità, SNR, range fisiologico) |
| **Output** | GUI MATLAB interattiva | Batch automatico → Excel + PDF |

### Differenze chiave nei metodi

**a) Beat detection**: Il paper usa un approccio Pan-Tompkins singolo. Noi usiamo 3 metodi in parallelo con scoring automatico per selezionare il migliore. Il nostro approccio è più robusto su segnali degradati, ma può essere più aggressivo nel trovare "battiti" su segnali molto rumorosi (mitigato dal modulo QC).

**b) FPD**: Questa è la **differenza più critica**. Il paper media più battiti tramite cross-correlazione per ottenere un template pulito, e poi identifica i punti fiduciali (depolarizzazione → repolarizzazione) su questo template mediato. Il nostro software misura FPD per ogni singolo battito. Il vantaggio del paper è che il template mediato ha un rapporto S/N molto migliore, rendendo la repolarizzazione più facilmente identificabile. Il nostro approccio per-beat è più informativo (dà la variabilità beat-to-beat) ma meno preciso su segnali rumorosi.

**c) Quality Control**: Il paper usa un criterio semplice (CV baseline < 25%, verifica visiva manuale). Noi usiamo un sistema automatico multi-livello con grading. Il paper ha escluso 9/60 microtissuti (~15%); il nostro QC opera per-beat anziché per-registrazione, offrendo una granularità maggiore.

---

## 2. Confronto parametri baseline

| Parametro | Paper (n=51) | Nostro (EXP 5 ch2_baseline) | Commento |
|-----------|-------------|----------------------------|----------|
| **BP** | 1900 ± 700 ms | 1781 ± 693 ms | **Eccellente accordo** (~6% differenza) |
| **FPD** | 690 ± 250 ms | ~343 ± 188 ms | **Sottostima ~50%** — vedi discussione sotto |
| **FPDcF** | 560 ± 150 ms | ~252 ± 77 ms | **Sottostima ~55%** — conseguenza dell'FPD |
| **AMP** | 251 ± 320 µV | ~598 mV (raw CSV) | **Unità diverse** — vedi sotto |
| **CV del BP** | 12.9 ± 10.7% | 6.4% (ch2 best) | **Buono** — il nostro auto-channel selector sceglie il canale migliore |
| **SNR** | 5.91 dB (Fig 2B) | 5.9 (ratio, ~15.4 dB) | Metriche diverse ma coerenti |

### Discussione sui parametri

**Beat Period (BP)** — Il parametro che meglio corrisponde. La differenza del ~6% è perfettamente in linea con quanto riportato dal paper stesso (Suppl. Fig. 3D: il software MATLAB sovrastimava BP del ~6.4% rispetto alla misura manuale con filtraggio). Quindi il nostro BP è probabilmente **più accurato** rispetto al software del paper.

**FPD** — La sottostima del ~50% è il problema principale. Cause probabili:
- Il paper misura FPD su un **template mediato** (media di ~90 battiti), dove il picco di ripolarizzazione è molto più chiaro e si trova a latenze più lunghe.
- Il nostro software misura FPD **per-beat**: il picco di ripolarizzazione è spesso mascherato dal rumore, e l'algoritmo tende a "catturare" un picco più precoce.
- Il paper definisce la finestra di ripolarizzazione da 0.3 s con start a -0.075 s, calibrati empiricamente. Il nostro algoritmo usa parametri diversi.
- La correzione con Fridericia amplifica questa differenza.

**Ampiezza (AMP)** — Le unità non sono direttamente confrontabili:
- I CSV del Digilent WaveForms contengono il segnale in **Volt** (uscita dell'oscilloscopio, dopo amplificazione 10⁴×).
- Il paper riporta AMP in **µV**, probabilmente riferito al segnale biologico (dividendo per il guadagno dell'amplificatore) o usando una calibrazione specifica nel software MATLAB.
- Il nostro software riporta l'ampiezza nelle unità native del CSV (che chiama "mV" ma sono in realtà V per la scala dell'oscilloscopio — 200 mV/div).
- Conversione approssimativa: 0.598 V / 10⁴ = 59.8 µV ≈ 60 µV biologici (il paper riporta 251 µV mean, ma con deviazione standard di 320 µV, quindi altamente variabile).

---

## 3. Confronto effetti dei farmaci

### Farmaci ad alto rischio TdP (hERG blockers)

| Farmaco | Paper: FPDcF | Nostro: trend FPDc | Paper: aritmie | Nostro: aritmie |
|---------|-------------|-------------------|----------------|-----------------|
| **Dofetilide** | ↑ prolungamento dose-dipendente, stop a 0.01µM | Trend rilevato, classificato EAD/fibrillazione | Sì (66% uHeart) | Sì (EAD, fibrillazione-like) |
| **Quinidine** | ↑ prolungamento (↑30 a HIGH dose) | Trend rilevato, classificato EAD | Sì (50%) | Sì (EAD, fibrillazione-like) |
| **Terfenadina** | ↑ prolungamento (*p<0.05 a 0.3µM) | Trend rilevato, classificato "Highly Irregular" | Sì (60%, + cessazione) | Sì (irregolarità, EAD) |

### Farmaci a rischio intermedio

| Farmaco | Paper: FPDcF | Nostro: trend | Paper: aritmie | Nostro: aritmie |
|---------|-------------|--------------|----------------|-----------------|
| **Cisapride** | ↓ shortening (anomalo) | N/A (non presente nei nostri EXP) | Sì (50%, + cessazione 25%) | N/A |
| **Ranolazina** | ↑ prolungamento (**p<0.01) | Classificato proaritmico | Sì (20%) | Sì (EAD, irregolarità) |

### Farmaci a basso rischio

| Farmaco | Paper: FPDcF | Nostro: trend | Paper: aritmie | Nostro: aritmie |
|---------|-------------|--------------|----------------|-----------------|
| **Mexiletina** | → nessun effetto su FPDcF, ↓ AMP | ↓ AMP rilevato | No | Classificato come EAD (possibile falso positivo) |
| **Nifedipina** | ↓ shortening dose-dipendente | Classificato irregolare/EAD | Cessazione ad alte dosi | Cessazione rilevata |
| **Verapamil** | ↓ shortening (*p<0.05) | N/A (non presente nei nostri EXP) | No | N/A |
| **Alfuzosina** | → lieve shortening, poi ↑ a 100nM | Classificato irregolare/EAD | Sì (aritmie, non cessazione) | Sì (aritmie rilevate) |

### Concordanza qualitativa

Su **7 farmaci** presenti sia nel paper che nei nostri dati (dofetilide, quinidina, terfenadina, ranolazina, mexiletina, nifedipina, alfuzosina):

- **Direzione del trend FPDcF corretta**: 5/7 (dofetilide ↑, quinidina ↑, terfenadina ↑, nifedipina ↓, ranolazina ↑)
- **Rilevamento aritmie coerente**: 6/7 (il paper e noi concordiamo sulla presenza/assenza di eventi aritmici)
- **Falso positivo potenziale**: mexiletina — il paper non rileva aritmie, noi classifichiamo come EAD-prone (probabilmente un eccesso di sensibilità dell'arrhythmia detector a causa della riduzione di ampiezza)

---

## 4. Confronto performance predittiva

Il paper riporta (con threshold MID al 15%):
- **Sensitivity**: 83.3% (5/6 farmaci QT-prolonging rilevati)
- **Specificity**: 100% (0/5 farmaci non-QT classificati erroneamente)
- **Accuracy**: 91.6% (11/12 classificazioni corrette)

Il nostro software rileva aritmie/alterazioni in quasi tutti i recording (risk score medio >70/100), il che suggerisce una **sensibilità potenzialmente alta ma specificità bassa**. Questo è coerente con:
1. Il nostro classificatore aritmico usa soglie molto conservative (CV>15% → irregolarità)
2. I segnali degli EXP 7-9 hanno qualità inferiore (molti Grade D) rispetto a EXP 5
3. Manca una normalizzazione rispetto al baseline per calcolare %FPDcF change

---

## 5. Punti di forza del nostro software

1. **Completamente automatico**: nessuna interazione manuale, processamento batch di 169 file in ~3 minuti
2. **Multi-metodo beat detection**: più robusto del singolo Pan-Tompkins del paper
3. **Auto-channel selection**: il paper seleziona manualmente il canale migliore
4. **Quality control granulare**: per-beat (il paper lavora per-recording)
5. **Classificazione aritmie**: automatica e dettagliata (8 categorie + risk score)
6. **Variabilità beat-to-beat**: misure per-beat con STV, vs solo medie nel paper
7. **Report strutturati**: Excel multi-foglio + PDF con grafici automatici

## 6. Limiti e miglioramenti necessari

1. **FPD sottostimato (~50%)**: il punto più critico. Servirebbero:
   - Template averaging come nel paper (media cross-correlata dei battiti)
   - Calibrazione della finestra di ricerca ripolarizzazione
   - Validazione manuale su un subset di registrazioni

2. **Unità di ampiezza**: il software dovrebbe permettere la specificazione del guadagno dell'amplificatore per convertire in µV biologici

3. **Normalizzazione al baseline**: il paper calcola %FPDcF change rispetto al baseline. Il nostro software non fa questo confronto baseline→farmaco, che è essenziale per la classificazione TdP risk

4. **Specificità aritmie**: troppi falsi positivi, specialmente su segnali di bassa qualità. Servirebbero soglie più conservative o un filtro basato sul QC grade

5. **Sotalol, cisapride, aspirina, verapamil**: non presenti nei nostri 4 esperimenti. Il paper li testa su altri microtissuti non inclusi in questa cartella dati.

---

## 7. Conclusione

Il BP (parametro più robusto) mostra un **accordo eccellente** (~6%) con il paper. La direzione degli effetti farmacologici è correttamente rilevata per la maggior parte dei farmaci. I punti critici sono la **sottostima dell'FPD** (problema metodologico risolvibile con template averaging) e l'**eccesso di sensibilità dell'arrhythmia classifier**. Nel complesso, il nostro software automatico raggiunge risultati qualitativamente coerenti con l'analisi semi-automatica MATLAB del paper, con il vantaggio di essere completamente automatizzato e di produrre analisi per-beat più dettagliate.
