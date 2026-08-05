# Bug: l'RR usato per FPDc e per il CV è calcolato sui battiti post-QC

**Data:** 5 agosto 2026
**Gravità:** alta — tocca l'endpoint primario (FPDc) su circa metà del dataset
**Stato:** verificato, non ancora corretto

---

## 1. Il meccanismo

Il QC scarta battiti per **ampiezza bassa** e **morfologia anomala**. Sono depolarizzazioni reali che non somigliano al template — non sono falsi positivi di detection. Il loro *timing* è valido.

Quando gli intervalli RR vengono ricalcolati sui soli battiti sopravvissuti, gli intervalli attraversano i buchi lasciati dai battiti scartati e risultano gonfiati. Se il QC elimina un battito su due, l'RR misurato raddoppia.

Catena, verificata riga per riga:

```
QC scarta battiti  →  bi_fpd ha buchi
  parameters.py:351   beat_periods = np.diff(bi_fpd)/fs        ← RR gonfiato
  parameters.py:428   rr = beat_periods[i-1]
  parameters.py:301   fpdc = fpd / rr**(1/3)                   ← FPDc sottostimato
  parameters.py:509   bp_ms → summary['beat_period_ms_cv']     ← CV gonfiato
```

Poiché `FPDc = FPD / RR^(1/3)`, un RR gonfiato **abbassa** l'FPDc riportato.

## 2. Il problema era già noto — e corretto a metà

`analyze.py:354-356`:

```python
# Beat period from ALL detected beats (timing is reliable even for
# morphologically marginal beats) — avoids artificial gaps from QC rejection.
bp = compute_beat_periods(bi, fs)
```

Il commento descrive esattamente questo bug, e la scelta giusta (usare i battiti grezzi) è già stata presa. Ma è stata applicata **solo** a `result['beat_periods']`. La chiamata a `extract_all_parameters(bd_fpd, btm_fpd, bi_fpd, ...)` continua a ricevere l'insieme filtrato, quindi `summary['beat_period_ms_*']` e tutta la catena FPDc restano con i buchi.

È di nuovo il pattern dello Sprint 0: **decisione giusta presa, collegata in un posto solo.**

## 3. Quanto pesa — 67 registrazioni misurate

Dataset: EXP 8/day6, EXP 5, EXP 7/ChipC.

**Gonfiaggio dell'RR** (filtrato / grezzo, 1.00 = nessun effetto)

| | valore |
|---|---:|
| mediana | 1.12× |
| p75 | 1.96× |
| p90 | 2.57× |
| max | 3.72× |
| file con RR gonfiato > 10 % | **35 / 67** |
| file con RR gonfiato > 50 % | **22 / 67** |

**Errore su FPDc** — di quanto il valore corretto è più alto di quello riportato oggi

| | valore |
|---|---:|
| mediana | **+4.0 %** |
| p75 | **+25.2 %** |
| p90 | **+37.0 %** |
| max | **+54.9 %** |
| file con errore > 5 % | 33 / 67 |
| file con errore > 10 % | **25 / 67** |

**Gonfiaggio del CV** (punti percentuali, summary − grezzo): mediana +7.5 pp, p90 +36.8 pp, max +251.7 pp.

## 4. Conseguenza sui criteri di inclusione

**21 registrazioni su 67 vengono spinte oltre la soglia CV del 25 % dal solo artefatto.** Alcune:

| File | CV grezzo | CV summary | battiti | QC |
|---|---:|---:|---:|:--:|
| `chipD_ch2_baseline_bis` | **8.8 %** | **170.6 %** | 28/71 | D |
| `chipD_ch2_ALFUS_10nM` | 11.5 % | 48.4 % | 24/96 | D |
| `chipD_ch2_ALFUS_1000nM_2` | 12.6 % | 28.9 % | 89/101 | B |
| `chipC_ch1_Alfus_100nM_2` | 14.5 % | 29.8 % | 59/66 | B |
| `chipD_ch2_Alfus_300uM` | 18.4 % | 77.5 % | 50/182 | D |
| **`chipD_ch1_baseline`** | **23.7 %** | **31.8 %** | 79/156 | D |

La prima riga è emblematica: un ritmo con CV **8.8 %** — regolarissimo — riportato come **170.6 %**.

L'ultima è **la baseline della dofetilide**.

## 5. Il punto 1 si risolve da sé

La baseline `chipD_ch1_baseline` di EXP 8/day6 ha un CV reale del **23.7 %**, cioè **sotto la soglia del 25 %**. Non andava esclusa.

Il gruppo dofetilide — 7 concentrazioni del controllo positivo CiPA canonico — è stato rimosso da un artefatto di calcolo, non da un difetto della preparazione.

**Non serve riacquisire nulla.** La mia conclusione precedente ("è un problema di dati") era sbagliata: è un bug software.

## 6. Cosa NON è in discussione, e cosa sì

**Non in discussione:** le due definizioni di beat period che convivono nel dizionario dei risultati sono incoerenti, e i consumatori a valle pescano ora dall'una ora dall'altra. `result['beat_periods']` usa i battiti grezzi, `summary['beat_period_ms_*']` quelli filtrati. Questo va unificato in ogni caso.

**Da decidere:** quale definizione è quella giusta per l'RR di correzione.

- *A favore dei battiti grezzi* — è la posizione già espressa dagli autori nel commento: il QC scarta per ampiezza e morfologia, non per timing implausibile, quindi quei battiti sono depolarizzazioni reali e il loro istante è valido. Ed è l'unica scelta che non produce RR fantasma.
- *A favore dei filtrati* — se una parte dei battiti scartati fosse rumore scambiato per battiti, includerli accorcerebbe artificialmente l'RR. Il filtro cluster-ampiezza a monte dovrebbe però già occuparsene.

La mia lettura è che i grezzi siano corretti, e che il caso `chipD_ch2_baseline_bis` (CV 8.8 % → 170.6 %) lo dimostri: nessuna interpretazione ragionevole rende 170 % la variabilità di quel ritmo.

Resta una terza via, più precisa di entrambe: mantenere per ogni battito accettato il suo RR **locale reale**, misurato sul segnale grezzo rispetto al battito immediatamente precedente — anche se quel precedente è stato scartato dal QC. Costa poco e non richiede di scegliere fra i due insiemi.

## 7. Impatto sui risultati già prodotti

Diverso da quello dello Sprint 0, e più grande.

Il confronto delle 206 registrazioni (`SPRINT0_reanalisi_dataset_storici.md`) resta valido: metteva a confronto due versioni che avevano **entrambe** questo bug, quindi le conclusioni di quel documento non cambiano.

Ma i **valori assoluti** di FPDc in tutti i risultati storici sono sottostimati, in misura che dipende da quanto ha scartato il QC — mediana 4 %, ma oltre il 10 % su un file su tre. E dato che il `%ΔFPDcF` è un rapporto fra due FPDc con frazioni di scarto QC diverse, l'errore **non si cancella** nella normalizzazione.

Questo va quantificato prima di correggere, con lo stesso metodo pre/post usato per lo Sprint 0.

---

## 8. Proposta

1. Unificare la definizione di beat period, con l'RR locale reale (§6, terza via).
2. Quantificare pre/post su tutto il dataset di calibrazione, come per lo Sprint 0.
3. Solo dopo, rivalutare i criteri di inclusione: la soglia CV va ritarata su CV veri, e l'analisi in `SPRINT1_soglia_CV_baseline.md` è basata su CV gonfiati — **le sue conclusioni numeriche vanno rifatte** (l'argomento concettuale sul CV come strumento sbagliato regge, i numeri no).
4. Verificare se la dofetilide, con l'RR corretto, arriva alla classificazione.
