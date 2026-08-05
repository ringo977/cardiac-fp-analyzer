# Sprint 2 — Impatto del fix RR locale sui dataset di calibrazione

**Data:** 5 agosto 2026
**Fix:** ogni battito accettato viene corretto per frequenza usando l'RR verso il **predecessore reale**, anche se scartato dal QC (`compute_local_rr`). Il beat-period del summary descrive il ritmo completo rilevato.
**Dataset:** 201 registrazioni, 8 sottoinsiemi.

---

## 1. Numeri

| Dataset | file | mediana \|ΔFPDc\| | max \|ΔFPDc\| | >5 % | call cambiate |
|---|--:|--:|--:|--:|--:|
| EXP 5 | 21 | 0.7 % | 65.2 % | 5 | 0 |
| EXP 7 / ChipC | 22 | 2.4 % | 41.3 % | 9 | 1 |
| EXP 7 / ChipE | 21 | 0.4 % | 49.4 % | 4 | 5 |
| EXP 7 / chipA | 9 | 2.6 % | 53.8 % | 4 | 1 |
| EXP 8 / day6 | 24 | **13.9 %** | 59.2 % | 17 | 1 |
| EXP 8 / day7 | 45 | **17.7 %** | 67.6 % | 30 | 2 |
| EXP 9 / day7 | 27 | 3.7 % | **81.6 %** | 13 | 2 |
| Signal Extr. / Exp4 | 32 | 2.7 % | 35.5 % | 10 | 0 |
| **totale** | **201** | | | **92** | **12** |

Per confronto, il sign-fix dello Sprint 0 dava mediana 0.0 %, max 11.9 % e **zero** call cambiate. Questo fix è di un ordine di grandezza più grande.

## 2. Il risultato che cercavamo

**La dofetilide è tornata.** In EXP 8/day6:

```
farmaci classificati    prima: [alfus]
                        dopo : [alfus, dofe]

dofe   +35.3 %,  positiva,  8 concentrazioni
file inclusi   15/24  →  23/24
```

Tutte e 7 le concentrazioni più la baseline passano da `False` a `True`. La baseline `chipD_ch1_baseline`:

| | prima | dopo |
|---|---:|---:|
| RR medio | 2272 ms | **1148 ms** |
| CV | 31.8 % | **23.7 %** ← sotto soglia |
| BPM | 26.4 | **52.3** |
| FPD | 617.8 ms | 617.8 ms (invariato, corretto) |
| FPDc | 483.3 ms | 602.6 ms (+24.7 %) |

Recuperata anche in EXP 7/ChipE, dove era assente.

## 3. Il risultato inatteso — e la sua causa

Delle 12 call cambiate, alcune sono **perdite**: nifedipina (ChipE, chipA) e mexiletina (ChipC) scompaiono dalla classificazione.

Non è il fix a perderle. Diagnosi su EXP 7/ChipE:

| File | FPDc prima | FPDc dopo | Δ |
|---|---:|---:|---:|
| `chipE_ch2_baseline` | 564 ms | **843 ms** | +49.4 % |

843 ms supera il limite superiore del filtro **`fpdc_physiol` = [350, 800] ms**, quindi la baseline viene esclusa e con lei tutto il gruppo nifedipina.

Prova diretta — stesso codice nuovo, solo quel filtro spento:

```
filtro fisiologico ON  → ['dofetilide']
filtro fisiologico OFF → ['dofetilide', 'nifedipine', 'nifedipine 1',
                          'nifedipine 10', 'nifedipine 5']
```

**Il filtro era tarato contro FPDc sistematicamente deflazionati.** I suoi limiti vengono dalla letteratura (Blinova/CiPA 350-800 ms per hiPSC-CM), ma venivano applicati a numeri che il bug abbassava del 4-55 %. Il filtro faceva quindi qualcosa di diverso da ciò che dichiarava. Ora fa quello che dice — ed è per questo che i suoi effetti cambiano.

## 4. La direzione del cambiamento è quella giusta

`comparison_with_paper.md` documentava che l'FPDcF del software era **~252 ± 77 ms contro il paper di riferimento, "Sottostima ~55 %"**.

Il fix sposta l'FPDc **verso l'alto**, cioè verso l'intervallo di letteratura. È evidenza indipendente che la correzione va nel verso corretto: parte della sottostima nota rispetto al paper era questo bug.

Questo non significa che ora tutti i valori siano giusti — significa che erano sbagliati per un motivo identificato, e che rimosso quel motivo si avvicinano al riferimento.

## 5. Cosa resta da fare

1. **Ritarare `fpdc_physiol`.** I limiti [350, 800] non sono più applicati agli stessi numeri di prima. Vanno verificati contro la distribuzione degli FPDc corretti su tutte le baseline, con lo stesso metodo usato per la soglia CV. Finché non è fatto, il filtro esclude gruppi validi (nifedipina, mexiletina).
2. **Rifare l'analisi della soglia CV.** `SPRINT1_soglia_CV_baseline.md` è costruito su CV gonfiati: le sue conclusioni numeriche non valgono più. L'argomento concettuale — il CV misura regolarità, non qualità — regge.
3. **Riconfrontare col paper.** Con l'FPDc corretto, `comparison_with_paper.md` va rifatto: la sottostima del 55 % andrebbe ricalcolata.
4. **Valutare l'isteresi.** La ripolarizzazione dipende da una media pesata degli ultimi cicli, non solo dall'RR immediatamente precedente. Non urgente, ma è il raffinamento naturale ora che l'RR è corretto.

## 6. Sui risultati storici

Cambia il quadro rispetto allo Sprint 0.

Lì i valori assoluti si muovevano poco e nessuna conclusione cambiava. Qui **12 classificazioni su 8 dataset cambiano**, e la mediana dello scostamento arriva al 17.7 % su un dataset.

I dati storici prodotti con la versione precedente hanno FPDc sottostimati in misura che dipende da quanto ha scartato il QC su ogni singola registrazione. Poiché il `%ΔFPDcF` è un rapporto fra due FPDc con frazioni di scarto diverse, **l'errore non si cancella nella normalizzazione**.

Diversamente dallo Sprint 0, qui una ri-analisi dei dataset che hai già presentato è **necessaria**, non facoltativa. Ma conviene farla dopo aver ritarato `fpdc_physiol` (punto 1), altrimenti si ri-analizza con un filtro non allineato.
