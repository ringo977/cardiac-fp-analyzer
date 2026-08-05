# Validazione contro i dati pubblicati (Visone et al. 2023)

**Riferimento:** Visone, Lozano-Juan et al., *"Predicting human cardiac QT alterations and pro-arrhythmic effects of compounds with a 3D beating heart-on-chip platform"*, Toxicological Sciences 2023, 191(1), 47-60.
**Confronto:** valori di baseline pubblicati (n = 51 microtessuti) contro le 36 baseline del dataset di calibrazione, analizzate col codice dopo i fix di oggi.

---

## 1. Il risultato

| Parametro | Paper (n=51) | Software (n=36) | scarto | |
|---|---:|---:|---:|---|
| BP | 1900 ± 700 ms | 1461 ± 578 ms | −29 % | dentro ±1 SD |
| **FPD** | **690 ± 250 ms** | **604 ± 189 ms** | **−12 %** | dentro ±1 SD |
| **FPDcF** | **560 ± 150 ms** | **552 ± 161 ms** | **−6 %** | dentro ±1 SD |
| CV del BP | 12.9 ± 10.7 % | 25 ± 24 % | +75 % | dentro ±1 SD |

## 2. Il confronto che conta

`comparison_with_paper.md`, scritto alla v3.3.0, documentava:

> **FPD** — La sottostima del ~50% è il problema principale.
> FPD ~343 ms contro 690 del paper; FPDcF ~252 ms contro 560.

Quel documento elencava fra i limiti: *"FPD sottostimato (~50%): il punto più critico"*, e ipotizzava servisse il template averaging come nel paper.

Dopo i fix di oggi:

```
FPDcF   prima: 252 ms  (−55 %)   →   ora: 528 ms  (−6 %)
FPD     prima: 343 ms  (−50 %)   →   ora: 610 ms  (−12 %)
```

**La sottostima del 55 % non richiedeva un cambio di metodo. Erano bug.** In ordine di peso: l'RR calcolato sui battiti post-QC (che deflazionava l'FPDc), il segno della ripolarizzazione non tracciato per-battito, e il degrado silenzioso da `tangent` a `peak` che ne conseguiva.

Questa è una verifica indipendente: nessuno dei fix è stato tarato su questi valori: sono stati derivati dal codice e dalla fisiologia, e il confronto col paper è arrivato dopo.

## 3. Le due discrepanze residue

### BP −29 % (1461 contro 1900 ms)

Le nostre preparazioni battono più rapidamente. Non è necessariamente un errore: sono esperimenti diversi da quelli del paper. Il documento precedente notava che sul singolo `EXP 5 ch2_baseline` l'accordo era del 6 %, quindi la differenza è fra popolazioni, non di metodo.

Vale la pena verificare se il sottoinsieme di segnali effettivamente comune al paper mostra lo stesso scarto: se sì, è metodo; se no, è biologia.

### CV del BP +75 % (25 % contro 12.9 %) — e cosa implica per la soglia

Questa è la più istruttiva, perché tocca direttamente il criterio di inclusione.

Il paper usa `CV baseline < 25 %` come criterio, **su una popolazione con mediana 12.9 %**. Per loro quella soglia taglia la coda: escludono 9 microtessuti su 60, circa il 15 %.

La nostra popolazione ha **mediana 25 %**. La stessa soglia numerica cade sulla mediana e taglia metà del dataset.

**Il numero è stato trapiantato senza il contesto che lo rendeva sensato.** E c'è una ragione strutturale per cui le due popolazioni differiscono: il paper seleziona **manualmente** il canale più pulito ed esclude i microtessuti dopo verifica visiva. La loro popolazione è pre-filtrata dal giudizio umano prima che il criterio CV la veda. La nostra no — selezione canale automatica, nessuna ispezione visiva.

Applicare la soglia di una popolazione pre-selezionata a una non pre-selezionata fa fare al criterio un lavoro diverso da quello per cui era stato scelto.

## 4. Conseguenza per la ricalibrazione

Rafforza la conclusione di `SPRINT3_ricalibrazione_CV.md`, e da una direzione indipendente.

Il 25 % non è un valore sacro derivato dalla fisiologia: è una soglia empirica valida per una popolazione visivamente pre-filtrata. Sui nostri dati, non pre-filtrati, esclude 5 riferimenti precisi su 36 senza intercettare nulla che il criterio di precisione non intercetti.

Restano aperte due strade, ora meglio informate:

1. **Sostituire il CV col criterio di precisione** (`rSEM ≤ 3 %`), come proposto. Misura direttamente ciò che serve e non dipende da quanto la popolazione sia stata pre-filtrata.
2. **Riprodurre il pre-filtro del paper** — ispezione visiva o un equivalente automatico — e *poi* applicare il CV al 25 %. Più fedele al metodo pubblicato, ma richiede il passo manuale che il software esiste per evitare.

La prima è coerente con lo scopo dello strumento. La seconda sarebbe l'unica difendibile se l'obiettivo fosse replicare esattamente la pipeline del paper.

## 5. Cosa questo non dimostra

L'accordo è sulle **medie di popolazione delle baseline**. Non è ancora verificato:

* che le curve dose-risposta riproducano le direzioni note (dofetilide ↑, nifedipina ↓, mexiletina →);
* che la sensibilità/specificità si avvicini all'83 %/100 % riportata dal paper;
* che i singoli segnali comuni diano gli stessi valori — il confronto è fra popolazioni, non appaiato.

Il terzo è il più informativo e il più fattibile: se nel dataset ci sono le registrazioni usate nel paper, un confronto file per file separerebbe definitivamente gli errori di metodo dalle differenze biologiche.

**Nota su una discrepanza già visibile:** `comparison_with_paper.md` riporta che il paper osserva per la nifedipina un **accorciamento** dose-dipendente dell'FPD. Nelle nostre analisi di oggi la nifedipina risulta **positiva**, cioè allungamento — direzione opposta. Per un bloccante dei canali del calcio l'attesa è l'accorciamento. Va indagato: è il candidato più promettente per trovare il prossimo bug.
