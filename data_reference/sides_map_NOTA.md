# Mappa dei lati (punto C, keystone) — stato 2026-08-06

## Cosa
Per ogni microtessuto, quale lato/elettrodo (`sx`=el1, `dx`=el2) gli autori hanno usato per i valori
pubblicati. Dedotto dagli Excel di riferimento con la regola del handoff: **il lato usato è quello il cui
blocco `RR array` grezzo contiene numeri**. I blocchi stanno a destra nei fogli (marker `sx` a col ~25,
`dx` a col ~49), con le concentrazioni come colonne.

File: `sides_resolved_clean.json` (67 microtessuti). Tally: **21 sx / 32 dx / 9 both / 5 none**.

## Reperto importante: `ground_truth.json` ha il bug del default-sx
Cross-check del resolver contro le etichette "certe" di `ground_truth.json`: **dissente su 10/23 casi, e
tutti sono `gt=sx → resolver=dx/both`, mai il contrario.** Asimmetria = le etichette `sx` di ground_truth
sono il default buggato dell'estrattore ("legge sempre il blocco sx", noto in memoria). Il resolver,
contando la densità reale del blocco RR-array, trova che molti "sx" sono in realtà dx o both. Ha risolto
37/44 microtessuti prima marcati `unknown`.

## Validazione diretta su Exp7 ChipE (decisiva)
Baseline ChipE su entrambi i canali contro il QT autori (684 ms):
- el1 (sx): FPD 514 ms, no_repol 9%
- el2 (dx): **FPD 576 ms (più vicino a 684), no_repol 4% (ripolarizzazione più pulita)**
Gli autori hanno quasi certamente usato **dx/el2** per ChipE. Il resolver (che dava dx/both) è più
corretto di ground_truth (che dava sx).

## Conseguenza sul punto B (da tenere presente)
La mia analisi del meccanismo residuo del punto B su Exp7 ChipE è stata fatta su **el1 (sx)**, ma gli
autori usavano **el2 (dx)**. Il *meccanismo* (find_peaks aggancia un after-potential su template piatto)
resta valido, ma i numeri specifici di ChipE (FPD 2/3/6nM, ratio baseline 16.8%) andrebbero **rifatti su
el2** per un confronto corretto col paper.

## Da rifinire prima di usarla in produzione
- La soglia `both` (rapporto densità <3) è troppo lasca: dà 9 both, ma ChipE (sx136/dx255) è
  probabilmente dx-dominante, non both. Tararla contro il test diretto "quale canale combacia col QT
  autori", NON contro un tally atteso.
- Verificare se Marco ha già lo script originale di /tmp/sides.json (handoff: "script in memoria" — non
  l'ho trovato nei file di memoria).

## Prossimo passo naturale (punto C completo)
Con la mappa lati affidabile: per ogni microtessuto girare la pipeline sul canale-autori (C1, isola
l'errore di MISURA) e su auto (C2). Metrica C2: quando auto sceglie un canale diverso, l'FPD è peggiore?
(concorda = nessuna info / diverso ma FPD vicino = selettore ok / diverso e FPD lontano = costa).
