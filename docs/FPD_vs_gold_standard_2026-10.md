# FPD contro il gold standard manuale — caratterizzazione dell'errore (ottobre 2026)

Dataset GG, split di sviluppo (Exp 5/8/10), pipeline v3.5.0. Confronto **battito per battito**: ogni battito del software accoppiato (±50 ms, dopo stima dell'offset) al battito dell'analista, FPD software vs `Rep. time − Dep. time` dell'analista. 1 755 coppie su 100 elettrodi analizzabili; una sola registrazione per voce gold (quella meglio allineata).

## 1. La definizione non è il problema

| metodo software | errore mediano per battito | entro ±10 % | entro ±20 % | scelte precoci (< −30 %) | tardive (> +30 %) |
|---|---|---|---|---|---|
| tangente (default) | +1.1 % | 60 % | 74 % | 15.6 % | 4.6 % |
| picco | 0.0 % | 63 % | 74 % | — | — |
| 50 % | +4.2 % | 52 % | 69 % | | |
| ritorno a baseline | +8.4 % | 33 % | 53 % | | |

Tangente e picco coincidono in mediana con l'analista: **non c'è un bias di convenzione**. Il problema sono le code: correlazione per battito 0.38, un battito su cinque fuori del ±20 %.

## 2. Due classi di errore: onde diverse, non punti diversi

A livello di elettrodo (grado A, n = 74): 44 entro ±10 %, **32 anticipi** (software ≈ 0.35× l'analista), **14 ritardi** (software 1.2–2.4×). Cambiando metodo non si recuperano: sui ritardi il metodo del picco resta a +14 %; sugli anticipi tutti i metodi danno −70 %. Un oracolo che scegliesse il metodo migliore per elettrodo arriverebbe al 61 %. Quindi software e analista stanno guardando **candidati diversi** nel tracciato: la questione è la *selezione del picco*, non il punto di misura sul picco scelto.

Anticipi: ritmi lenti (RR 2–4 s) con T larga a ~0.5×RR; il software sceglie un after-potential stretto a ~500 ms perché il criterio di scelta è la *prominenza*, che premia i picchi stretti. Provato il punteggio prominenza × larghezza: **peggiora** (anticipi 24 %) — le derive lente del tracciato hanno molta larghezza. Risultato negativo, non adottato.

Il tetto assoluto del minimo FPD adattivo (`max_adaptive_min_fpd_ms`, 350 ms) lasciava passare quei 500 ms a RR 3 s. Portato a 600 ms: anticipi 12.5 → 9.1 %, entro ±20 % per battito 76 → 80 %; a livello di elettrodo +2 punti su DEV, neutro su TEST. Entrambi i riferimenti reali (paper e analista, 268 blocchi) mostrano FPD/RR ≥ 0.3 praticamente sempre, quindi il floor `min(0.2×RR, 600 ms)` non esclude fisiologia osservata.

Allargare la finestra di ricerca (`search_end_pct_rr` 0.70 → 0.90): nessun effetto (la T vera era già dentro).

## 3. Il gold standard ha 5 blocchi impossibili

In 5 blocchi su 199 (Exp5 ×3, Exp7, Exp9) oltre metà dei battiti ha **FPD > RR**, impossibile per un singolo battito; `FPD − RR` dà valori plausibili (≈ 450–1 700 ms). È la firma di un `Rep. time` accoppiato al `Dep. time` del battito **precedente** (sfasamento di una riga nel foglio). Chiavi: (Exp9,G,2,baseline), (Exp7,E,3,dose 1), (Exp5,E,2,dose 3), (Exp5,D,3,baseline), (Exp5,C,3,dose 3). Su questi il software è probabilmente giusto e il riferimento no; vanno verificati dall'analista.

### Esito (2 ottobre)

L'analista ha ricontrollato: nei primi quattro blocchi mancava una ripolarizzazione (riga saltata → tutte le successive sfasate di un battito), aggiunta; nel quinto (Exp9 G ch2 baseline) ha rifatto la ricerca manuale dei picchi. Nessun altro valore è cambiato (verificato blocco per blocco). Con il riferimento corretto, stesso codice v3.5.1: DEV FPD entro ±10 % 43 → 45 %, entro ±20 % 56 → 58 %, grado A 59 → 61 %; TEST entro ±20 % 67 → 69 %. Sui cinque blocchi il software ora concorda in tre (−1 %, +5 %, +17 %); nei due restanti (grado B e D, detection povera) il torto è del software e resta nel conto.

## 4. Cosa farebbe la differenza

Un selettore del candidato che usi informazione **tra battiti** e **sul ritmo**: il candidato giusto è quello la cui latenza è stabile da battito a battito e cade in una frazione plausibile dell'RR (0.3–0.7 in entrambi i riferimenti); prominenza e larghezza da sole non bastano. Va costruito e tarato su DEV, verificato su TEST — con il rischio esplicito di cucire un prior sulla fisiologia che il farmaco può spostare (l'FPD prolungato *è* la misura).
