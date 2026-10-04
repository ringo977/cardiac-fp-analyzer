# FPD contro il gold standard manuale (ottobre 2026)

Dataset GG: 179 registrazioni, 313 elettrodi con misura manuale (tempi di depolarizzazione e ripolarizzazione per battito). Split per esperimento bilanciato per qualità del segnale: **sviluppo** Exp 5/8/10, **test** Exp 6/7/9, quest'ultimo eseguito una sola volta per versione. Strumento: `tools/compare_gold.py`.

---

## Parte I — Caratterizzazione (1 ottobre, v3.5.0–3.5.1)

### 1. Metodi di misura a confronto (per battito, DEV, 1 755 coppie su 100 elettrodi)

| metodo software | errore mediano per battito | entro ±10 % | entro ±20 % |
|---|---|---|---|
| tangente (default fino a v3.5.x) | +1.1 % | 60 % | 74 % |
| picco | 0.0 % | 63 % | 74 % |
| 50 % | +4.2 % | 52 % | 69 % |
| ritorno a baseline | +8.4 % | 33 % | 53 % |

Conclusione di allora: "nessun bias di convenzione". **Era sbagliata**, vedi §4: il bias della tangente (+2.4 %) esiste ma era mascherato dagli errori di selezione dell'onda, che spostano la mediana in entrambe le direzioni.

### 2. Due classi di errore

Il 1° ottobre ho scritto "grado A (n = 74): 44 entro ±10 %, 32 anticipi, 14 ritardi". **Errato**: avevo sommato un conteggio su 99 elettrodi di tutti i gradi a quello del solo grado A (44 + 32 + 14 ≠ 74). I numeri corretti, una registrazione per voce gold e riferimento corretto: grado A (59) — 43 entro ±10 %, **3 anticipi, 13 ritardi** (mediana +20 %); tutti i gradi (113) — 61 entro, 32 anticipi (rapporto ~0.4, quasi tutti nei gradi B e C), 20 ritardi. Due problemi diversi: ritardi sul segnale buono, anticipi su quello rumoroso.

Prove dell'epoca, rimaste valide: punteggio prominenza × larghezza **peggiora** (scartato); `search_end_pct_rr` 0.70 → 0.90 nessun effetto; tetto `max_adaptive_min_fpd_ms` 350 → 600 ms adottato in v3.5.1 (anticipi per battito 12.5 → 9.1 %).

### 3. Il gold standard aveva 5 blocchi impossibili

In 5 blocchi su 199 oltre metà dei battiti aveva FPD > RR. L'analista ha confermato: in quattro mancava una ripolarizzazione (riga saltata, tutte le successive sfasate di un battito), nel quinto (Exp9 G ch2 baseline) ha rifatto la ricerca dei picchi. Nessun altro valore è cambiato. Con il riferimento corretto e lo stesso codice v3.5.1: DEV FPD entro ±10 % 43 → 45 %; TEST entro ±20 % 67 → 69 %. Tutti i numeri che seguono usano il riferimento corretto.

---

## Parte II — Selezione dell'onda di ripolarizzazione (2 ottobre, v3.6.0)

Metodo: per ogni elettrodo DEV ho catturato il template esattamente come lo costruisce la pipeline ed elencato tutti i candidati che la ricerca vede (verificato: riproduco la scelta della pipeline in 121 casi su 121), etichettando quello che coincide con il valore dell'analista.

### 4. Cosa ha mostrato

- **Tetto raggiungibile**: tra i candidati già visti c'è l'onda dell'analista nel **78 %** degli elettrodi, contro il 53 % scelto. Il resto è recuperabile solo scegliendo meglio.
- **La ripolarizzazione è spesso bifasica** (lobo positivo e negativo a 60–200 ms di distanza) e l'analista segna il **lobo positivo** in 76 elettrodi su 87 — sia quando viene prima sia quando viene dopo — mentre il software prendeva il più prominente, spesso il negativo.
- **L'analista segna il picco**, non la fine dell'onda: sugli elettrodi con l'onda giusta, picco +0.1 %, tangente +2.4 %.
- **Finestra troncata**: in 23 dei 25 elettrodi senza alcun candidato vicino al valore dell'analista, l'onda vera stava oltre la fine della finestra di ricerca; in 22 perché l'RR usato per la finestra (treno completo, battiti spurii inclusi) era circa metà del vero. Una guardia in `analyze.py` ri-segmentava già i battiti con l'RR vero per rendere l'onda raggiungibile, ma la finestra usava l'altro RR — il template era abbastanza lungo, la finestra no.
- **Difetto nell'allineamento dei battiti** prima della mediana (`_align_beats_xcorr`, presente da sempre): uno sfasamento di indice faceva sì che battiti già allineati venissero spostati di 50 ms e il jitter fosse amplificato invece che rimosso. Ogni template era 50 ms in ritardo rispetto ai suoi battiti. Conseguenza a cascata: il test d'inversione di polarità per battito leggeva la "forma dello spike" del template dove c'era solo linea di base, e in ~40 % degli elettrodi dichiarava invertiti la maggior parte dei battiti (in 23 tutti), togliendo loro la guida del template.

### 5. Cosa è cambiato

1. Allineamento corretto (correlazione normalizzata, zero di lag al posto giusto).
2. Inversione per battito decisa per anti-correlazione con lo spike del template (≤ −0.5), non confrontando le deflessioni.
3. Regola `prefer_positive`: lobo positivo se prominenza ≥ 0.5× il massimo ed entro 400 ms da esso, altrimenti il massimo. Parametri scelti con *leave-one-experiment-out* dentro il DEV: in ogni piega gli stessi valori, accuratezza sull'esperimento escluso 67.9 % contro 68.8 % interna. È una convenzione di polarità, non un prior sulla latenza: non spinge l'FPD verso alcun valore.
4. Finestra del template con l'RR dei battiti che lo formano, quando è più lungo; l'estensione è controllata da una guardia di forma che la ferma prima di un evento simile allo spike (battito successivo), così la nuova finestra contiene sempre la vecchia.
5. Punto finale `peak` di default (era `tangent`).

Provato e non adottato: polarità per battito coerente con il template (−1 elettrodo; resta come opzione `per_beat_prefer_template_sign`).

### 6. Risultati (pipeline completa, FPD entro ±10 % tra gli elettrodi riportati)

Ablazione sul DEV:

| configurazione | ±10 % | ±20 % | grado A | errore mediano |
|---|---|---|---|---|
| v3.5.1 | 45 % | 58 % | 61 % | +0.5 % |
| + allineamento e inversione | 46 % | 57 % | 59 % | +1.2 % |
| + lobo positivo | 52 % | 61 % | 66 % | +1.5 % |
| + finestra con RR dei battiti del template | 55 % | 67 % | 66 % | +2.5 % |
| + punto finale al picco (= v3.6.0) | **61 %** | **67 %** | **74 %** | **0.0 %** |
| solo punto finale al picco, senza le altre | 46 % | 56 % | 62 % | −0.2 % |

Test tenuto da parte (Exp 6/7/9, una corsa):

| | v3.5.1 | v3.6.0 |
|---|---|---|
| FPD entro ±10 % | 48 % | **69 %** |
| FPD entro ±20 % | 69 % | **80 %** |
| errore mediano | −1.3 % | −0.1 % |
| grado A / B / C | 69 / 41 / 20 % | **84 / 59 / 50 %** |
| BP entro ±10 % | 69 % | 69 % (invariato: nessuna modifica alla detection) |

Corpus del paper (Visone 2023, 8 registrazioni): errore medio 3.4 % → 3.2 %, spostamenti di pochi ms. I cambiamenti agiscono dove il gold standard li ha motivati (onde bifasiche, finestre troncate) e non toccano i segnali puliti e monofasici.

### 7. Cosa resta

- Template **senza onda T visibile** (rumore dominante, gradi B/C): l'analista la vede sui singoli battiti, la mediana no. Serve un approccio per battito con vincolo di coerenza tra battiti, non una regola sul template.
- Elettrodi con **RR sbagliato** per sovra-rilevazione residua: sistemata la finestra, resta sbagliato il BP. Problema di detection.
- La convenzione del lobo positivo vale nell'87 % dei casi: negli altri l'analista segna il negativo, e la regola lo sbaglia.


## Parte III — Finestra di ricerca fino all'85 % del ciclo (4 ottobre, v3.15.0)

La finestra finiva al 70 % del RR; 5 elettrodi su 271 (4 DEV, 1 TEST) hanno l'FPD dell'analista oltre quel limite (fino al 78 % del ciclo) e il software li sbagliava tutti. Ora la finestra arriva all'85 % del RR su un ritmo regolare e si ferma 60 ms prima del 10° percentile degli intervalli su uno irregolare (`tools/compare_gold.py`, tag `new_w85` contro `v3.14.1_w70`).

| | DEV (189 analizzabili, 145 con FPD) | TEST (83, 63) — una esecuzione |
|---|---|---|
| FPD entro ±10 % (riportati) | 95 → 95 | 46 → 46 |
| FPD entro ±20 % | 105 → 106 | 52 → 52 |
| errore mediano | 4,1 → 2,9 % | 2,6 → 2,7 % |
| elettrodi con BP corretto: entro ±5 % / ±10 % | 59 → 61 / 71 → 72 su 90 | 32 → 33 / 38 → 38 su 44 |
| elettrodi con BP corretto: 90° percentile dell'errore | 35 → 30 % | 10,6 → 10,2 % |

Dei 4 FPD DEV oltre il 70 %: `Exp10_ChipE_ch2_TI12_…_A el1` da −49 % a +1,6 %; `Exp5_chipF_ch3_Ti02_…_C el1` da −61 % a +23 % e `Exp8_ChipE_…_C_post_rec el2` da −76 % a −64 % (periodo sbagliato in entrambi); `Exp8_ChipE_…_C el2` invariato (nessun FPD). Gli altri elettrodi che cambiano (70 DEV, 25 TEST) hanno quasi tutti il periodo sbagliato per sovra-rilevazione: lì la finestra, vecchia o nuova, misura un template senza senso. Due regressioni con periodo corretto: `Exp5_chipE_ch2_chipA_ch2_baseline el1` (1522 → 1060 ms, analista 1521: un plateau positivo e una deflessione negativa di prominenza quasi uguale, la regola `prefer_positive` cambia scelta con la pendenza del detrend) e `Exp7_ChipC_ch1_Ti01_…_C el1` (TEST, 18 → 38 %). È la fragilità già nota del §7, non della finestra.
