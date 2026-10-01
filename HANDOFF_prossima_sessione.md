# Passaggio di consegne — stato al 6 agosto 2026

Documento di ripresa. La memoria (`.auto-memory/`) contiene i dettagli; qui c'è **cosa fare e in che ordine**, con lo stretto necessario per capire perché.

---

## 1. Da dove ripartire — in una riga

**La misura dell'FPD è corretta all'1 % su file appaiato. Il problema è a monte: rileviamo circa il doppio dei battiti.**

Prossima azione concreta: **guardare il segnale** di `Experiments-Toxicol.Science/Exp8/Day6/chipD_ch1_baseline.csv` (canale `el1`) con sopra i battiti marcati dal nostro detector. 156 battiti dove il paper ne conta ~75. Accanto al CSV c'è il `.bmp` di riferimento degli autori.

---

## 2. Il reperto principale (step 0, 6 agosto)

File appaiato, stesso elettrodo, valore pubblicato contro il nostro:

| | pubblicato | nostro (el1) | Δ |
|---|---:|---:|---:|
| **FPD** | 609.1 ms | **615.0 ms** | **+1.0 %** |
| **RR** | 2115.6 ms | 1148.2 ms | **−45.7 %** |

Ha anche confermato la mappatura: **`el1` = lato `sx`, `el2` = lato `dx`** (colonna 2 e 3 del CSV).

**Decomposizione: misura FPD ottima, beat detection raddoppiata.** È l'opposto di dove stavo cercando il 5 agosto.

Questo spiega da solo le discrepanze di popolazione contro il paper (BP −29 %, CV +75 %) **e rende superflua** l'ipotesi che avevo costruito — che il paper pre-filtri visivamente i microtessuti. Quella era una spiegazione elegante per un fenomeno con causa meccanica.

### Ipotesi testata e SMENTITA

Avevo proposto che il detector contasse l'onda T come battito (darebbe esattamente 2×). **Il test di alternanza la smentisce:**

```
intervalli:  732  1317  1009  836  1622  1392  1177  973 …
posizioni pari 1170 ms   posizioni dispari 1126 ms   →  nessuna alternanza
intervalli corti: media 825 ms  (non 609 come sarebbe l'FPD)
```

Il 2× resta però reale: **la somma delle coppie di nostri intervalli fa 2293 ms contro i 2116 pubblicati** (+8 %). A coppie ricostruiamo il loro ritmo.

Quindi: sovra-rilevazione ~2× confermata, **meccanismo non identificato**. Candidati: le quattro passate di gap-filling in `beat_detection.py` che *inseriscono* battiti in posizioni ritmicamente attese, oppure una detection che aggancia feature secondarie in posizioni variabili.

---

## 3. Ordine di lavoro proposto

### A. Capire la sovra-rilevazione ⟵ **inizia da qui**
Plot del segnale con i battiti marcati, confronto col `.bmp`. Trenta secondi di occhio valgono più di ore di statistica indiretta. Poi disattivare selettivamente le passate di gap-filling e vedere quale produce il raddoppio.

### B. Fix del fallback argmax
Indipendente da A, e **ha già un criterio di successo oggettivo**: negli Excel degli autori, su 910 punti-concentrazione solo il **59 % ha un QT misurato** — hanno lasciato in bianco il 41 % perché non era misurabile. Dopo il fix, la nostra frazione di "non misurabile" deve avvicinarsi al 40 %, non restare vicina a zero.

Il fix: in `repolarization.py`, se `find_peaks` non trova un picco qualificante nella finestra, restituire `None`/NaN invece del `argmax(|seg_det|)`. ⚠️ Leggere prima `tests/test_bradycardia_fpd_robustness.py` e `tests/test_repolarization_bounds.py`: potrebbero dipendere dal fallback.

### C. Step 1 e 2 (piano di Marco)
1. Confronto col ground truth **forzando il canale usato dal paper** → isola l'errore di misura.
2. Rifare con **selezione automatica** → la differenza è attribuibile solo alla scelta del canale.

Per lo step 2, la metrica giusta non è "quanto spesso l'auto concorda con Roberta" ma: **quando non concorda, il risultato è peggiore?** La scelta umana non è la verità, è una scelta ragionevole fra due segnali. Tre esiti: concorda (nessuna informazione) / non concorda ma FPD vicino (**il selettore funziona**) / non concorda e FPD lontano (**qui costa**, casi da studiare).

### D. Solo dopo: soglie di inclusione
Tutta la calibrazione di `SPRINT3_ricalibrazione_CV.md` è costruita su un sottoinsieme di 206 CSV su 713 — probabilmente arricchito di casi difficili — e su misure poi rivelatesi distorte. **Non riprendere le soglie finché A e B non sono chiusi.**

---

## 4. Materiale disponibile

**Dataset completo** `Experiments-Toxicol.Science/` (23 GB): 713 CSV, 35 Excel, 584 BMP. Struttura `Exp<n>/<Day>/<Chip>/`. Ogni CSV ha 2 segnali (col 2 = sx = el1, col 3 = dx = el2).

**Ground truth estratto** `data_reference/ground_truth.json` — 67 microtessuti, per concentrazione: `RR interval (s)`, `QT interval (s)` (= FPD), `QT Fridericia (s)` (= FPDc), `Spike Amplitude (µV)`, note (`Arrh?`, `Difficile riconoscere`).
⚠️ Due difetti noti dell'estrattore, documentati in memoria: legge sempre il blocco `sx` (va corretto usando il lato dedotto), e include righe di riepilogo `mean`/`st dev` fra i punti (conteggi assoluti gonfiati ~2×, rapporti validi).

**Lato usato per microtessuto**: `/tmp/sides.json` (da rigenerare, script in memoria). Regola: **il lato usato è quello il cui blocco `RR array` contiene numeri.** Esito: 31 sx, 26 dx, 2 entrambi (DMSO), 8 nessuno.

**Appunti di Roberta** `data_reference/appunti_Roberta_microtessuti.md` — microtessuti per farmaco, CV di baseline, aritmie osservate. Trascrizione da foto, letture incerte marcate `[?]`.

**Strumenti** `tools/`: `reanalyze.py` (run + manifest di provenienza + `--diff`), `compare_pipeline_versions.py` (due versioni sugli stessi dati), `extract_reference_excel.py`.

---

## 5. Due domande aperte per Roberta

1. Il **`CV < 25 %`** era un criterio di *esclusione* applicato a tutti i microtessuti, o una *statistica riportata* con esclusioni decise per ispezione visiva? Nei suoi appunti compaiono microtessuti usati con CV fino al **39.5 %**, e negli Excel c'è un blocco "QC blinova" che indica **`Coeff RR <5%`** — nessuno dei due coincide col 25 % che il software eredita come "paper standard".
2. Nei fogli, i **blocchi grezzi `sx`/`dx` hanno a volte concentrazioni diverse fra loro** (es. dofetilide: sx 0/0.3/1/2/3/6/10, dx 0/0.1/1/2.5). Residui di template o dati di un altro microtessuto?

---

## 6. Due pattern ricorrenti — da cercare attivamente

Emersi da quattro bug in due giorni, non sono aneddoti.

**Meccanismo giusto, collegato in un posto solo.** Il segnale è un commento che descrive correttamente la soluzione mentre una chiamata lì accanto non la usa. `analyze.py:354` era esattamente così: *"Beat period from ALL detected beats — avoids artificial gaps from QC rejection"*, applicato a un solo consumatore su due.

**Finestre temporali assolute su fenomeni che scalano con la frequenza.** Il filtro `fpdc_physiol` a [350-800] ms e il tetto `max_adaptive_min_fpd_ms = 350`. Le finestre assolute sbagliano agli estremi di frequenza; i rapporti no.

E una regola di metodo, pagata tre volte: **prima di tarare una soglia, verificare sia che la misura sia corretta, sia che il campione sia rappresentativo.** Ho proposto tre cambi di default su basi che si sono poi rivelate viziate — dal bug dell'RR il primo, dal campione selezionato gli altri.

---

## 7. Stato del repo

Branch `feat/pyside-poc`, **19 commit** il 5 agosto, **714 test** (erano 628). Tutto committato tranne il materiale di oggi (`data_reference/`, `tools/extract_reference_excel.py`, questo documento).

Il commit che cambia i numeri storici è **`aa12d3f`** (polarità ripolarizzazione) e **`927568f`** (RR locale). Quest'ultimo rende **necessaria** la ri-analisi dei dataset storici: il `%ΔFPDcF` è un rapporto fra due FPDc con frazioni di scarto QC diverse, quindi l'errore non si cancella nella normalizzazione.

Risultato più solido finora: **FPDcF da −55 % a −6 %** contro i valori pubblicati, con fix derivati dal codice e confronto col paper arrivato dopo.
