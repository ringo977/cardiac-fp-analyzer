# Confronto farmaco per farmaco contro Visone et al. 2023

**Metodo:** pipeline attuale (dopo tutti i fix di oggi), config di default, dataset `µECG-Pharma Calibration - EXCEL DATA`.
**Riferimento:** direzioni pubblicate riportate in `comparison_with_paper.md` §3.

---

## 1. Tabella di sintesi

| Farmaco | Paper | Nostro | Direzione | Dose-risposta |
|---|---|---|:--:|:--:|
| **Chinidina** | ↑ prolungamento (+30 % dose alta) | +24 % → **+94 %** | ✅ | ✅ monotona |
| **Terfenadina** | ↑ prolungamento (p<0.05 a 0.3 µM) | +18 % → +60 % | ✅ | ✅ crescente |
| **Ranolazina** | ↑ prolungamento (p<0.01) | +4, +4, +77, +38, **−36**, +8 % | ~ | ❌ erratica |
| **Alfuzosina** | lieve ↓, poi ↑ a 100 nM | +59, −22, −27, +50, +54 % | ~ | ❌ erratica |
| **Dofetilide** | ↑ dose-dipendente, arresto a 10 nM | **da −48 % a +155 %** | ❌ | ❌ **incoerente** |
| **Nifedipina** | ↓ accorciamento dose-dipendente | *assente* (esclusa dal gate CV) | — | — |
| **Mexiletina** | → nessun effetto | *assente* (esclusa dal gate CV) | — | — |

Concordanza sulla direzione: **2 chiare su 5 valutabili.** Il documento del marzo scorso ne riportava 5 su 7, ma su un criterio qualitativo ("trend rilevato") anziché sul segno del `%ΔFPDcF` calcolato.

## 2. Il problema: la dofetilide non è coerente **dentro lo stesso chip**

La dofetilide è il bloccante hERG più pulito e riproducibile del pannello CiPA. Deve dare un prolungamento monotono da manuale. Invece:

```
EXP 7/ChipE       0.3 nM  −27.7 %      EXP 8/day7      0.3 nM  +10.3 %  +32.5 %
                  1.0 nM  −43.8 %                      1.0 nM  +42.7 %  +45.4 %
                  2.0 nM  −47.6 %                      2.0 nM  −31.8 %  +48.7 %
                  3.0 nM  +83.5 %                      3.0 nM  −28.5 %  +39.2 %
                  6.0 nM +155.2 %                      6.0 nM  −42.5 %  +33.6 %
                 10.0 nM  +48.6 %                     10.0 nM  −38.6 %  −38.5 %
```

Non è variabilità fra preparazioni: **il segno si ribalta lungo la scala di dosi dello stesso chip**, e in `day7` due repliche della stessa concentrazione danno −31.8 % e +48.7 %.

Nessuna interpretazione biologica regge. Un effetto reale può variare in ampiezza fra microtessuti, non invertirsi fra 2 e 3 nM per poi reinvertirsi a 10 nM.

**Questo è un difetto di misura, e la dose-risposta non lo può nascondere.** Vale per tutti e quattro i chip testati.

## 3. Perché conta più dell'accordo sulle medie

`VALIDAZIONE_vs_paper_Visone2023.md` mostra che le **medie di popolazione delle baseline** ora coincidono col paper entro il 6 % sull'FPDcF. È un risultato reale, ma dice solo che il valore *tipico* è giusto.

Questo confronto dice che i **valori individuali** non lo sono. Una misura può avere la media giusta e una dispersione tale da rendere inutile ogni singola osservazione — ed è esattamente ciò che si vede: ±100 punti percentuali sulla stessa concentrazione.

Le due cose non sono in contraddizione: la prima valida la *calibrazione*, la seconda invalida la *precisione*.

## 4. Ipotesi, in ordine di verificabilità

**a) Appaiamento baseline sbagliato.** `normalization.py` accoppia farmaco e baseline per gruppo esperimento/chip/camera/**elettrodo**, con un fallback che accoppia *fra elettrodi diversi* quando il selettore automatico ha scelto canali diversi per baseline e farmaco. `DOCUMENTATION.md:674` ammette che questo *"può introdurre un bias sistematico sull'FPDcF"*. Se una registrazione a 2 nM è normalizzata contro la baseline dell'elettrodo sbagliato, il segno può ribaltarsi. **Verificabile subito:** confrontare `norm.baseline_file` e il canale analizzato per ogni punto della curva.

**b) Instabilità della selezione automatica del canale.** Già osservata oggi: `Dofe_3nM` finiva in un gruppo diverso dai suoi fratelli, e `chipD_ch3baseline` è su el2 mentre il gemello è su el1. Se il canale cambia lungo la scala di dosi, si stanno confrontando elettrodi diversi. Il paper seleziona il canale **manualmente**.

**c) Sovra-rilevazione dei battiti.** BP 1461 ms contro 1900 del paper e CV 25 % contro 12.9 % — su quelle che dovrebbero essere le stesse preparazioni — indicano circa il 30 % di battiti in più. Entra nell'RR e quindi nell'FPDc.

**d) Dispersione residua dell'FPD per-battito.** Il paper misura l'FPD su un template mediato di ~90 battiti; noi per-battito. Su segnale rumoroso la nostra misura è più dispersa per costruzione.

La (a) e la (b) spiegherebbero i **ribaltamenti di segno**; la (c) e la (d) spiegherebbero l'ampiezza eccessiva. Probabilmente agiscono insieme.

## 5. Conseguenza sul lavoro di oggi

La ricalibrazione del criterio di inclusione (`SPRINT3_ricalibrazione_CV.md`) resta valida nel merito — il CV è dominato dal criterio di precisione — ma **va sospesa la decisione sul default**. Non ha senso ottimizzare quali registrazioni includere finché il `%ΔFPDcF` di una singola registrazione può cambiare segno per ragioni non biologiche.

Ordine che propongo:

1. **Verificare l'ipotesi (a)** — appaiamento baseline lungo le curve dofetilide. È un controllo sui dati già prodotti, non richiede nuove analisi.
2. **Verificare (b)** — quale canale è stato scelto per ogni punto delle stesse curve.
3. Solo dopo, riprendere la questione delle soglie.

Se (a) o (b) spiegano i ribaltamenti, il confronto col paper diventa il banco di prova naturale per il fix — come lo è stato oggi per l'FPD.

## 6. Nota su nifedipina e mexiletina

Sono i due farmaci che permetterebbero di testare la **specificità** — il paper riporta per la nifedipina un accorciamento e per la mexiletina nessun effetto. Entrambi assenti dalla classificazione perché le loro baseline cadono nel gate CV.

Con il criterio di precisione al posto del CV rientrerebbero (misurato: +7 farmaci, fra cui `nife` e `mexil`). È un'ulteriore ragione per fare quel cambio — ma dopo aver risolto i punti 1 e 2, non prima.
