# Appunti di Roberta — microtessuti usati per la validazione farmacologica

**Fonte:** scansione di appunti manoscritti, condivisa via email il 2026-08-05 insieme al link NAS dei Raw_Data.
**Stato:** trascrizione da immagine. ⚠️ **Alcune letture sono incerte** e sono marcate `[?]`. Da verificare contro gli Excel per esperimento prima di usarle come ground truth.

**Cosa contiene:** per ogni farmaco, i microtessuti (esperimento / chip / camera) usati, il **CV di baseline**, e le note sulle aritmie osservate da Roberta.

---

## Mexiletine (I_Na) — *no arrhythmias*

| Esperimento | Chip | Camera | CV |
|---|---|---|---:|
| EXP 7 | E | ch3 | 10 % |
| EXP 7 | A | ch3 | **33.28 %** |
| EXP 7 | B | ch1 | **32 %** |
| EXP 7 | C | ch3 | **39.1 %** |
| EXP 8 | A | ch2 | 8 % |

## Nifedipine (I_CaL)

| Esperimento | Chip | Camera | CV | note |
|---|---|---|---:|---|
| EXP 7 | E | ch2 | 10.6 % | "non venuto (eccitatore, medio?)" `[?]` |
| EXP 7 | A | ch1 | 20 % | |
| EXP 7 | B | ch3 | **33 %** | |
| EXP 8 | E | ch3 | 10 % | |
| EXP 9 | C | ch2 | 13 % | |

## Dofetilide (I_Kr) — 0.03 µM

| Esperimento | Chip | Camera | CV | aritmie | note |
|---|---|---|---:|---|---|
| EXP 8 | D | ch1 | **39.5 %** | ARR 1, ARR 6 ✗ | **no repol 6/10** |
| EXP 8 | A | ch1 | 17.20 % | ARR 0.1–10 ✗ | |
| EXP 8 | E | ch2 | 3.5 % | ARR 10? ✗ | difficult repolar. |
| EXP 9 | C | ch1 | 21 % | no ARRH | **stop beating** |

## Cisapride (I_Kr, I_to, I_Na) — *marcato con ✗ a margine*

| Esperimento | Chip | Camera | CV | aritmie | note |
|---|---|---|---:|---|---|
| EXP 5 | C | ch1 | 5.62 % | arrhythm 0.001 ✗ | 3 stop beat |
| EXP 6 | C | ch2 | 3.9 % | EAD @ 0.0025–0.005 ✗ | noise |
| EXP 6 | D | ch3 | 2.17 % | ARRH 0.003 | |

## Alfuzosine (I_Na?) `[?]`

| Esperimento | Chip | Camera | CV |
|---|---|---|---:|
| EXP 7 | C | ch1 | 4.2 % |
| EXP 7 | D | ch2 | 10.5 % |
| EXP 8 | D | ch2 | 6.6 % |
| EXP 8 | B | ch3 | 4.8 `[?]` |

## Quinidine (I_Kr, I_CaL, I_Na) — 10 µM

| Esperimento | Chip | Camera | CV | aritmie | note |
|---|---|---|---:|---|---|
| EXP 4 | 1 | ch1 | 9.5 % | | |
| EXP 4 | A | ch3 | **37.7 %** | | **stop beating** |
| EXP 5 | A | ch2 | 8.55 % | ARRHYT 0.1 ✗ | |
| EXP 5 | B | ch3 | 6.4 % | ARRHYTM 30? ✗ | |
| EXP 6 | B | ch3 | 12 % | no rep. 30 | |

## Ranolazine (I_Kr, I_Na) `[?]`

| Esperimento | Chip | Camera | CV | note |
|---|---|---|---:|---|
| EXP 4 | 1 | ch1 | — | |
| EXP 4 | 2 | ch1 | 9.8 % | "non venuto, ma vc rosso" `[?]` — 0.1 / 2 / 50 → ARRH \ EAD \ EAD |
| EXP 5 | A | ch3 | 5.3 % | |
| EXP 5 | B | ch1 | 8.2 % | |
| EXP 6 | B | ch2 | 19 % | |

## Terfenadine (I_Kr, I_CaL)

| Esperimento | Chip | Camera | CV | aritmie |
|---|---|---|---:|---|
| EXP 4 | 2 | ch2 | 7.3 % | 100 / 300 → DAD \ DAD |
| EXP 4 | 3 | ch1 | 14 % | EAD \ EAD \ EAD |
| EXP 5 | A | ch1 | 5.3 % | 0.04 / 0.1 / 0.3 → ARRHYTM |
| EXP 5 | B | ch2 | 6.65 % | |
| EXP 6 | E | ch1 | 8.18 % | |

---

## Osservazione che cambia una nostra assunzione

**Sei microtessuti hanno CV di baseline oltre il 25 %** e risultano comunque usati nella validazione:

```
Mexiletine   EXP7 chipC ch3   39.1 %
Dofetilide   EXP8 chipD ch1   39.5 %
Quinidine    EXP4 chipA  ch3   37.7 %
Mexiletine   EXP7 chipA ch3   33.28 %
Nifedipine   EXP7 chipB ch3   33 %
Mexiletine   EXP7 chipB ch1   32 %
```

Il criterio di inclusione che il software eredita è `max_cv_bp = 25.0`, descritto in `config.py` come *"paper standard"*, e `comparison_with_paper.md` riporta *"Inclusione: CV baseline BP < 25%"*.

**Le due cose non tornano.** Da chiarire con Roberta, e la domanda è precisa:

> Il `CV < 25 %` era un criterio di **esclusione** applicato a tutti i microtessuti, oppure una **statistica riportata** (con esclusioni decise invece per ispezione visiva)?

Se vale la seconda, la soglia al 25 % nel nostro software non ha la giustificazione che le attribuivamo, e l'analisi in `SPRINT3_ricalibrazione_CV.md` va riletta con questa informazione.

## Riscontro sulla diagnosi aperta

`EXP 8 / chip D / ch1` (dofetilide, CV 39.5 %) porta l'annotazione **"no repol 6/10"** — su 10 dosi, in 6 la ripolarizzazione non era leggibile.

È lo stesso gruppo che il nostro software escludeva e che abbiamo recuperato con il fix dell'RR. La nota di Roberta concorda con la diagnosi in `project_fpd_afterpotential_slow_rr`: **dove la ripolarizzazione non è leggibile la pipeline non deve produrre un FPD**, e qui c'è la conferma sperimentale indipendente che in quei punti non era leggibile davvero.

## Prossimi passi

1. Verificare questa trascrizione contro gli Excel per esperimento (che contengono i parametri misurati e il lato dx/sx usato).
2. Chiarire con Roberta la domanda sul CV.
3. Usare la colonna dx/sx come canale forzato, al posto dell'auto-selezione, e ri-verificare le dose-risposta incoerenti.
