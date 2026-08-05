# Sprint 0 — Ri-analisi dei dataset storici

**Data:** 5 agosto 2026
**Oggetto:** impatto dei tre fix scientifici (in particolare il bug di segno nella ripolarizzazione) sui risultati già prodotti
**Dataset:** `µECG-Pharma Calibration - EXCEL DATA/` — 206 registrazioni, 10 sottoinsiemi
**Metodo:** pipeline eseguita due volte (codice pre-fix da git HEAD vs codice post-fix), config di default, confronto per-file + per-farmaco

---

## 1. Risposta breve

**Nessuna conclusione scientifica cambia.** Su 206 registrazioni:

| Metrica | Valore |
|---|---|
| File confrontati | 206 |
| Mediana \|ΔFPDc\| | **0.0 %** |
| File con \|ΔFPDc\| > 5 % | **8** (3.9 %) |
| Massimo \|ΔFPDc\| | **11.9 %** |
| Registrazioni che attraversano la soglia 15 % | **0** |
| **Classificazioni di farmaco cambiate** | **0** |
| Cambiamenti in beat detection / QC / inclusione | **0** |

Il dettaglio per dataset:

| Dataset | File | mediana \|Δ\| | max \|Δ\| | >5 % | call cambiate |
|---|---:|---:|---:|---:|---:|
| EXP 5 | 21 | 0.0 % | 6.1 % | 1 | 0 |
| EXP 7 / ChipC | 22 | 0.0 % | 1.0 % | 0 | 0 |
| EXP 7 / ChipE | 21 | 0.0 % | 1.8 % | 0 | 0 |
| EXP 7 / chipA | 9 | 0.0 % | 3.1 % | 0 | 0 |
| EXP 8 / day6 | 24 | 1.3 % | 6.1 % | 2 | 0 |
| EXP 8 / day7 | 45 | 0.4 % | 11.9 % | 3 | 0 |
| EXP 9 / day7 | 27 | 2.2 % | 6.6 % | 2 | 0 |
| Signal Extraction / Exp4 | 32 | 0.1 % | 3.1 % | 0 | 0 |
| Signal Extraction / EXP5 | 2 | 0.0 % | 0.0 % | 0 | 0 |
| Signal Extraction / EXP6 | 3 | 0.6 % | 1.2 % | 0 | 0 |

**In pratica:** i dati pubblicati o presentati finora non vanno ritrattati. Il fix è comunque corretto e va tenuto — ma non c'è un'emergenza di ri-analisi.

---

## 2. Dove si concentra il cambiamento

L'effetto è **sparso, non sistematico**: la maggioranza dei file non si muove affatto (mediana 0.0 %), e quando si muove lo fa in entrambe le direzioni. Questo è coerente con il meccanismo del bug — si attivava solo sui battiti la cui onda T è invertita rispetto al template, che è una proprietà della singola registrazione, non della qualità generale.

I 12 file con la variazione maggiore:

| Dataset | File | \|Δ\| | QC | % senza repol | conf |
|---|---|---:|:--:|---:|---:|
| EXP8_day7 | `chipE_ch3_NIFE_100nM` | 11.9 % | F | 31 % | 0.62 |
| EXP8_day7 | `chipA_ch3_CTR3` | 9.2 % | D | 30 % | 0.82 |
| EXP9_day7 | `chipD_ch1_ALFU10nM` | 6.6 % | C | 28 % | 0.74 |
| EXP8_day7 | `chipE_ch2_DOFE_10nM` | 6.6 % | D | 52 % | 0.65 |
| EXP9_day7 | `chipB_ch2_baseline` | 6.1 % | C | 68 % | 0.66 |
| EXP5 | `chipA_ch3_ranolazine_100uM` | 6.1 % | F | 53 % | 0.74 |
| EXP8_day6 | `chipD_ch1_Dofe_0.3nM` | 6.1 % | **A** | **0 %** | 0.84 |
| EXP8_day6 | `chipD_ch1_Dofe_0_3nM` | 6.1 % | **A** | **0 %** | 0.84 |
| EXP8_day7 | `chipE_ch2_DOFE_3nM` | 4.7 % | B | 35 % | 0.77 |
| EXP8_day6 | `chipD_ch1_Dofe_3nM` | 4.3 % | C | 27 % | 0.75 |
| EXP8_day7 | `chipA_ch2_MEXI_50uM` | 4.2 % | D | 30 % | 0.87 |
| EXP9_day7 | `chipB_ch2_MEXI_1nM` | 4.2 % | B | 49 % | 0.80 |

Media di \|Δ\| per grado QC:

| QC | n | media \|Δ\| | max \|Δ\| |
|:--:|---:|---:|---:|
| A | 42 | 0.41 % | 6.1 % |
| B | 58 | 0.98 % | 4.7 % |
| C | 28 | 1.52 % | 6.6 % |
| D | 48 | 1.17 % | 9.2 % |
| F | 30 | 1.12 % | 11.9 % |

**Il punto che smentisce l'intuizione facile:** c'è una tendenza (A = 0.41 % contro C = 1.52 %), ma **non è una regola**. Due registrazioni di grado **A** con ripolarizzazione rilevata sul **100 % dei battiti** e confidence 0.84 si sono spostate del 6.1 %. Non si può quindi dire "il fix tocca solo i segnali brutti, quindi i dati buoni sono salvi": la polarità invertita è indipendente dalla qualità complessiva del segnale.

Sul dataset già visto nella sessione precedente (`Exp5_chipE_ch1`), il fix aveva ridotto la SD del 26 % — segno che la misura diventa più stabile, non solo diversa.

---

## 3. Reperti collaterali — non causati dai fix, ma emersi dalla ri-analisi

Questi erano già nel comportamento pre-fix. Li segnalo perché pesano più del fix stesso.

### 3.1 🔴 L'intera dose-risposta della dofetilide in EXP 8/day6 è esclusa

Dofetilide è il bloccante hERG di riferimento CiPA — il controllo positivo canonico. In `EXP 8/day6` **nessuna delle sue 7 concentrazioni raggiunge la classificazione**. Diagnosi:

```
chipD_ch1_baseline    passed=False   FPDc=481.6  conf=0.82   reason=Baseline CV=31.8% >= 25.0%
chipD_ch1_Dofe_0.3nM  passed=False   ...         reason=Baseline of group EXP 8/chipD_ch1/el1 failed
chipD_ch1_Dofe_1nM    passed=False   ...         reason=Baseline of group EXP 8/chipD_ch1/el1 failed
   ... (idem per 2nM, 6nM, 10nM, "nM")
```

La baseline è sana per FPDc (481.6 ms, in range) e per confidence (0.82). Fallisce **solo** sul CV del beat period: 31.8 % contro una soglia di 25 %. Quel singolo fallimento rimuove l'intero gruppo `chipD_ch1/el1`, e con esso tutta la curva dose-risposta.

Questo è esattamente il rischio di cascata segnalato al §3.7 dell'assessment. L'unico farmaco che arriva alla classificazione in day6 è `alfus`.

**Da decidere:** la soglia CV a 25 % su una baseline spontanea di hiPSC-CM è severa — il battito spontaneo è irregolare per natura. Vale la pena valutare (a) alzarla, (b) renderla un warning invece di un'esclusione, o (c) permettere che una baseline marginale resti utilizzabile con un flag di qualità propagato, come abbiamo fatto per `fpd_reliable`.

### 3.2 🟠 Un membro del gruppo escluso passa comunque

Nello stesso gruppo, `chipD_ch1_Dofe_3nM` risulta `passed=True` mentre tutti i suoi fratelli sono esclusi con "Baseline of group ... failed". Se il gruppo è stato rimosso, nessun membro dovrebbe sopravvivere.

L'ipotesi più probabile è che per quel file il selettore automatico abbia scelto **el2** invece di el1, mandandolo in un gruppo diverso (la chiave di gruppo include l'elettrodo) dove non c'è baseline. È lo stesso meccanismo del fallback cross-elettrodo che `DOCUMENTATION.md:674` ammette poter introdurre bias. Vale un'indagine mirata.

### 3.3 🟠 File duplicati con nomi incoerenti

Coppie che producono risultati identici bit a bit, cioè sono lo stesso segnale salvato due volte:

- `chipD_ch1_Dofe_0.3nM` e `chipD_ch1_Dofe_0_3nM` → FPDc 454.5 entrambi
- `chipD_ch1_Dofe_1nM` e `chipD_ch1_Dofe_nM` → FPDc 887.9 entrambi

Il secondo caso è più insidioso: `Dofe_nM` senza numero verrebbe letto come concentrazione mancante. Conviene ripulire la nomenclatura prima di rifare girare gli studi.

### 3.4 🟡 Terfenadina assente dalla classificazione in EXP 5

In EXP 5 arrivano a classificazione solo chinidina e ranolazina; i 6 file di terfenadina (`chipA_ch1_terfe_*`) non compaiono. Probabile stessa cascata da baseline `chipA_ch1`. Da verificare con lo stesso metodo del §3.1.

---

## 4. Conclusione e raccomandazioni

**Sul fix:** confermato corretto e a impatto contenuto. Nessuna ri-analisi obbligatoria dei risultati storici. Se rifai girare uno studio, i numeri si muoveranno di qualche punto percentuale su una minoranza di file, senza cambiare le conclusioni.

**Priorità reale, emersa dalla ri-analisi:** i criteri di inclusione stanno silenziosamente scartando curve dose-risposta intere, incluso il controllo positivo canonico CiPA. È un problema più grave del bug appena corretto, perché non produce numeri sbagliati — produce **assenza di numeri**, che è più difficile da notare.

Ordine suggerito:

1. Indagare la soglia CV 25 % sulle baseline (§3.1) e decidere se ammorbidirla o convertirla in flag.
2. Chiarire l'incoerenza di `Dofe_3nM` (§3.2) — probabile instabilità nella selezione dell'elettrodo.
3. Ripulire i nomi duplicati nel dataset (§3.3).
4. Aggiungere al report di batch un blocco esplicito **"gruppi rimossi e perché"**, così un'esclusione a cascata è visibile subito invece che deducibile.

---

## 5. Riproducibilità

Lo strumento di confronto è in repo: `tools/compare_pipeline_versions.py`.

```bash
# estrai la versione di riferimento
mkdir -p /tmp/orig && git archive <commit> cardiac_fp_analyzer | tar -x -C /tmp/orig

python3 tools/compare_pipeline_versions.py \
    --old-root /tmp/orig --new-root . \
    --data "µECG-Pharma Calibration - EXCEL DATA/EXP 5" \
    --out-dir /tmp/cmp --label EXP5
```

Produce un report testuale con le tre sezioni (per-file, normalizzazione, classificazione) e mette in cache i JSON, così rilanciare per modificare il report non ri-analizza. I report scritti dalla pipeline vengono reindirizzati fuori dalla cartella dati.

> **Nota:** un primo run fallito ha lasciato due file
> `cardiac_fp_analysis_20260805_112953.{pdf,xlsx}` dentro
> `µECG-Pharma Calibration - EXCEL DATA/EXP 5/analysis_results/`.
> Non ho i permessi per rimuoverli dal sandbox — cancellali tu.
