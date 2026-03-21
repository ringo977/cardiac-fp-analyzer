# cardiac_fp_analyzer

Analisi di **field potential (FP)** per registrazioni µECG / hiPSC-CM da file CSV **Digilent WaveForms** (due canali + tempo).

## Requisiti

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Dalla cartella che contiene il pacchetto `cardiac_fp_analyzer`:

```bash
python cardiac_fp_analyzer/analyze.py /percorso/cartella/dati --channel auto -o /percorso/output
```

Vengono letti tutti i `*.csv` in ricorsione e generati report Excel e PDF in `analysis_results/` (o nella cartella indicata con `-o`).

## Licenza

Uso interno / da definire.
