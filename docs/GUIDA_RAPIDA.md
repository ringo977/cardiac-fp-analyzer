---
title: "Cardiac FP Analyzer: guida rapida"
---

**A cosa serve.** Analizza i tracciati FP, propone depolarizzazioni e ripolarizzazioni, e permette di rivederli uno a uno prima di esportare gli Excel dei risultati. Il manuale completo è `MANUALE_Cardiac_FP_Analyzer.docx`.

## 1. Installazione (una volta sola)

Serve Python 3.12 (python.org, spuntare *Add python.exe to PATH*) e Git (git-scm.com). In alternativa a Git: GitHub ▸ *Code ▸ Download ZIP* e scompattare.

**Windows** (PowerShell):

```
cd $HOME\Documents
git clone https://github.com/ringo977/cardiac-fp-analyzer.git
cd cardiac-fp-analyzer
git checkout feat/pyside-poc
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[gui,reports,mcs,nwb]"
cardiac-fp-gui
```

**Mac** (Terminale): stessi comandi, con `python3 -m venv .venv` e `source .venv/bin/activate`.

Volte successive: `cd` nella cartella, riattivare l'ambiente (`.venv\Scripts\Activate.ps1` / `source .venv/bin/activate`), `cardiac-fp-gui`. Per aggiornare: `git pull` e di nuovo `pip install -e ...`. Se PowerShell rifiuta lo script di attivazione: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, risposta S.

## 2. Flusso di lavoro

1. **Organizzare il blocco.** Mettere in una cartella i tracciati da analizzare insieme, per esempio un composto con la baseline e tutte le concentrazioni degli stessi tessuti. Se l'esperimento ha una `samples.csv`, il software la usa per tessuto, dose e condizione.
2. **Coda di revisione** (*Revisione ▸ Coda di revisione*, Ctrl+Shift+R): *Aggiungi cartella…*. Il software fa la passata automatica su tutto il blocco e mette in lista ogni tessuto con il giudizio, le misure e i problemi. In cima le registrazioni con problemi.
3. **Rivedere un tracciato alla volta**: doppio clic sulla riga. Il pannello Revisione a destra mostra giudizio e misure.
4. **Correggere** (scheda *Segnale*, modalità *Edit*). Clic sul tracciato: il menu propone di aggiungere, spostare o togliere una depolarizzazione, e di spostare la ripolarizzazione (anche "e propaga a tutti i battiti"). Oppure si **trascina** il triangolo con il mouse: al rilascio si aggancia al picco (con Shift resta esattamente dove lo si lascia). Ctrl+Z annulla.
5. **Approvare** con la spunta *Approvato* (il nome e la data restano registrati) e **Salvare** con Ctrl+S. *Chiudi* blocca la registrazione.
6. **Esportare**: *Esporta Results…* nella coda produce l'Excel nel formato dei vostri fogli (foglio per chip, foglio *raw* con i battiti, foglio *samples*).

## 3. Come leggere il tracciato

- **Giudizio**: verde *Buono*, giallo *Da verificare*, rosso *Problemi*, con l'elenco dei motivi (*mostra tutto* apre l'elenco completo).
- **Tratti colorati sul tracciato**: arancio = battito forse mancante; blu = due depolarizzazioni troppo vicine; viola = ripolarizzazione assente tra due depolarizzazioni; grigio = rumore. *Prossimo tratto* (Ctrl+G) porta alla segnalazione seguente.
- **Viste**: *Filtrato* per controllare le depolarizzazioni, *1–40 Hz (operatori)* per le ripolarizzazioni (quel filtro sposta il picco apparente dello spike).
- **Tasti utili**: frecce = battito precedente/successivo, Canc = toglie il battito, P / N = ripolarizzazione sul picco positivo/negativo, Ctrl+S = salva.
- **Se depolarizzazioni e ripolarizzazioni sono scambiate**: *Correzioni su tutto il tracciato ▸ Scambia…* (sull'intervallo visibile o su tutto).

## 4. Che cosa segnalarmi

I marcatori automatici sbagliati, i giudizi che non condividete, ciò che nell'interfaccia rallenta, e le differenze tra i vostri Excel e quelli esportati. Basta anche uno screenshot.

Interfaccia in italiano: *Visualizza ▸ Lingua ▸ Italiano* (vale dal riavvio).
