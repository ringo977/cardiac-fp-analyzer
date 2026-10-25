# Cose da fare

## Correggere anche i tempi del picco (compensare il filtraggio)

Dall'ampiezza (25 ott 2026) si legge sul segnale non filtrato, ma i **tempi** della depolarizzazione che il
rilevamento automatico produce sono ancora quelli del tracciato filtrato. Il filtraggio (passa-banda 0,5–500 Hz,
Savitzky–Golay 7 campioni, notch) sposta il picco apparente rispetto al grezzo: 1–10 ms nei tessuti di Exp1
(su G Ch1 il marcatore automatico sta 8–13 ms prima di quello dell'operatore su tutti i battiti).

Oggi il tempo si corregge sul picco grezzo solo con i comandi manuali (clic, *Allinea*: ±8 ms, `ReviewState._refine_on_raw`).
Da fare, se serve: portare lo stesso passaggio nella pipeline, così `dep_s`, il BP e l'FPD (= ripolarizzazione − depolarizzazione)
sono letti sul grezzo come dagli operatori. Attenzione: cambia di qualche ms FPD e FPDc rispetto ai confronti già fatti con
l'operatore sul corpus di sviluppo, quindi va rifatto il confronto (DEV, poi TEST una volta) prima di adottarlo.
