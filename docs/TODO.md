# Cose da fare

## Tempi della depolarizzazione e filtraggio — misurato il 26 ottobre 2026

**Domanda.** L'ampiezza si legge sul segnale non filtrato (v3.23). Bisogna correggere anche i *tempi* del picco
per compensare il filtraggio?

**Cosa dicono i dati.** Ho confrontato i tempi della depolarizzazione del software con quelli dell'operatore, battito per
battito, sul corpus GG di sviluppo (1945 battiti) e sui 16 tessuti della baseline di Exp1.

- Il rilevamento automatico sta già sul picco: mediana 0,7 ms dal tempo dell'operatore su GG, 1,5 ms su Exp1. Il picco
  sul segnale grezzo, quello sul filtrato e il rilevamento coincidono entro 0,5 ms (stessa mediana −1,5 ms su Exp1).
- Rileggere i tempi sul picco grezzo (opzione `BeatDetectionConfig.refine_on_raw_ms`, ±8 ms) porta la mediana da 0,7 a
  0,4 ms su GG, ma l'FPD per registrazione entro ±10 % passa da 64,8 a 63,4 % (due registrazioni oltre la soglia): resta
  un'opzione, spenta.
- Il tempo dell'operatore **non** è il picco: è più vicino al punto di **discesa più ripida** dello spike (massima
  pendenza negativa, la definizione classica del tempo di attivazione): mediana −0,2 ms invece di −1,5 ms, 52 % dei
  battiti entro 1,5 ms invece del 42 %. Nemmeno il picco della vista 1–40 Hz lo spiega.
- Su tre tessuti di Exp1 con complessi larghi a due gobbe (G1, E2, F1) l'operatore segna un punto circa 11 ms dopo il
  picco, sulla seconda gobba; su D2 e G2 circa 6–10 ms dopo. (Spiegato dal loro codice: è la discesa più ripida calcolata
  sul passa-basso a 100 Hz, che riproduce anche questi casi — vedi sotto.)

**Effetto sull'FPD.** Il ritardo tipico di 1,5 ms allunga l'FPD del software dello 0,2 % circa; sui complessi larghi di
circa l'1 %. Gli scarti grandi con l'operatore (10–70 %) vengono dalla scelta dell'onda di ripolarizzazione, non dai tempi
della depolarizzazione.

**Fatto (v3.24).** Il codice MATLAB degli operatori (v1.5) conferma la discesa più ripida (minimo di dV/dt sul passa-basso a
100 Hz). È un'opzione: *Impostazioni ▸ Detection ▸ Tempo della depolarizzazione* (`dep_time_rule = 'steepest_fall'`,
`lab_convention.py`); riproduce i loro tempi di Exp1 allo stesso campione nel 98 % dei battiti. Il default resta il picco:
il gold GG del 2025 segue il picco (vedi DOCUMENTATION, v3.24.0). Da decidere con il laboratorio quale convenzione
usare negli studi nuovi.
