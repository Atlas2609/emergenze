# Sistema di Gestione Intelligente delle Emergenze Urbane

Caso di studio per il corso di **Ingegneria della Conoscenza** (Università di Bari, A.A. 2025/26).

**Autore:** Vincenzo Campanelli, matricola 758276, v.campanelli5@studenti.uniba.it

Il sistema assiste una centrale operativa che deve gestire più emergenze contemporanee in una città: decide quale mezzo inviare, su quale percorso e se serviranno rinforzi. Combina rappresentazione della conoscenza, ricerca, vincoli e apprendimento.

| Modulo | Argomento del corso | File |
|---|---|---|
| Base di conoscenza in Prolog: mappa, meteo, transitabilità, costi | Rappresentazione e ragionamento relazionale | `prolog/mappa_emergenze.pl`, `logica/kb_interface.py` |
| Ontologia OWL con ragionatore HermiT | Ontologie e logiche descrittive | `ontologia/ontologia_emergenze.py` |
| A* e ricerca a costo minimo | Ricerca informata in spazi di stati | `ricerca/astar_routing.py` |
| Allocazione mezzi: AC-3, backtracking, branch and bound | Ragionamento con vincoli | `ricerca/csp_backtracking.py` |
| Allocazione mezzi: Random Restart Hill Climbing, Simulated Annealing | Ricerca locale | `ricerca/csp_allocazione.py` |
| Classificatori con nested cross-validation | Apprendimento supervisionato | `machine_learning/modello_predittivo.py`, `validazione.py` |
| Rete bayesiana (struttura esperta e appresa con BIC) | Ragionamento e apprendimento con incertezza | `machine_learning/rete_bayesiana.py` |

La documentazione completa è in `docs/Relazione_Progetto.docx`.

## Prerequisiti

- Python 3.10 o successivo
- [SWI-Prolog](https://www.swi-prolog.org/download/stable), con la cartella `bin` nel PATH (serve a pyswip)
- Java 8 o successivo (serve al ragionatore HermiT di Owlready2)

## Installazione

```bash
cd emergenze_urbane
pip install -r requirements.txt
```

## Esecuzione

```bash
python main.py              # simulazione completa (circa 2 minuti)
python main.py --veloce     # nested CV ridotta, per una demo rapida
python esperimenti.py       # tutte le valutazioni riportate nella relazione
python figure_relazione.py  # diagrammi della relazione (docs/figure)
python -m pytest test       # 65 test
```

`main.py` esegue in sequenza:
1. carica la KB Prolog;
2. costruisce l'ontologia, la classifica con HermiT e ne verifica la coerenza con la KB;
3. genera lo scenario (emergenze, mezzi, meteo, fascia oraria) e lo asserisce nella KB;
4. calcola con A* la matrice dei costi mezzo → emergenza;
5. alloca i mezzi con il CSP (ottimo esatto) e lo confronta con la ricerca locale;
6. confronta i modelli di apprendimento con la nested CV e sceglie il migliore;
7. predice per ogni emergenza la probabilità che servano rinforzi, usando le feature lette dalla KB e il ritardo calcolato da A*.

Grafici e tabelle vengono salvati in `output/` (gli esperimenti in `output/esperimenti/`).

## Struttura

```
emergenze_urbane/
├── main.py                     # orchestratore della simulazione
├── esperimenti.py              # valutazione sperimentale (tabelle e grafici)
├── figure_relazione.py         # diagrammi: architettura, mappa, ontologia, reti bayesiane
├── prolog/mappa_emergenze.pl   # KB: fatti e regole
├── logica/kb_interface.py      # interfaccia Python-Prolog (pyswip)
├── ontologia/
│   ├── ontologia_emergenze.py  # TBox, ABox dalla KB, ragionamento HermiT
│   └── emergenze.owl           # ontologia generata (apribile con Protégé)
├── ricerca/
│   ├── astar_routing.py        # A* / costo minimo con query alla KB
│   ├── csp_allocazione.py      # modello CSP + RRHC + Simulated Annealing
│   └── csp_backtracking.py     # AC-3 + backtracking + branch and bound
├── machine_learning/
│   ├── dataset_synth.py        # dataset sintetico da modello causale esplicito
│   ├── modello_predittivo.py   # pipeline: imputazione, codifica, classificatori
│   ├── rete_bayesiana.py       # rete bayesiana (pgmpy)
│   └── validazione.py          # nested CV, metriche, grafici
├── simulazione/scenario_manager.py
├── test/                       # test unitari (pytest / unittest)
└── docs/                       # relazione (.docx, .pdf) e figure
```
