# Sistema di Gestione Intelligente delle Emergenze Urbane

Progetto per il corso di **Ingegneria della Conoscenza** che integra IA simbolica e sub-simbolica
per la gestione ottimale delle emergenze in ambito urbano.

## Architettura del Sistema

Il sistema integra tre pilastri dell'Intelligenza Artificiale:

1. **IA Simbolica (Prolog)**: Base di conoscenza con regole deduttive (clausole di Horn) per
   modellare la mappa cittadina, le condizioni meteo e la transitabilità delle strade.
2. **Ricerca e Ottimizzazione**: Algoritmo A* per il routing ottimo e CSP con ricerca locale
   (Random Restart Hill Climbing + Simulated Annealing) per l'allocazione mezzi-emergenze.
3. **Machine Learning (scikit-learn)**: Albero di decisione per predire la necessità di
   supporto aggiuntivo, validato con Nested Cross-Validation.

## Prerequisiti

- **Python 3.10+**
- **SWI-Prolog** installato e nel PATH di sistema
  - Windows: [Download SWI-Prolog](https://www.swi-prolog.org/download/stable)
  - Assicurarsi che `swipl` sia raggiungibile da terminale

## Installazione

```bash
# Clona o scarica il progetto
cd emergenze_urbane

# Installa le dipendenze Python
pip install -r requirements.txt
```

## Esecuzione

```bash
# Esecuzione completa del sistema
python main.py
```

Il sistema eseguirà in sequenza:
1. Caricamento della base di conoscenza Prolog
2. Generazione di uno scenario di emergenza multipla
3. Aggiornamento dinamico delle condizioni meteo nella KB
4. Calcolo dei percorsi ottimi con A*
5. Allocazione mezzi con CSP (confronto RRHC vs SA)
6. Addestramento e validazione ML con Nested CV
7. Predizione del supporto aggiuntivo per le emergenze simulate
8. Report finale

## Esecuzione dei Test

```bash
# Tutti i test
python -m pytest test/ -v

# Test singoli moduli
python -m pytest test/test_kb.py -v
python -m pytest test/test_astar.py -v
python -m pytest test/test_csp.py -v
python -m pytest test/test_ml.py -v

# Oppure con unittest
python -m unittest discover test/ -v
```

## Struttura del Progetto

```
emergenze_urbane/
├── requirements.txt          # Dipendenze Python
├── README.md                 # Questo file
├── main.py                   # Orchestratore principale
├── prolog/
│   └── mappa_emergenze.pl    # KB Prolog: mappa, regole deduttive
├── logica/
│   ├── __init__.py
│   └── kb_interface.py       # Wrapper pyswip: query, assert/retract
├── ricerca/
│   ├── __init__.py
│   ├── astar_routing.py      # A* con query dinamiche alla KB
│   └── csp_allocazione.py    # CSP + RRHC + Simulated Annealing
├── machine_learning/
│   ├── __init__.py
│   ├── dataset_synth.py      # Generazione dataset sintetico
│   ├── modello_predittivo.py # Pipeline DecisionTree
│   └── validazione.py        # Nested CV, ROC, confusion matrix
├── simulazione/
│   ├── __init__.py
│   └── scenario_manager.py   # Generazione scenari di emergenza
├── test/
│   ├── test_kb.py            # Test KB Prolog
│   ├── test_astar.py         # Test A*
│   ├── test_csp.py           # Test CSP
│   └── test_ml.py            # Test ML
└── output/                   # Grafici generati (ROC, confusion matrix)
```

## Principi di Ingegneria della Conoscenza Applicati

### Deduzione Logica (Prolog)
La KB Prolog non è un semplice database di fatti: contiene **regole deduttive** (clausole di Horn)
che derivano nuovi fatti per inferenza. Ad esempio, `transitabile/3` è *derivato* dalla
combinazione di `arco/3`, `strada_allagata/1` e `tipo_veicolo/2`, non elencato staticamente.

### Ricerca Informata (A*)
L'algoritmo A* usa un'euristica ammissibile (distanza euclidea) che garantisce l'ottimalità.
La peculiarità è che **interroga la KB Prolog ad ogni espansione di nodo**, rendendo il routing
sensibile allo stato dinamico della mappa.

### Ricerca Locale (CSP)
Il problema di allocazione è risolto con **Random Restart Hill Climbing** e **Simulated Annealing**,
non con backtracking esaustivo. La funzione obiettivo combina costi di distanza, penalità per
violazioni hard (compatibilità, capacità) e soft (distanza massima).

### Validazione Statistica (Nested CV)
La **Nested Cross-Validation** evita il data leakage tra tuning e valutazione:
- **Outer loop**: StratifiedKFold per stima onesta delle prestazioni
- **Inner loop**: GridSearchCV per ottimizzazione iperparametri
- Metriche riportate: accuracy ± std, log loss ± std, matrice di confusione, curva ROC con AUC

## Output Generati

- `output/confusion_matrix.png` — Matrice di confusione aggregata
- `output/roc_curve.png` — Curva ROC con AUC

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Ragionamento logico | SWI-Prolog via pyswip |
| Ricerca informata | A* (implementazione custom) |
| Ottimizzazione CSP | RRHC + Simulated Annealing |
| Machine Learning | scikit-learn (DecisionTreeClassifier) |
| Visualizzazione | matplotlib |
| Dati | pandas, numpy |

## Autore

Progetto sviluppato per il corso di Ingegneria della Conoscenza.
