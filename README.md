# Sistema di Gestione Intelligente delle Emergenze Urbane

Caso di studio per il corso di **Ingegneria della Conoscenza** (Università di Bari, A.A. 2025/26).

**Autore:** Vincenzo Campanelli, matricola 758276, v.campanelli5@studenti.uniba.it

Il sistema assiste una centrale operativa che gestisce più emergenze contemporanee: una KB Prolog e
un'ontologia OWL descrivono la città, A* calcola i percorsi dei mezzi, un CSP (backtracking con AC-3 e
branch and bound, confrontato con Random Restart Hill Climbing e Simulated Annealing) li alloca, e
classificatori e reti bayesiane valutati con nested cross-validation stimano se serviranno rinforzi.

Il codice, le istruzioni di installazione ed esecuzione sono in [`emergenze_urbane/`](emergenze_urbane/README.md).
La relazione è in [`emergenze_urbane/docs/Relazione_Progetto.pdf`](emergenze_urbane/docs/Relazione_Progetto.pdf).

```bash
cd emergenze_urbane
pip install -r requirements.txt
python main.py
```
