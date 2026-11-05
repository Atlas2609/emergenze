"""
rete_bayesiana.py — Rete bayesiana per il supporto aggiuntivo

Ragionamento sotto incertezza: la rete rappresenta la distribuzione
congiunta delle variabili con un grafo orientato aciclico e le tabelle
di probabilità condizionata (CPT) di ogni nodo dati i genitori.

Rispetto ai classificatori di modello_predittivo.py la rete:
  - rende esplicita la struttura delle dipendenze (leggibile e discutibile);
  - risponde a query con evidenza PARZIALE, ad esempio
    P(supporto | meteo = alluvione) senza conoscere orario o zona, perché
    l'inferenza (eliminazione delle variabili) somma sulle variabili
    non osservate.

Due strutture a confronto:
  - 'esperta': definita a mano dalla conoscenza del dominio;
  - 'appresa': stimata dai dati con ricerca locale (Hill Climbing) che
    massimizza il punteggio BIC.
Le CPT sono stimate con stimatore bayesiano e prior BDeu (pseudo-conteggi),
che evita probabilità nulle per combinazioni mai osservate.

La classe ReteBayesianaClassificatore espone l'interfaccia scikit-learn
(fit / predict / predict_proba) così la rete viene valutata con la
stessa nested cross-validation degli altri modelli.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.causal_discovery import HillClimbSearch
    from pgmpy.inference import VariableElimination

from machine_learning.dataset_synth import VALORI, FEATURE, TARGET, valori_noti


STRUTTURA_ESPERTA = [
    ("meteo", "ritardo_arrivo"),
    ("zona", "ritardo_arrivo"),
    ("orario", "ritardo_arrivo"),
    ("gravita", TARGET),
    ("tipo_emergenza", TARGET),
    ("ritardo_arrivo", TARGET),
    ("orario", TARGET),
]

NOMI_STATI = {**VALORI, TARGET: ["0", "1"]}


def _imputa_moda(df: pd.DataFrame, mode: dict) -> pd.DataFrame:
    """Valori mancanti o fuori dominio sostituiti con le mode dell'addestramento."""
    return valori_noti(df).fillna(value=mode)


def apprendi_struttura(df: pd.DataFrame) -> list:
    """
    Apprende la struttura con Hill Climbing sul punteggio BIC.
    BIC = log-verosimiglianza - (log N / 2) * numero di parametri:
    premia l'adattamento ai dati e penalizza le reti troppo complesse.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ricerca = HillClimbSearch(
            scoring_method="bic-d", return_type="dag", show_progress=False
        ).fit(df)
    return list(ricerca.causal_graph_.edges())


def costruisci_rete(df: pd.DataFrame, archi: list, equivalent_sample_size: int = 10):
    """Crea la rete con gli archi dati e ne stima le CPT (prior BDeu)."""
    rete = DiscreteBayesianNetwork(archi)
    rete.add_nodes_from(FEATURE + [TARGET])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stimatore = BayesianEstimator(rete, df, state_names=NOMI_STATI)
        cpd = stimatore.get_parameters(
            prior_type="BDeu", equivalent_sample_size=equivalent_sample_size
        )
    rete.add_cpds(*cpd)
    rete.check_model()
    return rete


class ReteBayesianaClassificatore(ClassifierMixin, BaseEstimator):
    """
    Classificatore compatibile con scikit-learn basato su rete bayesiana.

    Args:
        struttura: 'esperta' o 'appresa'.
        equivalent_sample_size: forza del prior BDeu (iperparametro).
    """

    def __init__(self, struttura: str = "esperta", equivalent_sample_size: int = 10):
        self.struttura = struttura
        self.equivalent_sample_size = equivalent_sample_size

    def fit(self, X: pd.DataFrame, y):
        X = valori_noti(X)
        self.mode_ = {c: X[c].mode()[0] for c in FEATURE}
        df = _imputa_moda(X, self.mode_).astype(str)
        df[TARGET] = np.asarray(y).astype(int).astype(str)

        if self.struttura == "esperta":
            self.archi_ = list(STRUTTURA_ESPERTA)
        else:
            self.archi_ = apprendi_struttura(df)

        self.rete_ = costruisci_rete(df, self.archi_, self.equivalent_sample_size)
        self.inferenza_ = VariableElimination(self.rete_)
        self.classes_ = np.array([0, 1])
        self._cache = {}
        return self

    def probabilita(self, evidenza: dict) -> float:
        """
        P(supporto = 1 | evidenza) con eliminazione delle variabili.
        L'evidenza può essere parziale; le variabili non presenti nella
        rete appresa (isolate) vengono ignorate.
        """
        nodi = set(self.rete_.nodes())
        evidenza = {k: v for k, v in evidenza.items()
                    if k in nodi and isinstance(v, str)}
        chiave = tuple(sorted(evidenza.items()))
        if chiave not in self._cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q = self.inferenza_.query(
                    [TARGET], evidence=evidenza, show_progress=False
                )
            self._cache[chiave] = float(q.values[q.state_names[TARGET].index("1")])
        return self._cache[chiave]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = _imputa_moda(pd.DataFrame(X, columns=FEATURE), self.mode_)
        p1 = np.array([self.probabilita(riga) for riga in X.to_dict("records")])
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def get_modelli_bayesiani() -> dict:
    """Le due reti da confrontare, con la griglia per il prior BDeu."""
    griglia = {"equivalent_sample_size": [5, 50, 200, 1000]}
    return {
        "Rete bayesiana (esperta)": (ReteBayesianaClassificatore("esperta"), griglia),
        "Rete bayesiana (appresa, BIC)": (ReteBayesianaClassificatore("appresa"), griglia),
    }


def esempi_query(modello: ReteBayesianaClassificatore) -> list:
    """
    Query con evidenza parziale, utili in relazione per mostrare il
    ragionamento probabilistico: restituisce (descrizione, probabilità).
    """
    query = [
        ("nessuna evidenza (probabilità a priori)", {}),
        ("meteo = alluvione", {"meteo": "alluvione"}),
        ("gravita = alta", {"gravita": "alta"}),
        ("gravita = alta, meteo = alluvione, orario = notte",
         {"gravita": "alta", "meteo": "alluvione", "orario": "notte"}),
        ("gravita = bassa, ritardo_arrivo = basso",
         {"gravita": "bassa", "ritardo_arrivo": "basso"}),
    ]
    return [(descr, modello.probabilita(ev)) for descr, ev in query]
