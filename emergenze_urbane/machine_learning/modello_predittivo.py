"""
modello_predittivo.py — Pipeline scikit-learn per la predizione
                        del supporto aggiuntivo nelle emergenze

Si confrontano più classificatori visti a lezione, ognuno in una
Pipeline che contiene anche il preprocessing (imputazione dei valori
mancanti + codifica delle feature categoriche). Tenere il
preprocessing nella pipeline garantisce che, durante la
cross-validation, la moda per l'imputazione sia calcolata solo sui
fold di training.

Modelli:
  - Albero di decisione (interpretabile, apprende regole)
  - Random Forest (ensemble di alberi, riduce la varianza)
  - Regressione logistica (modello lineare probabilistico)
  - Naive Bayes categorico (indipendenza condizionata delle feature)
La rete bayesiana è in rete_bayesiana.py e viene valutata con la
stessa procedura.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

from machine_learning.dataset_synth import VALORI, FEATURE, valori_noti


def crea_preprocessore(codifica: str = "onehot") -> Pipeline:
    """
    Preprocessing comune: valori fuori dominio trattati come mancanti,
    imputazione con la moda e codifica.

    Args:
        codifica: 'onehot' per i modelli che lavorano su feature binarie
                  (alberi, regressione logistica), 'ordinale' per il
                  Naive Bayes categorico, che vuole un intero per categoria.
    """
    categorie = [VALORI[col] for col in FEATURE]
    if codifica == "onehot":
        encoder = OneHotEncoder(categories=categorie, handle_unknown="ignore")
    else:
        encoder = OrdinalEncoder(categories=categorie)
    return Pipeline([
        ("valori_noti", FunctionTransformer(valori_noti)),
        ("imputazione", SimpleImputer(strategy="most_frequent")),
        ("codifica", encoder),
    ])


def crea_pipeline(classificatore=None, codifica: str = "onehot") -> Pipeline:
    """
    Crea la pipeline preprocessing + classificatore.
    Senza argomenti restituisce l'albero di decisione.
    """
    if classificatore is None:
        classificatore = DecisionTreeClassifier(random_state=42)
    return Pipeline([
        ("preprocessing", crea_preprocessore(codifica)),
        ("classificatore", classificatore),
    ])


def get_spazio_iperparametri() -> dict:
    """Spazio di ricerca dell'albero di decisione (usato anche dai test)."""
    return {
        "classificatore__max_depth": [3, 5, 7, 10, None],
        "classificatore__min_samples_leaf": [1, 5, 10, 20],
        "classificatore__criterion": ["gini", "entropy"],
    }


def get_modelli() -> dict:
    """
    Catalogo dei modelli da confrontare: nome -> (pipeline, griglia).

    Le griglie sono volutamente piccole: l'obiettivo è il confronto
    tra famiglie di modelli, non l'ottimizzazione spinta di ciascuno.
    """
    return {
        "Albero di decisione": (
            crea_pipeline(DecisionTreeClassifier(random_state=42)),
            get_spazio_iperparametri(),
        ),
        "Random Forest": (
            crea_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
            {
                "classificatore__max_depth": [5, 10, None],
                "classificatore__min_samples_leaf": [1, 5, 10],
            },
        ),
        "Regressione logistica": (
            crea_pipeline(LogisticRegression(max_iter=2000)),
            {"classificatore__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Naive Bayes": (
            crea_pipeline(
                CategoricalNB(min_categories=[len(VALORI[c]) for c in FEATURE]),
                codifica="ordinale",
            ),
            {"classificatore__alpha": [0.1, 0.5, 1.0, 2.0]},
        ),
    }


def descrivi_modello(pipeline_addestrata: Pipeline) -> str:
    """Descrizione testuale del modello addestrato e dei suoi iperparametri."""
    clf = pipeline_addestrata.named_steps["classificatore"]
    linee = [f"=== Modello selezionato: {type(clf).__name__} ==="]
    if isinstance(clf, DecisionTreeClassifier):
        linee += [
            f"  Criterio: {clf.criterion}",
            f"  Profondità massima: {clf.max_depth}",
            f"  Min campioni per foglia: {clf.min_samples_leaf}",
            f"  Numero foglie: {clf.get_n_leaves()}",
        ]
    else:
        parametri = {k: v for k, v in clf.get_params().items()
                     if k in ("C", "alpha", "max_depth", "min_samples_leaf", "n_estimators")}
        linee.append(f"  Iperparametri: {parametri}")
    return "\n".join(linee)
