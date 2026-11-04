"""
modello_predittivo.py — Pipeline scikit-learn per la predizione
                        del supporto aggiuntivo nelle emergenze

Il modello usa un albero di decisione (DecisionTreeClassifier), un
classificatore interpretabile che apprende regole di decisione dalla
struttura dei dati. La pipeline include preprocessing e classificatore
in un unico oggetto scikit-learn, garantendo che le trasformazioni
siano applicate correttamente durante la cross-validation.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def crea_pipeline() -> Pipeline:
    """
    Crea la pipeline scikit-learn con preprocessing e classificatore.

    La pipeline include:
    1. StandardScaler: normalizzazione delle feature (non strettamente
       necessaria per un albero di decisione, ma buona pratica per
       una pipeline generica riusabile con altri classificatori)
    2. DecisionTreeClassifier: classificatore ad albero di decisione

    Returns:
        Pipeline scikit-learn pronta per il fit.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classificatore", DecisionTreeClassifier(random_state=42))
    ])
    return pipeline


def get_spazio_iperparametri() -> dict:
    """
    Definisce lo spazio di iperparametri per il tuning via GridSearchCV.

    Iperparametri esplorati:
    - max_depth: profondità massima dell'albero (controlla l'overfitting)
    - min_samples_split: campioni minimi per effettuare uno split
    - min_samples_leaf: campioni minimi in ogni foglia
    - criterion: funzione di impurità (gini vs entropia)

    Returns:
        Dizionario con i nomi dei parametri (prefissati con il nome
        dello step della pipeline) e i relativi valori da esplorare.
    """
    spazio = {
        "classificatore__max_depth": [3, 5, 7, 10, None],
        "classificatore__min_samples_split": [2, 5, 10, 20],
        "classificatore__min_samples_leaf": [1, 3, 5, 10],
        "classificatore__criterion": ["gini", "entropy"],
    }
    return spazio


def descrivi_modello(pipeline_addestrata: Pipeline) -> str:
    """
    Genera una descrizione testuale del modello addestrato,
    inclusi gli iperparametri selezionati.

    Args:
        pipeline_addestrata: pipeline scikit-learn dopo il fit.

    Returns:
        Stringa con la descrizione del modello.
    """
    clf = pipeline_addestrata.named_steps["classificatore"]
    linee = [
        "=== Modello Predittivo: Albero di Decisione ===",
        f"  Criterio: {clf.criterion}",
        f"  Profondità massima: {clf.max_depth}",
        f"  Min campioni per split: {clf.min_samples_split}",
        f"  Min campioni per foglia: {clf.min_samples_leaf}",
        f"  Numero foglie: {clf.get_n_leaves()}",
        f"  Profondità effettiva: {clf.get_depth()}",
    ]
    return "\n".join(linee)
