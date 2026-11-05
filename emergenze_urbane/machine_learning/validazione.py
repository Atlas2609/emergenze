"""
validazione.py — Validazione statistica dei modelli con Nested Cross-Validation

La Nested Cross-Validation fornisce una stima non ottimistica delle
prestazioni. Il design a doppio ciclo separa la scelta degli
iperparametri dalla stima dell'errore:

  - Outer loop (StratifiedKFold): ogni fold esterno fa da test set mai
    visto durante il tuning; restituisce una misura per fold, da cui
    media e deviazione standard.
  - Inner loop (GridSearchCV): sceglie gli iperparametri usando solo i
    dati di training del fold esterno corrente.

Tutti i modelli (classificatori scikit-learn e reti bayesiane) passano
per la stessa procedura con gli stessi fold, quindi i risultati sono
direttamente confrontabili.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

from machine_learning.modello_predittivo import crea_pipeline, get_spazio_iperparametri


METRICHE = ["accuracy", "precision", "recall", "f1", "log_loss", "roc_auc"]

# Metrica ottimizzata nel loop interno e usata per scegliere il modello:
# AUC-ROC. Con classi bilanciate F1 premia anche un classificatore banale
# che risponde sempre "supporto" (recall = 1): la baseline lo dimostra.
# L'AUC misura invece quanto bene il modello ordina i casi per rischio,
# indipendentemente dalla soglia, ed è ciò che serve a un operatore.
SCORING_INTERNO = "roc_auc"


def esegui_nested_cv(
    X,
    y: np.ndarray,
    n_fold_outer: int = 5,
    n_fold_inner: int = 3,
    random_state: int = 42,
    output_dir: str = ".",
    modello=None,
    griglia: dict = None,
    nome: str = "Albero di decisione",
    grafici: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Esegue la Nested Cross-Validation di un modello.

    Args:
        X: feature (DataFrame categorico, vedi dataset_synth.prepara_features).
        y: vettore target binario.
        n_fold_outer / n_fold_inner: fold dei due cicli.
        random_state: seed per la riproducibilità (stessi fold per tutti i modelli).
        output_dir: directory per i grafici.
        modello, griglia: estimator e spazio degli iperparametri; se None,
                          albero di decisione con il suo spazio.
        nome: nome del modello per stampe e grafici.
        grafici: se True salva matrice di confusione e curva ROC.

    Returns:
        Dizionario con, per ogni metrica, valori per fold, media e std,
        più matrice di confusione aggregata e iperparametri per fold.
    """
    if modello is None:
        modello, griglia = crea_pipeline(), get_spazio_iperparametri()
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)

    outer_cv = StratifiedKFold(n_splits=n_fold_outer, shuffle=True, random_state=random_state)
    per_fold = {m: [] for m in METRICHE}
    migliori_parametri = []
    tutte_y_true, tutte_y_pred, tutte_y_prob = [], [], []
    inizio = time.time()

    if verbose:
        print(f"\n  --- Nested CV: {nome} "
              f"({n_fold_outer} fold esterni x {n_fold_inner} interni) ---")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_fold_inner, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            estimator=clone(modello),
            param_grid=griglia,
            cv=inner_cv,
            scoring=SCORING_INTERNO,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        migliori_parametri.append(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        per_fold["accuracy"].append(accuracy_score(y_test, y_pred))
        per_fold["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        per_fold["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        per_fold["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        per_fold["log_loss"].append(log_loss(y_test, y_prob, labels=[0, 1]))
        per_fold["roc_auc"].append(roc_auc_score(y_test, y_prob))

        tutte_y_true.extend(y_test)
        tutte_y_pred.extend(y_pred)
        tutte_y_prob.extend(y_prob)

        if verbose:
            print(f"      Fold {fold_idx + 1}: AUC = {per_fold['roc_auc'][-1]:.4f} | "
                  f"F1 = {per_fold['f1'][-1]:.4f} | Accuracy = {per_fold['accuracy'][-1]:.4f} | "
                  f"parametri = {_parametri_brevi(grid_search.best_params_)}")

    risultati = {"nome": nome, "migliori_parametri": migliori_parametri,
                 "tempo_s": time.time() - inizio}
    for m in METRICHE:
        risultati[f"{m}_fold"] = per_fold[m]
        risultati[f"{m}_media"] = float(np.mean(per_fold[m]))
        risultati[f"{m}_std"] = float(np.std(per_fold[m]))

    tutte_y_true = np.array(tutte_y_true)
    tutte_y_pred = np.array(tutte_y_pred)
    tutte_y_prob = np.array(tutte_y_prob)
    risultati["confusion_matrix"] = confusion_matrix(tutte_y_true, tutte_y_pred)
    risultati["y_true_aggregato"] = tutte_y_true
    risultati["y_prob_aggregato"] = tutte_y_prob

    if verbose:
        print(f"      => AUC {risultati['roc_auc_media']:.4f} ± {risultati['roc_auc_std']:.4f} | "
              f"F1 {risultati['f1_media']:.4f} ± {risultati['f1_std']:.4f} | "
              f"Accuracy {risultati['accuracy_media']:.4f} ± {risultati['accuracy_std']:.4f}")

    if grafici:
        os.makedirs(output_dir, exist_ok=True)
        _plot_confusion_matrix(risultati["confusion_matrix"], nome, output_dir)
        _plot_roc_curve(tutte_y_true, tutte_y_prob, nome, output_dir)

    return risultati


def confronta_modelli(
    X,
    y: np.ndarray,
    modelli: dict,
    n_fold_outer: int = 5,
    n_fold_inner: int = 3,
    random_state: int = 42,
    output_dir: str = ".",
) -> tuple:
    """
    Valuta ogni modello del dizionario {nome: (estimator, griglia)} con
    la nested CV (stessi fold per tutti) e una baseline che predice
    sempre la classe più frequente con la probabilità a priori.

    Returns:
        Tupla (tabella DataFrame "media ± std", dizionario dei risultati).
    """
    tutti = dict(modelli)
    tutti["Baseline (probabilità a priori)"] = (
        DummyClassifier(strategy="prior"), {"strategy": ["prior"]}
    )

    risultati = {}
    for nome, (modello, griglia) in tutti.items():
        risultati[nome] = esegui_nested_cv(
            X, y, n_fold_outer, n_fold_inner, random_state, output_dir,
            modello=modello, griglia=griglia, nome=nome, grafici=False,
        )

    tabella = pd.DataFrame([
        {"Modello": nome, **{
            m: f"{r[f'{m}_media']:.3f} ± {r[f'{m}_std']:.3f}" for m in METRICHE
        }, "tempo (s)": f"{r['tempo_s']:.1f}"}
        for nome, r in risultati.items()
    ])

    os.makedirs(output_dir, exist_ok=True)
    tabella.to_csv(os.path.join(output_dir, "confronto_modelli.csv"), index=False)
    _plot_boxplot_auc(risultati, output_dir)
    _plot_roc_confronto(risultati, output_dir)
    nome_migliore = migliore_modello(risultati)
    _plot_confusion_matrix(risultati[nome_migliore]["confusion_matrix"], nome_migliore, output_dir)
    return tabella, risultati


def migliore_modello(risultati: dict) -> str:
    """Nome del modello con AUC-ROC media più alta (baseline esclusa)."""
    candidati = {n: r for n, r in risultati.items() if not n.startswith("Baseline")}
    return max(candidati, key=lambda n: candidati[n]["roc_auc_media"])


def addestra_modello_finale(X, y: np.ndarray, modello=None, griglia: dict = None):
    """
    Addestra il modello scelto su tutto il dataset, selezionando gli
    iperparametri con GridSearchCV. È il modello usato in simulazione;
    la sua stima di prestazione è quella della nested CV, non questa.
    """
    if modello is None:
        modello, griglia = crea_pipeline(), get_spazio_iperparametri()
    grid_search = GridSearchCV(
        estimator=clone(modello),
        param_grid=griglia,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring=SCORING_INTERNO,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(pd.DataFrame(X).reset_index(drop=True), np.asarray(y))
    print(f"\n  Modello finale addestrato su {len(y)} campioni")
    print(f"  Migliori parametri: {_parametri_brevi(grid_search.best_params_)}")
    return grid_search.best_estimator_


def _parametri_brevi(params: dict) -> dict:
    return {k.split("__")[-1]: v for k, v in params.items()}


def _nome_file(nome: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in nome.lower()).strip("_")


def _plot_confusion_matrix(cm: np.ndarray, nome: str, output_dir: str):
    """Matrice di confusione aggregata sui fold esterni."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=["No supporto", "Supporto"]).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"Matrice di confusione - {nome} (nested CV)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{_nome_file(nome)}.png"), dpi=150)
    plt.close()


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, nome: str, output_dir: str):
    """Curva ROC sulle predizioni aggregate dei fold esterni."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{nome} (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Classificatore casuale")
    ax.set_xlabel("Tasso di falsi positivi")
    ax.set_ylabel("Tasso di veri positivi")
    ax.set_title(f"Curva ROC - {nome} (nested CV)", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_{_nome_file(nome)}.png"), dpi=150)
    plt.close()


def _plot_roc_confronto(risultati: dict, output_dir: str):
    """Curve ROC di tutti i modelli (predizioni aggregate dei fold esterni)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome, r in risultati.items():
        if nome.startswith("Baseline"):
            continue
        fpr, tpr, _ = roc_curve(r["y_true_aggregato"], r["y_prob_aggregato"])
        ax.plot(fpr, tpr, lw=1.5, label=f"{nome} (AUC {r['roc_auc_media']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Classificatore casuale")
    ax.set_xlabel("Tasso di falsi positivi")
    ax.set_ylabel("Tasso di veri positivi")
    ax.set_title("Curve ROC - nested CV")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_modelli.png"), dpi=150)
    plt.close()


def _plot_boxplot_auc(risultati: dict, output_dir: str):
    """Distribuzione dell'AUC-ROC sui fold esterni per ogni modello."""
    nomi = list(risultati)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([risultati[n]["roc_auc_fold"] for n in nomi])
    ax.set_xticks(range(1, len(nomi) + 1))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in nomi], fontsize=8)
    ax.set_ylabel("AUC-ROC sui fold esterni")
    ax.set_title("Confronto modelli - nested CV")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confronto_modelli_auc.png"), dpi=150)
    plt.close()
