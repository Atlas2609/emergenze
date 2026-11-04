"""
validazione.py — Validazione statistica del modello con Nested Cross-Validation

La Nested Cross-Validation fornisce una stima onesta (non ottimistica)
delle prestazioni del modello. Il design a doppio ciclo evita il data
leakage che si verificherebbe usando lo stesso set per il tuning degli
iperparametri e la valutazione:

  - Outer loop (StratifiedKFold): stima le prestazioni generalizzabili
    del modello. Ogni fold esterno simula un "nuovo dataset" su cui
    testare il modello ottimizzato.
  - Inner loop (GridSearchCV): ottimizza gli iperparametri usando solo
    i dati di training del fold esterno corrente.

Questo approccio è statisticamente più rigoroso di un singolo
train/test split perché riduce la varianza della stima delle
prestazioni, evita l'ottimismo da selezione del modello e produce
media e deviazione standard delle metriche.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
)

from machine_learning.modello_predittivo import crea_pipeline, get_spazio_iperparametri


def esegui_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_fold_outer: int = 5,
    n_fold_inner: int = 3,
    random_state: int = 42,
    output_dir: str = ".",
) -> dict:
    """
    Esegue la Nested Cross-Validation completa.

    L'outer loop (StratifiedKFold) partiziona i dati in fold di test e
    training, preservando la distribuzione delle classi. Per ogni fold
    esterno, l'inner loop (GridSearchCV) trova i migliori iperparametri
    usando solo i dati di training. Il modello ottimizzato viene poi
    valutato sul fold di test, mai visto durante il tuning.

    Args:
        X: matrice delle feature (n_campioni × n_feature).
        y: vettore target binario.
        n_fold_outer: numero di fold per il loop esterno.
        n_fold_inner: numero di fold per GridSearchCV (loop interno).
        random_state: seed per la riproducibilità.
        output_dir: directory dove salvare i grafici.

    Returns:
        Dizionario con tutte le metriche e i risultati.
    """
    print("\n" + "=" * 60)
    print("  NESTED CROSS-VALIDATION")
    print("=" * 60)
    print(f"  Outer folds: {n_fold_outer} (StratifiedKFold)")
    print(f"  Inner folds: {n_fold_inner} (GridSearchCV)")
    print(f"  Dataset: {X.shape[0]} campioni, {X.shape[1]} feature")
    print(f"  Distribuzione classi: {np.bincount(y)}")

    outer_cv = StratifiedKFold(
        n_splits=n_fold_outer, shuffle=True, random_state=random_state
    )

    spazio_parametri = get_spazio_iperparametri()

    accuratezze = []
    log_losses = []
    migliori_parametri = []

    tutte_y_true = []
    tutte_y_pred = []
    tutte_y_prob = []

    print(f"\n  Avvio dei {n_fold_outer} fold esterni...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  --- Fold esterno {fold_idx + 1}/{n_fold_outer} ---")
        print(f"      Training: {len(train_idx)} campioni | "
              f"Test: {len(test_idx)} campioni")

        # Inner loop: il tuning avviene solo sui dati di training del
        # fold esterno, evitando data leakage verso il test set
        pipeline = crea_pipeline()
        inner_cv = StratifiedKFold(
            n_splits=n_fold_inner, shuffle=True, random_state=random_state
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=spazio_parametri,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        migliori_parametri.append(best_params)
        print(f"      Migliori parametri: {best_params}")
        print(f"      Miglior score inner CV: {grid_search.best_score_:.4f}")

        # Il modello è già stato ri-addestrato su tutto X_train con i best params
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)

        accuratezze.append(acc)
        log_losses.append(ll)

        tutte_y_true.extend(y_test)
        tutte_y_pred.extend(y_pred)
        tutte_y_prob.extend(y_prob[:, 1])

        print(f"      Accuracy: {acc:.4f} | Log Loss: {ll:.4f}")

    print("\n" + "=" * 60)
    print("  RISULTATI NESTED CROSS-VALIDATION")
    print("=" * 60)

    acc_media = np.mean(accuratezze)
    acc_std = np.std(accuratezze)
    ll_media = np.mean(log_losses)
    ll_std = np.std(log_losses)

    print(f"\n  Accuracy:  {acc_media:.4f} ± {acc_std:.4f}")
    print(f"  Log Loss:  {ll_media:.4f} ± {ll_std:.4f}")
    print(f"\n  Accuracy per fold: {[f'{a:.4f}' for a in accuratezze]}")
    print(f"  Log Loss per fold: {[f'{l:.4f}' for l in log_losses]}")

    tutte_y_true = np.array(tutte_y_true)
    tutte_y_pred = np.array(tutte_y_pred)
    tutte_y_prob = np.array(tutte_y_prob)

    cm = confusion_matrix(tutte_y_true, tutte_y_pred)
    print(f"\n  Matrice di Confusione (aggregata su tutti i fold):")
    print(f"  {cm}")

    print(f"\n  Report di Classificazione:")
    report = classification_report(
        tutte_y_true, tutte_y_pred,
        target_names=["No Supporto", "Supporto Aggiuntivo"]
    )
    print(report)

    _plot_confusion_matrix(cm, output_dir)
    roc_auc = _plot_roc_curve(tutte_y_true, tutte_y_prob, output_dir)

    print(f"  AUC-ROC: {roc_auc:.4f}")
    print(f"\n  Grafici salvati in: {output_dir}/")

    print(f"\n  Iperparametri selezionati per ogni fold esterno:")
    for i, params in enumerate(migliori_parametri):
        params_brevi = {k.split("__")[1]: v for k, v in params.items()}
        print(f"    Fold {i + 1}: {params_brevi}")

    risultati = {
        "accuracy_media": acc_media,
        "accuracy_std": acc_std,
        "log_loss_media": ll_media,
        "log_loss_std": ll_std,
        "accuratezze_fold": accuratezze,
        "log_losses_fold": log_losses,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "migliori_parametri": migliori_parametri,
        "y_true_aggregato": tutte_y_true,
        "y_pred_aggregato": tutte_y_pred,
        "y_prob_aggregato": tutte_y_prob,
    }

    return risultati


def _plot_confusion_matrix(cm: np.ndarray, output_dir: str):
    """Genera e salva il grafico della matrice di confusione."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Supporto", "Supporto Agg."]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Matrice di Confusione — Nested CV (Aggregata)", fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Matrice di confusione salvata: {path}")


def _plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, output_dir: str
) -> float:
    """
    Genera e salva la curva ROC con AUC.

    La curva ROC mostra il trade-off tra tasso di veri positivi
    (sensibilità) e tasso di falsi positivi al variare della soglia
    di decisione. AUC = 1.0 è perfetto, AUC = 0.5 è casuale.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
            label="Random classifier (AUC = 0.5)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Tasso di Falsi Positivi (FPR)", fontsize=12)
    ax.set_ylabel("Tasso di Veri Positivi (TPR)", fontsize=12)
    ax.set_title("Curva ROC — Nested CV (Aggregata)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Curva ROC salvata: {path}")

    return roc_auc


def addestra_modello_finale(X: np.ndarray, y: np.ndarray) -> object:
    """
    Addestra il modello finale su tutto il dataset con GridSearchCV
    per selezionare i migliori iperparametri. Questo modello è usato
    per le predizioni nella simulazione.

    Returns:
        Pipeline scikit-learn addestrata con i migliori parametri.
    """
    pipeline = crea_pipeline()
    spazio_parametri = get_spazio_iperparametri()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=spazio_parametri,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X, y)

    print(f"\n  Modello finale addestrato su {X.shape[0]} campioni")
    print(f"  Migliori parametri: {grid_search.best_params_}")
    print(f"  Miglior score CV: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
