"""
test_ml.py — Test del modulo Machine Learning

Verifica:
- Generazione del dataset sintetico
- Encoding delle feature
- Pipeline del modello
- Nested Cross-Validation
"""

import unittest
import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from machine_learning.dataset_synth import genera_dataset, prepara_features
from machine_learning.modello_predittivo import (
    crea_pipeline, get_spazio_iperparametri, descrivi_modello,
)
from machine_learning.validazione import esegui_nested_cv, addestra_modello_finale


class TestDatasetSynth(unittest.TestCase):
    """Test per la generazione del dataset sintetico."""

    def test_dimensione_dataset(self):
        """Verifica la dimensione del dataset generato."""
        df = genera_dataset(n_campioni=500)
        self.assertEqual(df.shape[0], 500)
        self.assertGreater(df.shape[1], 1)

    def test_colonne_presenti(self):
        """Verifica che tutte le colonne richieste siano presenti."""
        df = genera_dataset(n_campioni=100)
        colonne_richieste = [
            "meteo", "orario", "zona", "gravita",
            "tipo_emergenza", "supporto_aggiuntivo"
        ]
        for col in colonne_richieste:
            self.assertIn(col, df.columns, f"Colonna mancante: {col}")

    def test_target_binario(self):
        """Verifica che il target sia binario (0 o 1)."""
        df = genera_dataset(n_campioni=500)
        valori_unici = df["supporto_aggiuntivo"].unique()
        self.assertTrue(set(valori_unici).issubset({0, 1}))

    def test_distribuzione_non_degenere(self):
        """
        Verifica che il target NON sia tutto 0 o tutto 1
        (il segnale deve essere apprendibile, non degenere).
        """
        df = genera_dataset(n_campioni=500, random_state=42)
        conteggi = df["supporto_aggiuntivo"].value_counts()
        self.assertGreater(conteggi.get(0, 0), 50, "Troppi pochi campioni negativi")
        self.assertGreater(conteggi.get(1, 0), 50, "Troppi pochi campioni positivi")

    def test_riproducibilita(self):
        """Verifica la riproducibilità con lo stesso seed."""
        df1 = genera_dataset(n_campioni=100, random_state=42)
        df2 = genera_dataset(n_campioni=100, random_state=42)
        self.assertTrue(df1.equals(df2))

    def test_encoding_features(self):
        """Verifica l'encoding One-Hot delle feature."""
        df = genera_dataset(n_campioni=100)
        X, y, nomi = prepara_features(df)
        self.assertEqual(X.shape[0], 100)
        self.assertGreater(X.shape[1], 5, "One-Hot dovrebbe espandere le colonne")
        self.assertEqual(len(y), 100)
        self.assertEqual(len(nomi), X.shape[1])


class TestModelloPredittivo(unittest.TestCase):
    """Test per la pipeline del modello."""

    def test_crea_pipeline(self):
        """Verifica la creazione della pipeline."""
        pipeline = crea_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertIn("scaler", pipeline.named_steps)
        self.assertIn("classificatore", pipeline.named_steps)

    def test_spazio_iperparametri(self):
        """Verifica lo spazio degli iperparametri."""
        spazio = get_spazio_iperparametri()
        self.assertIn("classificatore__max_depth", spazio)
        self.assertIn("classificatore__criterion", spazio)
        self.assertGreater(len(spazio), 0)

    def test_addestramento_pipeline(self):
        """Verifica che la pipeline possa essere addestrata."""
        df = genera_dataset(n_campioni=200, random_state=42)
        X, y, _ = prepara_features(df)
        pipeline = crea_pipeline()
        pipeline.fit(X, y)
        predizioni = pipeline.predict(X)
        self.assertEqual(len(predizioni), 200)

    def test_predizioni_probabilistiche(self):
        """Verifica le predizioni probabilistiche."""
        df = genera_dataset(n_campioni=200, random_state=42)
        X, y, _ = prepara_features(df)
        pipeline = crea_pipeline()
        pipeline.fit(X, y)
        prob = pipeline.predict_proba(X)
        self.assertEqual(prob.shape, (200, 2))
        # Le probabilità devono sommare a 1
        somme = np.sum(prob, axis=1)
        np.testing.assert_allclose(somme, 1.0, atol=1e-10)


class TestValidazione(unittest.TestCase):
    """Test per la Nested Cross-Validation."""

    def test_nested_cv(self):
        """
        Test della Nested CV (versione ridotta per velocità nei test).
        """
        df = genera_dataset(n_campioni=200, random_state=42)
        X, y, _ = prepara_features(df)

        output_dir = os.path.join(PROJECT_DIR, "test", "output_test")
        os.makedirs(output_dir, exist_ok=True)

        risultati = esegui_nested_cv(
            X, y,
            n_fold_outer=3,  # Ridotto per velocità
            n_fold_inner=2,
            random_state=42,
            output_dir=output_dir,
        )

        # Verifica che le metriche siano presenti
        self.assertIn("accuracy_media", risultati)
        self.assertIn("accuracy_std", risultati)
        self.assertIn("log_loss_media", risultati)
        self.assertIn("roc_auc", risultati)

        # Verifica intervalli ragionevoli
        self.assertGreater(risultati["accuracy_media"], 0.3, "Accuracy troppo bassa")
        self.assertLess(risultati["accuracy_media"], 1.0, "Accuracy sospettosamente perfetta")
        self.assertGreater(risultati["roc_auc"], 0.4, "AUC troppo bassa")

    def test_modello_finale(self):
        """Test dell'addestramento del modello finale."""
        df = genera_dataset(n_campioni=200, random_state=42)
        X, y, _ = prepara_features(df)
        modello = addestra_modello_finale(X, y)
        self.assertIsNotNone(modello)

        # Verifica che il modello possa predire
        pred = modello.predict(X[:10])
        self.assertEqual(len(pred), 10)


if __name__ == "__main__":
    unittest.main()
