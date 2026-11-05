"""
test_ml.py — Test del modulo Machine Learning

Verifica:
- Generazione del dataset sintetico
- Valori mancanti e loro imputazione nella pipeline
- Pipeline dei modelli e rete bayesiana
- Nested Cross-Validation
"""

import unittest
import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from machine_learning.dataset_synth import (
    genera_dataset, prepara_features, ritardo_da_costo, limite_teorico,
)
from machine_learning.modello_predittivo import (
    crea_pipeline, get_spazio_iperparametri, get_modelli,
)
from machine_learning.rete_bayesiana import ReteBayesianaClassificatore
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

    def test_prepara_features(self):
        """Feature categoriche separate dal target, una colonna per feature."""
        df = genera_dataset(n_campioni=100)
        X, y, nomi = prepara_features(df)
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(list(X.columns), nomi)

    def test_valori_mancanti_presenti(self):
        """Il generatore inserisce valori mancanti da gestire nel cleaning."""
        df = genera_dataset(n_campioni=1000, quota_mancanti=0.05)
        self.assertGreater(df.isna().sum().sum(), 0)
        self.assertEqual(df["supporto_aggiuntivo"].isna().sum(), 0)

    def test_ritardo_da_costo(self):
        """Discretizzazione del costo A* in ritardo di arrivo."""
        self.assertEqual(ritardo_da_costo(3.0), "basso")
        self.assertEqual(ritardo_da_costo(10.0), "medio")
        self.assertEqual(ritardo_da_costo(float("inf")), "alto")

    def test_limite_teorico(self):
        """Il classificatore ottimo ha AUC tra 0.5 e 1 (rumore irriducibile)."""
        df = genera_dataset(n_campioni=2000, con_probabilita_vera=True)
        limite = limite_teorico(df)
        self.assertGreater(limite["roc_auc"], 0.6)
        self.assertLess(limite["roc_auc"], 0.9)


class TestModelloPredittivo(unittest.TestCase):
    """Test per la pipeline del modello."""

    def test_crea_pipeline(self):
        """Verifica la creazione della pipeline."""
        pipeline = crea_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertIn("preprocessing", pipeline.named_steps)
        self.assertIn("classificatore", pipeline.named_steps)

    def test_tutti_i_modelli_gestiscono_mancanti(self):
        """Ogni pipeline del catalogo si addestra su dati con NaN."""
        df = genera_dataset(n_campioni=300, quota_mancanti=0.1)
        X, y, _ = prepara_features(df)
        for nome, (pipeline, _) in get_modelli().items():
            pipeline.fit(X, y)
            self.assertEqual(len(pipeline.predict(X)), 300, nome)

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
        self.assertIn("roc_auc_media", risultati)
        self.assertIn("f1_std", risultati)

        # Verifica intervalli ragionevoli
        self.assertGreater(risultati["accuracy_media"], 0.3, "Accuracy troppo bassa")
        self.assertLess(risultati["accuracy_media"], 1.0, "Accuracy sospettosamente perfetta")
        self.assertGreater(risultati["roc_auc_media"], 0.5, "AUC non migliore del caso")

    def test_modello_finale(self):
        """Test dell'addestramento del modello finale."""
        df = genera_dataset(n_campioni=200, random_state=42)
        X, y, _ = prepara_features(df)
        modello = addestra_modello_finale(X, y)
        self.assertIsNotNone(modello)

        # Verifica che il modello possa predire
        pred = modello.predict(X[:10])
        self.assertEqual(len(pred), 10)


class TestReteBayesiana(unittest.TestCase):
    """Test per la rete bayesiana."""

    @classmethod
    def setUpClass(cls):
        df = genera_dataset(n_campioni=1000, random_state=1)
        cls.X, cls.y, _ = prepara_features(df)
        cls.rete = ReteBayesianaClassificatore("esperta").fit(cls.X, cls.y)

    def test_probabilita_valide(self):
        prob = self.rete.predict_proba(self.X.head(50))
        self.assertEqual(prob.shape, (50, 2))
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-9)

    def test_evidenza_parziale(self):
        """La rete risponde anche senza conoscere tutte le variabili."""
        p_alta = self.rete.probabilita({"gravita": "alta"})
        p_bassa = self.rete.probabilita({"gravita": "bassa"})
        self.assertGreater(p_alta, p_bassa)

    def test_struttura_appresa(self):
        """La ricerca della struttura produce una rete valida."""
        rete = ReteBayesianaClassificatore("appresa").fit(self.X, self.y)
        self.assertEqual(len(rete.predict(self.X.head(10))), 10)


class TestRobustezza(unittest.TestCase):
    """Valori fuori dominio non devono far fallire nessun modello."""

    def test_valori_sconosciuti(self):
        import pandas as pd
        from machine_learning.rete_bayesiana import get_modelli_bayesiani
        X, y, _ = prepara_features(genera_dataset(n_campioni=400))
        riga = pd.DataFrame([{"meteo": "sconosciuta", "orario": "mattina", "zona": None,
                              "gravita": "alta", "tipo_emergenza": "medica",
                              "ritardo_arrivo": "alto"}])
        modelli = {**get_modelli(), **get_modelli_bayesiani()}
        for nome, (modello, _) in modelli.items():
            prob = modello.fit(X, y).predict_proba(riga)
            self.assertEqual(prob.shape, (1, 2), nome)


if __name__ == "__main__":
    unittest.main()
