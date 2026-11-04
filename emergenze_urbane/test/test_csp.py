"""
test_csp.py — Test del modulo CSP per l'allocazione mezzi-emergenze

Verifica:
- Correttezza della funzione obiettivo
- Funzionamento di Random Restart Hill Climbing
- Funzionamento di Simulated Annealing
- Rispetto dei vincoli (compatibilità, capacità)
"""

import unittest
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ricerca.csp_allocazione import (
    Emergenza, Mezzo, ProblemaCSP,
    genera_soluzione_casuale, funzione_obiettivo,
    genera_vicino, risolvi_random_restart,
    risolvi_simulated_annealing, formatta_soluzione,
)


def _crea_problema_test():
    """Crea un problema CSP di test semplice."""
    emergenze = [
        Emergenza("EM_01", "ospedale", "medica", "alta"),
        Emergenza("EM_02", "stadio", "incendio", "media"),
        Emergenza("EM_03", "porto", "allagamento", "bassa"),
    ]
    mezzi = [
        Mezzo("MZ_01", "ospedale", "ambulanza", 1),
        Mezzo("MZ_02", "stazione_pompieri", "autopompa", 2),
        Mezzo("MZ_03", "stazione_pompieri", "autoscala", 1),
    ]
    matrice_costi = {
        ("MZ_01", "EM_01"): 0.0,
        ("MZ_01", "EM_02"): 8.0,
        ("MZ_01", "EM_03"): 15.0,
        ("MZ_02", "EM_01"): 10.0,
        ("MZ_02", "EM_02"): 5.0,
        ("MZ_02", "EM_03"): 7.0,
        ("MZ_03", "EM_01"): 10.0,
        ("MZ_03", "EM_02"): 5.0,
        ("MZ_03", "EM_03"): 7.0,
    }
    compatibilita = {
        ("ambulanza", "medica"): True,
        ("ambulanza", "incendio"): False,
        ("ambulanza", "allagamento"): False,
        ("ambulanza", "incidente"): True,
        ("autopompa", "medica"): False,
        ("autopompa", "incendio"): True,
        ("autopompa", "allagamento"): True,
        ("autopompa", "incidente"): True,
        ("autoscala", "medica"): False,
        ("autoscala", "incendio"): True,
        ("autoscala", "allagamento"): False,
        ("autoscala", "soccorso_altezza"): True,
    }
    return ProblemaCSP(
        emergenze=emergenze,
        mezzi=mezzi,
        matrice_costi=matrice_costi,
        compatibilita=compatibilita,
    )


class TestCSP(unittest.TestCase):
    """Test per il CSP e gli algoritmi di ricerca locale."""

    def setUp(self):
        self.problema = _crea_problema_test()

    def test_genera_soluzione_casuale(self):
        """Verifica che la soluzione casuale abbia il formato corretto."""
        sol = genera_soluzione_casuale(self.problema)
        self.assertEqual(len(sol), 3, "Una variabile per ogni emergenza")
        for em_id in sol:
            self.assertIn(em_id, ["EM_01", "EM_02", "EM_03"])

    def test_funzione_obiettivo(self):
        """Verifica che la funzione obiettivo restituisca un valore finito."""
        sol = genera_soluzione_casuale(self.problema)
        costo = funzione_obiettivo(self.problema, sol)
        self.assertIsInstance(costo, float)
        self.assertGreaterEqual(costo, 0)

    def test_soluzione_ottima_nota(self):
        """
        Una soluzione con assegnamenti compatibili dovrebbe avere
        un costo inferiore a una con violazioni.
        """
        # Soluzione con compatibilità corretta
        sol_buona = {
            "EM_01": "MZ_01",  # ambulanza → medica (compatibile)
            "EM_02": "MZ_02",  # autopompa → incendio (compatibile)
            "EM_03": "MZ_02",  # autopompa → allagamento (compatibile, capacità 2)
        }
        # Soluzione con violazione di compatibilità
        sol_cattiva = {
            "EM_01": "MZ_02",  # autopompa → medica (NON compatibile)
            "EM_02": "MZ_01",  # ambulanza → incendio (NON compatibile)
            "EM_03": "MZ_03",  # autoscala → allagamento (NON compatibile)
        }
        costo_buona = funzione_obiettivo(self.problema, sol_buona)
        costo_cattiva = funzione_obiettivo(self.problema, sol_cattiva)
        self.assertLess(
            costo_buona, costo_cattiva,
            "Soluzione compatibile deve costare meno di una con violazioni"
        )

    def test_genera_vicino(self):
        """Verifica che il vicino differisca per un solo assegnamento."""
        sol = genera_soluzione_casuale(self.problema)
        vicino = genera_vicino(self.problema, sol)
        differenze = sum(1 for k in sol if sol[k] != vicino[k])
        self.assertLessEqual(differenze, 1, "Il vicino deve differire per al più 1 assegnamento")

    def test_random_restart_hill_climbing(self):
        """Verifica che RRHC trovi una soluzione valida."""
        sol, costo = risolvi_random_restart(
            self.problema, n_restart=10, max_iterazioni_per_restart=200
        )
        self.assertIsNotNone(sol)
        self.assertIsInstance(costo, float)
        self.assertGreaterEqual(costo, 0)
        self.assertEqual(len(sol), 3)

    def test_simulated_annealing(self):
        """Verifica che SA trovi una soluzione valida."""
        sol, costo = risolvi_simulated_annealing(
            self.problema,
            temperatura_iniziale=50.0,
            raffreddamento=0.99,
            max_iterazioni=5000,
        )
        self.assertIsNotNone(sol)
        self.assertIsInstance(costo, float)
        self.assertGreaterEqual(costo, 0)
        self.assertEqual(len(sol), 3)

    def test_rrhc_migliora_random(self):
        """Verifica che RRHC produca soluzioni migliori di quelle casuali."""
        costi_random = []
        for _ in range(20):
            sol = genera_soluzione_casuale(self.problema)
            costi_random.append(funzione_obiettivo(self.problema, sol))
        media_random = sum(costi_random) / len(costi_random)

        sol_rrhc, costo_rrhc = risolvi_random_restart(
            self.problema, n_restart=20
        )
        self.assertLessEqual(
            costo_rrhc, media_random,
            "RRHC dovrebbe trovare soluzioni migliori della media casuale"
        )

    def test_formatta_soluzione(self):
        """Verifica che la formattazione produca output leggibile."""
        sol = {"EM_01": "MZ_01", "EM_02": "MZ_02", "EM_03": "non_assegnato"}
        output = formatta_soluzione(self.problema, sol, 42.0)
        self.assertIn("42.00", output)
        self.assertIn("NON ASSEGNATO", output)


if __name__ == "__main__":
    unittest.main()
