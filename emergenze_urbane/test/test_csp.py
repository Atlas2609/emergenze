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

from ricerca.csp_backtracking import risolvi_branch_and_bound, domini_iniziali, ac3
from ricerca.csp_allocazione import (
    Emergenza, Mezzo, ProblemaCSP,
    genera_soluzione_casuale, funzione_obiettivo,
    genera_vicino, risolvi_random_restart,
    risolvi_simulated_annealing, formatta_soluzione, conta_violazioni,
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

    def test_branch_and_bound_ottimo(self):
        """Il branch and bound coincide con la ricerca esaustiva."""
        import itertools
        sol, costo, _ = risolvi_branch_and_bound(self.problema)
        valori = [m.id for m in self.problema.mezzi] + ["non_assegnato"]
        ids = [e.id for e in self.problema.emergenze]
        migliore = min(
            funzione_obiettivo(self.problema, dict(zip(ids, a)))
            for a in itertools.product(valori, repeat=len(ids))
            if conta_violazioni(self.problema, dict(zip(ids, a))) == 0
        )
        self.assertAlmostEqual(costo, migliore)
        self.assertEqual(conta_violazioni(self.problema, sol), 0)

    def test_domini_consistenza_di_nodo(self):
        """Ambulanza esclusa dagli incendi; emergenze gravi sempre assegnate."""
        domini = domini_iniziali(self.problema)
        self.assertEqual(domini["EM_01"], ["MZ_01"])  # medica, gravità alta
        self.assertNotIn("MZ_01", domini["EM_02"])    # incendio
        self.assertIn("non_assegnato", domini["EM_03"])

    def test_ac3_rileva_insoddisfacibilita(self):
        """Due emergenze gravi e una sola ambulanza: AC-3 svuota un dominio."""
        domini = {"A": ["MZ_01"], "B": ["MZ_01"]}
        consistente, _ = ac3(domini, {"MZ_01": 1})
        self.assertFalse(consistente)

    def test_ac3_capacita_due(self):
        """Un mezzo di capacità 2 può servire due emergenze."""
        domini = {"A": ["MZ_02"], "B": ["MZ_02"]}
        consistente, _ = ac3(domini, {"MZ_02": 2})
        self.assertTrue(consistente)

    def test_violazione_costa_piu_di_non_assegnare(self):
        """Lasciare non assegnata un'emergenza media conviene più che violare un vincolo."""
        sol_non_assegnata = {"EM_01": "MZ_01", "EM_02": "non_assegnato", "EM_03": "MZ_02"}
        sol_incompatibile = {"EM_01": "MZ_01", "EM_02": "MZ_01", "EM_03": "MZ_02"}
        self.assertLess(funzione_obiettivo(self.problema, sol_non_assegnata),
                        funzione_obiettivo(self.problema, sol_incompatibile))

    def test_problema_senza_emergenze(self):
        vuoto = ProblemaCSP(emergenze=[], mezzi=self.problema.mezzi,
                            matrice_costi={}, compatibilita={})
        self.assertEqual(risolvi_random_restart(vuoto), ({}, 0.0))
        self.assertEqual(risolvi_simulated_annealing(vuoto), ({}, 0.0))
        self.assertEqual(risolvi_branch_and_bound(vuoto)[:2], ({}, 0.0))


if __name__ == "__main__":
    unittest.main()
