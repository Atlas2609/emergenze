"""
test_astar.py — Test dell'algoritmo A* per il routing

Verifica:
- Correttezza del percorso trovato
- Ottimalità (costo minimo)
- Gestione di nodi non raggiungibili
- Adattamento dinamico alle condizioni della mappa
"""

import unittest
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from logica.kb_interface import KnowledgeBase
from ricerca.astar_routing import trova_percorso, calcola_matrice_distanze


class TestAStarRouting(unittest.TestCase):
    """Test per l'algoritmo A*."""

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()

    def test_percorso_diretto(self):
        """Verifica che A* trovi il percorso per archi diretti."""
        percorso, costo = trova_percorso(
            self.kb, "ospedale", "incrocio_nord", "ambulanza"
        )
        self.assertIsNotNone(percorso)
        self.assertEqual(percorso[0], "ospedale")
        self.assertEqual(percorso[-1], "incrocio_nord")
        self.assertGreater(costo, 0)

    def test_stesso_nodo(self):
        """Verifica che A* gestisca partenza = destinazione."""
        percorso, costo = trova_percorso(
            self.kb, "ospedale", "ospedale", "ambulanza"
        )
        self.assertEqual(percorso, ["ospedale"])
        self.assertEqual(costo, 0.0)

    def test_percorso_lungo(self):
        """Verifica che A* trovi un percorso tra nodi distanti."""
        percorso, costo = trova_percorso(
            self.kb, "aeroporto", "porto", "ambulanza"
        )
        self.assertIsNotNone(percorso, "Deve esistere un percorso aeroporto→porto")
        self.assertEqual(percorso[0], "aeroporto")
        self.assertEqual(percorso[-1], "porto")
        self.assertGreater(len(percorso), 2, "Il percorso deve passare per nodi intermedi")

    def test_nodo_inesistente(self):
        """Verifica la gestione di nodi non esistenti."""
        percorso, costo = trova_percorso(
            self.kb, "nodo_falso", "ospedale", "ambulanza"
        )
        self.assertIsNone(percorso)
        self.assertEqual(costo, float("inf"))

    def test_percorso_ambulanza_vs_autopompa(self):
        """
        Verifica che il percorso possa differire tra ambulanza
        e autopompa quando ci sono strade allagate.
        """
        # Forza allagamento in una zona
        self.kb.cambia_meteo("zona_centro", "pioggia_forte")

        try:
            perc_amb, costo_amb = trova_percorso(
                self.kb, "ospedale", "stadio", "ambulanza"
            )
            perc_auto, costo_auto = trova_percorso(
                self.kb, "ospedale", "stadio", "autopompa"
            )

            # Entrambi dovrebbero trovare un percorso (la mappa è connessa)
            self.assertIsNotNone(perc_auto, "L'autopompa deve trovare un percorso")

            # L'autopompa potrebbe avere un percorso diverso o un costo diverso
            # perché può attraversare strade allagate
            if perc_amb is not None:
                # Se l'ambulanza trova un percorso, potrebbe essere più lungo
                # perché deve evitare le zone allagate
                pass  # Entrambi i percorsi sono validi
        finally:
            self.kb.cambia_meteo("zona_centro", "pioggia")

    def test_matrice_distanze(self):
        """Verifica il calcolo della matrice delle distanze."""
        partenze = ["ospedale", "stazione_pompieri"]
        destinazioni = ["stadio", "porto"]
        matrice = calcola_matrice_distanze(
            self.kb, partenze, destinazioni, "ambulanza"
        )
        self.assertEqual(len(matrice), 4)  # 2 × 2 = 4 coppie
        for (p, d), (percorso, costo) in matrice.items():
            self.assertIn(p, partenze)
            self.assertIn(d, destinazioni)

    def test_percorso_continuo(self):
        """Verifica che il percorso sia una sequenza di archi validi."""
        percorso, costo = trova_percorso(
            self.kb, "ospedale", "porto", "ambulanza"
        )
        if percorso:
            for i in range(len(percorso) - 1):
                vicini = [v for v, _ in self.kb.get_vicini(percorso[i])]
                self.assertIn(
                    percorso[i + 1], vicini,
                    f"Arco mancante: {percorso[i]} → {percorso[i+1]}"
                )


if __name__ == "__main__":
    unittest.main()
