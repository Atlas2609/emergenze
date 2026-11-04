"""
test_kb.py — Test del modulo logica/kb_interface.py

Verifica il funzionamento dell'interfaccia con la KB Prolog:
- Caricamento della KB
- Query sui nodi e archi
- Regole deduttive (transitabilità, costo arco)
- Assert/retract dinamici
"""

import unittest
import os
import sys

# Aggiungi la directory del progetto al path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from logica.kb_interface import KnowledgeBase, PrologQueryError


class TestKnowledgeBase(unittest.TestCase):
    """Test per la classe KnowledgeBase."""

    @classmethod
    def setUpClass(cls):
        """Carica la KB una volta per tutti i test."""
        cls.kb = KnowledgeBase()

    def test_caricamento_kb(self):
        """Verifica che la KB sia stata caricata correttamente."""
        nodi = self.kb.get_tutti_nodi()
        self.assertGreaterEqual(len(nodi), 10, "La mappa deve avere almeno 10 nodi")
        self.assertIn("ospedale", nodi)
        self.assertIn("stazione_pompieri", nodi)

    def test_coordinate_nodi(self):
        """Verifica che i nodi abbiano coordinate valide."""
        coord = self.kb.get_coordinate("ospedale")
        self.assertIsNotNone(coord)
        self.assertEqual(len(coord), 2)
        self.assertIsInstance(coord[0], float)
        self.assertIsInstance(coord[1], float)

    def test_vicini(self):
        """Verifica la query dei vicini di un nodo."""
        vicini = self.kb.get_vicini("piazza_centrale")
        self.assertGreater(len(vicini), 0, "piazza_centrale deve avere vicini")
        for vicino, distanza in vicini:
            self.assertIsInstance(vicino, str)
            self.assertGreater(distanza, 0)

    def test_transitabilita_senza_allagamento(self):
        """
        Test regola deduttiva: con meteo sereno, tutti i veicoli
        possono transitare su tutti gli archi esistenti.
        """
        # Con condizioni iniziali (sereno/pioggia, no pioggia_forte)
        # ospedale è in zona_nord (sereno), incrocio_nord è in zona_nord (sereno)
        # Entrambi i tipi di veicolo dovrebbero poter transitare
        self.assertTrue(
            self.kb.is_transitabile("ospedale", "incrocio_nord", "ambulanza"),
            "L'ambulanza dovrebbe transitare su arco non allagato"
        )
        self.assertTrue(
            self.kb.is_transitabile("ospedale", "incrocio_nord", "autopompa"),
            "L'autopompa dovrebbe transitare su arco non allagato"
        )

    def test_transitabilita_con_allagamento(self):
        """
        Test regola deduttiva: con strada allagata, solo i mezzi
        pesanti possono transitare (eccezione esplicita in Prolog).
        """
        # Cambia il meteo a pioggia_forte per zona_nord
        self.kb.cambia_meteo("zona_nord", "pioggia_forte")

        try:
            # L'ambulanza (leggero) NON dovrebbe poter transitare
            self.assertFalse(
                self.kb.is_transitabile("ospedale", "incrocio_nord", "ambulanza"),
                "L'ambulanza NON dovrebbe transitare su strada allagata"
            )

            # L'autopompa (mezzo_pesante) DOVREBBE poter transitare
            self.assertTrue(
                self.kb.is_transitabile("ospedale", "incrocio_nord", "autopompa"),
                "L'autopompa dovrebbe transitare anche su strada allagata"
            )
        finally:
            # Ripristina condizioni originali
            self.kb.cambia_meteo("zona_nord", "sereno")

    def test_costo_arco(self):
        """Verifica il calcolo del costo dell'arco."""
        costo = self.kb.costo_arco("ospedale", "incrocio_nord", "ambulanza")
        self.assertGreater(costo, 0, "Il costo deve essere positivo")
        self.assertLess(costo, float("inf"), "Il costo non deve essere infinito")

    def test_costo_arco_non_transitabile(self):
        """Verifica che un arco non transitabile abbia costo infinito."""
        costo = self.kb.costo_arco("ospedale", "porto", "ambulanza")
        self.assertEqual(costo, float("inf"), "Archi non collegati devono avere costo infinito")

    def test_distanza_euclidea(self):
        """Verifica il calcolo della distanza euclidea."""
        dist = self.kb.distanza_euclidea("ospedale", "stazione_pompieri")
        self.assertGreater(dist, 0)
        # Distanza da ospedale (2,8) a stazione_pompieri (8,8) = 6.0
        self.assertAlmostEqual(dist, 6.0, places=1)

    def test_assert_retract(self):
        """Test dell'asserzione e ritrattazione dinamica."""
        # Asserisci una condizione
        self.kb.cambia_meteo("zona_nord", "alluvione")
        condizione = self.kb.get_condizione_meteo("zona_nord")
        self.assertEqual(condizione, "alluvione")

        # Ripristina
        self.kb.cambia_meteo("zona_nord", "sereno")
        condizione = self.kb.get_condizione_meteo("zona_nord")
        self.assertEqual(condizione, "sereno")

    def test_compatibilita_mezzo(self):
        """Verifica la compatibilità mezzo-emergenza."""
        self.assertTrue(self.kb.is_compatibile("ambulanza", "medica"))
        self.assertTrue(self.kb.is_compatibile("autopompa", "incendio"))
        self.assertFalse(self.kb.is_compatibile("ambulanza", "incendio"))

    def test_strada_allagata_deduzione(self):
        """
        Test della deduzione: strada_allagata/1 NON è un fatto,
        è DERIVATA dalla regola che combina zona/2 e condizione_meteo/2.
        """
        # Inizialmente non allagato
        self.assertFalse(self.kb.is_strada_allagata("ospedale"))

        # Cambia meteo → la regola deve derivare il fatto
        self.kb.cambia_meteo("zona_nord", "pioggia_forte")
        self.assertTrue(
            self.kb.is_strada_allagata("ospedale"),
            "ospedale (zona_nord) dovrebbe risultare allagato con pioggia_forte"
        )

        # Ripristina
        self.kb.cambia_meteo("zona_nord", "sereno")
        self.assertFalse(self.kb.is_strada_allagata("ospedale"))


if __name__ == "__main__":
    unittest.main()
