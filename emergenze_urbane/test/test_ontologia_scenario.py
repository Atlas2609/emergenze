"""
test_ontologia_scenario.py — Test dell'ontologia OWL e del generatore di scenari

Verifica:
- Classificazione dell'ontologia con il ragionatore HermiT
- Coerenza tra ontologia (mondo aperto) e KB Prolog (mondo chiuso)
- Generazione di scenari coerenti con la KB
"""

import unittest
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from logica.kb_interface import KnowledgeBase
from ontologia.ontologia_emergenze import (
    crea_ontologia, ragiona, sottoclassi_inferite, verifica_coerenza_con_kb,
)
from simulazione.scenario_manager import genera_scenario


class TestOntologia(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()
        cls.onto = ragiona(crea_ontologia(cls.kb))

    def test_classificazione_inferita(self):
        """HermiT deduce le sottoclassi delle classi definite."""
        self.assertEqual(sottoclassi_inferite(self.onto, "MezzoAntincendio"),
                         ["Autopompa", "Autoscala"])
        self.assertEqual(sottoclassi_inferite(self.onto, "MezzoSoccorsoSanitario"),
                         ["Ambulanza"])

    def test_coerenza_con_kb(self):
        self.assertEqual(verifica_coerenza_con_kb(self.onto, self.kb), [])

    def test_abox_popolata_dalla_kb(self):
        nodi = {i.name for i in self.onto.Nodo.instances()}
        self.assertEqual(nodi, set(self.kb.get_tutti_nodi()))


class TestScenario(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()

    def test_composizione_flotta(self):
        """6 mezzi: 3 ambulanze, 2 autopompe, 1 autoscala."""
        _, mezzi = genera_scenario(self.kb, n_emergenze=3, n_mezzi=6, random_state=0)
        tipi = [m.tipo for m in mezzi]
        self.assertEqual(tipi.count("ambulanza"), 3)
        self.assertEqual(tipi.count("autopompa"), 2)
        self.assertEqual(tipi.count("autoscala"), 1)

    def test_scenario_coerente_con_kb(self):
        emergenze, mezzi = genera_scenario(self.kb, 10, 8, random_state=1)
        nodi = set(self.kb.get_tutti_nodi())
        for em in emergenze:
            self.assertIn(em.posizione, nodi)
            self.assertIn(em.tipo, self.kb.get_tipi_emergenza())
        for m in mezzi:
            self.assertIn(m.posizione, self.kb.get_basi(m.tipo))
            self.assertEqual(m.capacita, self.kb.get_capacita(m.tipo))

    def test_riproducibilita(self):
        a = genera_scenario(self.kb, 5, 5, random_state=3)
        b = genera_scenario(self.kb, 5, 5, random_state=3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
