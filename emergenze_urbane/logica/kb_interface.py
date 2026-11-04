"""
kb_interface.py — Interfaccia Python ↔ SWI-Prolog via pyswip

Questo modulo incapsula l'interazione con il motore di inferenza
Prolog, permettendo al sistema Python di sfruttare la deduzione
logica della KB senza duplicare la logica in Python. Ogni query di
transitabilità o costo è risolta dal motore Prolog attraverso le
clausole di Horn definite nella KB.
"""

import os
from pyswip import Prolog


class PrologQueryError(Exception):
    """Eccezione per errori nelle query Prolog."""
    pass


class KnowledgeBase:
    """
    Wrapper attorno a pyswip.Prolog che carica la KB e espone
    metodi Python per interrogare e modificare dinamicamente
    lo stato della mappa cittadina.
    """

    def __init__(self, prolog_file: str = None):
        """
        Inizializza il motore Prolog e carica il file .pl della mappa.

        Args:
            prolog_file: percorso al file Prolog. Se None, usa il
                         percorso di default relativo al progetto.
        """
        self.prolog = Prolog()

        if prolog_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prolog_file = os.path.join(base_dir, "prolog", "mappa_emergenze.pl")

        prolog_file = prolog_file.replace("\\", "/")

        if not os.path.exists(prolog_file.replace("/", os.sep)):
            raise FileNotFoundError(
                f"File Prolog non trovato: {prolog_file}"
            )

        try:
            self.prolog.consult(prolog_file)
        except Exception as e:
            raise PrologQueryError(
                f"Errore nel caricamento della KB Prolog: {e}"
            )

    def _query(self, query_str: str) -> list:
        """
        Esegue una query Prolog e restituisce la lista dei risultati.

        Args:
            query_str: stringa della query Prolog (senza punto finale).

        Returns:
            Lista di dizionari con le variabili istanziate.

        Raises:
            PrologQueryError: se la query è malformata o fallisce.
        """
        try:
            results = list(self.prolog.query(query_str))
            return results
        except Exception as e:
            raise PrologQueryError(
                f"Errore nella query Prolog '{query_str}': {e}"
            )

    def _query_bool(self, query_str: str) -> bool:
        """
        Esegue una query Prolog e restituisce True se ha almeno
        una soluzione, False altrimenti.
        """
        try:
            results = list(self.prolog.query(query_str))
            return len(results) > 0
        except Exception:
            return False

    # ================================================================
    # METODI DI INTERROGAZIONE (sfruttano le regole deduttive Prolog)
    # ================================================================

    def is_transitabile(self, nodo1: str, nodo2: str, tipo_veicolo: str) -> bool:
        """
        Verifica se l'arco (nodo1, nodo2) è transitabile per il tipo
        di veicolo specificato. La transitabilità è dedotta dalla KB
        Prolog tramite le clausole di Horn, non è un fatto statico:
        il motore applica la regola transitabile/3 combinando arco/3,
        strada_allagata/1 e tipo_veicolo/2.
        """
        query = f"transitabile({nodo1}, {nodo2}, {tipo_veicolo})"
        return self._query_bool(query)

    def costo_arco(self, nodo1: str, nodo2: str, tipo_veicolo: str) -> float:
        """
        Calcola il costo dell'arco (nodo1, nodo2) per il tipo di veicolo.
        Il costo è derivato per deduzione dalla regola costo_arco/4,
        che compone distanza base, fattore meteo e fattore allagamento.

        Returns:
            Costo dell'arco come float, o float('inf') se non transitabile.
        """
        query = f"costo_arco({nodo1}, {nodo2}, {tipo_veicolo}, Costo)"
        results = self._query(query)
        if results:
            return float(results[0]["Costo"])
        return float("inf")

    def get_vicini(self, nodo: str) -> list:
        """
        Restituisce la lista dei nodi adiacenti al nodo dato,
        con le relative distanze base.

        Returns:
            Lista di tuple (vicino, distanza).
        """
        query = f"vicino({nodo}, Vicino, Distanza)"
        results = self._query(query)
        vicini = []
        for r in results:
            vicini.append((str(r["Vicino"]), float(r["Distanza"])))
        return vicini

    def get_vicini_transitabili(self, nodo: str, tipo_veicolo: str) -> list:
        """
        Restituisce solo i vicini raggiungibili dal tipo di veicolo,
        con il costo effettivo (maggiorato dalle condizioni).

        Returns:
            Lista di tuple (vicino, costo_effettivo).
        """
        query = f"costo_arco({nodo}, Vicino, {tipo_veicolo}, Costo)"
        results = self._query(query)
        vicini = []
        seen = set()
        for r in results:
            vicino = str(r["Vicino"])
            if vicino not in seen:
                seen.add(vicino)
                vicini.append((vicino, float(r["Costo"])))
        return vicini

    def get_coordinate(self, nodo: str) -> tuple:
        """
        Recupera le coordinate (X, Y) di un nodo dalla KB Prolog.

        Returns:
            Tupla (x, y) come float, o None se il nodo non esiste.
        """
        query = f"coordinate({nodo}, X, Y)"
        results = self._query(query)
        if results:
            return (float(results[0]["X"]), float(results[0]["Y"]))
        return None

    def distanza_euclidea(self, nodo1: str, nodo2: str) -> float:
        """
        Calcola la distanza euclidea tra due nodi usando il predicato
        Prolog distanza_euclidea/3.
        """
        query = f"distanza_euclidea({nodo1}, {nodo2}, Dist)"
        results = self._query(query)
        if results:
            return float(results[0]["Dist"])
        return float("inf")

    def get_tutti_nodi(self) -> list:
        """Restituisce la lista di tutti i nodi della mappa."""
        results = self._query("nodo(X)")
        return [str(r["X"]) for r in results]

    def is_compatibile(self, tipo_mezzo: str, tipo_emergenza: str) -> bool:
        """
        Verifica la compatibilità tra tipo di mezzo e tipo di emergenza
        tramite la regola Prolog compatibile/2.
        """
        query = f"mezzo_compatibile({tipo_mezzo}, {tipo_emergenza})"
        return self._query_bool(query)

    def get_capacita(self, tipo_mezzo: str) -> int:
        """Restituisce la capacità di un tipo di mezzo."""
        query = f"capacita_mezzo({tipo_mezzo}, C)"
        results = self._query(query)
        if results:
            return int(results[0]["C"])
        return 0

    def get_tipo_veicolo(self, nome_mezzo: str) -> str:
        """Restituisce la categoria del veicolo (leggero/mezzo_pesante)."""
        query = f"tipo_veicolo({nome_mezzo}, Tipo)"
        results = self._query(query)
        if results:
            return str(results[0]["Tipo"])
        return None

    # ================================================================
    # METODI DI MODIFICA DINAMICA (assert/retract)
    # La KB è dinamica: il mondo cambia durante la simulazione e la
    # KB riflette questi cambiamenti tramite assert/retract runtime.
    # ================================================================

    def assert_evento(self, predicato: str):
        """
        Asserisce un nuovo fatto nella KB Prolog.
        Es: assert_evento("condizione_meteo(zona_nord, pioggia_forte)")
        """
        try:
            list(self.prolog.query(f"assert({predicato})"))
        except Exception as e:
            raise PrologQueryError(
                f"Errore nell'asserzione di '{predicato}': {e}"
            )

    def retract_evento(self, predicato: str):
        """
        Ritratta (rimuove) un fatto dalla KB Prolog.
        Es: retract_evento("condizione_meteo(zona_nord, sereno)")
        """
        try:
            list(self.prolog.query(f"retract({predicato})"))
        except Exception as e:
            raise PrologQueryError(
                f"Errore nella ritrattazione di '{predicato}': {e}"
            )

    def cambia_meteo(self, zona: str, nuova_condizione: str):
        """
        Cambia la condizione meteo di una zona: ritratta la vecchia
        condizione e asserisce la nuova. Questo causa una cascata di
        inferenze nelle regole strada_allagata, transitabile e costo_arco.
        """
        try:
            list(self.prolog.query(
                f"retract(condizione_meteo({zona}, _))"
            ))
        except Exception:
            pass

        self.assert_evento(f"condizione_meteo({zona}, {nuova_condizione})")

    def get_condizione_meteo(self, zona: str) -> str:
        """Restituisce la condizione meteo corrente di una zona."""
        query = f"condizione_meteo({zona}, Cond)"
        results = self._query(query)
        if results:
            return str(results[0]["Cond"])
        return "sconosciuta"

    def is_strada_allagata(self, nodo: str) -> bool:
        """
        Verifica se un nodo è su una strada allagata.
        Fatto derivato dalla regola strada_allagata/1, non memorizzato.
        """
        return self._query_bool(f"strada_allagata({nodo})")
