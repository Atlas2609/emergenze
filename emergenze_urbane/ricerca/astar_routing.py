"""
astar_routing.py — Algoritmo A* per il routing dei veicoli di soccorso

Ricerca informata: A* combina il costo del cammino già percorso (g)
con una stima euristica del costo rimanente (h) per trovare il
percorso ottimo. L'euristica usata è la distanza euclidea: è
ammissibile perché ogni arco della KB ha distanza >= della distanza
euclidea tra i suoi estremi e i fattori meteo/allagamento sono >= 1,
quindi h(n) non sovrastima mai il costo reale. Per la disuguaglianza
triangolare è anche consistente (monotona), per cui un nodo estratto
dalla frontiera non deve mai essere riaperto.

Con usa_euristica=False la stessa procedura diventa la ricerca a costo
minimo (lowest-cost-first, h = 0), usata come termine di confronto.

Il costo e la transitabilità di ogni arco vengono chiesti dinamicamente
alla KB Prolog ad ogni step di espansione, così il routing riflette lo
stato in tempo reale della mappa (strade allagate, condizioni meteo,
tipo di veicolo).
"""

import heapq
from logica.kb_interface import KnowledgeBase


class NodoAStar:
    """Nodo per la coda a priorità di A*."""

    def __init__(self, nome: str, g: float, h: float, genitore=None):
        self.nome = nome
        self.g = g
        self.h = h
        self.f = g + h
        self.genitore = genitore

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.nome == other.nome

    def __hash__(self):
        return hash(self.nome)


def _euristica(kb: KnowledgeBase, nodo: str, destinazione: str) -> float:
    """
    Calcola l'euristica h(n) come distanza euclidea in linea d'aria.
    L'euristica euclidea è ammissibile e consistente, il che garantisce
    che A* trovi il percorso di costo minimo.

    Args:
        kb: istanza KnowledgeBase per ottenere le coordinate.
        nodo: nodo corrente.
        destinazione: nodo obiettivo.

    Returns:
        Distanza euclidea come float.
    """
    return kb.distanza_euclidea(nodo, destinazione)


def _ricostruisci_percorso(nodo: NodoAStar) -> list:
    """Ricostruisce il percorso dal nodo obiettivo al nodo iniziale."""
    percorso = []
    corrente = nodo
    while corrente is not None:
        percorso.append(corrente.nome)
        corrente = corrente.genitore
    percorso.reverse()
    return percorso


def trova_percorso(
    kb: KnowledgeBase,
    partenza: str,
    destinazione: str,
    tipo_veicolo: str
) -> tuple:
    """
    Interfaccia principale: restituisce (percorso, costo_totale).
    Vedi ricerca_percorso per i dettagli dell'algoritmo.
    """
    risultato = ricerca_percorso(kb, partenza, destinazione, tipo_veicolo)
    return (risultato["percorso"], risultato["costo"])


def ricerca_percorso(
    kb: KnowledgeBase,
    partenza: str,
    destinazione: str,
    tipo_veicolo: str,
    usa_euristica: bool = True,
) -> dict:
    """
    Trova il percorso ottimo da partenza a destinazione per il
    tipo di veicolo specificato usando l'algoritmo A*.

    Ad ogni espansione di nodo, la transitabilità e il costo dell'arco
    vengono chiesti alla KB Prolog: se durante la ricerca le condizioni
    cambiano (es. strada allagata), A* si adatta automaticamente perché
    interroga la KB in tempo reale.

    Args:
        kb: istanza KnowledgeBase con la mappa caricata.
        partenza: nome del nodo di partenza.
        destinazione: nome del nodo di destinazione.
        tipo_veicolo: tipo di veicolo (es. 'ambulanza', 'autopompa').

        usa_euristica: se False, h = 0 (ricerca a costo minimo).

    Returns:
        Dizionario con 'percorso' (lista di nodi, None se non esiste),
        'costo' (float, inf se non esiste) e 'nodi_espansi' (int).
    """
    def esito(percorso, costo, espansi):
        return {"percorso": percorso, "costo": costo, "nodi_espansi": espansi}

    if partenza == destinazione:
        return esito([partenza], 0.0, 0)

    nodi = kb.get_tutti_nodi()
    if partenza not in nodi:
        print(f"  [A*] ERRORE: nodo partenza '{partenza}' non trovato nella mappa")
        return esito(None, float("inf"), 0)
    if destinazione not in nodi:
        print(f"  [A*] ERRORE: nodo destinazione '{destinazione}' non trovato nella mappa")
        return esito(None, float("inf"), 0)

    def h(nodo):
        return _euristica(kb, nodo, destinazione) if usa_euristica else 0.0

    h_start = h(partenza)
    nodo_start = NodoAStar(partenza, g=0.0, h=h_start)

    # Coda a priorità (open set): min-heap su f = g + h
    open_set = [nodo_start]
    heapq.heapify(open_set)

    closed_set = set()
    g_score = {partenza: 0.0}
    nodi_espansi = 0

    while open_set:
        corrente = heapq.heappop(open_set)

        if corrente.nome in closed_set:
            continue

        nodi_espansi += 1

        if corrente.nome == destinazione:
            percorso = _ricostruisci_percorso(corrente)
            return esito(percorso, corrente.g, nodi_espansi)

        closed_set.add(corrente.nome)

        # Interrogazione della KB Prolog: qui avviene l'integrazione
        # tra ricerca e ragionamento logico, Prolog decide transitabilità e costo
        vicini = kb.get_vicini_transitabili(corrente.nome, tipo_veicolo)

        for vicino_nome, costo_arco in vicini:
            if vicino_nome in closed_set:
                continue

            g_tentativo = corrente.g + costo_arco

            if g_tentativo < g_score.get(vicino_nome, float("inf")):
                g_score[vicino_nome] = g_tentativo
                nodo_vicino = NodoAStar(
                    vicino_nome, g=g_tentativo, h=h(vicino_nome), genitore=corrente
                )
                heapq.heappush(open_set, nodo_vicino)

    return esito(None, float("inf"), nodi_espansi)


def calcola_matrice_distanze(
    kb: KnowledgeBase,
    nodi_partenza: list,
    nodi_destinazione: list,
    tipo_veicolo: str
) -> dict:
    """
    Calcola la matrice delle distanze minime tra un set di nodi partenza
    e un set di nodi destinazione, usando A* per ogni coppia.

    Utile per il CSP di allocazione: pre-calcola i costi di assegnazione
    mezzo→emergenza.

    Args:
        kb: istanza KnowledgeBase.
        nodi_partenza: lista di nomi nodi partenza (posizioni mezzi).
        nodi_destinazione: lista di nomi nodi destinazione (posizioni emergenze).
        tipo_veicolo: tipo di veicolo per il calcolo dei costi.

    Returns:
        Dizionario {(partenza, destinazione): (percorso, costo)} per
        ogni coppia. Se non raggiungibile, costo = float('inf').
    """
    matrice = {}
    for p in nodi_partenza:
        for d in nodi_destinazione:
            percorso, costo = trova_percorso(kb, p, d, tipo_veicolo)
            matrice[(p, d)] = (percorso, costo)
    return matrice
