"""
csp_allocazione.py — Allocazione mezzi-emergenze come CSP con ricerca locale

Il problema di allocare M mezzi a N emergenze è modellato come un
Constraint Satisfaction Problem (CSP). La ricerca della soluzione
avviene tramite algoritmi di ricerca locale:
  - Random Restart Hill Climbing: esplora vicinati locali partendo da
    configurazioni casuali, riavviando per sfuggire ai minimi locali.
  - Simulated Annealing: accetta soluzioni peggiorative con probabilità
    decrescente nel tempo, ispirandosi al raffreddamento dei metalli,
    per esplorare lo spazio delle soluzioni in modo più ampio.

Non si usa backtracking esaustivo né solver esterni "black box".
"""

import random
import math
import copy
from dataclasses import dataclass, field


@dataclass
class Emergenza:
    """Rappresenta un'emergenza attiva nella città."""
    id: str
    posizione: str
    tipo: str
    gravita: str
    richiede_supporto: bool = False


@dataclass
class Mezzo:
    """Rappresenta un mezzo di soccorso disponibile."""
    id: str
    posizione: str
    tipo: str
    capacita: int = 1


@dataclass
class ProblemaCSP:
    """
    Definizione formale del CSP per l'allocazione mezzi-emergenze.

    - Variabili: le emergenze attive (a ciascuna va assegnato un mezzo)
    - Domini: i mezzi disponibili (+ opzione 'non_assegnato')
    - Vincoli:
      1. Compatibilità mezzo/emergenza (hard)
      2. Capacità del mezzo non superata (hard)
      3. Emergenze di gravità alta obbligatoriamente coperte (hard)
      4. Distanza massima accettabile (soft, penalizzata)

    Nella ricerca locale i vincoli hard sono rilassati in penalità
    (peso_violazione_hard per ogni violazione); il risolutore esatto in
    csp_backtracking.py li tratta invece come vincoli veri.
    """
    emergenze: list
    mezzi: list
    matrice_costi: dict
    compatibilita: dict
    distanza_massima: float = 30.0
    # Se False le emergenze gravi possono restare scoperte (vincolo
    # rilassato quando le risorse non bastano a coprirle tutte)
    obbligo_gravita_alta: bool = True

    peso_distanza: float = 1.0
    # La penalità di una violazione hard deve superare qualsiasi costo
    # ammissibile (la peggiore non assegnazione vale 50 * 3 = 150, una
    # distanza tipica 20-60): con 100 la ricerca locale trovava
    # conveniente violare un vincolo invece di lasciare un'emergenza
    # media non assegnata (entrambe costavano 100).
    peso_violazione_hard: float = 1000.0
    peso_non_assegnato: float = 50.0
    peso_gravita: float = field(default_factory=lambda: {
        "alta": 3.0, "media": 2.0, "bassa": 1.0
    })


def genera_soluzione_casuale(problema: ProblemaCSP) -> dict:
    """
    Genera una soluzione casuale: assegna ad ogni emergenza un mezzo
    scelto a caso dal dominio (inclusa l'opzione 'non_assegnato').

    Returns:
        Dizionario {emergenza_id: mezzo_id o 'non_assegnato'}
    """
    soluzione = {}
    mezzi_ids = [m.id for m in problema.mezzi] + ["non_assegnato"]
    for em in problema.emergenze:
        soluzione[em.id] = random.choice(mezzi_ids)
    return soluzione


def funzione_obiettivo(problema: ProblemaCSP, soluzione: dict) -> float:
    """
    Calcola il valore della funzione obiettivo (da minimizzare).

    La funzione combina:
    1. Costo totale delle distanze (pesato per gravità dell'emergenza)
    2. Penalità per violazioni di vincoli hard (incompatibilità, sovracapacità)
    3. Penalità per emergenze non assegnate

    Returns:
        Valore float della funzione obiettivo (più basso = migliore).
    """
    costo_totale = 0.0
    violazioni = 0

    mezzi_map = {m.id: m for m in problema.mezzi}
    emergenze_map = {e.id: e for e in problema.emergenze}

    assegnazioni_mezzo = {}
    for em_id, mezzo_id in soluzione.items():
        if mezzo_id != "non_assegnato":
            assegnazioni_mezzo.setdefault(mezzo_id, []).append(em_id)

    for em_id, mezzo_id in soluzione.items():
        emergenza = emergenze_map[em_id]
        peso_grav = problema.peso_gravita.get(emergenza.gravita, 1.0)

        if mezzo_id == "non_assegnato":
            costo_totale += problema.peso_non_assegnato * peso_grav
            # Vincolo hard: le emergenze gravi devono avere un mezzo
            if emergenza.gravita == "alta" and problema.obbligo_gravita_alta:
                violazioni += 1
            continue

        mezzo = mezzi_map.get(mezzo_id)
        if mezzo is None:
            violazioni += 1
            continue

        # Vincolo hard: compatibilità mezzo-emergenza
        chiave_compat = (mezzo.tipo, emergenza.tipo)
        if not problema.compatibilita.get(chiave_compat, False):
            violazioni += 1

        costo_distanza = problema.matrice_costi.get(
            (mezzo.id, em_id), float("inf")
        )
        if costo_distanza == float("inf"):
            violazioni += 1
        else:
            costo_totale += costo_distanza * peso_grav

            # Vincolo soft: penalità extra oltre la distanza massima
            if costo_distanza > problema.distanza_massima:
                eccesso = costo_distanza - problema.distanza_massima
                costo_totale += eccesso * peso_grav * 0.5

    # Vincolo hard: capacità del mezzo non superata
    for mezzo_id, em_ids in assegnazioni_mezzo.items():
        mezzo = mezzi_map.get(mezzo_id)
        if mezzo and len(em_ids) > mezzo.capacita:
            eccesso_cap = len(em_ids) - mezzo.capacita
            violazioni += eccesso_cap

    costo_totale += violazioni * problema.peso_violazione_hard

    return costo_totale


def contributo(problema: ProblemaCSP, emergenza: Emergenza, mezzo_id: str) -> float:
    """
    Contributo di un singolo assegnamento ammissibile alla funzione
    obiettivo (distanza pesata per gravità + penalità soft, oppure
    penalità di non assegnazione). Usato dal branch and bound.
    """
    peso_grav = problema.peso_gravita.get(emergenza.gravita, 1.0)
    if mezzo_id == "non_assegnato":
        return problema.peso_non_assegnato * peso_grav
    costo = problema.matrice_costi[(mezzo_id, emergenza.id)]
    valore = costo * peso_grav
    if costo > problema.distanza_massima:
        valore += (costo - problema.distanza_massima) * peso_grav * 0.5
    return valore


def conta_violazioni(problema: ProblemaCSP, soluzione: dict) -> int:
    """Numero di vincoli hard violati da una soluzione (0 = ammissibile)."""
    mezzi_map = {m.id: m for m in problema.mezzi}
    emergenze_map = {e.id: e for e in problema.emergenze}
    violazioni = 0
    uso = {}
    for em_id, mezzo_id in soluzione.items():
        em = emergenze_map[em_id]
        if mezzo_id == "non_assegnato":
            violazioni += em.gravita == "alta" and problema.obbligo_gravita_alta
            continue
        uso[mezzo_id] = uso.get(mezzo_id, 0) + 1
        mezzo = mezzi_map[mezzo_id]
        if not problema.compatibilita.get((mezzo.tipo, em.tipo), False):
            violazioni += 1
        if problema.matrice_costi.get((mezzo_id, em_id), float("inf")) == float("inf"):
            violazioni += 1
    for mezzo_id, n in uso.items():
        violazioni += max(0, n - mezzi_map[mezzo_id].capacita)
    return violazioni


def genera_vicino(problema: ProblemaCSP, soluzione: dict) -> dict:
    """
    Genera una soluzione vicina modificando l'assegnazione di
    una emergenza scelta a caso (mossa minima nello spazio di ricerca).
    Il vicinato è l'insieme di soluzioni che differiscono dalla
    corrente per un solo assegnamento.
    """
    vicino = copy.copy(soluzione)
    emergenze_ids = list(vicino.keys())
    mezzi_ids = [m.id for m in problema.mezzi] + ["non_assegnato"]

    em_scelta = random.choice(emergenze_ids)
    mezzo_attuale = vicino[em_scelta]

    alternative = [m for m in mezzi_ids if m != mezzo_attuale]
    if alternative:
        vicino[em_scelta] = random.choice(alternative)

    return vicino


def risolvi_random_restart(
    problema: ProblemaCSP,
    n_restart: int = 20,
    max_iterazioni_per_restart: int = 500,
    verbose: bool = False
) -> tuple:
    """
    Risolve il CSP con Random Restart Hill Climbing.

    Hill Climbing è un algoritmo di ricerca locale che ad ogni passo si
    muove verso il vicino con valore migliore, ma può bloccarsi in
    minimi locali. Random Restart mitiga il problema eseguendo HC da
    N punti di partenza casuali e tenendo la migliore soluzione trovata
    globalmente.

    Args:
        problema: istanza ProblemaCSP.
        n_restart: numero di riavvii casuali.
        max_iterazioni_per_restart: passi massimi per ogni run di HC.
        verbose: se True, stampa informazioni di debug.

    Returns:
        Tupla (migliore_soluzione, migliore_costo).
    """
    if not problema.emergenze:
        return {}, 0.0

    migliore_soluzione = None
    migliore_costo = float("inf")

    for restart in range(n_restart):
        soluzione_corrente = genera_soluzione_casuale(problema)
        costo_corrente = funzione_obiettivo(problema, soluzione_corrente)

        iterazioni_senza_miglioramento = 0
        for i in range(max_iterazioni_per_restart):
            vicino = genera_vicino(problema, soluzione_corrente)
            costo_vicino = funzione_obiettivo(problema, vicino)

            # Discesa: accetta solo se il vicino è strettamente migliore
            if costo_vicino < costo_corrente:
                soluzione_corrente = vicino
                costo_corrente = costo_vicino
                iterazioni_senza_miglioramento = 0
            else:
                iterazioni_senza_miglioramento += 1

            if iterazioni_senza_miglioramento > 50:
                break

        if costo_corrente < migliore_costo:
            migliore_soluzione = copy.copy(soluzione_corrente)
            migliore_costo = costo_corrente

        if verbose:
            print(f"  [RRHC] Restart {restart + 1}/{n_restart}: "
                  f"costo = {costo_corrente:.2f} "
                  f"(migliore globale = {migliore_costo:.2f})")

    return migliore_soluzione, migliore_costo


def risolvi_simulated_annealing(
    problema: ProblemaCSP,
    temperatura_iniziale: float = 100.0,
    raffreddamento: float = 0.995,
    temperatura_minima: float = 0.01,
    max_iterazioni: int = 10000,
    verbose: bool = False
) -> tuple:
    """
    Risolve il CSP con Simulated Annealing.

    A temperature alte, l'algoritmo accetta anche soluzioni peggiori
    con alta probabilità (esplorazione), mentre a temperature basse si
    comporta come Hill Climbing greedy (sfruttamento), permettendo di
    sfuggire ai minimi locali. La probabilità di accettare una
    soluzione peggiorativa segue il criterio di Metropolis:

        P(accetta) = exp(-ΔE / T)

    dove ΔE è il peggioramento e T la temperatura corrente.

    Args:
        problema: istanza ProblemaCSP.
        temperatura_iniziale: temperatura iniziale T₀.
        raffreddamento: fattore di raffreddamento α (0 < α < 1),
                        con schema geometrico T(t+1) = α · T(t).
        temperatura_minima: temperatura sotto la quale si arresta.
        max_iterazioni: numero massimo di iterazioni.
        verbose: se True, stampa informazioni di debug.

    Returns:
        Tupla (migliore_soluzione, migliore_costo).
    """
    if not problema.emergenze:
        return {}, 0.0

    soluzione_corrente = genera_soluzione_casuale(problema)
    costo_corrente = funzione_obiettivo(problema, soluzione_corrente)

    migliore_soluzione = copy.copy(soluzione_corrente)
    migliore_costo = costo_corrente

    temperatura = temperatura_iniziale

    for i in range(max_iterazioni):
        if temperatura < temperatura_minima:
            break

        vicino = genera_vicino(problema, soluzione_corrente)
        costo_vicino = funzione_obiettivo(problema, vicino)

        delta_e = costo_vicino - costo_corrente

        if delta_e < 0:
            soluzione_corrente = vicino
            costo_corrente = costo_vicino
        else:
            probabilita_accettazione = math.exp(-delta_e / temperatura)
            if random.random() < probabilita_accettazione:
                soluzione_corrente = vicino
                costo_corrente = costo_vicino

        if costo_corrente < migliore_costo:
            migliore_soluzione = copy.copy(soluzione_corrente)
            migliore_costo = costo_corrente

        temperatura *= raffreddamento

        if verbose and (i + 1) % 1000 == 0:
            print(f"  [SA] Iterazione {i + 1}: T = {temperatura:.4f}, "
                  f"costo = {costo_corrente:.2f}, "
                  f"migliore = {migliore_costo:.2f}")

    return migliore_soluzione, migliore_costo


def formatta_soluzione(
    problema: ProblemaCSP,
    soluzione: dict,
    costo: float
) -> str:
    """
    Formatta la soluzione del CSP in un report leggibile.

    Returns:
        Stringa con il report dell'allocazione.
    """
    mezzi_map = {m.id: m for m in problema.mezzi}

    linee = []
    linee.append(f"  Costo obiettivo: {costo:.2f}")
    linee.append(f"  {'Emergenza':<20} {'Mezzo Assegnato':<20} "
                 f"{'Tipo Mezzo':<15} {'Costo Percorso':<15}")
    linee.append("  " + "-" * 70)

    for em_id, mezzo_id in soluzione.items():
        if mezzo_id == "non_assegnato":
            linee.append(f"  {em_id:<20} {'NON ASSEGNATO':<20} "
                         f"{'-':<15} {'-':<15}")
        else:
            mezzo = mezzi_map.get(mezzo_id)
            tipo = mezzo.tipo if mezzo else "?"
            costo_p = problema.matrice_costi.get(
                (mezzo_id, em_id), float("inf")
            )
            costo_str = f"{costo_p:.2f}" if costo_p < float("inf") else "∞"
            linee.append(f"  {em_id:<20} {mezzo_id:<20} "
                         f"{tipo:<15} {costo_str:<15}")

    return "\n".join(linee)
