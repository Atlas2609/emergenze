"""
csp_backtracking.py — Risoluzione esatta del CSP di allocazione

Il problema di csp_allocazione.py viene risolto in modo completo:

  1. Consistenza di nodo (vincoli unari): dal dominio di ogni emergenza
     si tolgono i mezzi incompatibili (regola compatibile/2 della KB) o
     che non possono raggiungerla (costo A* infinito) e, per le
     emergenze di gravità alta, l'opzione 'non_assegnato'.
  2. Arc consistency (AC-3) sui vincoli binari di capacità: due
     emergenze non possono usare lo stesso mezzo di capacità 1.
     Se un dominio si svuota il problema è insoddisfacibile e lo si
     scopre senza alcuna ricerca.
  3. Backtracking con euristica MRV (variabile con il dominio più
     piccolo) e forward checking: quando un mezzo esaurisce la
     capacità viene tolto dai domini delle variabili non ancora
     assegnate.
  4. Branch and bound: tra le soluzioni ammissibili cerca quella di
     costo minimo, potando i rami il cui costo parziale più una stima
     ottimistica del costo restante supera la migliore soluzione nota.

Il risultato è l'ottimo globale, usato come riferimento per misurare
la qualità delle soluzioni di Random Restart Hill Climbing e Simulated
Annealing, che non danno garanzie di ottimalità.
"""

import time
from collections import deque

from ricerca.csp_allocazione import ProblemaCSP, contributo, funzione_obiettivo

NON_ASSEGNATO = "non_assegnato"


def domini_iniziali(problema: ProblemaCSP) -> dict:
    """Consistenza di nodo: applica i vincoli unari ai domini."""
    domini = {}
    for em in problema.emergenze:
        valori = [
            m.id for m in problema.mezzi
            if problema.compatibilita.get((m.tipo, em.tipo), False)
            and problema.matrice_costi.get((m.id, em.id), float("inf")) < float("inf")
        ]
        if em.gravita != "alta" or not problema.obbligo_gravita_alta:
            valori.append(NON_ASSEGNATO)
        domini[em.id] = valori
    return domini


def _coppia_consistente(a: str, b: str, capacita: dict) -> bool:
    """Vincolo binario: lo stesso mezzo di capacità 1 non serve due emergenze."""
    return not (a == b and a != NON_ASSEGNATO and capacita[a] < 2)


def ac3(domini: dict, capacita: dict) -> tuple:
    """
    Algoritmo AC-3 sui vincoli binari di capacità.

    Ogni arco (Xi, Xj) è consistente se per ogni valore di Xi esiste un
    valore di Xj compatibile. Quando il dominio di Xi viene ridotto, gli
    archi (Xk, Xi) vanno ricontrollati.

    Returns:
        Tupla (consistente, valori_rimossi). I domini sono modificati
        sul posto.
    """
    variabili = list(domini)
    coda = deque((xi, xj) for xi in variabili for xj in variabili if xi != xj)
    rimossi = 0
    while coda:
        xi, xj = coda.popleft()
        da_togliere = [
            a for a in domini[xi]
            if not any(_coppia_consistente(a, b, capacita) for b in domini[xj])
        ]
        if da_togliere:
            domini[xi] = [a for a in domini[xi] if a not in da_togliere]
            rimossi += len(da_togliere)
            if not domini[xi]:
                return False, rimossi
            coda.extend((xk, xi) for xk in variabili if xk not in (xi, xj))
    return True, rimossi


def risolvi_branch_and_bound(problema: ProblemaCSP) -> tuple:
    """
    Trova l'assegnamento ammissibile di costo minimo.

    Returns:
        Tupla (soluzione, costo, statistiche). Se il problema non ha
        soluzioni ammissibili, soluzione è None e costo è inf: le risorse
        non bastano a coprire tutte le emergenze gravi.
    """
    inizio = time.time()
    emergenze = {e.id: e for e in problema.emergenze}
    capacita = {m.id: m.capacita for m in problema.mezzi}

    domini = domini_iniziali(problema)
    stat = {"nodi": 0, "potature_bound": 0, "rimossi_ac3": 0, "tempo_s": 0.0}

    vuoti = [v for v, d in domini.items() if not d]
    consistente, stat["rimossi_ac3"] = (False, 0) if vuoti else ac3(domini, capacita)
    if not consistente:
        stat["tempo_s"] = time.time() - inizio
        return None, float("inf"), stat

    # Contributo di ogni coppia (variabile, valore), calcolato una volta
    peso = {
        (v, a): contributo(problema, emergenze[v], a)
        for v, d in domini.items() for a in d
    }

    migliore = {"soluzione": None, "costo": float("inf")}

    def stima_ottimistica(non_assegnate, domini_correnti):
        # Ogni variabile libera paga almeno il minimo del suo dominio:
        # la stima non sovrastima mai, quindi la potatura è sicura
        return sum(min(peso[(v, a)] for a in domini_correnti[v]) for v in non_assegnate)

    def backtrack(assegnamento, costo_parziale, domini_correnti, uso):
        stat["nodi"] += 1
        non_assegnate = [v for v in domini_correnti if v not in assegnamento]
        if not non_assegnate:
            if costo_parziale < migliore["costo"]:
                migliore["soluzione"] = dict(assegnamento)
                migliore["costo"] = costo_parziale
            return

        if costo_parziale + stima_ottimistica(non_assegnate, domini_correnti) >= migliore["costo"]:
            stat["potature_bound"] += 1
            return

        # MRV: prima la variabile con meno valori possibili
        var = min(non_assegnate, key=lambda v: len(domini_correnti[v]))
        for valore in sorted(domini_correnti[var], key=lambda a: peso[(var, a)]):
            nuovo_uso = dict(uso)
            if valore != NON_ASSEGNATO:
                nuovo_uso[valore] = nuovo_uso.get(valore, 0) + 1

            # Forward checking: un mezzo pieno sparisce dagli altri domini
            nuovi_domini = dict(domini_correnti)
            fallito = False
            if valore != NON_ASSEGNATO and nuovo_uso[valore] >= capacita[valore]:
                for v in non_assegnate:
                    if v != var and valore in nuovi_domini[v]:
                        nuovi_domini[v] = [a for a in nuovi_domini[v] if a != valore]
                        if not nuovi_domini[v]:
                            fallito = True
                            break
            if fallito:
                continue

            assegnamento[var] = valore
            backtrack(assegnamento, costo_parziale + peso[(var, valore)], nuovi_domini, nuovo_uso)
            del assegnamento[var]

    backtrack({}, 0.0, domini, {})
    stat["tempo_s"] = time.time() - inizio

    if migliore["soluzione"] is None:
        return None, float("inf"), stat
    soluzione = {em.id: migliore["soluzione"][em.id] for em in problema.emergenze}
    # Ricontrollo con la stessa funzione obiettivo della ricerca locale
    return soluzione, funzione_obiettivo(problema, soluzione), stat
