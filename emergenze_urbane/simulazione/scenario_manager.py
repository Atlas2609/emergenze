"""
scenario_manager.py — Generazione di scenari di emergenza multipla

Questo modulo genera scenari per la simulazione, creando emergenze
simultanee in posizioni diverse della città e mezzi di soccorso
disponibili con caratteristiche diverse.

Nodi, zone, tipi di mezzo, basi operative, capacità e tipi di emergenza
sono letti dalla KB Prolog: lo scenario è sempre coerente con la
conoscenza del dominio e non ne duplica i fatti.
"""

import random
from ricerca.csp_allocazione import Emergenza, Mezzo


GRAVITA_POSSIBILI = ["bassa", "media", "alta"]
FASCE_ORARIE = ["notte", "mattina", "pomeriggio", "sera"]
CONDIZIONI_METEO = ["sereno", "pioggia", "nebbia", "pioggia_forte", "alluvione"]
PESI_METEO = [0.35, 0.25, 0.15, 0.15, 0.10]

# Parametri dello scenario (non conoscenza del dominio). La flotta si
# compone prendendo i tipi in quest'ordine, ripetuto se serve: i primi
# tre garantiscono un mezzo per tipo, poi prevalgono le ambulanze
# (es. 6 mezzi -> 3 ambulanze, 2 autopompe, 1 autoscala).
ORDINE_FLOTTA = ["ambulanza", "autopompa", "autoscala",
                 "ambulanza", "autopompa", "ambulanza", "ambulanza", "autoscala"]
SIGLE_MEZZO = {"ambulanza": "AMB", "autopompa": "POM", "autoscala": "SCA"}


def genera_scenario(
    kb,
    n_emergenze: int = 4,
    n_mezzi: int = 5,
    random_state: int = None,
) -> tuple:
    """
    Genera uno scenario di emergenza multipla con N emergenze
    simultanee e M mezzi disponibili.

    Args:
        kb: KnowledgeBase da cui leggere nodi, mezzi e tipi di emergenza.
        n_emergenze: numero di emergenze da generare.
        n_mezzi: numero di mezzi disponibili.
        random_state: seed per la riproducibilità. Se None, casuale.

    Returns:
        Tupla (lista_emergenze, lista_mezzi).
    """
    rng = random.Random(random_state)

    nodi_mappa = kb.get_tutti_nodi()
    tipi_emergenza = kb.get_tipi_emergenza()
    tipi_mezzo = kb.get_tipi_mezzo()

    emergenze = []
    nodi_usati = set()
    for i in range(n_emergenze):
        nodi_disponibili = [n for n in nodi_mappa if n not in nodi_usati] or nodi_mappa
        posizione = rng.choice(nodi_disponibili)
        nodi_usati.add(posizione)

        emergenze.append(Emergenza(
            id=f"EM_{i + 1:02d}",
            posizione=posizione,
            tipo=rng.choice(tipi_emergenza),
            gravita=rng.choices(GRAVITA_POSSIBILI, weights=[0.3, 0.4, 0.3], k=1)[0],
        ))

    ordine = [t for t in ORDINE_FLOTTA if t in tipi_mezzo] or tipi_mezzo
    tipi_da_generare = [ordine[i % len(ordine)] for i in range(n_mezzi)]
    rng.shuffle(tipi_da_generare)

    mezzi = []
    for i, tipo_mezzo in enumerate(tipi_da_generare):
        basi = kb.get_basi(tipo_mezzo) or nodi_mappa
        mezzi.append(Mezzo(
            id=f"MZ_{i + 1:02d}_{SIGLE_MEZZO.get(tipo_mezzo, tipo_mezzo[:3].upper())}",
            posizione=rng.choice(basi),
            tipo=tipo_mezzo,
            capacita=kb.get_capacita(tipo_mezzo),
        ))

    return emergenze, mezzi


def genera_condizioni_meteo_avverse(zone: list, random_state: int = None) -> dict:
    """
    Genera condizioni meteo casuali per le zone della città,
    incluse condizioni avverse che alterano la mappa.

    Returns:
        Dizionario {zona: condizione_meteo}.
    """
    rng = random.Random(random_state)
    return {
        zona: rng.choices(CONDIZIONI_METEO, weights=PESI_METEO, k=1)[0]
        for zona in zone
    }


def genera_fascia_oraria(random_state: int = None) -> str:
    """Estrae la fascia oraria dello scenario."""
    return random.Random(random_state).choice(FASCE_ORARIE)


def applica_scenario_alla_kb(kb, meteo: dict, fascia_oraria: str):
    """Aggiorna la KB con meteo e fascia oraria dello scenario (assert/retract)."""
    for zona, condizione in meteo.items():
        kb.cambia_meteo(zona, condizione)
    kb.set_fascia_oraria(fascia_oraria)


def stampa_scenario(emergenze: list, mezzi: list):
    """Stampa un riepilogo leggibile dello scenario generato."""
    print("\n  === SCENARIO DI EMERGENZA ===")
    print(f"\n  Emergenze attive: {len(emergenze)}")
    print(f"  {'ID':<10} {'Posizione':<25} {'Tipo':<18} {'Gravità':<10}")
    print("  " + "-" * 63)
    for em in emergenze:
        print(f"  {em.id:<10} {em.posizione:<25} {em.tipo:<18} {em.gravita:<10}")

    print(f"\n  Mezzi disponibili: {len(mezzi)}")
    print(f"  {'ID':<25} {'Posizione':<25} {'Tipo':<15} {'Capacità':<10}")
    print("  " + "-" * 75)
    for m in mezzi:
        print(f"  {m.id:<25} {m.posizione:<25} {m.tipo:<15} {m.capacita:<10}")
