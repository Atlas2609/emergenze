"""
scenario_manager.py — Generazione di scenari di emergenza multipla

Questo modulo genera scenari realistici per la simulazione, creando
emergenze simultanee in posizioni diverse della città e mezzi di
soccorso disponibili con caratteristiche diverse.
"""

import random
from ricerca.csp_allocazione import Emergenza, Mezzo


TIPI_EMERGENZA = ["medica", "incendio", "allagamento", "incidente"]
GRAVITA_POSSIBILI = ["bassa", "media", "alta"]
TIPI_MEZZO = ["ambulanza", "autopompa", "autoscala"]

# Nodi della mappa cittadina: devono corrispondere alla KB Prolog
NODI_MAPPA = [
    "ospedale", "stazione_pompieri", "piazza_centrale",
    "incrocio_nord", "incrocio_sud", "incrocio_est", "incrocio_ovest",
    "zona_industriale", "porto", "aeroporto", "universita",
    "stadio", "parco", "centro_commerciale", "stazione_treni",
]

BASI_MEZZI = {
    "ambulanza": ["ospedale", "piazza_centrale", "stazione_treni"],
    "autopompa": ["stazione_pompieri", "zona_industriale"],
    "autoscala": ["stazione_pompieri"],
}

CAPACITA_MEZZO = {
    "ambulanza": 1,
    "autopompa": 2,
    "autoscala": 1,
}


def genera_scenario(
    n_emergenze: int = 4,
    n_mezzi: int = 5,
    random_state: int = None,
) -> tuple:
    """
    Genera uno scenario di emergenza multipla con N emergenze
    simultanee e M mezzi disponibili.

    Args:
        n_emergenze: numero di emergenze da generare.
        n_mezzi: numero di mezzi disponibili.
        random_state: seed per la riproducibilità. Se None, casuale.

    Returns:
        Tupla (lista_emergenze, lista_mezzi) con le emergenze
        e i mezzi generati per la simulazione.
    """
    if random_state is not None:
        random.seed(random_state)

    emergenze = []
    nodi_usati_emergenze = set()

    for i in range(n_emergenze):
        nodi_disponibili = [n for n in NODI_MAPPA if n not in nodi_usati_emergenze]
        if not nodi_disponibili:
            nodi_disponibili = NODI_MAPPA

        posizione = random.choice(nodi_disponibili)
        nodi_usati_emergenze.add(posizione)

        tipo = random.choice(TIPI_EMERGENZA)
        gravita = random.choices(
            GRAVITA_POSSIBILI,
            weights=[0.3, 0.4, 0.3],
            k=1
        )[0]

        emergenza = Emergenza(
            id=f"EM_{i + 1:02d}",
            posizione=posizione,
            tipo=tipo,
            gravita=gravita,
        )
        emergenze.append(emergenza)

    mezzi = []

    tipi_da_generare = []
    for i in range(n_mezzi):
        if i < n_mezzi // 2:
            tipi_da_generare.append("ambulanza")
        elif i < n_mezzi * 3 // 4:
            tipi_da_generare.append("autopompa")
        else:
            tipi_da_generare.append("autoscala")

    if n_mezzi >= 3:
        tipi_da_generare[0] = "ambulanza"
        tipi_da_generare[1] = "autopompa"
        tipi_da_generare[2] = "autoscala"

    random.shuffle(tipi_da_generare)

    for i in range(n_mezzi):
        tipo_mezzo = tipi_da_generare[i] if i < len(tipi_da_generare) else random.choice(TIPI_MEZZO)

        basi = BASI_MEZZI.get(tipo_mezzo, NODI_MAPPA)
        posizione = random.choice(basi)

        mezzo = Mezzo(
            id=f"MZ_{i + 1:02d}_{tipo_mezzo[:3].upper()}",
            posizione=posizione,
            tipo=tipo_mezzo,
            capacita=CAPACITA_MEZZO.get(tipo_mezzo, 1),
        )
        mezzi.append(mezzo)

    return emergenze, mezzi


def genera_condizioni_meteo_avverse(
    random_state: int = None,
) -> dict:
    """
    Genera condizioni meteo casuali per le zone della città,
    incluse condizioni avverse che alterano la mappa.

    Returns:
        Dizionario {zona: condizione_meteo}.
    """
    if random_state is not None:
        random.seed(random_state)

    zone = ["zona_nord", "zona_centro", "zona_sud", "zona_est"]
    condizioni = ["sereno", "pioggia", "nebbia", "pioggia_forte", "alluvione"]
    pesi = [0.35, 0.25, 0.15, 0.15, 0.10]

    meteo = {}
    for zona in zone:
        meteo[zona] = random.choices(condizioni, weights=pesi, k=1)[0]

    return meteo


def stampa_scenario(emergenze: list, mezzi: list):
    """Stampa un riepilogo leggibile dello scenario generato."""
    print("\n  === SCENARIO DI EMERGENZA ===")
    print(f"\n  Emergenze attive: {len(emergenze)}")
    print(f"  {'ID':<10} {'Posizione':<25} {'Tipo':<15} {'Gravità':<10}")
    print("  " + "-" * 60)
    for em in emergenze:
        print(f"  {em.id:<10} {em.posizione:<25} {em.tipo:<15} {em.gravita:<10}")

    print(f"\n  Mezzi disponibili: {len(mezzi)}")
    print(f"  {'ID':<25} {'Posizione':<25} {'Tipo':<15} {'Capacità':<10}")
    print("  " + "-" * 75)
    for m in mezzi:
        print(f"  {m.id:<25} {m.posizione:<25} {m.tipo:<15} {m.capacita:<10}")
