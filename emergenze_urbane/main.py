"""
main.py — Orchestratore del Sistema di Gestione Intelligente delle Emergenze Urbane

Integra tutti i componenti in un'unica simulazione:
1. Base di conoscenza Prolog (fatti, regole, assert/retract)
2. Ontologia OWL (classificazione con HermiT, coerenza con la KB)
3. Scenario: emergenze, mezzi, meteo e fascia oraria asseriti nella KB
4. A* (ricerca informata che interroga la KB a ogni espansione)
5. CSP di allocazione: ottimo esatto (AC-3 + backtracking + branch and
   bound) confrontato con la ricerca locale (RRHC, Simulated Annealing)
6. Apprendimento: confronto di classificatori e reti bayesiane con
   nested cross-validation
7. Predizione del supporto aggiuntivo: le feature di ogni emergenza
   vengono dalla KB (meteo, zona, fascia oraria) e da A* (ritardo di
   arrivo del mezzo assegnato)

Uso:
    python main.py            simulazione completa
    python main.py --veloce   nested CV ridotta (3x2 fold), per una demo rapida
"""

import argparse
import os
import sys
from dataclasses import replace

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# La console di Windows usa cp1252: senza questa riga stampare caratteri
# come "±" o "→" può interrompere il programma con UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from logica.kb_interface import KnowledgeBase
from ontologia.ontologia_emergenze import (
    costruisci_e_salva, sottoclassi_inferite, verifica_coerenza_con_kb,
)
from ricerca.astar_routing import trova_percorso
from ricerca.csp_allocazione import (
    ProblemaCSP, risolvi_random_restart, risolvi_simulated_annealing,
    formatta_soluzione, conta_violazioni,
)
from ricerca.csp_backtracking import risolvi_branch_and_bound
from machine_learning.dataset_synth import genera_dataset, prepara_features, ritardo_da_costo
from machine_learning.modello_predittivo import get_modelli, descrivi_modello
from machine_learning.rete_bayesiana import get_modelli_bayesiani, esempi_query
from machine_learning.validazione import (
    confronta_modelli, migliore_modello, addestra_modello_finale,
)
from simulazione.scenario_manager import (
    genera_scenario, genera_condizioni_meteo_avverse, genera_fascia_oraria,
    applica_scenario_alla_kb, stampa_scenario,
)

SEED_SCENARIO = 42


def _separatore(titolo: str):
    print("\n" + "=" * 70)
    print(f"  {titolo}")
    print("=" * 70)


def fase1_inizializza_kb() -> KnowledgeBase:
    """FASE 1: caricamento della KB Prolog."""
    _separatore("FASE 1: BASE DI CONOSCENZA PROLOG")
    kb = KnowledgeBase()
    nodi = kb.get_tutti_nodi()
    print(f"\n  KB caricata: {len(nodi)} nodi, {len(kb.get_archi())} archi, "
          f"zone {kb.get_zone()}")
    print(f"  Tipi di mezzo: {kb.get_tipi_mezzo()}")
    print(f"  Tipi di emergenza: {kb.get_tipi_emergenza()}")
    return kb


def fase2_ontologia(kb: KnowledgeBase):
    """FASE 2: ontologia OWL, ragionamento con HermiT e coerenza con la KB."""
    _separatore("FASE 2: ONTOLOGIA OWL E RAGIONAMENTO (HermiT)")
    onto = costruisci_e_salva(kb)
    print(f"\n  Ontologia salvata in ontologia/emergenze.owl "
          f"({len(list(onto.individuals()))} individui dalla KB)")
    print("  Sottoclassi dedotte dal ragionatore per le classi definite:")
    for classe in ["MezzoAntincendio", "MezzoSoccorsoSanitario", "MezzoPolivalente"]:
        print(f"    {classe}: {', '.join(sottoclassi_inferite(onto, classe))}")
    discrepanze = verifica_coerenza_con_kb(onto, kb)
    if discrepanze:
        print(f"  ATTENZIONE: ontologia e KB discordano su {discrepanze}")
    else:
        print("  Compatibilità mezzo/emergenza: ontologia e KB Prolog concordano.")


def fase3_scenario(kb: KnowledgeBase) -> tuple:
    """FASE 3: scenario di emergenza, meteo e fascia oraria asseriti nella KB."""
    _separatore("FASE 3: SCENARIO, METEO E FASCIA ORARIA")
    emergenze, mezzi = genera_scenario(kb, n_emergenze=5, n_mezzi=6,
                                       random_state=SEED_SCENARIO)
    stampa_scenario(emergenze, mezzi)

    meteo = genera_condizioni_meteo_avverse(kb.get_zone(), random_state=123)
    fascia = genera_fascia_oraria(random_state=SEED_SCENARIO)
    applica_scenario_alla_kb(kb, meteo, fascia)
    print("\n  Aggiornamento della KB (retract/assert):")
    for zona, condizione in meteo.items():
        print(f"    condizione_meteo({zona}, {condizione})")
    print(f"    fascia_oraria_corrente({kb.get_fascia_oraria()})")

    allagati = [n for n in kb.get_tutti_nodi() if kb.is_strada_allagata(n)]
    print(f"\n  Nodi allagati dedotti dalla KB: {allagati or 'nessuno'}")
    return emergenze, mezzi


def fase4_routing_astar(kb: KnowledgeBase, emergenze: list, mezzi: list) -> tuple:
    """FASE 4: matrice dei costi mezzo -> emergenza calcolata con A*."""
    _separatore("FASE 4: PERCORSI CON A*")
    matrice_costi, matrice_percorsi = {}, {}
    for mezzo in mezzi:
        for em in emergenze:
            percorso, costo = trova_percorso(kb, mezzo.posizione, em.posizione, mezzo.tipo)
            matrice_costi[(mezzo.id, em.id)] = costo
            matrice_percorsi[(mezzo.id, em.id)] = percorso

    print(f"\n  {'Mezzo':<20}" + "".join(f"{em.id:>10}" for em in emergenze))
    for mezzo in mezzi:
        riga = "".join(
            f"{matrice_costi[(mezzo.id, em.id)]:>10.2f}"
            if matrice_costi[(mezzo.id, em.id)] < float("inf") else f"{'inf':>10}"
            for em in emergenze
        )
        print(f"  {mezzo.id:<20}{riga}")
    print("  (inf = destinazione non raggiungibile per quel tipo di veicolo)")
    return matrice_costi, matrice_percorsi


def fase5_allocazione_csp(kb, emergenze, mezzi, matrice_costi) -> tuple:
    """FASE 5: allocazione con ottimo esatto e ricerca locale a confronto."""
    _separatore("FASE 5: ALLOCAZIONE MEZZI (CSP)")
    compatibilita = {
        (tm, te): kb.is_compatibile(tm, te)
        for tm in kb.get_tipi_mezzo() for te in kb.get_tipi_emergenza()
    }
    problema = ProblemaCSP(emergenze=emergenze, mezzi=mezzi,
                           matrice_costi=matrice_costi, compatibilita=compatibilita,
                           distanza_massima=25.0)

    sol_bb, costo_bb, stat = risolvi_branch_and_bound(problema)
    if sol_bb is None:
        # Le risorse non bastano a coprire tutte le emergenze gravi: si
        # rilassa solo quel vincolo e si cerca la migliore allocazione
        # ammissibile; le gravi rimaste scoperte vanno a rinforzi esterni
        print("\n  Nessuna allocazione copre tutte le emergenze gravi: "
              "vincolo rilassato.")
        problema = replace(problema, obbligo_gravita_alta=False)
        sol_bb, costo_bb, stat = risolvi_branch_and_bound(problema)
        scoperte = [e.id for e in emergenze
                    if e.gravita == "alta" and sol_bb[e.id] == "non_assegnato"]
        print(f"  Emergenze gravi da affidare a rinforzi esterni: {', '.join(scoperte)}")

    print(f"\n  Ottimo con AC-3 + backtracking + branch and bound "
          f"({stat['nodi']} nodi, {stat['potature_bound']} potature, "
          f"{stat['rimossi_ac3']} valori rimossi da AC-3):")
    print(formatta_soluzione(problema, sol_bb, costo_bb))
    soluzione, costo = sol_bb, costo_bb

    sol_rrhc, costo_rrhc = risolvi_random_restart(problema, n_restart=30)
    sol_sa, costo_sa = risolvi_simulated_annealing(problema)
    print("\n  Confronto con la ricerca locale:")
    for nome, sol, c in [("Branch and bound", sol_bb, costo_bb),
                         ("Random Restart HC", sol_rrhc, costo_rrhc),
                         ("Simulated Annealing", sol_sa, costo_sa)]:
        violazioni = conta_violazioni(problema, sol) if sol else "-"
        print(f"    {nome:<22} costo = {c:>8.2f}   vincoli violati = {violazioni}")
    return soluzione, costo


def fase6_apprendimento(output_dir: str, veloce: bool) -> tuple:
    """FASE 6: confronto dei modelli con nested CV e scelta del migliore."""
    _separatore("FASE 6: APPRENDIMENTO - CONFRONTO MODELLI (NESTED CV)")
    df = genera_dataset(random_state=42)
    X, y, _ = prepara_features(df)
    print(f"\n  Dataset: {len(y)} esempi, valori mancanti: "
          f"{int(df.isna().sum().sum())}, quota positivi: {y.mean():.2f}")

    modelli = {**get_modelli(), **get_modelli_bayesiani()}
    fold = (3, 2) if veloce else (5, 3)
    tabella, risultati = confronta_modelli(
        X, y, modelli, n_fold_outer=fold[0], n_fold_inner=fold[1], output_dir=output_dir
    )
    print("\n  Risultati (media ± deviazione standard sui fold esterni):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabella.to_string(index=False))

    nome = migliore_modello(risultati)
    print(f"\n  Modello scelto (AUC-ROC media più alta): {nome}")
    modello, griglia = modelli[nome]
    finale = addestra_modello_finale(X, y, modello, griglia)
    if hasattr(finale, "named_steps"):
        print(descrivi_modello(finale))

    rete = addestra_modello_finale(X, y, *get_modelli_bayesiani()["Rete bayesiana (esperta)"])
    print("\n  Query alla rete bayesiana (evidenza parziale, eliminazione delle variabili):")
    for descrizione, p in esempi_query(rete):
        print(f"    P(supporto | {descrizione}) = {p:.3f}")
    return finale


def fase7_predizione(kb, modello, emergenze, mezzi, soluzione, matrice_costi) -> dict:
    """
    FASE 7: predizione del supporto aggiuntivo per le emergenze dello
    scenario. Meteo, zona e fascia oraria sono letti dalla KB; il
    ritardo deriva dal costo A* del mezzo assegnato dal CSP.
    """
    _separatore("FASE 7: PREDIZIONE DEL SUPPORTO AGGIUNTIVO")
    righe = []
    for em in emergenze:
        mezzo_id = soluzione.get(em.id, "non_assegnato")
        costo = matrice_costi.get((mezzo_id, em.id), float("inf"))
        righe.append({
            "meteo": kb.get_meteo_nodo(em.posizione),
            "orario": kb.get_fascia_oraria(),
            "zona": kb.get_zona(em.posizione),
            "gravita": em.gravita,
            "tipo_emergenza": em.tipo,
            "ritardo_arrivo": ritardo_da_costo(costo),
        })
    X_scenario = pd.DataFrame(righe)
    probabilita = modello.predict_proba(X_scenario)[:, 1]

    print(f"\n  {'Emergenza':<10} {'Tipo':<17} {'Gravità':<8} {'Meteo':<14} "
          f"{'Ritardo':<8} {'P(supporto)':>11}  Esito")
    predizioni = {}
    for em, riga, p in zip(emergenze, righe, probabilita):
        em.richiede_supporto = bool(p >= 0.5)
        critica = kb.is_emergenza_critica(em.posizione, em.gravita)
        esito = "INVIARE RINFORZI" if em.richiede_supporto else "-"
        if critica:
            esito += " [critica per la KB]"
        predizioni[em.id] = {"probabilita": float(p), "supporto": em.richiede_supporto}
        print(f"  {em.id:<10} {em.tipo:<17} {em.gravita:<8} {riga['meteo']:<14} "
              f"{riga['ritardo_arrivo']:<8} {p:>11.3f}  {esito}")
    return predizioni


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--veloce", action="store_true",
                        help="nested CV ridotta (3 fold esterni, 2 interni)")
    args = parser.parse_args()

    print("#" * 70)
    print("#   SISTEMA DI GESTIONE INTELLIGENTE DELLE EMERGENZE URBANE")
    print("#   Ingegneria della Conoscenza - A.A. 2025/26")
    print("#" * 70)

    output_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    kb = fase1_inizializza_kb()
    fase2_ontologia(kb)
    emergenze, mezzi = fase3_scenario(kb)
    matrice_costi, _ = fase4_routing_astar(kb, emergenze, mezzi)
    soluzione, _ = fase5_allocazione_csp(kb, emergenze, mezzi, matrice_costi)
    modello = fase6_apprendimento(output_dir, args.veloce)
    fase7_predizione(kb, modello, emergenze, mezzi, soluzione, matrice_costi)

    _separatore("SIMULAZIONE COMPLETATA")
    print(f"  Grafici e tabelle in: {output_dir}")


if __name__ == "__main__":
    main()
