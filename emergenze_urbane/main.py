"""
main.py — Orchestratore del Sistema di Gestione Intelligente delle Emergenze Urbane

Questo modulo è il punto di ingresso del sistema e integra tutti i componenti:
1. Base di Conoscenza Prolog (deduzione logica)
2. Algoritmo A* (ricerca informata con interrogazione dinamica alla KB)
3. CSP con ricerca locale (allocazione mezzi-emergenze)
4. Machine Learning (predizione supporto aggiuntivo con Nested CV)

Il sistema integra IA simbolica (Prolog, regole deduttive), IA
sub-simbolica (albero di decisione ML) e tecniche di ricerca (A*,
Hill Climbing, Simulated Annealing) per risolvere un problema
complesso del mondo reale.
"""

import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from logica.kb_interface import KnowledgeBase
from ricerca.astar_routing import trova_percorso, calcola_matrice_distanze
from ricerca.csp_allocazione import (
    ProblemaCSP, Emergenza, Mezzo,
    risolvi_random_restart, risolvi_simulated_annealing,
    formatta_soluzione,
)
from machine_learning.dataset_synth import genera_dataset, prepara_features
from machine_learning.validazione import esegui_nested_cv, addestra_modello_finale
from machine_learning.modello_predittivo import descrivi_modello
from simulazione.scenario_manager import (
    genera_scenario, genera_condizioni_meteo_avverse, stampa_scenario,
)


def _separatore(titolo: str):
    """Stampa un separatore con titolo per il report."""
    print("\n" + "=" * 70)
    print(f"  {titolo}")
    print("=" * 70)


def fase1_inizializza_kb() -> KnowledgeBase:
    """
    FASE 1: Inizializzazione della Base di Conoscenza Prolog.

    Carica la KB dal file .pl e verifica il funzionamento.
    """
    _separatore("FASE 1: INIZIALIZZAZIONE BASE DI CONOSCENZA PROLOG")

    kb = KnowledgeBase()
    nodi = kb.get_tutti_nodi()
    print(f"\n  KB Prolog caricata con successo.")
    print(f"  Nodi nella mappa: {len(nodi)}")
    print(f"  Nodi: {', '.join(nodi[:8])}...")

    print(f"\n  Condizioni meteo iniziali:")
    for zona in ["zona_nord", "zona_centro", "zona_sud", "zona_est"]:
        condizione = kb.get_condizione_meteo(zona)
        print(f"    {zona}: {condizione}")

    return kb


def fase2_scenario_e_meteo(kb: KnowledgeBase) -> tuple:
    """
    FASE 2: Generazione dello scenario e aggiornamento condizioni meteo.

    Genera emergenze multiple, mezzi disponibili e condizioni meteo
    avverse, aggiornando dinamicamente la KB Prolog.
    """
    _separatore("FASE 2: GENERAZIONE SCENARIO E CONDIZIONI METEO")

    emergenze, mezzi = genera_scenario(
        n_emergenze=4, n_mezzi=5, random_state=42
    )
    stampa_scenario(emergenze, mezzi)

    meteo = genera_condizioni_meteo_avverse(random_state=123)
    print(f"\n  Aggiornamento condizioni meteo nella KB Prolog:")
    for zona, condizione in meteo.items():
        kb.cambia_meteo(zona, condizione)
        print(f"    {zona} → {condizione}")

    print(f"\n  Effetto sulle strade:")
    for nodo in kb.get_tutti_nodi():
        if kb.is_strada_allagata(nodo):
            print(f"    ⚠ {nodo}: STRADA ALLAGATA")

    print(f"\n  Verifica transitabilità (deduzione Prolog):")
    test_archi = [
        ("ospedale", "incrocio_nord", "ambulanza"),
        ("ospedale", "incrocio_nord", "autopompa"),
        ("piazza_centrale", "incrocio_est", "ambulanza"),
        ("piazza_centrale", "incrocio_est", "autopompa"),
    ]
    for n1, n2, veicolo in test_archi:
        transitabile = kb.is_transitabile(n1, n2, veicolo)
        stato = "✓ TRANSITABILE" if transitabile else "✗ BLOCCATO"
        costo = kb.costo_arco(n1, n2, veicolo) if transitabile else "N/A"
        print(f"    {n1} → {n2} ({veicolo}): {stato} | costo: {costo}")

    return emergenze, mezzi


def fase3_routing_astar(
    kb: KnowledgeBase, emergenze: list, mezzi: list
) -> dict:
    """
    FASE 3: Calcolo dei percorsi ottimi con A*.

    Per ogni coppia (mezzo, emergenza), calcola il percorso di costo
    minimo usando A* con interrogazione dinamica alla KB Prolog.
    """
    _separatore("FASE 3: CALCOLO PERCORSI A* (ROUTING)")

    print(f"\n  Calcolo della matrice delle distanze mezzo→emergenza...")
    print(f"  (A* interroga la KB Prolog ad ogni espansione di nodo)")

    matrice_costi = {}
    matrice_percorsi = {}

    for mezzo in mezzi:
        for emergenza in emergenze:
            percorso, costo = trova_percorso(
                kb, mezzo.posizione, emergenza.posizione, mezzo.tipo
            )
            matrice_costi[(mezzo.id, emergenza.id)] = costo
            matrice_percorsi[(mezzo.id, emergenza.id)] = percorso

    print(f"\n  Matrice delle distanze (costo A*):")
    header = f"  {'Mezzo':<25}" + "".join(
        f"{em.id:<12}" for em in emergenze
    )
    print(header)
    print("  " + "-" * (25 + 12 * len(emergenze)))
    for mezzo in mezzi:
        riga = f"  {mezzo.id:<25}"
        for emergenza in emergenze:
            costo = matrice_costi[(mezzo.id, emergenza.id)]
            if costo < float("inf"):
                riga += f"{costo:<12.2f}"
            else:
                riga += f"{'∞':<12}"
        print(riga)

    print(f"\n  Percorsi dettagliati (primi 3):")
    count = 0
    for (mezzo_id, em_id), percorso in matrice_percorsi.items():
        if percorso and count < 3:
            costo = matrice_costi[(mezzo_id, em_id)]
            print(f"    {mezzo_id} → {em_id}: "
                  f"{' → '.join(percorso)} (costo: {costo:.2f})")
            count += 1

    return matrice_costi


def fase4_allocazione_csp(
    emergenze: list, mezzi: list, matrice_costi: dict, kb: KnowledgeBase
) -> tuple:
    """
    FASE 4: Allocazione ottima mezzi-emergenze con CSP e ricerca locale.

    Il problema è modellato come CSP e risolto con due algoritmi di
    ricerca locale a confronto: Random Restart Hill Climbing e
    Simulated Annealing.
    """
    _separatore("FASE 4: ALLOCAZIONE CSP (RICERCA LOCALE)")

    compatibilita = {}
    tipi_mezzo = ["ambulanza", "autopompa", "autoscala"]
    tipi_emergenza = ["medica", "incendio", "allagamento", "incidente", "soccorso_altezza"]
    for tm in tipi_mezzo:
        for te in tipi_emergenza:
            compatibilita[(tm, te)] = kb.is_compatibile(tm, te)

    problema = ProblemaCSP(
        emergenze=emergenze,
        mezzi=mezzi,
        matrice_costi=matrice_costi,
        compatibilita=compatibilita,
        distanza_massima=25.0,
    )

    print(f"\n  Risoluzione con Random Restart Hill Climbing...")
    sol_rrhc, costo_rrhc = risolvi_random_restart(
        problema, n_restart=30, max_iterazioni_per_restart=500, verbose=False
    )
    print(f"\n  Risultato RRHC:")
    print(formatta_soluzione(problema, sol_rrhc, costo_rrhc))

    print(f"\n  Risoluzione con Simulated Annealing...")
    sol_sa, costo_sa = risolvi_simulated_annealing(
        problema,
        temperatura_iniziale=100.0,
        raffreddamento=0.995,
        max_iterazioni=10000,
        verbose=False,
    )
    print(f"\n  Risultato SA:")
    print(formatta_soluzione(problema, sol_sa, costo_sa))

    print(f"\n  Confronto algoritmi:")
    print(f"    Random Restart HC:   costo = {costo_rrhc:.2f}")
    print(f"    Simulated Annealing: costo = {costo_sa:.2f}")

    if costo_rrhc <= costo_sa:
        print(f"    → RRHC ha trovato la soluzione migliore")
        soluzione_migliore = sol_rrhc
        costo_migliore = costo_rrhc
    else:
        print(f"    → SA ha trovato la soluzione migliore")
        soluzione_migliore = sol_sa
        costo_migliore = costo_sa

    return soluzione_migliore, costo_migliore


def fase5_machine_learning(output_dir: str) -> tuple:
    """
    FASE 5: Predizione supporto aggiuntivo con ML e Nested CV.

    Validazione statistica rigorosa con Nested CV (non un singolo
    train/test split) per una stima onesta delle prestazioni
    generalizzabili del modello.
    """
    _separatore("FASE 5: MACHINE LEARNING — PREDIZIONE SUPPORTO AGGIUNTIVO")

    print(f"\n  Generazione dataset sintetico...")
    df = genera_dataset(n_campioni=800, random_state=42)
    X, y, nomi_feature = prepara_features(df)
    print(f"  Dataset: {X.shape[0]} campioni, {X.shape[1]} feature")
    print(f"  Distribuzione target: {np.bincount(y)}")
    print(f"  Feature: {nomi_feature[:5]}... (totale: {len(nomi_feature)})")

    risultati_cv = esegui_nested_cv(
        X, y,
        n_fold_outer=5,
        n_fold_inner=3,
        random_state=42,
        output_dir=output_dir,
    )

    print(f"\n  Addestramento modello finale su tutto il dataset...")
    modello_finale = addestra_modello_finale(X, y)
    print(descrivi_modello(modello_finale))

    return modello_finale, risultati_cv, nomi_feature


def fase6_predizione_emergenze(
    modello, emergenze: list, nomi_feature: list
) -> dict:
    """
    FASE 6: Predizione del supporto aggiuntivo per le emergenze simulate.

    Usa il modello ML addestrato per predire se ciascuna emergenza
    dello scenario simulato richiederà supporto aggiuntivo.
    """
    _separatore("FASE 6: PREDIZIONE ML SULLE EMERGENZE SIMULATE")

    import pandas as pd

    predizioni = {}

    # In uno scenario reale queste feature verrebbero lette da KB/sensori
    condizioni_emergenze = []
    for em in emergenze:
        cond = {
            "meteo": _meteo_da_posizione(em.posizione),
            "orario": "mattina",
            "zona": _zona_da_posizione(em.posizione),
            "gravita": em.gravita,
            "tipo_emergenza": em.tipo,
        }
        condizioni_emergenze.append(cond)

    df_em = pd.DataFrame(condizioni_emergenze)

    X_em = pd.get_dummies(df_em, drop_first=False)

    # Allineamento delle colonne con quelle viste in training
    for col in nomi_feature:
        if col not in X_em.columns:
            X_em[col] = 0
    X_em = X_em[nomi_feature]

    y_pred = modello.predict(X_em.values.astype(float))
    y_prob = modello.predict_proba(X_em.values.astype(float))

    print(f"\n  Predizioni per le emergenze simulate:")
    print(f"  {'Emergenza':<12} {'Posizione':<20} {'Tipo':<15} "
          f"{'Gravità':<10} {'Supporto':<12} {'Probabilità':<12}")
    print("  " + "-" * 81)

    for i, em in enumerate(emergenze):
        supporto = "SÌ" if y_pred[i] == 1 else "NO"
        prob = y_prob[i][1]
        em.richiede_supporto = bool(y_pred[i] == 1)
        predizioni[em.id] = {
            "supporto": bool(y_pred[i] == 1),
            "probabilita": float(prob),
        }
        print(f"  {em.id:<12} {em.posizione:<20} {em.tipo:<15} "
              f"{em.gravita:<10} {supporto:<12} {prob:<12.4f}")

    return predizioni


def _meteo_da_posizione(posizione: str) -> str:
    """Associa una condizione meteo alla posizione (semplificato)."""
    mappa = {
        "ospedale": "sereno", "stazione_pompieri": "nebbia",
        "piazza_centrale": "pioggia", "incrocio_nord": "sereno",
        "incrocio_sud": "sereno", "incrocio_est": "pioggia",
        "incrocio_ovest": "pioggia", "zona_industriale": "sereno",
        "porto": "sereno", "aeroporto": "sereno",
        "universita": "pioggia", "stadio": "sereno",
        "parco": "sereno", "centro_commerciale": "nebbia",
        "stazione_treni": "sereno",
    }
    return mappa.get(posizione, "sereno")


def _zona_da_posizione(posizione: str) -> str:
    """Associa una zona alla posizione (semplificato)."""
    mappa = {
        "ospedale": "zona_nord", "stazione_pompieri": "zona_est",
        "piazza_centrale": "zona_centro", "incrocio_nord": "zona_nord",
        "incrocio_sud": "zona_sud", "incrocio_est": "zona_centro",
        "incrocio_ovest": "zona_centro", "zona_industriale": "zona_sud",
        "porto": "zona_sud", "aeroporto": "zona_nord",
        "universita": "zona_centro", "stadio": "zona_sud",
        "parco": "zona_nord", "centro_commerciale": "zona_est",
        "stazione_treni": "zona_sud",
    }
    return mappa.get(posizione, "zona_centro")


def fase7_riepilogo_finale(
    emergenze: list, mezzi: list, soluzione: dict,
    costo_allocazione: float, predizioni: dict, risultati_cv: dict
):
    """FASE 7: Riepilogo finale della simulazione."""
    _separatore("RIEPILOGO FINALE DELLA SIMULAZIONE")

    mezzi_map = {m.id: m for m in mezzi}
    emergenze_map = {e.id: e for e in emergenze}

    print(f"\n  {'='*60}")
    print(f"  ALLOCAZIONE FINALE MEZZI → EMERGENZE")
    print(f"  {'='*60}")
    print(f"  Costo totale allocazione: {costo_allocazione:.2f}")

    for em_id, mezzo_id in soluzione.items():
        emergenza = emergenze_map[em_id]
        pred = predizioni.get(em_id, {})
        supporto = "SÌ" if pred.get("supporto", False) else "NO"
        prob = pred.get("probabilita", 0.0)

        if mezzo_id == "non_assegnato":
            print(f"\n  {em_id} ({emergenza.tipo}, {emergenza.gravita}) "
                  f"@ {emergenza.posizione}")
            print(f"    Mezzo: NON ASSEGNATO ⚠")
            print(f"    Supporto aggiuntivo predetto: {supporto} "
                  f"(prob: {prob:.2%})")
        else:
            mezzo = mezzi_map.get(mezzo_id)
            tipo_mezzo = mezzo.tipo if mezzo else "?"
            print(f"\n  {em_id} ({emergenza.tipo}, {emergenza.gravita}) "
                  f"@ {emergenza.posizione}")
            print(f"    Mezzo: {mezzo_id} ({tipo_mezzo}) "
                  f"da {mezzo.posizione if mezzo else '?'}")
            print(f"    Supporto aggiuntivo predetto: {supporto} "
                  f"(prob: {prob:.2%})")

    print(f"\n  {'='*60}")
    print(f"  METRICHE ML (Nested Cross-Validation)")
    print(f"  {'='*60}")
    print(f"  Accuracy:  {risultati_cv['accuracy_media']:.4f} "
          f"± {risultati_cv['accuracy_std']:.4f}")
    print(f"  Log Loss:  {risultati_cv['log_loss_media']:.4f} "
          f"± {risultati_cv['log_loss_std']:.4f}")
    print(f"  AUC-ROC:   {risultati_cv['roc_auc']:.4f}")

    print(f"\n  {'='*60}")
    print(f"  SIMULAZIONE COMPLETATA CON SUCCESSO")
    print(f"  {'='*60}")


def main():
    """Funzione principale: orchestra tutte le fasi del sistema."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   SISTEMA DI GESTIONE INTELLIGENTE DELLE EMERGENZE URBANE" + " " * 9 + "#")
    print("#   Ingegneria della Conoscenza — IA Simbolica e Sub-Simbolica" + " " * 7 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    output_dir = os.path.join(PROJECT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    kb = fase1_inizializza_kb()
    emergenze, mezzi = fase2_scenario_e_meteo(kb)
    matrice_costi = fase3_routing_astar(kb, emergenze, mezzi)
    soluzione, costo_allocazione = fase4_allocazione_csp(
        emergenze, mezzi, matrice_costi, kb
    )
    modello, risultati_cv, nomi_feature = fase5_machine_learning(output_dir)
    predizioni = fase6_predizione_emergenze(modello, emergenze, nomi_feature)
    fase7_riepilogo_finale(
        emergenze, mezzi, soluzione,
        costo_allocazione, predizioni, risultati_cv
    )


if __name__ == "__main__":
    main()
