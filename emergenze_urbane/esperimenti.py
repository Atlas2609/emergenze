"""
esperimenti.py — Valutazione sperimentale di tutti i moduli

Produce le tabelle riportate nella relazione (cartella output/esperimenti):

  1. A* contro ricerca a costo minimo (h = 0) su tutte le coppie di nodi,
     per ogni tipo di veicolo e per più scenari meteo: verifica di
     ottimalità, ammissibilità dell'euristica, nodi espansi, tempi.
  2. CSP di allocazione su 30 scenari casuali: branch and bound (ottimo)
     contro Random Restart Hill Climbing e Simulated Annealing.
  3. Apprendimento: confronto dei modelli con nested CV, limite teorico
     del classificatore bayesiano ottimo e curva di apprendimento delle
     reti bayesiane al crescere degli esempi.

Uso:
    python esperimenti.py            tutti gli esperimenti (alcuni minuti)
    python esperimenti.py --solo astar|csp|ml
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from logica.kb_interface import KnowledgeBase
from ricerca.astar_routing import ricerca_percorso, trova_percorso
from ricerca.csp_allocazione import (
    ProblemaCSP, risolvi_random_restart, risolvi_simulated_annealing, conta_violazioni,
)
from ricerca.csp_backtracking import risolvi_branch_and_bound
from machine_learning.dataset_synth import genera_dataset, prepara_features, limite_teorico
from machine_learning.modello_predittivo import get_modelli
from machine_learning.rete_bayesiana import get_modelli_bayesiani
from machine_learning.validazione import confronta_modelli, esegui_nested_cv
from simulazione.scenario_manager import (
    genera_scenario, genera_condizioni_meteo_avverse, applica_scenario_alla_kb,
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "esperimenti")


def _ms(valori):
    """Formatta media ± deviazione standard."""
    return f"{np.mean(valori):.2f} ± {np.std(valori):.2f}"


def _salva_figura(nome: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{nome}.png"), dpi=150)
    plt.close()


def _salva(tabella: pd.DataFrame, nome: str):
    tabella.to_csv(os.path.join(OUTPUT_DIR, f"{nome}.csv"), index=False)
    print(tabella.to_string(index=False))


# ======================================================================
# 1. A* contro ricerca a costo minimo
# ======================================================================

def esperimento_astar(kb: KnowledgeBase, n_scenari_meteo: int = 5):
    print("\n=== 1. A* contro ricerca a costo minimo ===")
    nodi = kb.get_tutti_nodi()
    righe = {nome: {"espansi": [], "tempo": []} for nome in ("A*", "Costo minimo")}
    differenze_costo, violazioni_h, coppie, irraggiungibili = 0, 0, 0, 0

    for seed in range(n_scenari_meteo):
        meteo = genera_condizioni_meteo_avverse(kb.get_zone(), random_state=seed)
        applica_scenario_alla_kb(kb, meteo, "mattina")
        for veicolo in kb.get_tipi_mezzo():
            for partenza in nodi:
                for arrivo in nodi:
                    if partenza == arrivo:
                        continue
                    esiti = {}
                    for nome, usa_h in (("A*", True), ("Costo minimo", False)):
                        t0 = time.perf_counter()
                        esiti[nome] = ricerca_percorso(kb, partenza, arrivo, veicolo, usa_h)
                        righe[nome]["tempo"].append((time.perf_counter() - t0) * 1000)
                        righe[nome]["espansi"].append(esiti[nome]["nodi_espansi"])
                    coppie += 1
                    costo_vero = esiti["Costo minimo"]["costo"]
                    if costo_vero == float("inf"):
                        irraggiungibili += 1
                        continue
                    if abs(esiti["A*"]["costo"] - costo_vero) > 1e-9:
                        differenze_costo += 1
                    if kb.distanza_euclidea(partenza, arrivo) > costo_vero + 1e-9:
                        violazioni_h += 1

    tabella = pd.DataFrame([
        {"Algoritmo": nome,
         "Nodi espansi": _ms(r["espansi"]),
         "Tempo (ms)": _ms(r["tempo"])}
        for nome, r in righe.items()
    ])
    _salva(tabella, "astar_vs_costo_minimo")
    riduzione = 1 - np.mean(righe["A*"]["espansi"]) / np.mean(righe["Costo minimo"]["espansi"])

    fig, ax = plt.subplots(figsize=(7, 4))
    massimo = max(max(r["espansi"]) for r in righe.values())
    ax.hist([righe["A*"]["espansi"], righe["Costo minimo"]["espansi"]],
            bins=range(0, massimo + 2), label=["A*", "Costo minimo (h = 0)"])
    ax.set_xlabel("Nodi espansi per ricerca")
    ax.set_ylabel("Numero di ricerche")
    ax.set_title("Nodi espansi: A* contro ricerca a costo minimo")
    ax.legend()
    _salva_figura("astar_nodi_espansi")
    print(f"Ricerche: {coppie} per algoritmo ({n_scenari_meteo} scenari meteo x "
          f"{len(kb.get_tipi_mezzo())} veicoli x {len(nodi) * (len(nodi) - 1)} coppie), "
          f"destinazioni irraggiungibili: {irraggiungibili}")
    print(f"Coppie in cui A* non trova il costo ottimo: {differenze_costo}")
    print(f"Coppie in cui h(n) supera il costo reale: {violazioni_h}")
    print(f"Riduzione dei nodi espansi grazie all'euristica: {riduzione:.1%}")
    applica_scenario_alla_kb(kb, {z: "sereno" for z in kb.get_zone()}, "mattina")


# ======================================================================
# 2. CSP: ottimo esatto contro ricerca locale
# ======================================================================

def esperimento_csp(kb: KnowledgeBase, n_scenari: int = 30,
                    n_emergenze: int = 7, n_mezzi: int = 7):
    print(f"\n=== 2. CSP: {n_scenari} scenari, {n_emergenze} emergenze, {n_mezzi} mezzi ===")
    compatibilita = {(tm, te): kb.is_compatibile(tm, te)
                     for tm in kb.get_tipi_mezzo() for te in kb.get_tipi_emergenza()}
    risultati = {n: {"costo": [], "gap": [], "ottimo": 0, "ammissibile": 0, "tempo": []}
                 for n in ("Branch and bound", "Random Restart HC", "Simulated Annealing")}
    nodi_bb, insoddisfacibili, dettaglio = [], 0, []

    for seed in range(n_scenari):
        meteo = genera_condizioni_meteo_avverse(kb.get_zone(), random_state=1000 + seed)
        applica_scenario_alla_kb(kb, meteo, "mattina")
        emergenze, mezzi = genera_scenario(kb, n_emergenze, n_mezzi, random_state=seed)
        matrice = {(m.id, e.id): trova_percorso(kb, m.posizione, e.posizione, m.tipo)[1]
                   for m in mezzi for e in emergenze}
        problema = ProblemaCSP(emergenze, mezzi, matrice, compatibilita, distanza_massima=25.0)

        t0 = time.perf_counter()
        sol_bb, costo_bb, stat = risolvi_branch_and_bound(problema)
        t_bb = time.perf_counter() - t0
        if sol_bb is None:
            insoddisfacibili += 1
            continue
        nodi_bb.append(stat["nodi"])

        esiti = {"Branch and bound": (sol_bb, costo_bb, t_bb)}
        random.seed(seed)
        t0 = time.perf_counter()
        sol, c = risolvi_random_restart(problema, n_restart=30)
        esiti["Random Restart HC"] = (sol, c, time.perf_counter() - t0)
        t0 = time.perf_counter()
        sol, c = risolvi_simulated_annealing(problema)
        esiti["Simulated Annealing"] = (sol, c, time.perf_counter() - t0)

        for nome, (sol, costo, tempo) in esiti.items():
            r = risultati[nome]
            r["costo"].append(costo)
            r["gap"].append(100 * (costo - costo_bb) / max(costo_bb, 1e-9))
            r["ottimo"] += abs(costo - costo_bb) < 1e-6
            r["ammissibile"] += conta_violazioni(problema, sol) == 0
            r["tempo"].append(tempo * 1000)
            dettaglio.append({"scenario": seed, "algoritmo": nome, "costo": costo,
                              "scarto_%": r["gap"][-1], "tempo_ms": tempo * 1000})

    risolti = n_scenari - insoddisfacibili
    tabella = pd.DataFrame([
        {"Algoritmo": nome,
         "Costo": _ms(r["costo"]),
         "Scarto dall'ottimo (%)": _ms(r["gap"]),
         "Ottimo trovato": f"{r['ottimo']}/{risolti}",
         "Soluzione ammissibile": f"{r['ammissibile']}/{risolti}",
         "Tempo (ms)": _ms(r["tempo"])}
        for nome, r in risultati.items()
    ])
    _salva(tabella, "csp_confronto")
    pd.DataFrame(dettaglio).to_csv(os.path.join(OUTPUT_DIR, "csp_dettaglio.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    nomi = list(risultati)
    ax1.boxplot([risultati[n]["gap"] for n in nomi])
    ax1.set_xticks(range(1, len(nomi) + 1))
    ax1.set_xticklabels(nomi, fontsize=8)
    ax1.set_ylabel("Scarto dall'ottimo (%)")
    ax1.set_title("Qualità della soluzione")
    ax2.bar(nomi, [np.mean(risultati[n]["tempo"]) for n in nomi],
            yerr=[np.std(risultati[n]["tempo"]) for n in nomi], capsize=4)
    ax2.set_ylabel("Tempo medio (ms)")
    ax2.set_title("Tempo di calcolo")
    ax2.tick_params(axis="x", labelsize=8)
    _salva_figura("csp_confronto")
    print(f"Scenari insoddisfacibili (emergenze gravi non coperte): {insoddisfacibili}/{n_scenari}")
    print(f"Nodi visitati dal branch and bound: {_ms(nodi_bb)}")
    applica_scenario_alla_kb(kb, {z: "sereno" for z in kb.get_zone()}, "mattina")


# ======================================================================
# 3. Apprendimento
# ======================================================================

def esperimento_ml():
    print("\n=== 3. Apprendimento: confronto modelli (nested CV 5x3) ===")
    df = genera_dataset(random_state=42, con_probabilita_vera=True)
    limite = limite_teorico(df)
    X, y, _ = prepara_features(df.drop(columns="p_vera"))

    tabella, _ = confronta_modelli(
        X, y, {**get_modelli(), **get_modelli_bayesiani()},
        n_fold_outer=5, n_fold_inner=3, output_dir=OUTPUT_DIR,
    )
    _salva(tabella, "ml_confronto_modelli")
    print(f"Limite teorico (classificatore bayesiano ottimo): "
          f"accuracy {limite['accuracy']:.3f}, AUC {limite['roc_auc']:.3f}")

    print("\n=== 3b. Curva di apprendimento (nested CV 3x2, AUC-ROC) ===")
    modelli = {**get_modelli_bayesiani(),
               "Regressione logistica": get_modelli()["Regressione logistica"]}
    righe = []
    for n in (500, 2000, 8000, 20000):
        df_n = genera_dataset(n_campioni=n, random_state=7)
        X_n, y_n, _ = prepara_features(df_n)
        riga = {"Esempi": n}
        for nome, (modello, griglia) in modelli.items():
            r = esegui_nested_cv(X_n, y_n, 3, 2, 42, OUTPUT_DIR, modello, griglia,
                                 nome, grafici=False, verbose=False)
            riga[nome] = f"{r['roc_auc_media']:.3f} ± {r['roc_auc_std']:.3f}"
        righe.append(riga)
        print(f"  n = {n}: fatto")
    _salva(pd.DataFrame(righe), "ml_curva_apprendimento")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    esempi = [r["Esempi"] for r in righe]
    for nome in modelli:
        medie = [float(r[nome].split(" ± ")[0]) for r in righe]
        std = [float(r[nome].split(" ± ")[1]) for r in righe]
        ax.errorbar(esempi, medie, yerr=std, marker="o", capsize=3, label=nome)
    ax.axhline(limite_teorico(genera_dataset(200000, random_state=0,
                                             con_probabilita_vera=True))["roc_auc"],
               color="gray", linestyle="--", label="Limite teorico")
    ax.set_xscale("log")
    ax.set_xlabel("Numero di esempi (scala logaritmica)")
    ax.set_ylabel("AUC-ROC (nested CV)")
    ax.set_title("Curva di apprendimento")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _salva_figura("ml_curva_apprendimento")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo", choices=["astar", "csp", "ml"])
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.solo in (None, "astar", "csp"):
        kb = KnowledgeBase()
    if args.solo in (None, "astar"):
        esperimento_astar(kb)
    if args.solo in (None, "csp"):
        esperimento_csp(kb)
    if args.solo in (None, "ml"):
        esperimento_ml()
    print(f"\nTabelle salvate in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
