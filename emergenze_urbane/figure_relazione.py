"""
figure_relazione.py — Diagrammi per la relazione (cartella docs/figure)

  - architettura.png      flusso dei moduli del sistema
  - mappa.png             mappa della KB e percorsi A* in uno scenario allagato
  - ontologia.png         gerarchia delle classi e classificazione di HermiT
  - reti_bayesiane.png    struttura esperta e struttura appresa con BIC

I grafici dei risultati sperimentali sono prodotti da esperimenti.py.

Uso:
    python figure_relazione.py
"""

import os
import sys
import warnings

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

from logica.kb_interface import KnowledgeBase
from ricerca.astar_routing import trova_percorso

FIGURE_DIR = os.path.join(PROJECT_DIR, "docs", "figure")
COLORI_ZONA = {"zona_nord": "#cfe2f3", "zona_centro": "#fff2cc",
               "zona_sud": "#d9ead3", "zona_est": "#f4cccc"}


def _salva(nome: str):
    plt.savefig(os.path.join(FIGURE_DIR, nome), dpi=150, bbox_inches="tight")
    plt.close()


def _box(ax, x, y, w, h, testo, colore):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", fc=colore, ec="#444444"))
    ax.text(x, y, testo, ha="center", va="center", fontsize=8.5)


def _freccia(ax, x1, y1, x2, y2, testo=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    if testo:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, testo, ha="center",
                fontsize=7, color="#555555", style="italic")


def figura_architettura():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(-0.1, 11)
    ax.set_ylim(-0.1, 5.2)
    ax.axis("off")

    _box(ax, 1.4, 4.2, 2.3, 0.9, "Scenario\n(emergenze, mezzi,\nmeteo, fascia oraria)", "#eeeeee")
    _box(ax, 1.4, 2.5, 2.3, 0.9, "Base di conoscenza\nProlog\n(fatti + regole)", "#cfe2f3")
    _box(ax, 1.4, 0.8, 2.3, 0.9, "Ontologia OWL\n(HermiT)", "#cfe2f3")
    _box(ax, 4.9, 4.2, 2.3, 0.9, "A*\n(percorsi e costi)", "#fff2cc")
    _box(ax, 4.9, 2.5, 2.3, 0.9, "CSP allocazione\nAC-3 + B&B\nRRHC / SA", "#fff2cc")
    _box(ax, 8.6, 4.2, 2.3, 0.9, "Dataset storico\n(sintetico)", "#eeeeee")
    _box(ax, 8.6, 2.5, 2.3, 0.9, "Classificatori e\nreti bayesiane\n(nested CV)", "#d9ead3")
    _box(ax, 8.6, 0.8, 2.3, 0.9, "Predizione\nsupporto aggiuntivo", "#d9ead3")
    _box(ax, 4.9, 0.8, 2.3, 0.9, "Decisione per la centrale:\nallocazione + rinforzi", "#f4cccc")

    _freccia(ax, 1.4, 3.75, 1.4, 2.95, "assert / retract")
    _freccia(ax, 1.4, 1.25, 1.4, 2.05, "coerenza")
    _freccia(ax, 2.55, 2.75, 3.75, 4.0, "transitabile, costo_arco")
    _freccia(ax, 2.55, 2.5, 3.75, 2.5, "compatibile")
    _freccia(ax, 4.9, 3.75, 4.9, 2.95, "matrice dei costi")
    _freccia(ax, 8.6, 3.75, 8.6, 2.95, "addestramento")
    _freccia(ax, 8.6, 2.05, 8.6, 1.25, "modello scelto")
    _freccia(ax, 6.05, 2.3, 7.45, 1.0, "")
    ax.text(6.95, 1.85, "ritardo\n(costo A*)", fontsize=7, color="#555555", style="italic")
    _freccia(ax, 4.9, 2.05, 4.9, 1.25, "allocazione")
    _freccia(ax, 7.45, 0.8, 6.05, 0.8, "P(supporto)")
    # KB -> predizione: percorso a gomito sotto i riquadri
    ax.plot([0.25, 0.05, 0.05, 8.6], [2.5, 2.5, 0.05, 0.05], color="#444444", lw=1.2)
    _freccia(ax, 8.6, 0.05, 8.6, 0.35)
    ax.text(3.0, 0.1, "meteo, zona, fascia oraria (dalla KB)", fontsize=7,
            color="#555555", style="italic")
    ax.set_title("Architettura del sistema", fontsize=12)
    _salva("architettura.png")


def _disegna_mappa(ax, kb, titolo):
    nodi = kb.get_tutti_nodi()
    coord = {n: kb.get_coordinate(n) for n in nodi}
    for n1, n2, d in kb.get_archi():
        (x1, y1), (x2, y2) = coord[n1], coord[n2]
        ax.plot([x1, x2], [y1, y2], color="#999999", lw=1, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{d:g}", fontsize=6.5, color="#666666",
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.1))
    basi = {n for t in kb.get_tipi_mezzo() for n in kb.get_basi(t)}
    for n in nodi:
        x, y = coord[n]
        bordo = "red" if kb.is_strada_allagata(n) else "#333333"
        ax.scatter(x, y, s=170, c=COLORI_ZONA[kb.get_zona(n)], edgecolors=bordo,
                   linewidths=2.5 if bordo == "red" else 1, zorder=3,
                   marker="s" if n in basi else "o")
        ax.text(x, y - 0.55, n.replace("_", " "), fontsize=6.5, ha="center", zorder=4)
    ax.set_xlim(-1, 11.5)
    ax.set_ylim(-1.2, 10.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titolo, fontsize=10)
    return coord


def figura_mappa():
    kb = KnowledgeBase()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.3))

    for zona in kb.get_zone():
        kb.cambia_meteo(zona, "sereno")
    _disegna_mappa(ax1, kb, "(a) Mappa della KB: zone, distanze, basi dei mezzi (quadrati)")
    legenda = [Patch(fc=c, ec="#333333", label=z.replace("_", " ")) for z, c in COLORI_ZONA.items()]
    ax1.legend(handles=legenda, loc="upper right", fontsize=7)

    kb.cambia_meteo("zona_centro", "pioggia_forte")
    kb.cambia_meteo("zona_est", "nebbia")
    coord = _disegna_mappa(ax2, kb, "(b) Pioggia forte nel centro, nebbia a est:\n"
                                    "percorsi A* da stazione treni a parco")
    percorsi = {"ambulanza": "#1f77b4", "autopompa": "#d62728"}
    linee = []
    for i, (veicolo, colore) in enumerate(percorsi.items()):
        percorso, costo = trova_percorso(kb, "stazione_treni", "parco", veicolo)
        offset = 0.12 * (1 if i else -1)
        xs = [coord[n][0] + offset for n in percorso]
        ys = [coord[n][1] + offset for n in percorso]
        ax2.plot(xs, ys, color=colore, lw=2.5, zorder=2)
        linee.append(Line2D([], [], color=colore, lw=2.5,
                            label=f"{veicolo}: costo {costo:.1f}"))
    linee.append(Line2D([], [], marker="o", color="white", markeredgecolor="red",
                        markeredgewidth=2, markersize=9, label="nodo allagato"))
    ax2.legend(handles=linee, loc="upper right", fontsize=7)
    for zona in kb.get_zone():
        kb.cambia_meteo(zona, "sereno")
    _salva("mappa.png")


def figura_ontologia():
    from ontologia.ontologia_emergenze import crea_ontologia, ragiona, sottoclassi_inferite
    onto = ragiona(crea_ontologia())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    pos = {
        "Thing": (6, 5.0), "Mezzo": (2.5, 3.9), "Emergenza": (8.6, 3.9), "Luogo": (11.2, 3.9),
        "MezzoLeggero": (1.2, 2.7), "MezzoPesante": (3.8, 2.7),
        "Ambulanza": (1.2, 1.5), "Autopompa": (3.2, 1.5), "Autoscala": (4.6, 1.5),
        "EmergenzaMedica": (6.6, 2.7), "Incendio": (7.6, 2.0), "Allagamento": (8.6, 2.7),
        "Incidente": (9.6, 2.0), "SoccorsoInAltezza": (10.3, 2.7),
        "Nodo": (10.7, 3.0), "Zona": (11.6, 3.0),
        "MezzoAntincendio": (3.2, 0.3), "MezzoSoccorsoSanitario": (1.0, 0.3),
        "MezzoPolivalente": (5.4, 0.3),
    }
    asseriti = [("Mezzo", "Thing"), ("Emergenza", "Thing"), ("Luogo", "Thing"),
                ("MezzoLeggero", "Mezzo"), ("MezzoPesante", "Mezzo"),
                ("Ambulanza", "MezzoLeggero"), ("Autopompa", "MezzoPesante"),
                ("Autoscala", "MezzoPesante"), ("Nodo", "Luogo"), ("Zona", "Luogo")]
    asseriti += [(c, "Emergenza") for c in ["EmergenzaMedica", "Incendio", "Allagamento",
                                            "Incidente", "SoccorsoInAltezza"]]
    for figlio, padre in asseriti:
        (x1, y1), (x2, y2) = pos[figlio], pos[padre]
        ax.annotate("", xy=(x2, y2 - 0.2), xytext=(x1, y1 + 0.2),
                    arrowprops=dict(arrowstyle="->", color="#333333"))
    for definita in ["MezzoAntincendio", "MezzoSoccorsoSanitario", "MezzoPolivalente"]:
        for sotto in sottoclassi_inferite(onto, definita):
            (x1, y1), (x2, y2) = pos[sotto], pos[definita]
            ax.annotate("", xy=(x2, y2 + 0.2), xytext=(x1, y1 - 0.2),
                        arrowprops=dict(arrowstyle="->", color="#cc0000", linestyle="--"))
    for nome, (x, y) in pos.items():
        definita = nome.startswith("Mezzo") and nome not in ("Mezzo", "MezzoLeggero", "MezzoPesante")
        ax.text(x, y, nome, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="#fde9d9" if definita else "#e8eef7",
                          ec="#cc0000" if definita else "#555555"))
    ax.text(0.1, 5.2, "→ sottoclasse dichiarata nella TBox\n"
                      "⇢ sottoclasse dedotta da HermiT (in rosso: classi definite)",
            fontsize=8, va="top")
    ax.text(9.0, 0.9, "Autopompa ⊑ ∃puoGestire.Incendio\n"
                      "MezzoAntincendio ≡ Mezzo ⊓ ∃puoGestire.Incendio",
            fontsize=8, ha="center", family="monospace")
    ax.set_title("Gerarchia delle classi dell'ontologia", fontsize=12)
    _salva("ontologia.png")


def figura_reti_bayesiane():
    import networkx as nx
    from machine_learning.dataset_synth import genera_dataset, prepara_features, TARGET
    from machine_learning.rete_bayesiana import ReteBayesianaClassificatore, STRUTTURA_ESPERTA

    X, y, _ = prepara_features(genera_dataset(random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        appresa = ReteBayesianaClassificatore("appresa").fit(X, y).archi_

    pos = {"meteo": (0, 2), "zona": (1, 2), "orario": (2, 2), "ritardo_arrivo": (1, 1),
           "gravita": (3.4, 1.0), "tipo_emergenza": (2.9, 2), TARGET: (2, 0)}
    fig, assi = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, archi, titolo in [(assi[0], STRUTTURA_ESPERTA, "(a) Struttura esperta"),
                              (assi[1], appresa, "(b) Struttura appresa (Hill Climbing + BIC)")]:
        g = nx.DiGraph()
        g.add_nodes_from(pos)
        g.add_edges_from(archi)
        nx.draw_networkx(g, pos, ax=ax, node_color="#e8eef7", edgecolors="#555555",
                         node_size=2600, font_size=7.5, arrowsize=15,
                         labels={n: n.replace("_", "\n") for n in pos})
        ax.set_title(titolo, fontsize=10)
        ax.axis("off")
        ax.margins(0.15)
    _salva("reti_bayesiane.png")
    return appresa


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    figura_architettura()
    figura_mappa()
    figura_ontologia()
    archi = figura_reti_bayesiane()
    print(f"Figure salvate in {FIGURE_DIR}")
    print(f"Archi della struttura appresa: {archi}")


if __name__ == "__main__":
    main()
