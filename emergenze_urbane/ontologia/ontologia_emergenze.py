"""
ontologia_emergenze.py — Ontologia OWL del dominio delle emergenze

L'ontologia descrive il vocabolario condiviso del dominio con una
logica descrittiva (OWL 2), costruita in Python con Owlready2:

  - TBox (scritta a mano): gerarchie di classi (Mezzo, Emergenza, Luogo),
    proprietà con dominio e codominio, classi disgiunte e restrizioni
    esistenziali (es. Autopompa ⊑ ∃puoGestire.Incendio);
  - ABox (popolata dalla KB Prolog): nodi della mappa e zone, così i
    fatti non vengono riscritti a mano in due posti.

Il ragionatore HermiT classifica l'ontologia: ad esempio deduce che
Autopompa e Autoscala sono sottoclassi della classe definita
MezzoAntincendio ≡ Mezzo ⊓ ∃puoGestire.Incendio, senza che lo si sia
scritto esplicitamente, e verifica la consistenza (classi disgiunte).

Infine verifica_coerenza_con_kb confronta l'ontologia con la regola
compatibile/2 della KB Prolog. Le due rappresentazioni ragionano in modo
diverso: Prolog adotta l'ipotesi di mondo chiuso (ciò che non è
derivabile è falso: negazione come fallimento), OWL quella di mondo
aperto (ciò che non è affermato è solo sconosciuto). Per questo da OWL
si possono derivare le compatibilità, ma non le incompatibilità.
"""

import os

from owlready2 import (
    Thing, ObjectProperty, AllDisjoint, get_ontology, sync_reasoner,
)
import owlready2.reasoning

# HermiT gira su Java: un heap contenuto funziona anche con JVM a 32 bit
owlready2.reasoning.JAVA_MEMORY = 512

IRI = "http://www.uniba.it/icon/emergenze.owl"
FILE_OWL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergenze.owl")

# Corrispondenza tra nomi Prolog e classi OWL
CLASSI_MEZZO = {"ambulanza": "Ambulanza", "autopompa": "Autopompa", "autoscala": "Autoscala"}
CLASSI_EMERGENZA = {
    "medica": "EmergenzaMedica",
    "incendio": "Incendio",
    "allagamento": "Allagamento",
    "incidente": "Incidente",
    "soccorso_altezza": "SoccorsoInAltezza",
}

# Capacità operative di ogni tipo di mezzo (restrizioni esistenziali)
CAPACITA_OPERATIVE = {
    "Ambulanza": ["EmergenzaMedica", "Incidente"],
    "Autopompa": ["Incendio", "Allagamento", "Incidente"],
    "Autoscala": ["Incendio", "SoccorsoInAltezza"],
}


def crea_ontologia(kb=None):
    """
    Costruisce l'ontologia; se kb è data, popola l'ABox con nodi e zone.

    Returns:
        L'oggetto ontologia di Owlready2.
    """
    onto = get_ontology(IRI)

    with onto:
        # --- Gerarchia dei luoghi ---
        class Luogo(Thing): pass
        class Nodo(Luogo): pass
        class Zona(Luogo): pass
        AllDisjoint([Nodo, Zona])

        class situatoIn(ObjectProperty):
            domain = [Nodo]
            range = [Zona]

        # --- Gerarchia delle emergenze ---
        class Emergenza(Thing): pass
        classi_em = {nome: type(nome, (Emergenza,), {}) for nome in CLASSI_EMERGENZA.values()}
        AllDisjoint(list(classi_em.values()))

        class avvieneIn(ObjectProperty):
            domain = [Emergenza]
            range = [Nodo]

        # --- Gerarchia dei mezzi ---
        class Mezzo(Thing): pass
        class MezzoLeggero(Mezzo): pass
        class MezzoPesante(Mezzo): pass
        AllDisjoint([MezzoLeggero, MezzoPesante])

        class Ambulanza(MezzoLeggero): pass
        class Autopompa(MezzoPesante): pass
        class Autoscala(MezzoPesante): pass

        class puoGestire(ObjectProperty):
            domain = [Mezzo]
            range = [Emergenza]

        class haBase(ObjectProperty):
            domain = [Mezzo]
            range = [Nodo]

        classi_mezzo = {"Ambulanza": Ambulanza, "Autopompa": Autopompa, "Autoscala": Autoscala}
        for nome_mezzo, gestite in CAPACITA_OPERATIVE.items():
            for nome_em in gestite:
                classi_mezzo[nome_mezzo].is_a.append(puoGestire.some(classi_em[nome_em]))

        # --- Classi definite: il ragionatore ne trova le sottoclassi ---
        class MezzoAntincendio(Thing):
            equivalent_to = [Mezzo & puoGestire.some(classi_em["Incendio"])]

        class MezzoSoccorsoSanitario(Thing):
            equivalent_to = [Mezzo & puoGestire.some(classi_em["EmergenzaMedica"])]

        class MezzoPolivalente(Thing):
            # gestisce sia incendi sia incidenti stradali
            equivalent_to = [Mezzo & puoGestire.some(classi_em["Incendio"])
                             & puoGestire.some(classi_em["Incidente"])]

        # Classi "gestibile da" per ogni tipo di emergenza, usate per
        # interrogare l'ontologia sulla compatibilità
        for nome_em, classe_em in classi_em.items():
            type(f"MezzoPer{nome_em}", (Thing,), {
                "equivalent_to": [Mezzo & puoGestire.some(classe_em)]
            })

        # --- ABox dalla KB Prolog ---
        if kb is not None:
            zone = {z: Zona(z) for z in kb.get_zone()}
            for nodo in kb.get_tutti_nodi():
                individuo = Nodo(nodo)
                zona = kb.get_zona(nodo)
                if zona in zone:
                    individuo.situatoIn = [zone[zona]]

    return onto


def ragiona(onto):
    """Esegue HermiT: classificazione della TBox e verifica di consistenza."""
    with onto:
        sync_reasoner(onto, infer_property_values=True, debug=0)
    return onto


def sottoclassi_inferite(onto, nome_classe: str) -> list:
    """Classi di mezzo che il ragionatore ha classificato sotto nome_classe."""
    classe = onto[nome_classe]
    return sorted(c.name for c in classe.descendants()
                  if c is not classe and c.name in CLASSI_MEZZO.values())


def compatibile_owl(onto, tipo_mezzo: str, tipo_emergenza: str) -> bool:
    """True se dall'ontologia si deriva che il mezzo può gestire l'emergenza."""
    classe_mezzo = onto[CLASSI_MEZZO[tipo_mezzo]]
    classe_query = onto[f"MezzoPer{CLASSI_EMERGENZA[tipo_emergenza]}"]
    return classe_query in classe_mezzo.ancestors()


def verifica_coerenza_con_kb(onto, kb) -> list:
    """
    Confronta compatibile/2 (Prolog, mondo chiuso) con l'ontologia
    (OWL, mondo aperto) su tutte le coppie tipo mezzo / tipo emergenza.

    Returns:
        Lista di discrepanze (tipo_mezzo, tipo_emergenza, prolog, owl);
        vuota se le due basi di conoscenza concordano.
    """
    discrepanze = []
    for tipo_mezzo in CLASSI_MEZZO:
        for tipo_em in CLASSI_EMERGENZA:
            p = kb.is_compatibile(tipo_mezzo, tipo_em)
            o = compatibile_owl(onto, tipo_mezzo, tipo_em)
            if p != o:
                discrepanze.append((tipo_mezzo, tipo_em, p, o))
    return discrepanze


def costruisci_e_salva(kb=None) -> object:
    """Crea l'ontologia, la salva in emergenze.owl (apribile con Protégé) e ragiona."""
    onto = crea_ontologia(kb)
    onto.save(file=FILE_OWL, format="rdfxml")
    return ragiona(onto)


if __name__ == "__main__":
    onto = costruisci_e_salva()
    print(f"Ontologia salvata in {FILE_OWL}")
    for classe in ["MezzoAntincendio", "MezzoSoccorsoSanitario", "MezzoPolivalente"]:
        print(f"  {classe}: {sottoclassi_inferite(onto, classe)}")
