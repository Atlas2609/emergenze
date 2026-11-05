"""
dataset_synth.py — Generazione di un dataset sintetico per la predizione
                   del supporto aggiuntivo nelle emergenze urbane

Non disponendo di uno storico reale degli interventi, il dataset è
generato da un modello causale esplicito (descritto in genera_dataset),
così che le dipendenze tra le variabili siano note e documentabili:

    meteo, zona, orario  ->  ritardo_arrivo
    gravita, tipo_emergenza, ritardo_arrivo, orario  ->  supporto_aggiuntivo

Questa scelta è dichiarata nella relazione: le prestazioni misurate
valgono per questo modello generativo e non per dati reali.

Il generatore introduce anche una piccola quota di valori mancanti
(dati raccolti in modo incompleto dagli operatori), che vengono gestiti
nella fase di data cleaning (pulisci_dataset).
"""

import numpy as np
import pandas as pd


VALORI = {
    "meteo": ["sereno", "pioggia", "nebbia", "pioggia_forte", "alluvione"],
    "orario": ["notte", "mattina", "pomeriggio", "sera"],
    "zona": ["zona_nord", "zona_centro", "zona_sud", "zona_est"],
    "gravita": ["bassa", "media", "alta"],
    "tipo_emergenza": ["medica", "incendio", "allagamento", "incidente", "soccorso_altezza"],
    "ritardo_arrivo": ["basso", "medio", "alto"],
}
FEATURE = ["meteo", "orario", "zona", "gravita", "tipo_emergenza", "ritardo_arrivo"]
TARGET = "supporto_aggiuntivo"

# Colonne in cui il generatore inserisce valori mancanti
COLONNE_CON_MANCANTI = ["meteo", "orario", "zona"]

# Soglie (in unità di costo A*) per discretizzare il costo del percorso
# del mezzo assegnato nella variabile ritardo_arrivo
SOGLIA_RITARDO_MEDIO = 8.0
SOGLIA_RITARDO_ALTO = 15.0


def ritardo_da_costo(costo: float) -> str:
    """Discretizza il costo A* del percorso in basso/medio/alto."""
    if costo < SOGLIA_RITARDO_MEDIO:
        return "basso"
    if costo < SOGLIA_RITARDO_ALTO:
        return "medio"
    return "alto"


def genera_dataset(
    n_campioni: int = 2000,
    random_state: int = 42,
    quota_mancanti: float = 0.03,
    con_probabilita_vera: bool = False,
) -> pd.DataFrame:
    """
    Genera il dataset sintetico secondo il modello causale del modulo.

    Modello del ritardo di arrivo (dipende da meteo, zona e orario):
        r = effetto_meteo + effetto_zona + effetto_orario
        P(alto) = 0.10 + 0.55 r,  P(medio) = 0.35,  P(basso) = il resto

    Modello del supporto aggiuntivo:
        p = base_gravita + effetto_tipo + effetto_ritardo + effetto_orario
        supporto ~ Bernoulli(clip(p, 0.03, 0.97))

    Args:
        n_campioni: numero di istanze da generare.
        random_state: seed per la riproducibilità.
        quota_mancanti: frazione di celle mancanti nelle colonne
                        COLONNE_CON_MANCANTI.
        con_probabilita_vera: se True aggiunge la colonna p_vera con la
                        probabilità di supporto usata dal generatore
                        (serve solo per calcolare il limite teorico).

    Returns:
        DataFrame pandas con feature (con eventuali NaN) e target.
    """
    rng = np.random.RandomState(random_state)

    meteo = rng.choice(VALORI["meteo"], size=n_campioni, p=[0.38, 0.27, 0.13, 0.14, 0.08])
    orario = rng.choice(VALORI["orario"], size=n_campioni, p=[0.20, 0.30, 0.30, 0.20])
    zona = rng.choice(VALORI["zona"], size=n_campioni, p=[0.25, 0.30, 0.25, 0.20])
    gravita = rng.choice(VALORI["gravita"], size=n_campioni, p=[0.35, 0.40, 0.25])
    tipo = rng.choice(VALORI["tipo_emergenza"], size=n_campioni,
                      p=[0.32, 0.18, 0.12, 0.28, 0.10])

    eff_meteo = {"sereno": 0.0, "pioggia": 0.20, "nebbia": 0.25,
                 "pioggia_forte": 0.40, "alluvione": 0.60}
    eff_zona = {"zona_nord": 0.0, "zona_centro": 0.05, "zona_sud": 0.20, "zona_est": 0.10}
    eff_orario_traffico = {"notte": 0.0, "mattina": 0.15, "pomeriggio": 0.10, "sera": 0.20}

    ritardo = []
    for m, z, o in zip(meteo, zona, orario):
        r = eff_meteo[m] + eff_zona[z] + eff_orario_traffico[o]
        p_alto = 0.10 + 0.55 * r
        p_basso = max(1.0 - p_alto - 0.35, 0.02)
        p = np.array([p_basso, 0.35, p_alto])
        ritardo.append(rng.choice(VALORI["ritardo_arrivo"], p=p / p.sum()))
    ritardo = np.array(ritardo)

    base_gravita = {"bassa": 0.08, "media": 0.28, "alta": 0.60}
    eff_tipo = {"medica": 0.0, "incidente": 0.08, "allagamento": 0.10,
                "incendio": 0.15, "soccorso_altezza": 0.12}
    eff_ritardo = {"basso": 0.0, "medio": 0.10, "alto": 0.22}
    # Di notte il personale in servizio è ridotto
    eff_orario_personale = {"notte": 0.12, "mattina": 0.0, "pomeriggio": 0.0, "sera": 0.04}

    p_supporto = np.array([
        base_gravita[g] + eff_tipo[t] + eff_ritardo[r] + eff_orario_personale[o]
        for g, t, r, o in zip(gravita, tipo, ritardo, orario)
    ])
    p_supporto = np.clip(p_supporto, 0.03, 0.97)
    supporto = (rng.random_sample(n_campioni) < p_supporto).astype(int)

    df = pd.DataFrame({
        "meteo": meteo,
        "orario": orario,
        "zona": zona,
        "gravita": gravita,
        "tipo_emergenza": tipo,
        "ritardo_arrivo": ritardo,
        TARGET: supporto,
    })

    if con_probabilita_vera:
        df["p_vera"] = p_supporto

    for col in COLONNE_CON_MANCANTI:
        mancanti = rng.random_sample(n_campioni) < quota_mancanti
        df.loc[mancanti, col] = np.nan

    return df


def valori_noti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sostituisce con NaN i valori non previsti da VALORI (es. un meteo
    'sconosciuta' letto da una fonte incompleta), così vengono trattati
    come mancanti e imputati invece di far fallire il modello.
    """
    df = pd.DataFrame(df, columns=FEATURE).copy()
    for col in FEATURE:
        df[col] = df[col].where(df[col].isin(VALORI[col]))
    return df


def limite_teorico(df: pd.DataFrame) -> dict:
    """
    Prestazioni del classificatore bayesiano ottimo, che conosce le vere
    probabilità del generatore: nessun modello può fare meglio in media,
    perché il target contiene rumore irriducibile (è estratto a caso).
    Richiede un dataset generato con con_probabilita_vera=True.
    """
    from sklearn.metrics import roc_auc_score
    p, y = df["p_vera"].values, df[TARGET].values
    return {
        "accuracy": float(np.mean(np.maximum(p, 1 - p))),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def prepara_features(df: pd.DataFrame) -> tuple:
    """
    Separa feature e target. Le righe senza target vengono scartate;
    i valori mancanti delle feature restano NaN e sono imputati DENTRO
    la pipeline (vedi modello_predittivo.crea_preprocessore), così la
    moda usata per l'imputazione è calcolata solo sui fold di training
    e non c'è fuga di informazione verso i fold di test.

    Le righe duplicate NON vengono rimosse: in un registro di eventi due
    righe uguali sono due interventi distinti e legittimi.

    Returns:
        Tupla (X, y, nomi_feature) con X DataFrame categorico.
    """
    df = df.dropna(subset=[TARGET])
    return df[FEATURE].copy(), df[TARGET].values.astype(int), list(FEATURE)


if __name__ == "__main__":
    df = genera_dataset()
    print(f"Dataset generato: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"Valori mancanti per colonna:\n{df.isna().sum()}")
    print(f"\nDistribuzione target:\n{df[TARGET].value_counts(normalize=True)}")
    X, y, nomi = prepara_features(df)
    print(f"\nFeature codificate: {X.shape[1]} colonne")
