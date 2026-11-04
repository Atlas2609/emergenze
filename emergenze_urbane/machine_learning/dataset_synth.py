"""
dataset_synth.py — Generazione di un dataset sintetico per la predizione
                   del supporto aggiuntivo nelle emergenze urbane

Il dataset non è generato in modo puramente casuale: la variabile
target "supporto_aggiuntivo" è derivata da una logica probabilistica
che lega le feature (meteo, orario, zona, gravità) al bisogno effettivo
di rinforzi, così che il modello di ML abbia un segnale reale da
apprendere e non rumore.
"""

import numpy as np
import pandas as pd


def genera_dataset(n_campioni: int = 800, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintetico con feature plausibili per la predizione
    del supporto aggiuntivo in scenari di emergenza urbana.

    Feature:
    - meteo: condizione meteorologica (sereno, pioggia, nebbia, pioggia_forte)
    - orario: fascia oraria (notte, mattina, pomeriggio, sera)
    - zona: zona della città (zona_nord, zona_centro, zona_sud, zona_est)
    - gravita: livello di gravità (bassa, media, alta)
    - tipo_emergenza: tipo di emergenza (medica, incendio, allagamento, incidente)

    Target:
    - supporto_aggiuntivo: 0/1 (se l'emergenza richiederà supporto extra)

    Logica probabilistica per il target:
    - Gravità alta → alta probabilità di supporto
    - Meteo avverso → aumenta la probabilità
    - Orario notturno → risorse ridotte, più probabile supporto
    - Zona sud/industriale → più distante dai centri di soccorso
    - Tipo incendio/allagamento → più probabile supporto

    Args:
        n_campioni: numero di istanze da generare.
        random_state: seed per la riproducibilità.

    Returns:
        DataFrame pandas con le feature e il target.
    """
    rng = np.random.RandomState(random_state)

    valori_meteo = ["sereno", "pioggia", "nebbia", "pioggia_forte"]
    valori_orario = ["notte", "mattina", "pomeriggio", "sera"]
    valori_zona = ["zona_nord", "zona_centro", "zona_sud", "zona_est"]
    valori_gravita = ["bassa", "media", "alta"]
    valori_tipo = ["medica", "incendio", "allagamento", "incidente"]

    prob_meteo = [0.40, 0.30, 0.15, 0.15]
    prob_orario = [0.20, 0.30, 0.30, 0.20]
    prob_zona = [0.25, 0.30, 0.25, 0.20]
    prob_gravita = [0.35, 0.40, 0.25]
    prob_tipo = [0.35, 0.20, 0.15, 0.30]

    meteo = rng.choice(valori_meteo, size=n_campioni, p=prob_meteo)
    orario = rng.choice(valori_orario, size=n_campioni, p=prob_orario)
    zona = rng.choice(valori_zona, size=n_campioni, p=prob_zona)
    gravita = rng.choice(valori_gravita, size=n_campioni, p=prob_gravita)
    tipo_emergenza = rng.choice(valori_tipo, size=n_campioni, p=prob_tipo)

    # La probabilità di supporto è una combinazione lineare di fattori
    # di dominio, non un valore casuale indipendente dalle feature
    probabilita_supporto = np.zeros(n_campioni)

    mappa_gravita = {"bassa": 0.10, "media": 0.30, "alta": 0.65}
    for i in range(n_campioni):
        probabilita_supporto[i] = mappa_gravita[gravita[i]]

    mappa_meteo = {"sereno": 0.0, "pioggia": 0.10, "nebbia": 0.05, "pioggia_forte": 0.20}
    for i in range(n_campioni):
        probabilita_supporto[i] += mappa_meteo[meteo[i]]

    mappa_orario = {"notte": 0.15, "mattina": 0.0, "pomeriggio": 0.0, "sera": 0.05}
    for i in range(n_campioni):
        probabilita_supporto[i] += mappa_orario[orario[i]]

    mappa_zona = {"zona_nord": 0.0, "zona_centro": -0.05, "zona_sud": 0.10, "zona_est": 0.05}
    for i in range(n_campioni):
        probabilita_supporto[i] += mappa_zona[zona[i]]

    mappa_tipo = {"medica": 0.0, "incendio": 0.15, "allagamento": 0.10, "incidente": 0.05}
    for i in range(n_campioni):
        probabilita_supporto[i] += mappa_tipo[tipo_emergenza[i]]

    probabilita_supporto = np.clip(probabilita_supporto, 0.05, 0.95)
    supporto_aggiuntivo = (rng.random(n_campioni) < probabilita_supporto).astype(int)

    df = pd.DataFrame({
        "meteo": meteo,
        "orario": orario,
        "zona": zona,
        "gravita": gravita,
        "tipo_emergenza": tipo_emergenza,
        "supporto_aggiuntivo": supporto_aggiuntivo,
    })

    return df


def prepara_features(df: pd.DataFrame) -> tuple:
    """
    Prepara le feature per il modello ML: encoding delle variabili
    categoriche tramite One-Hot Encoding (pd.get_dummies).

    Args:
        df: DataFrame con le feature categoriche e il target.

    Returns:
        Tupla (X, y, nomi_feature) dove:
        - X: numpy array con le feature codificate
        - y: numpy array con il target
        - nomi_feature: lista dei nomi delle colonne codificate
    """
    feature_cols = ["meteo", "orario", "zona", "gravita", "tipo_emergenza"]
    target_col = "supporto_aggiuntivo"

    X_encoded = pd.get_dummies(df[feature_cols], drop_first=False)

    X = X_encoded.values.astype(float)
    y = df[target_col].values.astype(int)
    nomi_feature = list(X_encoded.columns)

    return X, y, nomi_feature


if __name__ == "__main__":
    df = genera_dataset(n_campioni=1000)
    print(f"Dataset generato: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"\nDistribuzione target:")
    print(df["supporto_aggiuntivo"].value_counts(normalize=True))
    print(f"\nPrime 5 righe:")
    print(df.head())

    X, y, nomi = prepara_features(df)
    print(f"\nFeature codificate: {X.shape[1]} colonne")
    print(f"Nomi: {nomi}")
