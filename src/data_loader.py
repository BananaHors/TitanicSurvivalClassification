"""Ucitavanje dataset-a i par pomocnih funkcija za pregled."""

import pandas as pd

from . import config


def load_raw():
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Nema dataset-a: {config.DATA_FILE}")
    return pd.read_csv(config.DATA_FILE)


def missing_report(df):
    """Broj i procenat praznih polja po koloni (samo kolone koje imaju prazna)."""
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    out = pd.DataFrame({"nedostaje_n": total, "nedostaje_%": pct})
    return out[out["nedostaje_n"] > 0].sort_values("nedostaje_n", ascending=False)


def overview(df):
    return {
        "broj_redova": int(df.shape[0]),
        "broj_kolona": int(df.shape[1]),
        "kolone": list(df.columns),
        "stopa_prezivljavanja": round(float(df[config.TARGET].mean()), 4),
    }
