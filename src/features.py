"""
Feature engineering - izvlacenje korisnih atributa iz sirovih kolona.
Sve je row-wise i deterministicko, pa moze pre podele na train/test.

Napomena: presavijanje retkih titula/paluba (count < prag) NIJE ovde, nego u
preprocessing.FeatureBuilder, jer zavisi od ucestanosti i mora da se uci samo
na trening podacima (bez curenja informacija).
"""

# Jezicke varijante iste titule -> normalizacija (npr. francusko Mlle = Miss).
TITLE_NORMALIZE = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]


def _title(name):
    # format imena je uvek "Prezime, Titula. Ostalo"
    try:
        raw = name.split(",")[1].split(".")[0].strip()
    except IndexError:
        return "Rare"
    return TITLE_NORMALIZE.get(raw, raw)


def engineer_features(df):
    df = df.copy()

    df["Title"] = df["Name"].apply(_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # iz "C85" nas zanima samo paluba "C"; prazno -> U (unknown)
    df["Deck"] = df["Cabin"].astype("string").str[0].fillna("U").astype(str)

    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])
