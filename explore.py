"""
Inventar podataka - izlistava sve sirove vrednosti po koloni i koliko ih ima.
Sluzi da odluke o obradi budu POTKREPLJENE brojevima iz podataka, a ne rucnim
pregledom (npr. dokaz da paluba 'T' ima samo 1 putnika pa je presavijamo u 'U').

Pokretanje: python explore.py
"""

import pandas as pd

from src import config
from src.features import TITLE_NORMALIZE

SMALL = config.MIN_CATEGORY  # isti prag koji model koristi za presavijanje
# Previse jedinstvenih za izlistavanje (Cabin posebno obradjujemo kao palube nize).
HIGH_CARD = {"PassengerId", "Name", "Ticket", "Cabin"}


def profile_column(s, name):
    miss = int(s.isna().sum())
    nun = int(s.nunique(dropna=True))
    print(f"\n--- {name} ---  tip={s.dtype}, prazno={miss}, razlicitih={nun}")

    if name in HIGH_CARD:
        print(f"    (visoka kardinalnost - {nun} jedinstvenih; primeri: "
              f"{list(s.dropna().unique()[:3])})")
    elif pd.api.types.is_numeric_dtype(s) and nun > 20:
        # neprekidna numericka kolona -> statistike umesto liste
        print(f"    min={s.min():.2f}  median={s.median():.2f}  "
              f"mean={s.mean():.2f}  max={s.max():.2f}")
    else:
        for val, cnt in s.value_counts(dropna=False).items():
            mark = "   <-- malo" if cnt < SMALL else ""
            print(f"      {str(val):>14}: {cnt}{mark}")


def raw_decks(df):
    """Sirova paluba = prvo slovo iz Cabin, BEZ ikakve transformacije (ukljucuje T)."""
    deck = df["Cabin"].astype("string").str[0]
    return deck.value_counts(dropna=False)


def _norm_title(name):
    raw = name.split(",")[1].split(".")[0].strip()
    return TITLE_NORMALIZE.get(raw, raw)


def title_summary(df):
    """Normalizovana titula -> broj -> u sta se presavija (Rare ako < prag)."""
    vc = df["Name"].apply(_norm_title).value_counts()
    rows = [{"titula": t, "broj": int(c), "->": (t if c >= SMALL else "Rare")}
            for t, c in vc.items()]
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(config.DATA_FILE)
    print(f"Dataset: {len(df)} putnika, {df.shape[1]} kolona")

    print("\n" + "=" * 60)
    print("INVENTAR SVIH KOLONA (sirove vrednosti)")
    print("=" * 60)
    for col in df.columns:
        profile_column(df[col], col)

    print("\n" + "=" * 60)
    print("POTKREPLJENJE ODLUKA O OBRADI")
    print("=" * 60)

    print(f"\nPravilo: kategorije sa < {SMALL} clanova se presavijaju "
          f"(palube -> 'U', titule -> 'Rare').")

    print("\nSirove palube (prvo slovo Cabin):")
    for deck, cnt in raw_decks(df).items():
        mark = "   <-- presavija se u 'U'" if cnt < SMALL else ""
        print(f"      {str(deck):>6}: {cnt}{mark}")

    print("\nTitule (posle normalizacije Mlle/Mme/Ms):")
    print(title_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
