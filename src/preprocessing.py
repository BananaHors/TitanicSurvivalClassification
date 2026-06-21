"""
Imputacija nedostajucih vrednosti i kodiranje, sve kao sklearn transformeri.
Posto je sve u pipeline-u, statistike se uce samo na trening delu (nema leakage-a).
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import engineer_features

NUMERIC_COLS = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass"]
CATEGORICAL_COLS = ["Sex", "Embarked", "Title", "Deck"]
PASSTHROUGH_COLS = ["IsAlone"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Izvodi atribute i popunjava prazna polja (Age po tituli, Embarked/Fare standardno)."""

    def fit(self, X, y=None):
        df = engineer_features(X)
        # Age zavisi od titule (Master su deca, Mr odrasli...), pa medijana po grupi.
        self.age_by_title_ = df.groupby("Title")["Age"].median()
        self.age_global_ = df["Age"].median()
        self.embarked_mode_ = df["Embarked"].mode(dropna=True).iloc[0]
        self.fare_median_ = df["Fare"].median()

        # Ceste kategorije (>= prag) - sve retke se kasnije presavijaju u Rare / U.
        # Granica se uci samo na trening podacima.
        tc = df["Title"].value_counts()
        self.common_titles_ = set(tc[tc >= config.MIN_CATEGORY].index)
        dc = df["Deck"].value_counts()
        self.common_decks_ = set(dc[dc >= config.MIN_CATEGORY].index)
        return self

    def transform(self, X):
        df = engineer_features(X)

        df["Age"] = df["Age"].fillna(df["Title"].map(self.age_by_title_))
        df["Age"] = df["Age"].fillna(self.age_global_)  # za neviđenu titulu
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        df["Fare"] = df["Fare"].fillna(self.fare_median_)

        # Retke titule -> "Rare", retke palube -> "U" (granica nauceena u fit).
        df["Title"] = df["Title"].where(df["Title"].isin(self.common_titles_), "Rare")
        df["Deck"] = df["Deck"].where(df["Deck"].isin(self.common_decks_), "U")

        return df[NUMERIC_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS]


def build_column_transformer():
    # Pclass ostaje numericki jer 1<2<3 ima smisla; ostalo kategorijsko -> one-hot.
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("pass", "passthrough", PASSTHROUGH_COLS),
    ])
