"""Grafici - EDA i evaluacija modela. Sve se snima u outputs/figures."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from . import config

sns.set_theme(style="whitegrid", palette="muted")
SURV_COLORS = {0: "#d1495b", 1: "#2e7d8a"}


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(config.ROOT_DIR))


def plot_eda(df):
    """EDA grafici. Ocekuje izvedene kolone (Title, FamilySize, IsAlone)."""
    saved = []

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Survived", hue="Survived",
                  palette=SURV_COLORS, legend=False, ax=ax)
    ax.set_title("Raspodela prezivljavanja (0 = nije, 1 = jeste)")
    ax.set_xlabel("Survived"); ax.set_ylabel("Broj putnika")
    saved.append(_save(fig, "01_raspodela_prezivljavanja.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=df, x="Sex", y="Survived", errorbar=None, ax=ax)
    ax.set_title("Stopa prezivljavanja po polu")
    ax.set_ylabel("Udeo prezivelih")
    saved.append(_save(fig, "02_prezivljavanje_po_polu.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None, ax=ax)
    ax.set_title("Stopa prezivljavanja po klasi karte")
    ax.set_xlabel("Pclass"); ax.set_ylabel("Udeo prezivelih")
    saved.append(_save(fig, "03_prezivljavanje_po_klasi.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30,
                 palette=SURV_COLORS, ax=ax)
    ax.set_title("Raspodela starosti po ishodu")
    ax.set_xlabel("Starost"); ax.set_ylabel("Broj putnika")
    saved.append(_save(fig, "04_raspodela_starosti.png"))

    # odsecam ekstremno skupe karte da grafik bude citljiv
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df[df["Fare"] < 300], x="Fare", hue="Survived", kde=True,
                 bins=40, palette=SURV_COLORS, ax=ax)
    ax.set_title("Raspodela cene karte (Fare < 300) po ishodu")
    ax.set_xlabel("Fare"); ax.set_ylabel("Broj putnika")
    saved.append(_save(fig, "05_raspodela_cene_karte.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    order = df.groupby("Title")["Survived"].mean().sort_values().index
    sns.barplot(data=df, x="Title", y="Survived", order=order, errorbar=None, ax=ax)
    ax.set_title("Stopa prezivljavanja po tituli")
    ax.set_ylabel("Udeo prezivelih")
    saved.append(_save(fig, "06_prezivljavanje_po_tituli.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=df, x="IsAlone", y="Survived", errorbar=None, ax=ax)
    ax.set_title("Prezivljavanje: sam (1) vs sa porodicom (0)")
    ax.set_xlabel("IsAlone"); ax.set_ylabel("Udeo prezivelih")
    saved.append(_save(fig, "07_prezivljavanje_sam_vs_porodica.png"))

    num = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(num.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Korelaciona matrica (numericki atributi)")
    saved.append(_save(fig, "08_korelaciona_matrica.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=df.dropna(subset=["Embarked"]), x="Embarked", y="Survived",
                errorbar=None, ax=ax)
    ax.set_title("Stopa prezivljavanja po luci ukrcavanja")
    ax.set_ylabel("Udeo prezivelih")
    saved.append(_save(fig, "09_prezivljavanje_po_luci.png"))

    return saved


def plot_confusion(y_true, y_pred, name="10_matrica_konfuzije.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["Stvarno: 0", "Stvarno: 1"], ax=ax)
    ax.set_title("Matrica konfuzije (test skup)")
    return _save(fig, name)


def plot_roc(model, X_test, y_test, name="11_roc_kriva.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC kriva (test skup)")
    return _save(fig, name)


def plot_feature_importance(names, importances, name="12_vaznost_atributa.png", top=15):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=np.array(importances)[order], y=np.array(names)[order],
                color="#2e7d8a", ax=ax)
    ax.set_title(f"Vaznost atributa (top {top})")
    ax.set_xlabel("Vaznost"); ax.set_ylabel("Atribut")
    return _save(fig, name)


def plot_port_confound(df, name="13_luka_unutar_klase.png"):
    """
    Objasnjava NISKU vaznost atributa Embarked: razlaze prezivljavanje po luci
    po klasi. Kad se fiksira klasa, razlike izmedju luka se uglavnom istope -
    znaci je veza luka<->prezivljavanje uglavnom posredna (preko klase/cene).
    """
    d = df.dropna(subset=["Embarked"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=d, x="Embarked", y="Survived", hue="Pclass",
                errorbar=None, ax=ax)
    ax.set_title("Prezivljavanje po luci, razlozeno po klasi")
    ax.set_ylabel("Udeo prezivelih")
    ax.legend(title="Pclass")
    return _save(fig, name)
