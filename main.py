"""
Pokrece ceo pipeline: ucitavanje -> EDA -> treniranje -> evaluacija -> izvestaji.
Pokretanje: python main.py   (rezultati zavrsavaju u outputs/)
"""

import json
import joblib
from sklearn.model_selection import train_test_split

from src import config, data_loader, model, visualization
from src.features import engineer_features


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def build_interpretation(eda_df, importances, metrics, best_name):
    """Sklapa tekstualno tumacenje iz stvarnih brojeva (ide u izvestaj.txt)."""
    names, vals = importances
    ranked = sorted(zip(names, vals), key=lambda t: t[1], reverse=True)
    top3 = ", ".join(f"{n} ({v:.3f})" for n, v in ranked[:3])

    by_sex = eda_df.groupby("Sex")["Survived"].mean()
    by_class = eda_df.groupby("Pclass")["Survived"].mean()

    return "\n".join([
        "TUMACENJE REZULTATA",
        "=" * 70,
        "",
        f"Najbolji model: {best_name}",
        f"  Tacnost (accuracy):   {metrics['accuracy']}",
        f"  Preciznost (precision): {metrics['precision']}",
        f"  Odziv (recall):       {metrics['recall']}",
        f"  F1-skor:              {metrics['f1']}",
        f"  ROC-AUC:              {metrics['roc_auc']}",
        "",
        "Najuticajniji atributi (permutaciona vaznost): " + top3,
        "",
        "Kljucni nalazi iz podataka:",
        f"  - Pol je najjaci pojedinacni faktor. Zene prezivljavaju "
        f"{by_sex.get('female', 0):.1%}, muskarci {by_sex.get('male', 0):.1%} "
        f"(princip 'zene i deca prvi').",
        f"  - Klasa karte: 1. klasa {by_class.get(1, 0):.1%}, 2. klasa "
        f"{by_class.get(2, 0):.1%}, 3. klasa {by_class.get(3, 0):.1%} - visi "
        f"polozaj, veca sansa.",
        "  - Titula sazima pol, uzrast i status; 'Master' (decaci) i 'Mrs/Miss' "
        "imaju visoke stope.",
        "  - Cena karte prati klasu: skuplja karta -> veca sansa.",
        "  - Velicina porodice je nelinearna: samci i velike porodice prezivljavaju "
        "redje od malih (2-4 clana).",
        "",
        "Preporuke za dalje:",
        "  - Optimizacija hiperparametara (GridSearch) i podesavanje praga "
        "(preciznost vs odziv).",
        "  - Jos atributa: velicina grupe po Ticket-u, interakcije Pclass*Sex.",
        "  - Robusnija imputacija i analiza gresaka modela.",
    ])


def main():
    config.ensure_dirs()

    header("1) UCITAVANJE I PREGLED PODATAKA")
    df = data_loader.load_raw()
    ov = data_loader.overview(df)
    print(f"Dataset: {ov['broj_redova']} redova x {ov['broj_kolona']} kolona")
    print(f"Stopa prezivljavanja: {ov['stopa_prezivljavanja']:.2%}")
    miss = data_loader.missing_report(df)
    print("\nNedostajuce vrednosti:")
    print(miss.to_string() if not miss.empty else "  (nema)")

    header("2) EKSPLORATIVNA ANALIZA (EDA)")
    eda_df = engineer_features(df)
    eda_df["Survived"] = df["Survived"]
    saved_eda = visualization.plot_eda(eda_df)
    print(f"Snimljeno {len(saved_eda)} EDA grafika u outputs/figures/")

    header("3) PODELA NA TRAIN / TEST")
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} (stratifikovano)")

    header("4) POREDJENJE MODELA (5-fold CV, F1)")
    models = model.candidate_models()
    cv_results = model.cross_validate_models(models, X_train, y_train)
    for name, r in cv_results.items():
        print(f"  {name:20s}  F1 = {r['f1_mean']:.4f} (+/- {r['f1_std']:.4f})")
    best_name = max(cv_results, key=lambda n: cv_results[n]["f1_mean"])
    print(f"\nNajbolji po CV: {best_name}")

    header("5) TRENIRANJE I EVALUACIJA NA TEST SKUPU")
    best = models[best_name]
    best.fit(X_train, y_train)
    metrics = model.evaluate_on_test(best, X_test, y_test)
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall   : {metrics['recall']}")
    print(f"  F1       : {metrics['f1']}")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")
    print("\n" + metrics["classification_report"])

    header("6) VAZNOST ATRIBUTA I GRAFICI EVALUACIJE")
    names, vals = model.get_feature_importance(best, X_test, y_test)
    visualization.plot_confusion(y_test, metrics["y_pred"])
    visualization.plot_roc(best, X_test, y_test)
    visualization.plot_feature_importance(names, vals)
    # Tumacenje: zasto je luka (Embarked) niska iako grafik 09 pokazuje skok.
    visualization.plot_port_confound(eda_df)
    print("Snimljeni grafici: matrica konfuzije, ROC kriva, vaznost atributa, "
          "luka-unutar-klase.")

    header("7) CUVANJE MODELA I IZVESTAJA")
    model_path = config.MODELS_DIR / "best_model.joblib"
    joblib.dump(best, model_path)

    metrics_json = {
        "pregled": ov,
        "nedostajuce_vrednosti": miss.reset_index(names="kolona").to_dict("records"),
        "cv_rezultati": cv_results,
        "najbolji_model": best_name,
        "test_metrike": {k: v for k, v in metrics.items()
                         if k not in ("classification_report", "y_pred")},
        "vaznost_atributa": sorted(
            ({"atribut": n, "vaznost": round(float(v), 5)} for n, v in zip(names, vals)),
            key=lambda d: d["vaznost"], reverse=True),
    }
    with open(config.METRICS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    interpretation = build_interpretation(eda_df, (names, vals), metrics, best_name)
    with open(config.METRICS_DIR / "izvestaj.txt", "w", encoding="utf-8") as f:
        f.write(interpretation + "\n\n")
        f.write("DETALJAN IZVESTAJ KLASIFIKACIJE\n")
        f.write("=" * 70 + "\n")
        f.write(metrics["classification_report"])

    print(f"Model: {model_path.relative_to(config.ROOT_DIR)}")
    print("Metrike: outputs/metrics/metrics.json | Tumacenje: outputs/metrics/izvestaj.txt")
    header("GOTOVO")


if __name__ == "__main__":
    main()
