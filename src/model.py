"""Treniranje, poredjenje modela i evaluacija."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import FeatureBuilder, build_column_transformer


def _pipeline(clf):
    return Pipeline([
        ("build", FeatureBuilder()),
        ("encode", build_column_transformer()),
        ("clf", clf),
    ])


def candidate_models():
    rs = config.RANDOM_STATE
    return {
        "LogisticRegression": _pipeline(LogisticRegression(max_iter=1000, random_state=rs)),
        "RandomForest": _pipeline(RandomForestClassifier(n_estimators=300, random_state=rs)),
        "GradientBoosting": _pipeline(GradientBoostingClassifier(random_state=rs)),
    }


def cross_validate_models(models, X, y):
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
        results[name] = {"f1_mean": round(float(scores.mean()), 4),
                         "f1_std": round(float(scores.std()), 4)}
    return results


def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Nije preziveo", "Preziveo"]),
        "y_pred": y_pred,
    }


def get_feature_importance(model, X_test, y_test):
    """
    Permutaciona vaznost na test skupu, na nivou izvedenih atributa (ne pojedinacnih
    one-hot kolona) - citljivije. Radi za bilo koji model.
    """
    X_built = model.named_steps["build"].transform(X_test)
    sub = Pipeline([("encode", model.named_steps["encode"]),
                    ("clf", model.named_steps["clf"])])
    result = permutation_importance(sub, X_built, y_test, n_repeats=15,
                                    random_state=config.RANDOM_STATE, scoring="f1")
    return list(X_built.columns), result.importances_mean
