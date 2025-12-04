# -*- coding: utf-8 -*-
"""
SVM RBF binaire (MIT-BIH, lead MLII) avec GroupKFold (groupé par record_x).
- Charge les CSV train/test déjà séparés par patients
- CIBLE = colonne 'binary' (0 = normal, 1 = anormal)
- GridSearchCV (F1) + ré-entraînement + figures (CM + barres + tableau report)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1) Dossiers du projet
# =========================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "Modelisation_SVM"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2) Chargement (cible = 'binary')
# =========================
def load_data_binary():
    train_path = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
    test_path  = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"

    print("Lecture train :", train_path)
    df_train = pd.read_csv(train_path, low_memory=False)

    print("Lecture test  :", test_path)
    df_test = pd.read_csv(test_path, low_memory=False)

    # Sécurité : la colonne 'binary' doit exister
    for name, df_ in [("train", df_train), ("test", df_test)]:
        if "binary" not in df_.columns:
            raise ValueError(
                f"Le CSV {name} ne contient pas la colonne 'binary'. "
                f"Assure-toi d'avoir créé cette colonne (0 normal / 1 anormal)."
            )
        if "record_x" not in df_.columns:
            raise ValueError(f"Le CSV {name} doit contenir la colonne 'record_x' (identifiant patient).")

    # Colonnes features : '0'..'186'
    feature_cols = sorted([c for c in df_train.columns if c.isdigit()], key=lambda x: int(x))
    print("Nb features :", len(feature_cols))

    # X / y / groups
    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train["binary"].astype(int).to_numpy()
    X_test  = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_test  = df_test["binary"].astype(int).to_numpy()

    groups_train = df_train["record_x"].astype(str).to_numpy()

    # Sanity check : pas de patient commun
    inter = set(df_train["record_x"].unique()).intersection(set(df_test["record_x"].unique()))
    assert len(inter) == 0, f"Des patients sont à la fois en train et en test: {sorted(list(inter))[:5]}"

    print("\nRépartition y_train (binary):")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index())
    print("\nRépartition y_test (binary):")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())

    return X_train, y_train, X_test, y_test, feature_cols, groups_train

# =========================
# 3) GridSearchCV SVM RBF binaire (GroupKFold)
# =========================
def tune_svm_binary(X_train, y_train, groups_train):
    svm_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42)),
    ])

    param_grid = {
        "clf__C": [1, 2, 4, 8],
        "clf__gamma": [0.001, 0.01, 0.1],
    }

    cv = GroupKFold(n_splits=3)

    grid = GridSearchCV(
        estimator=svm_pipe,
        param_grid=param_grid,
        scoring="f1",          # F1 de la classe positive (1 = anormal)
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print("\n=== GridSearchCV SVM RBF (F1, GroupKFold) ===")
    grid.fit(X_train, y_train, **{"groups": groups_train})
    print("Meilleurs hyperparamètres :", grid.best_params_)
    print("Meilleur F1 CV (groupé)   :", grid.best_score_)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# =========================
# 4) Plots utilitaires
# =========================
def plot_confusion_matrix_norm(y_true, y_pred, labels, display_labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                cbar=False, xticklabels=display_labels, yticklabels=display_labels, square=True)
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Sauvé :", out_path)

def plot_scores_bar(y_true, y_pred, labels, display_labels, title, out_path):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    precisions = [rep[str(l)]["precision"] for l in labels] + [rep["macro avg"]["precision"]]
    recalls    = [rep[str(l)]["recall"]    for l in labels] + [rep["macro avg"]["recall"]]
    f1s        = [rep[str(l)]["f1-score"]  for l in labels] + [rep["macro avg"]["f1-score"]]
    x_labels   = display_labels + ["macro avg"]
    x = np.arange(len(x_labels)); w = 0.25

    plt.figure(figsize=(1.8*len(x_labels), 5))
    plt.bar(x - w, precisions, w, label="precision")
    plt.bar(x,     recalls,    w, label="recall")
    plt.bar(x + w, f1s,        w, label="f1-score")
    plt.xticks(x, x_labels); plt.ylim(0, 1)
    plt.ylabel("Score"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print("Sauvé :", out_path)

def plot_classification_report_table(y_true, y_pred, labels, title, out_path):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(rep).T[["precision", "recall", "f1-score", "support"]]
    plt.figure(figsize=(8, 4)); plt.axis("off"); plt.title(title, pad=20)
    table = plt.table(cellText=np.round(df_rep.values, 3),
                      rowLabels=df_rep.index, colLabels=df_rep.columns,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.4)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print("Sauvé :", out_path)

# =========================
# 5) Entraînement + évaluation
# =========================
def main():
    X_train, y_train, X_test, y_test, feature_cols, groups_train = load_data_binary()
    best_model, best_params, best_cv_f1 = tune_svm_binary(X_train, y_train, groups_train)

    print("\nRé-entraînement sur tout le train…")
    best_model.fit(X_train, y_train)

    model_path = PROJECT_ROOT / "models" / "svm_mit_mlii_binary.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print("Modèle sauvegardé :", model_path)

    print("\n=== Évaluation sur le TEST (patients jamais vus) ===")
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print("F1 test (classe 1) :", f1)
    print("Meilleur F1 CV     :", best_cv_f1)
    print("\nClassification report :")
    print(classification_report(y_test, y_pred, digits=3))
    print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))

    labels = [0, 1]; display = ["Normal (0)", "Anormal (1)"]
    plot_confusion_matrix_norm(y_test, y_pred, labels, display,
                               "Matrice de confusion normalisée (SVM RBF binaire)",
                               FIG_DIR / "svm_cm_mit_test_binary.png")
    plot_scores_bar(y_test, y_pred, labels, display,
                    "Scores sur le test (SVM RBF binaire)",
                    FIG_DIR / "svm_scores_mit_test_binary.png")
    plot_classification_report_table(y_test, y_pred, labels,
                    "Classification report SVM RBF binaire",
                    FIG_DIR / "svm_report_mit_test_binary.png")

if __name__ == "__main__":
    main()
