# -*- coding: utf-8 -*-
"""
SVM RBF multi-classes (MIT-BIH, lead MLII) avec GroupKFold pour éviter toute fuite patient.
- Chargement des CSV train/test (split niveau patient déjà fait en amont)
- GridSearchCV avec GroupKFold(n_splits=3) en groupant par record_x
- Entraînement final + évaluation + figures
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
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
# 2) Chargement (5 classes)
# =========================

def load_data_multiclass():
    """
    Charge les CSV train/test MIT-BIH (split patient-level)
    et renvoie X_train, y_train, X_test, y_test, feature_cols, groups_train.
    groups_train = identifiant patient (record_x) pour GroupKFold.
    """

    train_path = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
    test_path  = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"

    print("Lecture train :", train_path)
    df_train = pd.read_csv(train_path, low_memory=False)

    print("Lecture test  :", test_path)
    df_test = pd.read_csv(test_path, low_memory=False)

    print("Shape train (battements) :", df_train.shape)
    print("Shape test  (battements) :", df_test.shape)

    # Features = colonnes "0".."186"
    feature_cols = [c for c in df_train.columns if c.isdigit()]
    feature_cols = sorted(feature_cols, key=lambda x: int(x))
    print("\nNombre de features :", len(feature_cols))

    # Matrices X / vecteurs y
    X_train = df_train[feature_cols].values
    X_test  = df_test[feature_cols].values
    y_train = df_train["label"].astype(int).values
    y_test  = df_test["label"].astype(int).values

    # Groupes = patient/record (pour GroupKFold)
    # record_x est du type 'x_100', 'x_101', etc.
    if "record_x" not in df_train.columns:
        raise ValueError("La colonne 'record_x' est requise dans le train pour grouper par patient.")
    groups_train = df_train["record_x"].values

    print("\nRépartition des classes (train) :")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index())

    print("\nRépartition des classes (test) :")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())

    # Sanity check : pas de mélange patient entre train/test
    inter = set(df_train["record_x"].unique()).intersection(set(df_test["record_x"].unique()))
    assert len(inter) == 0, f"Des patients sont à la fois en train et en test: {sorted(list(inter))[:5]}"

    return X_train, y_train, X_test, y_test, feature_cols, groups_train


# ==========================================
# 3) GridSearchCV SVM RBF avec GroupKFold
# ==========================================

def tune_svm_multiclass(X_train, y_train, groups_train):
    """
    Pipeline StandardScaler + SVC RBF, GridSearchCV (F1-macro),
    CV = GroupKFold(n_splits=3) groupé par patient (record_x).
    """

    svm_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",  # important vu le déséquilibre
                probability=False,
                random_state=42,
            )),
        ]
    )

    param_grid = {
        "clf__C": [1, 2, 4, 8],
        "clf__gamma": [0.001, 0.01, 0.1],
    }

    # CV groupée par patient → aucune fuite entre train/val
    cv = GroupKFold(n_splits=3)

    grid = GridSearchCV(
        estimator=svm_pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print("\n=== GridSearchCV SVM RBF (score = F1 macro, GroupKFold) ===")
    # IMPORTANT : passer groups=groups_train
    grid.fit(X_train, y_train, **{"groups": groups_train})

    print("\nMeilleurs hyperparamètres SVM :")
    print(grid.best_params_)

    print("\nMeilleur F1-macro (CV, groupée patient) :", grid.best_score_)

    best_model = grid.best_estimator_
    return best_model, grid.best_params_, grid.best_score_


# ==================================
# 4) Fonctions de visualisation
# ==================================

def plot_confusion_matrix_norm(y_true, y_pred, labels, display_labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,
        cbar=False,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
    )
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Matrice de confusion sauvegardée : {out_path}")


def plot_scores_bar(y_true, y_pred, labels, display_labels, title, out_path):
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    precisions = [report_dict[str(l)]["precision"] for l in labels]
    recalls    = [report_dict[str(l)]["recall"]    for l in labels]
    f1s        = [report_dict[str(l)]["f1-score"]  for l in labels]
    # macro avg
    precisions.append(report_dict["macro avg"]["precision"])
    recalls.append(report_dict["macro avg"]["recall"])
    f1s.append(report_dict["macro avg"]["f1-score"])

    x_labels = display_labels + ["macro avg"]
    x = np.arange(len(x_labels))
    width = 0.25

    plt.figure(figsize=(1.8 * len(x_labels), 5))
    plt.bar(x - width, precisions, width, label="precision")
    plt.bar(x,         recalls,    width, label="recall")
    plt.bar(x + width, f1s,        width, label="f1-score")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Barplot scores sauvegardé : {out_path}")


def plot_classification_report_table(y_true, y_pred, labels, title, out_path):
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).T
    df_report = df_report[["precision", "recall", "f1-score", "support"]]

    plt.figure(figsize=(10, 3 + 0.4 * len(df_report)))
    plt.axis("off")
    plt.title(title, pad=20)
    table = plt.table(
        cellText=np.round(df_report.values, 3),
        rowLabels=df_report.index,
        colLabels=df_report.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Tableau classification report sauvegardé : {out_path}")


# ==================================
# 5) Entraînement final + test
# ==================================

def main():
    # Chargement
    X_train, y_train, X_test, y_test, feature_cols, groups_train = load_data_multiclass()

    # Recherche d'hyperparamètres (CV groupée patient)
    best_model, best_params, best_cv_f1 = tune_svm_multiclass(X_train, y_train, groups_train)

    # Ré-entraînement final sur tout le train (pipeline scaler + SVM)
    print("\nRé-entraînement du meilleur SVM sur tout le train...")
    best_model.fit(X_train, y_train)

    # Sauvegarde du modèle
    model_path = PROJECT_ROOT / "models" / "svm_mit_mlii_5classes.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print("Modèle SVM 5 classes sauvegardé dans :", model_path)

    # Évaluation sur le test (patients jamais vus)
    print("\n=== Évaluation SVM 5 classes sur le test (patients jamais vus) ===")
    y_pred = best_model.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    print("F1-macro (SVM) sur le test :", f1_macro)
    print("Meilleur F1-macro (CV groupée) :", best_cv_f1)

    labels = [0, 1, 2, 3, 4]

    print("\nMatrice de confusion brute :")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    print("\nClassification report :")
    print(classification_report(y_test, y_pred, digits=3))

    print("\nRépartition des prédictions :")
    print(pd.Series(y_pred).value_counts(normalize=True).sort_index())

    # Figures
    plot_confusion_matrix_norm(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        display_labels=[str(c) for c in labels],
        title="Matrice de confusion normalisée (SVM RBF, 5 classes)",
        out_path=FIG_DIR / "svm_cm_mit_test_5classes.png",
    )

    plot_scores_bar(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        display_labels=[str(c) for c in labels],
        title="Scores sur le test (SVM RBF, 5 classes)",
        out_path=FIG_DIR / "svm_scores_mit_test_5classes.png",
    )

    plot_classification_report_table(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        title="Classification report SVM RBF (5 classes)",
        out_path=FIG_DIR / "svm_report_mit_test_5classes.png",
    )


if __name__ == "__main__":
    main()
