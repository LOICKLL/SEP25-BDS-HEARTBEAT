# -*- coding: utf-8 -*-
"""
SVM RBF binaire (MIT-BIH, lead MLII) avec GroupKFold pour éviter la fuite patient.
- Chargement des CSV train/test (split patient-level déjà fait)
- Binarisation : 0 = normal, 1 = anormal (regroupe 1,2,3,4)
- GridSearchCV (F1) avec GroupKFold(n_splits=3) groupé par record_x
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
# 2) Chargement & binarisation
# =========================

def load_data_binary():
    """
    Charge les CSV train/test MIT-BIH (split patient-level),
    garde les 187 points MLII et binarise le label :
    0 = normal, 1 = anormal (classes 1,2,3,4 regroupées).
    Renvoie aussi groups_train = record_x pour GroupKFold.
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

    X_train = df_train[feature_cols].values
    X_test  = df_test[feature_cols].values

    # Label multi-classes d'origine
    y_train_multi = df_train["label"].astype(int).values
    y_test_multi  = df_test["label"].astype(int).values

    # Binarisation : 0 = normal, 1 = anormal
    y_train = (y_train_multi != 0).astype(int)
    y_test  = (y_test_multi  != 0).astype(int)

    print("\nRépartition y_bin (train) :")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index())

    print("\nRépartition y_bin (test) :")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())

    # Groupes = patient/record (pour GroupKFold)
    if "record_x" not in df_train.columns:
        raise ValueError("La colonne 'record_x' est requise dans le train pour grouper par patient.")
    groups_train = df_train["record_x"].values

    # Sanity check : aucun patient commun train/test
    inter = set(df_train["record_x"].unique()).intersection(set(df_test["record_x"].unique()))
    assert len(inter) == 0, f"Des patients sont à la fois en train et en test: {sorted(list(inter))[:5]}"

    return X_train, y_train, X_test, y_test, feature_cols, groups_train


# =========================
# 3) GridSearchCV SVM RBF binaire (GroupKFold)
# =========================

def tune_svm_binary(X_train, y_train, groups_train):
    """
    Pipeline StandardScaler + SVC RBF,
    GridSearchCV sur C et gamma avec F1 (classe positive),
    CV = GroupKFold(n_splits=3) groupée par patient (record_x).
    """
    svm_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",   # utile sur déséquilibre
                probability=True,          # pratique si on veut ROC/PR plus tard
                random_state=42,
            )),
        ]
    )

    param_grid = {
        "clf__C": [1, 2, 4, 8],
        "clf__gamma": [0.001, 0.01, 0.1],
    }

    # CV groupée par patient → aucune fuite patient entre train/val
    cv = GroupKFold(n_splits=3)

    grid = GridSearchCV(
        estimator=svm_pipe,
        param_grid=param_grid,
        scoring="f1",   # F1 sur la classe positive (anormal=1)
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print("\n=== GridSearchCV SVM RBF (score = F1, GroupKFold) ===")
    # IMPORTANT : passer groups=groups_train
    grid.fit(X_train, y_train, **{"groups": groups_train})

    print("\nMeilleurs hyperparamètres SVM :")
    print(grid.best_params_)

    print("\nMeilleur F1 (CV, groupée patient) :", grid.best_score_)

    best_model = grid.best_estimator_
    return best_model, grid.best_params_, grid.best_score_


# =========================
# 4) Fonctions de visualisation
# =========================

def plot_confusion_matrix_norm(y_true, y_pred, labels, display_labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(4, 4))
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

    # Ajout macro avg
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

    plt.figure(figsize=(8, 4))
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


# =========================
# 5) Entraînement final + test
# =========================

def main():
    # Chargement
    X_train, y_train, X_test, y_test, feature_cols, groups_train = load_data_binary()

    # GridSearch (CV groupée patient)
    best_model, best_params, best_cv_f1 = tune_svm_binary(X_train, y_train, groups_train)

    # Ré-entraînement sur tout le train
    print("\nRé-entraînement du meilleur SVM sur tout le train (par sécurité)...")
    best_model.fit(X_train, y_train)

    # Sauvegarde du modèle
    model_path = PROJECT_ROOT / "models" / "svm_mit_mlii_binary.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print("Modèle SVM binaire sauvegardé dans :", model_path)

    # Évaluation sur le test (patients jamais vus)
    print("\n=== Évaluation SVM binaire sur le test (patients jamais vus) ===")
    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print("F1 (classe anormale, SVM) :", f1)
    print("Meilleur F1 (CV groupée)  :", best_cv_f1)

    print("\nMatrice de confusion brute :")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)

    print("\nClassification report :")
    print(classification_report(y_test, y_pred, digits=3))

    print("\nRépartition des prédictions :")
    print(pd.Series(y_pred).value_counts(normalize=True).sort_index())

    # Figures
    labels = [0, 1]
    display_labels = ["Normal (0)", "Anormal (1)"]

    plot_confusion_matrix_norm(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        display_labels=display_labels,
        title="Matrice de confusion normalisée (SVM RBF binaire)",
        out_path=FIG_DIR / "svm_cm_mit_test_binary.png",
    )

    plot_scores_bar(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        display_labels=display_labels,
        title="Scores sur le test (SVM RBF binaire)",
        out_path=FIG_DIR / "svm_scores_mit_test_binary.png",
    )

    plot_classification_report_table(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        title="Classification report SVM RBF binaire",
        out_path=FIG_DIR / "svm_report_mit_test_binary.png",
    )


if __name__ == "__main__":
    main()
