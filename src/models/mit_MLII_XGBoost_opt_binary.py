"""
XGBoost binaire sur MIT-BIH (lead MLII, split patient-level)
Objectif : prédire y_bin = 0 (normal) / 1 (anormal)
- CV avec GroupKFold(n_splits=3) pour éviter toute fuite patient
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# 1) Dossiers du projet
# ----------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ----------------------------
# 2) Chargement & binarisation
# ----------------------------
def load_data_binary():
    train_path = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
    test_path  = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"

    print("Lecture train :", train_path)
    df_train = pd.read_csv(train_path, low_memory=False)

    print("Lecture test  :", test_path)
    df_test = pd.read_csv(test_path, low_memory=False)

    print("Shape train (battements) :", df_train.shape)
    print("Shape test  (battements) :", df_test.shape)

    # Features = colonnes "0".."186" triées numériquement
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

    # Sanity check : aucun patient en commun train/test
    inter = set(df_train["record_x"].unique()).intersection(set(df_test["record_x"].unique()))
    assert len(inter) == 0, f"Des patients sont à la fois en train et en test: {sorted(list(inter))[:5]}"

    return X_train, y_train, X_test, y_test, groups_train


def compute_sample_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("\nPoids de classes (balanced) :", cw_dict)
    sample_weights = np.array([cw_dict[int(yi)] for yi in y], dtype=float)
    return sample_weights


# ----------------------------
# 3) RandomizedSearchCV binaire (GroupKFold)
# ----------------------------
def tune_xgb_binary(X_train, y_train, sample_weights, groups_train):
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",   # garde "logloss" pour l'entraînement ; la métrique de sélection est ci-dessous
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    param_distributions = {
        "n_estimators":      [200, 300, 400, 500, 600],
        "learning_rate":     [0.03, 0.05, 0.07, 0.1],
        "max_depth":         [3, 4, 5, 6, 7],
        "min_child_weight":  [1, 3, 5],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
        "gamma":             [0.0, 0.1, 0.2],
        "reg_lambda":        [0.5, 1.0, 1.5, 2.0],
    }

    # CV groupée par patient → aucune fuite entre train/val
    cv = GroupKFold(n_splits=3)

    rnd_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1",        # F1 sur la classe positive (anormal)
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=42,
        refit=True,
    )

    print("\n=== Recherche d'hyperparamètres (binaire, score = F1, GroupKFold) ===")
    # IMPORTANT : passer groups=groups_train et sample_weight
    rnd_search.fit(X_train, y_train, groups=groups_train, sample_weight=sample_weights)

    print("\nMeilleurs hyperparamètres :")
    print(rnd_search.best_params_)
    print("\nMeilleur F1 (CV groupée patient) :", rnd_search.best_score_)

    best_model = rnd_search.best_estimator_
    return best_model, rnd_search.best_params_, rnd_search.best_score_


# ----------------------------
# 4) Entraînement final + test
# ----------------------------
def main():
    X_train, y_train, X_test, y_test, groups_train = load_data_binary()

    sample_weights = compute_sample_weights(y_train)

    best_model, best_params, best_cv_f1 = tune_xgb_binary(
        X_train, y_train, sample_weights, groups_train
    )

    # Ré-entraînement sur tout le train
    print("\nRé-entraînement du meilleur modèle sur tout le train...")
    best_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Sauvegarde modèle
    model_path = PROJECT_ROOT / "models" / "xgb_mit_mlii_binary.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print("Modèle sauvegardé dans :", model_path)

    # Évaluation sur le test (patients jamais vus)
    print("\n=== Évaluation sur le test (patients jamais vus) ===")
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Analyse de plusieurs seuils
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    print("\n=== Analyse des seuils (classe anormale = 1) ===")
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_thr, pos_label=1, average="binary", zero_division=0
        )
        print(
            f"Seuil = {thr:.2f} -> "
            f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}"
        )

    # Seuil choisi (ajuste au besoin)
    chosen_thr = 0.6
    print(f"\n=== Détails pour le seuil choisi : {chosen_thr:.2f} ===")
    y_pred = (y_proba >= chosen_thr).astype(int)

    f1 = f1_score(y_test, y_pred)
    print("F1 (classe anormale, seuil choisi) :", f1)
    print("Meilleur F1 (CV groupée)           :", best_cv_f1)

    # Matrice de confusion (non normalisée)
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion :")
    print(cm)

    # Rapport texte
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    print("\nClassification report :")
    print(classification_report(y_test, y_pred, digits=3))

    print("\nRépartition des prédictions :")
    print(pd.Series(y_pred).value_counts(normalize=True).sort_index())

    # ----------------------------
    # Figures
    # ----------------------------
    img_dir = PROJECT_ROOT / "reports" / "figures" / "Modelisation_XGboost"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) Matrice de confusion normalisée
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,
        cbar=False,
        xticklabels=["Normal (0)", "Anormal (1)"],
        yticklabels=["Normal (0)", "Anormal (1)"],
    )
    plt.title("Matrice de confusion normalisée (test)")
    plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout()
    img_cm_path = img_dir / "xgb_cm_mit_test.png"
    plt.savefig(img_cm_path, dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée dans : {img_cm_path}")

    # 2) Scores (precision/recall/F1)
    report_df = pd.DataFrame(report_dict).T
    scores_plot = report_df.loc[["0", "1", "macro avg"], ["precision", "recall", "f1-score"]]
    ax = scores_plot.plot(kind="bar", figsize=(6, 4))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.ylim(0, 1.0)
    plt.title("Scores sur le test (XGBoost binaire)")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    img_scores_path = img_dir / "xgb_scores_mit_test.png"
    plt.savefig(img_scores_path, dpi=150)
    plt.close()
    print(f"Scores (precision/recall/F1) sauvegardés dans : {img_scores_path}")

    # 3) Tableau de scores
    table_df = report_df.loc[
        ["0", "1", "accuracy", "macro avg", "weighted avg"],
        ["precision", "recall", "f1-score", "support"],
    ].copy()
    for col in ["precision", "recall", "f1-score"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")
    table_df["support"] = table_df["support"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    plt.title("Classification report XGBOOST Binaire", pad=10)
    plt.tight_layout()
    img_report_path = img_dir / "xgb_report_mit_test.png"
    plt.savefig(img_report_path, dpi=150)
    plt.close()
    print(f"Tableau de scores sauvegardé dans : {img_report_path}")


if __name__ == "__main__":
    main()
