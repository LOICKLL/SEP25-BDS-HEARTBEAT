# -*- coding: utf-8 -*-
"""
Réseau de neurones fully-connected (MLP) binaire sur MIT-BIH (lead MLII, split patient-level)
Objectif : prédire y_bin = 0 (normal) / 1 (anormal)

Entrée : vecteur de 187 features (points du battement)
Sauvegardes :
  - modèle Keras : models/nn_mit_mlii_binary.keras
  - historique d'entraînement : models/nn_mit_mlii_binary_history.pkl
  - scaler : models/scaler_mit_mlii_binary.pkl
  - ordre des features : models/nn_mit_mlii_binary_features.json
  - figures : reports/figures/Modelisation_NN/*
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models


# ----------------------------
# 1) Dossiers du projet
# ----------------------------
THIS_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR   = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR    = PROJECT_ROOT / "reports" / "figures" / "Modelisation_NN"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


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

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    X_test  = df_test[feature_cols].to_numpy(dtype=np.float32)

    # Label multi-classes d'origine -> binaire
    y_train_multi = df_train["label"].astype(int).values
    y_test_multi  = df_test["label"].astype(int).values
    y_train = (y_train_multi != 0).astype(int)
    y_test  = (y_test_multi  != 0).astype(int)

    print("\nRépartition y_bin (train) :")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index())
    print("\nRépartition y_bin (test) :")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())

    return X_train, y_train, X_test, y_test, feature_cols


def compute_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("\nPoids de classes (balanced) :", cw_dict)
    return cw_dict


# ----------------------------
# 3) Modèle NN (MLP)
# ----------------------------
def build_nn_model(input_dim):
    model = models.Sequential(name="nn_mit_mlii_binary")
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()
    return model


# ----------------------------
# 4) Entraînement + test
# ----------------------------
def main():
    # Reproductibilité
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, y_train, X_test, y_test, feature_cols = load_data_binary()

    # ===== Normalisation (StandardScaler) =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # >>> Sauvegarder le scaler et l'ordre des features <<<
    scaler_path = MODELS_DIR / "scaler_mit_mlii_binary.pkl"
    joblib.dump(scaler, scaler_path)
    print("Scaler sauvegardé :", scaler_path)

    features_path = MODELS_DIR / "nn_mit_mlii_binary_features.json"
    features_path.write_text(json.dumps(feature_cols))
    print("Ordre des features sauvegardé :", features_path)

    # Split interne train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train,
    )

    # Poids de classes
    class_weights = compute_class_weights(y_tr)

    # Modèle
    input_dim = X_train_scaled.shape[1]
    model = build_nn_model(input_dim)

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=5, mode="max", restore_best_weights=True,
    )
    model_path = MODELS_DIR / "nn_mit_mlii_binary.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path), monitor="val_auc", mode="max",
        save_best_only=True, verbose=1,
    )

    # Entraînement
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint_cb],
        verbose=2,
    )
    print(f"\nMeilleur modèle NN sauvegardé dans : {model_path}")

    # Historique d'entraînement
    history_path = MODELS_DIR / "nn_mit_mlii_binary_history.pkl"
    joblib.dump(history.history, history_path)
    print("Historique d'entraînement sauvegardé dans :", history_path)

    # Évaluation sur le test
    print("\n=== Évaluation NN sur le test (patients jamais vus) ===")
    best_model = tf.keras.models.load_model(model_path)
    y_proba = best_model.predict(X_test_scaled).ravel()

    # Analyse de seuils
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    print("\n=== Analyse des seuils (classe anormale = 1) ===")
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_thr, pos_label=1, average="binary", zero_division=0
        )
        print(f"Seuil = {thr:.2f} -> precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

    chosen_thr = 0.5
    print(f"\n=== Détails pour le seuil choisi : {chosen_thr:.2f} ===")
    y_pred = (y_proba >= chosen_thr).astype(int)

    f1 = f1_score(y_test, y_pred)
    print("F1 (classe anormale, seuil choisi, NN) :", f1)

    # Matrice de confusion & report
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion NN :\n", cm)
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    print("\nClassification report NN :")
    print(classification_report(y_test, y_pred, digits=3))

    # ------- Figures -------
    hist = history.history

    # Loss
    plt.figure(figsize=(5, 3))
    plt.plot(hist["loss"], label="loss train")
    plt.plot(hist["val_loss"], label="loss val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Évolution de la loss - NN (MLP)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "nn_loss.png", dpi=150); plt.close()

    # Accuracy
    if "accuracy" in hist:
        plt.figure(figsize=(5, 3))
        plt.plot(hist["accuracy"], label="acc train")
        plt.plot(hist["val_accuracy"], label="acc val")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        plt.title("Évolution de l'accuracy - NN (MLP)")
        plt.tight_layout(); plt.savefig(FIG_DIR / "nn_accuracy.png", dpi=150); plt.close()

    # CM normalisée
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=["Normal (0)", "Anormal (1)"],
                yticklabels=["Normal (0)", "Anormal (1)"])
    plt.title("Matrice de confusion normalisée (test) - NN (MLP)")
    plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout(); plt.savefig(FIG_DIR / "nn_cm_mit_test.png", dpi=150); plt.close()

    # Barres de scores
    report_df = pd.DataFrame(report_dict).T
    scores_plot = report_df.loc[["0", "1", "macro avg"], ["precision", "recall", "f1-score"]]
    ax = scores_plot.plot(kind="bar", figsize=(6, 4))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.ylim(0, 1.0); plt.title("Scores sur le test (NN binaire)"); plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "nn_scores_mit_test.png", dpi=150); plt.close()

    # Tableau de scores
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
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.2, 1.4)
    plt.title("Classification report NN Binaire (MLP)", pad=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "nn_report_mit_test.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
