"""
CNN 1D binaire sur MIT-BIH (lead MLII, split patient-level)
Objectif : prédire y_bin = 0 (normal) / 1 (anormal)
Architecture simple de CNN 1D avec EarlyStopping.
"""

import pandas as pd
import numpy as np
from pathlib import Path

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
# 1) "Dossiers du projet"
#    
# ----------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
FIG_DIR      = PROJECT_ROOT / "reports" / "figures" / "Modelisation_CNN"
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

    X_train = df_train[feature_cols].values.astype("float32")
    X_test  = df_test[feature_cols].values.astype("float32")

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

    return X_train, y_train, X_test, y_test


def compute_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("\nPoids de classes (balanced) :", cw_dict)
    return cw_dict


# ----------------------------
# 3) Construction du modèle CNN
# ----------------------------
def build_cnn_model(input_shape):
    model = models.Sequential(name="cnn_mit_mlii_binary")

    # Bloc 1
    model.add(layers.Conv1D(filters=32, kernel_size=5, activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Bloc 2
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    # Bloc 3
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling1D())

    # Dense
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))  # sortie binaire

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
    # Pour un peu de reproductibilité
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, y_train, X_test, y_test = load_data_binary()
    
    # ============================
    # Normalisation (StandardScaler)
    # ============================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Reshape pour CNN 1D : (n_samples, 187, 1)
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_test_cnn  = X_test_scaled[..., np.newaxis]

    # Split interne train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_cnn,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # Poids de classes pour gérer le déséquilibre
    class_weights = compute_class_weights(y_tr)

    # Modèle
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
    model = build_cnn_model(input_shape)

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True,
    )

    checkpoint_path = MODELS_DIR / "cnn_mit_mlii_binary_best.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # Entraînement
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint_cb],
        verbose=2,
    )

    print(f"\nMeilleur modèle sauvegardé dans : {checkpoint_path}")

    # Évaluation sur le test
    print("\n=== Évaluation CNN sur le test (patients jamais vus) ===")
    y_proba = model.predict(X_test_cnn).ravel()

    # Analyse de seuils
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

    chosen_thr = 0.5
    print(f"\n=== Détails pour le seuil choisi : {chosen_thr:.2f} ===")
    y_pred = (y_proba >= chosen_thr).astype(int)

    f1 = f1_score(y_test, y_pred)
    print("F1 (classe anormale, seuil choisi, CNN) :", f1)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion CNN :")
    print(cm)

    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    print("\nClassification report CNN :")
    print(classification_report(y_test, y_pred, digits=3))

    # ----------------------------
    # Figures : courbes + confusion
    # ----------------------------
    # 1) Courbes loss / metrics
    hist = history.history
    # Loss
    plt.figure(figsize=(5, 3))
    plt.plot(hist["loss"], label="loss train")
    plt.plot(hist["val_loss"], label="loss val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Évolution de la loss - CNN")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cnn_loss.png", dpi=150)
    plt.show()

    # Accuracy
    if "accuracy" in hist:
        plt.figure(figsize=(5, 3))
        plt.plot(hist["accuracy"], label="acc train")
        plt.plot(hist["val_accuracy"], label="acc val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Évolution de l'accuracy - CNN")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cnn_accuracy.png", dpi=150)
        plt.show()

    # 2) Matrice de confusion normalisée
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
    plt.title("Matrice de confusion normalisée (test) - CNN")
    plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cnn_cm_mit_test.png", dpi=150)
    plt.show()

    # 3) Tableau de scores
    report_df = pd.DataFrame(report_dict).T
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
    plt.title("Classification report CNN Binaire", pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cnn_report_mit_test.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
