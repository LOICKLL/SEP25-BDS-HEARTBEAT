# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import tensorflow as tf

# ================== PATHS ==================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_IN = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
TARGET_COL = "binary"

SCALER_PATH = MODELS_DIR / "scaler_mit_mlii_binary.pkl"  # scaler entraîné sur les features
FEATURES_PATH = MODELS_DIR / "nn_mit_mlii_binary_features.json"  # liste des colonnes de features

# ================== CONFIG MODELES ==================
MODEL_CONFIG = {
    "logreg_mit_binary": {
        "filename": "LogReg_mit_binary.pkl",
        "framework": "sklearn",
        "type": "classique",
    },
    "svm_mit_binary": {
        "filename": "svm_mit_mlii_binary.pkl",
        "framework": "sklearn",
        "type": "classique",
    },
    "rf_mit_binary": {
        "filename": "rf_mit_mlii_binary.pkl",
        "framework": "sklearn",
        "type": "classique",
    },
    "xgb_mit_binary": {
        "filename": "XGB_mit_binary.pkl",
        "framework": "sklearn",
        "type": "classique",
    },
    "nn_mit_binary": {
        "filename": "nn_mit_mlii_binary.keras",
        "framework": "keras_dense",
        "type": "deep",
    },
    "cnn_mit_binary": {
        "filename": "cnn_mit_mlii_binary_best.keras",
        "framework": "keras_cnn",
        "type": "deep",
    },
}


def main():
    # ================== 1) Lecture des données ==================
    print(f"Lecture des données : {CSV_IN}")
    df = pd.read_csv(CSV_IN)

    # Vérifier que la colonne cible existe
    assert TARGET_COL in df.columns, f"Colonne cible {TARGET_COL} introuvable."

    # 1.a Charger la liste des features utilisées à l'entraînement
    print(f"Lecture des features depuis : {FEATURES_PATH}")
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    print("Nombre de features :", len(feature_cols))
    print("Exemples de colonnes :", feature_cols[:5])

    # Vérifier que toutes les features sont bien dans le CSV
    missing = [c for c in feature_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Colonnes de features manquantes dans le CSV : {missing}")

    # X = uniquement les colonnes de features, y = cible binaire
    X_test = df[feature_cols].astype("float32").values
    y_test = df[TARGET_COL].values

    print("Shape X_test :", X_test.shape)
    print("Shape y_test :", y_test.shape)

    # ================== 2) Scaling ==================
    if SCALER_PATH.exists():
        print(f"Chargement du scaler : {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("⚠️ Pas de scaler trouvé, on utilise X_test brut.")
        X_test_scaled = X_test

    # Préparation des entrées pour les modèles Keras
    X_test_dense = X_test_scaled                              # (n_samples, n_features)
    X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)  # (n_samples, n_features, 1)

    # ================== 3) Boucle sur les modèles ==================
    results = []

    for model_id, cfg in MODEL_CONFIG.items():
        model_path = MODELS_DIR / cfg["filename"]
        print(f"\n=== {model_id} ===")
        print(f"Chemin modèle : {model_path}")

        if not model_path.exists():
            print("  -> Fichier introuvable, on saute.")
            continue

        # ---------- Chargement ----------
        try:
            if cfg["framework"] == "sklearn":
                model = joblib.load(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"  -> Erreur de chargement : {e}")
            continue

        # ---------- Prédictions ----------
        try:
            if cfg["framework"] == "sklearn":
                X_input = X_test_scaled
                y_pred = model.predict(X_input)
                # Si le modèle renvoie des proba ou un 2D, on gère
                if y_pred.ndim > 1:
                    if y_pred.shape[1] == 2:
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        y_pred = (y_pred.ravel() >= 0.5).astype(int)
            elif cfg["framework"] == "keras_dense":
                X_input = X_test_dense
                y_proba = model.predict(X_input, verbose=0).ravel()
                y_pred = (y_proba >= 0.5).astype(int)
            elif cfg["framework"] == "keras_cnn":
                X_input = X_test_cnn
                y_proba = model.predict(X_input, verbose=0).ravel()
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                print("  -> Framework non géré.")
                continue
        except Exception as e:
            print(f"  -> Erreur de prédiction : {e}")
            continue

        # ---------- Métriques ----------
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  Précision : {prec:.4f}")
        print(f"  Rappel : {rec:.4f}")

        results.append(
            {
                "model_id": model_id,
                "model_file": cfg["filename"],
                "framework": cfg["framework"],
                "type": cfg["type"],
                "n_test": len(y_test),
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
            }
        )

    # ================== 4) Sauvegarde CSV ==================
    if results:
        scores_df = pd.DataFrame(results)
        out_csv = REPORTS_DIR / "mit_models_scores.csv"
        scores_df.to_csv(out_csv, index=False)
        print("\nFichier de scores écrit dans :", out_csv)
    else:
        print("\nAucun résultat, rien à écrire.")


if __name__ == "__main__":
    main()
