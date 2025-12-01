# apply_xgb_on_ptb_beats_with_label_minimal.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ======================================================================
# Chemins
# ======================================================================

THIS_DIR = Path(__file__).resolve().parent

# Racine du projet
PROJECT_ROOT = THIS_DIR.parent.parent

# Modèle XGBoost entraîné sur MIT (MLII b0..b186)
MODEL_PATH = PROJECT_ROOT/"models"/ "xgb_mit_mlii_binary.pkl"

# Fichier de battements PTB + healthy_label (produit par build_ptb_beats_with_labels.py)
PTB_BEATS_PATH = PROJECT_ROOT/"data"/"processed" / "ptb_beats_all_with_healthy_label.csv"

# Fichier de sortie minimal (sans lead/fs/n_samples/r_index_360/b0..b186)
OUT_PATH = PROJECT_ROOT/"data"/"processed" / "ptb_beats_xgb_minimal_labels_only.csv"


def main():
    # ------------------------------------------------------------------
    # 1) Chargement du modèle
    # ------------------------------------------------------------------
    print(f"Chargement du modèle XGBoost : {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print("!! Le fichier de modèle n'existe pas, vérifie le chemin.")
        return

    xgb_clf = joblib.load(MODEL_PATH)

    # ------------------------------------------------------------------
    # 2) Lecture du CSV de battements PTB
    # ------------------------------------------------------------------
    print(f"\nLecture des battements PTB : {PTB_BEATS_PATH}")
    if not PTB_BEATS_PATH.exists():
        print("!! Le fichier de battements n'existe pas, vérifie le chemin.")
        return

    beats = pd.read_csv(PTB_BEATS_PATH)

    print("\nColonnes disponibles au départ :")
    print(list(beats.columns[:20]))
    print("...")
    print("Nombre total de lignes :", len(beats))

    # ------------------------------------------------------------------
    # 3) Sélection des features b0..b186 pour la prédiction
    # ------------------------------------------------------------------
    feat_cols = [c for c in beats.columns if c.startswith("b")]
    feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))

    print("\nNombre de features par battement :", len(feat_cols))

    X = beats[feat_cols].values

        # ------------------------------------------------------------------
    # 4) Application du modèle XGBoost
    # ------------------------------------------------------------------
    print("\nPrédiction des probabilités avec XGBoost...")
    proba = xgb_clf.predict_proba(X)
    proba_malade = proba[:, 1]  # proba classe "malade/anormale"

    # ------------------------------------------------------------------
    # 5) Construction des labels GT et des prédictions
    #    healthy_label : 1 = sain, 0 = malade
    #    gt_malade     : 1 = malade, 0 = sain  (cible pour le modèle)
    # ------------------------------------------------------------------
    if "healthy_label" not in beats.columns:
        print("\nATTENTION : la colonne 'healthy_label' n'est pas présente dans le CSV !")
        return

    # GT en mode malade/anormal
    beats["gt_malade"] = 1 - beats["healthy_label"]

    # Probabilité et prédiction binaire "malade"
    beats["xgb_proba_malade"] = proba_malade
    threshold = 0.5
    beats["xgb_pred_malade"] = (beats["xgb_proba_malade"] >= threshold).astype(int)

    # Colonne pratique : 1 si prédiction correcte, 0 sinon
    beats["correct"] = (beats["xgb_pred_malade"] == beats["gt_malade"]).astype(int)

    # ------------------------------------------------------------------
    # 6) Ne garder QUE les colonnes demandées
    # ------------------------------------------------------------------
    cols_final = [
        "patient",
        "record",
        "idx_local",
        "healthy_label",      # 1 = sain, 0 = malade
        "gt_malade",          # 1 = malade, 0 = sain
        "xgb_proba_malade",
        "xgb_pred_malade",
        "correct",
    ]

    cols_final = [c for c in cols_final if c in beats.columns]
    beats_min = beats[cols_final].copy()


    # ------------------------------------------------------------------
    # 7) Sauvegarde du CSV minimal
    # ------------------------------------------------------------------
    beats_min.to_csv(OUT_PATH, index=False)

    print("\n=== Résumé ===")
    print("Nombre de lignes (battements) :", len(beats_min))
    print("Colonnes de sortie :")
    print(beats_min.columns.tolist())
    print("Fichier sauvegardé dans :", OUT_PATH)


if __name__ == "__main__":
    main()
