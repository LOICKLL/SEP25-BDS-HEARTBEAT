import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

# ----- Entrée NN : battements + proba/pred du NN -----
BEATS_CSV = PROJECT_ROOT / "data" / "processed" / "ptb_beats_nn_minimal_labels_only.csv"

# ----- Sortie : 1 ligne par visite (patient + record) -----
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "ptb_patient_pred_malade_nn.csv"

# Règle de décision visite : au moins N battements NN==malade pour classer la visite malade
THRESHOLD_ABNORMAL_BEATS = 1

def _pick_col(df: pd.DataFrame, *candidates, required=False):
    """Renvoie la première colonne existante parmi candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Colonnes manquantes : {candidates}")
    return None

def main():
    # 1) Lecture
    print(f"Lecture des battements (NN) : {BEATS_CSV}")
    beats = pd.read_csv(BEATS_CSV)

    # Vérifications de base
    need = {"patient", "healthy_label"}
    miss = need - set(beats.columns)
    if miss:
        raise ValueError(f"Colonnes manquantes dans le CSV NN : {miss}")

    # Colonne 'record' (pour la notion de visite). Si absente, on bascule en agrégation par patient.
    has_record = "record" in beats.columns
    group_keys = ["patient", "record"] if has_record else ["patient"]

    # 2) Obtenir une prédiction binaire NN par battement

    pred_col = _pick_col(beats, "nn_pred_malade", "pred_malade", required=False)
    proba_col = _pick_col(beats, "nn_proba_malade", "proba_malade", required=False)

    if pred_col is None:
        if proba_col is None:
            raise ValueError(
                "Aucune des colonnes de prédiction/proba NN n'a été trouvée "
                "(essayé: nn_pred_malade / pred_malade / nn_proba_malade / proba_malade)."
            )
        print(f"Aucune colonne de prédiction binaire trouvée -> seuillage de '{proba_col}' à 0.5")
        beats["nn_pred_malade"] = (pd.to_numeric(beats[proba_col], errors="coerce").fillna(0) >= 0.5).astype(int)
        pred_col = "nn_pred_malade"
    else:
        # Nettoyage sécurité
        beats[pred_col] = pd.to_numeric(beats[pred_col], errors="coerce").fillna(0).astype(int)

    # 3) Agrégation par visite (patient+record) ou par patient (fallback)
    grp = (
        beats
        .groupby(group_keys, as_index=False)
        .agg(
            n_beats      = (pred_col, "size"),
            n_pred_malade= (pred_col, "sum"),
            healthy_label= ("healthy_label", "first")  # 1=sain, 0=malade
        )
    )

    # GT visite/patient : 1 si malade, 0 sinon
    grp["gt_malade"] = 1 - grp["healthy_label"].astype(int)

    # 4) Règle de décision au niveau visite/patient
    grp["nn_pred_malade"] = (grp["n_pred_malade"] >= THRESHOLD_ABNORMAL_BEATS).astype(int)

    # Pour info/commodité
    grp["gt_healthy"]      = 1 - grp["gt_malade"]
    grp["nn_pred_healthy"] = 1 - grp["nn_pred_malade"]

    # 5) Sauvegarde CSV
    grp.to_csv(OUT_CSV, index=False)
    print("\n=== CSV visite/patient sauvegardé ===")
    print("Nombre de lignes :", len(grp))
    print("Colonnes :", grp.columns.tolist())
    print("Fichier :", OUT_CSV)

    # 6) Matrice de confusion (gt_malade vs nn_pred_malade)
    y_true = grp["gt_malade"].values
    y_pred = grp["nn_pred_malade"].values

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    img_dir = PROJECT_ROOT / "reports" / "figures" / "ptb_eval_nn_pat" 
    img_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
        xticklabels=["Prédit sain", "Prédit malade"],
        yticklabels=["Vrai sain", "Vrai malade"],
    )
    title = "Matrice de confusion patients (sain/malade)" if has_record else "Matrice de confusion patients (sain/malade)"
    plt.title(title)
    plt.xlabel("Prédiction NN")
    plt.ylabel("Vérité terrain (gt_malade)")

    img_path = img_dir / ("ptb_patients_confusion_malade_vs_sain.png" if has_record else "ptb_patients_confusion_malade_vs_sain.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    print(f"\nMatrice de confusion sauvegardée dans : {img_path}")


if __name__ == "__main__":
    main()
