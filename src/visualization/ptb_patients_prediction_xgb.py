import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

THIS_DIR = Path(__file__).resolve().parent
# Racine du projet
PROJECT_ROOT = THIS_DIR.parent.parent

# Fichier en entrée : battements avec prédictions XGBoost
BEATS_CSV = PROJECT_ROOT/"data"/"processed" / "ptb_beats_xgb_minimal_labels_only.csv"

# Fichier en sortie : 1 ligne par visite (patient + record)
OUT_CSV = PROJECT_ROOT/"data"/"processed" / "ptb_patient_pred_malade_xgb.csv"

def main():
    
    # 1) Lecture du CSV de battements
    
    print(f"Lecture des battements : {BEATS_CSV}")
    beats = pd.read_csv(BEATS_CSV)

    # On vérifie  les colonnes 
    expected_cols = {"patient", "healthy_label", "xgb_pred_malade"}
    missing = expected_cols - set(beats.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV de battements : {missing}")

    
    # 2) Agrégation par patient
    
    # On compte le nombre total de battements et de battements anormaux prédits
    grp = (
        beats
        .groupby("patient")
        .agg(
            n_beats=("xgb_pred_malade", "size"),
            n_pred_malade=("xgb_pred_malade", "sum"),
            healthy_label=("healthy_label", "first")  
        )
        .reset_index()
    )

    # Label ground truth patient : gt_malade = 1 - healthy_label
    grp["gt_malade"] = 1 - grp["healthy_label"]

    
    # 3) Règle de décision patient
    
    # Ici seuil = 1 battement anormal → patient prédit malade
    threshold_abnormal_beats = 1
    grp["xgb_pred_malade"] = (grp["n_pred_malade"] >= threshold_abnormal_beats).astype(int)


    # Pour info, on peut aussi définir les versions "healthy"
    grp["gt_healthy"] = 1 - grp["gt_malade"]
    grp["xgb_pred_healthy"] = 1 - grp["xgb_pred_malade"]

    
    # 4) Sauvegarde du CSV patient
    
    grp.to_csv(OUT_CSV, index=False)

    print("\n=== CSV patient sauvegardé ===")
    print("Nombre de patients :", len(grp))
    print("Colonnes :", grp.columns.tolist())
    print("Fichier :", OUT_CSV)

   
    # 5) MATRICE DE CONFUSION PATIENTS (gt_malade vs xgb_pred_malade)
    
    y_true = grp["gt_malade"].values          # 0 = sain, 1 = malade (réalité)
    y_pred = grp["xgb_pred_malade"].values    # 0 = sain, 1 = malade (XGBoost patient)

    # Matrice de confusion normalisée par ligne
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    # Création du dossier images/
    img_dir = PROJECT_ROOT/"reports"/"figures" /"ptb_eval_xgb_pat"
    img_dir.mkdir(exist_ok=True)

    # Tracé
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=["Prédit sain", "Prédit malade"],
        yticklabels=["Vrai sain", "Vrai malade"],
    )
    plt.title("Matrice de confusion patients (sain/malade)")
    plt.xlabel("Prédiction XGBoost")
    plt.ylabel("Vérité terrain (gt_malade)")

    img_path = img_dir / "ptb_patients_confusion_malade_vs_sain.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    print(f"\nMatrice de confusion sauvegardée dans : {img_path}")


if __name__ == "__main__":
    main()
