import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# 1) Dossiers du projet

# Dossier où se trouve ce script
THIS_DIR = Path(__file__).resolve().parent
# Racine du projet
PROJECT_ROOT = THIS_DIR.parent.parent
# Dossier qui contient le csv mitbih_187pts_MLII.csv
DATA_DIR = PROJECT_ROOT / "data" / "processed"
# Dossier de sortie pour les CSV
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Chemins 
    
    csv_in = DATA_DIR / "mitbih_187pts_MLII.csv"
    csv_train = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
    csv_test = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"

    print("Lecture du fichier :", csv_in)
    df = pd.read_csv(csv_in, low_memory=False)
    print("Shape totale (battements) :", df.shape)

    # 2) Liste des patients 
    patients = df["record_x"].unique()
    n_patients = len(patients)
    print(f"Nombre total de patients (records) : {n_patients}")

 
    # 3) Classification des patients par "bin" selon abnormal_ratio
    print("\n=== Construction des bins de patients selon abnormal_ratio ===")

    # Table patient x label : nombre de battements par label pour chaque patient
    tab = df.groupby(["record_x", "label"]).size().unstack(fill_value=0)

    # On calcule le total de battements par patient
    tab["total"] = tab.sum(axis=1)

    # Proportion de battements normaux (label 0), puis anormaux
    tab["N_ratio"] = tab[0] / tab["total"].replace(0, np.nan)
    tab["abnormal_ratio"] = 1 - tab["N_ratio"]

    # Bins 
    bins = [0, 0.01, 0.05, 0.2, 1.0]  # quasi normal / un peu / mixte / très anormal
    labels_bins = ["quasi_normaux", "un_peu_anormaux", "mixtes", "tres_anormaux"]

    tab["type"] = pd.cut(
        tab["abnormal_ratio"],
        bins=bins,
        labels=labels_bins,
        include_lowest=True
    )

    print("\nRépartition des patients par type (bin) :")
    print(tab["type"].value_counts(dropna=False))

    
    # 4) Split PATIENTS PAR BIN : 40 % train / 60 % test
  
    patients_train = []
    patients_test = []

    for name, grp in tab.groupby("type"):
        if pd.isna(name):
            # au cas où certains patients n'ont pas pu être binés
            print("\nType NaN (patients sans bin) : ignoré pour le split par bin.")
            continue

        record_per_bin = grp.index.to_numpy()
        if len(record_per_bin) == 0:
            continue

        print(f"\nBin '{name}': {len(record_per_bin)} patients")

        # Gestion du cas où il n'y a qu'UN seul patient dans le bin
        if len(record_per_bin) == 1:
            train_p = record_per_bin
            test_p = np.array([], dtype=record_per_bin.dtype)
            print(f"  -> 1 seul patient, mis dans le TRAIN.")
        else:
            train_p, test_p = train_test_split(
                record_per_bin,
                test_size=0.6,        # 60 % des patients de ce bin en test
                random_state=42,
                shuffle=True,
            )

        patients_train.extend(train_p.tolist())
        patients_test.extend(test_p.tolist())

        print(f"  -> train : {len(train_p)}, test : {len(test_p)}")

    patients_train = list(set(patients_train))
    patients_test = list(set(patients_test))

    print("\nAprès split par bins :")
    print(f"Patients train : {len(patients_train)}")
    print(f"Patients test  : {len(patients_test)}")

    
    # 5) On s'assure que chaque label 0..4 est présent dans le train

    labels = sorted(df["label"].unique())
    print("\nLabels présents dans le dataset :", labels)

    def labels_in_patients(patient_list):
        sub = df[df["record_x"].isin(patient_list)]
        return set(sub["label"].unique())

    train_labels = labels_in_patients(patients_train)
    print("Labels présents dans le train (au début) :", train_labels)

    for lbl in labels:
        if lbl in train_labels:
            continue

        print(f"\nLabel {lbl} absent du train -> on déplace un patient depuis le test")

        patients_test_with_lbl = df[
            (df["record_x"].isin(patients_test)) & (df["label"] == lbl)
        ]["record_x"].unique()

        if len(patients_test_with_lbl) == 0:
            print(f"  Aucun patient test ne contient le label {lbl} -> on ne peut rien faire.")
            continue

        chosen_patient = patients_test_with_lbl[0]
        print(f"  Patient choisi pour ce label : {chosen_patient}")

        patients_test.remove(chosen_patient)
        patients_train.append(chosen_patient)

        train_labels = labels_in_patients(patients_train)
        print("  Labels dans le train maintenant :", train_labels)

    # Vérif finale : intersection des patients train/test doit être vide
    inter = set(patients_train).intersection(set(patients_test))
    print("\nIntersection patients train/test (doit être vide) :", inter)

  
    # 6) Construction des DataFrames train/test (avant undersampling)
    
    df_train = df[df["record_x"].isin(patients_train)].copy()
    df_test = df[df["record_x"].isin(patients_test)].copy()

    print("\nShape train AVANT undersampling (battements) :", df_train.shape)
    print("Shape test  (battements) :", df_test.shape)

    print("\nRépartition des labels (train AVANT undersampling) :")
    print(df_train["label"].value_counts(normalize=True).sort_index())

    print("\nRépartition des labels (test) :")
    print(df_test["label"].value_counts(normalize=True).sort_index())

 
    # 7) UNDERSAMPLING sur le TRAIN
   
    MAX_PER_PATIENT_LABEL = 300     # max battements par (patient, label)
    MAX_PER_LABEL_GLOBAL = 10_000   # max battements pour un label (surtout 0)

    # Limitation par (record_x, label)
    def sample_per_patient_label(group):
        n = min(len(group), MAX_PER_PATIENT_LABEL)
        return group.sample(n=n, random_state=42)

    df_train_limited = (
        df_train.groupby(["record_x", "label"], group_keys=False)
                .apply(sample_per_patient_label)
                .reset_index(drop=True)
    )

    # Limitation globale par label (surtout pour 0)
    def sample_per_label(group):
        label = group.name
        if label == 0:
            n = min(len(group), MAX_PER_LABEL_GLOBAL)
            return group.sample(n=n, random_state=42)
        else:
            return group

    df_train_balanced = (
        df_train_limited.groupby("label", group_keys=False)
                        .apply(sample_per_label)
                        .reset_index(drop=True)
    )

    print("\nShape train APRÈS undersampling (battements) :", df_train_balanced.shape)
    print("\nRépartition des labels (train APRÈS undersampling) :")
    print(df_train_balanced["label"].value_counts().sort_index())
    print("\nProportions (train APRÈS undersampling) :")
    print(df_train_balanced["label"].value_counts(normalize=True).sort_index())

    
    # 8) Sauvegarde
    
    df_train_balanced.to_csv(csv_train, index=False)
    df_test.to_csv(csv_test, index=False)

    print("\nFichiers créés :")
    print("  Train (≈40% des patients, undersamplé) :", csv_train)
    print("  Test  (≈60% des patients)             :", csv_test)


if __name__ == "__main__":
    main()
