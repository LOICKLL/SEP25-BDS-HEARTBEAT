# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================== PATHS ==============================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CSV_IN  = DATA_DIR / "mitbih_187pts_MLII.csv"   
CSV_TRN = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
CSV_TST = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"

print("Lecture :", CSV_IN)
df = pd.read_csv(CSV_IN, low_memory=False)

#############################################################################################################
## NETTOYAGE MINIMAL (on NE filtre PAS par lead, et on NE retire PAS label == -1)
#############################################################################################################

print("... nettoyage de la base (minimal) ... ")

# Tri (optionnel)
df = df.sort_values(["record_x"]).reset_index(drop=True)

# Colonnes signal robustes (présentes dans le CSV)
signal_cols = [str(i) for i in range(187) if str(i) in df.columns]
# Convertit en float si besoin
df[signal_cols] = df[signal_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)

print("Records restants :", df["record_x"].nunique())
print("Labels uniques   :", sorted(df["label"].dropna().unique().tolist()))

##############################################################################################################
## CLASSIFICATION BINAIRE MALADE/SAIN (les -1 comptent comme anormal)
##############################################################################################################

print(
    """
##############################################################################################################
## CLASSIFICATION BINAIRE MALADE/SAIN ##
##############################################################################################################
"""
)

df_binary = df.copy()
df_binary["binary"] = (df_binary["label"] != 0).astype(int)  # -1 => 1 (anormal)

# Split par patients (évite le data leakage)
patients = df_binary["record_x"].unique()
train_pat, test_pat = train_test_split(
    patients, test_size=0.2, random_state=42
)

df_train_binary = df_binary[df_binary["record_x"].isin(train_pat)].copy()
df_test_binary  = df_binary[df_binary["record_x"].isin(test_pat)].copy()

# Undersampling dans le TRAIN : ratio 3:1 pour les normaux
normal  = df_train_binary[df_train_binary["binary"] == 0]
anormal = df_train_binary[df_train_binary["binary"] == 1]

target_normals = min(len(normal), 3 * len(anormal))
if len(normal) > 0 and len(anormal) > 0:
    normal_sampled = normal.sample(n=target_normals, random_state=42)
    df_train_binary = pd.concat([normal_sampled, anormal], ignore_index=True)
else:
    # Cas bord : si une classe vide dans le train, on garde tel quel
    df_train_binary = pd.concat([normal, anormal], ignore_index=True)

# Shuffle final
df_train_binary = df_train_binary.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nTaille train équilibré :", len(df_train_binary))
print("Répartition binaire (train) :\n", df_train_binary["binary"].value_counts())

# --- Colonnes à conserver et ordre exact ---
meta_cols    = ["record_x", "sample", "t_sec", "lead", "symbol", "label"]
signal_cols  = [str(i) for i in range(187) if str(i) in df.columns]
order_cols   = meta_cols +["binary"]  + signal_cols   # on garde aussi la cible binair

# Constructions des sorties
df_train_out = df_train_binary[order_cols].copy()
df_test_out  = df_test_binary[order_cols].copy()

# Sauvegarde
df_train_out.to_csv(CSV_TRN, index=False)
df_test_out.to_csv(CSV_TST, index=False)


print("\nFichiers créés :")
print(" - Train :", CSV_TRN)
print(" - Test  :", CSV_TST)
print("\nShapes :")
print(" - Train :", df_train_out.shape)
print(" - Test  :", df_test_out.shape)
