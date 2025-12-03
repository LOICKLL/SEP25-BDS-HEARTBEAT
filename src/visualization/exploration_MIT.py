# -*- coding: utf-8 -*-
"""
Mini EDA MIT-BIH + graphiques, avec sauvegarde des figures
dans reports/figures/exploration_MIT.

Entrée attendue : un CSV "full" contenant :
- colonnes de signal "0".."186"
- meta: record_num, age, sexe ('M'/'F' ou 'male'/'female'), lead, label, bpm_moyen
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ----------------------------- PATHS -----------------------------
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "mitbih_187pts_fullmeta.csv"

OUT_DIR = PROJECT_ROOT / "reports" / "figures" / "exploration_MIT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str, dpi: int = 150):
    path = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Sauvé : {path}")

# ----------------------------- LOAD -----------------------------
print("... chargement du dataframe ...")
mitbih_data = pd.read_csv(DATA_PATH, low_memory=False)
mitbih_data = mitbih_data.loc[:, ~mitbih_data.columns.duplicated()].copy()

print(mitbih_data.head())

# Normalisation douce du sexe
if "sexe" in mitbih_data.columns:
    mitbih_data["sexe"] = (
        mitbih_data["sexe"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"m": "M", "male": "M", "homme": "M",
              "f": "F", "female": "F", "femme": "F"})
        .fillna(mitbih_data["sexe"])
    )

# ---------------------- RÉPARTITION PAR TYPE --------------------
# Mapping des labels
mapping = {
    -1: "Autres",
     0: "N (Normal)",
     1: "S (Supraventriculaire)",
     2: "E (Ectopique)",
     3: "V (Ventriculaire)",
     4: "F (Fusion)"
}
if "label" in mitbih_data.columns:
    mitbih_data["label_name"] = mitbih_data["label"].map(mapping).fillna("Autres")
else:
    raise KeyError("Colonne 'label' absente du CSV.")

label_counts = mitbih_data["label_name"].value_counts(dropna=False)

plt.figure(figsize=(7,7))
plt.pie(
    label_counts.values,
    labels=label_counts.index,
    autopct="%1.1f%%",
    shadow=True
)
plt.title("Répartition des battements par type", fontsize=14)
savefig("repartition_labels_global")

# ------------- RÉPARTITION PAR SEXE × TYPE DE BATTEMENT ---------
if {"label", "sexe"}.issubset(mitbih_data.columns):
    cross = pd.crosstab(mitbih_data["label"], mitbih_data["sexe"])
    ax = cross.plot(kind="bar", figsize=(8,5))
    ax.set_title("Répartition des types de battements par sexe", fontsize=14)
    ax.set_xlabel("Type de battement (label)")
    ax.set_ylabel("Nombre de battements")
    ax.legend(title="Sexe")
    ax.grid(axis="y", alpha=0.3)
    savefig("repartition_labels_par_sexe")

# ------------- RÉPARTITION PAR TRANCHE D'ÂGE × TYPE -------------
if "age" in mitbih_data.columns:
    bins = [0, 25, 35, 50, 120]
    age_labels = ["18-25", "26-35", "36-50", "50+"]

    mitbih_data["age"] = pd.to_numeric(mitbih_data["age"], errors="coerce")
    mitbih_data["Tranche âge"] = pd.cut(
        mitbih_data["age"], bins=bins, labels=age_labels, right=False
    )

    prop_age = pd.crosstab(
        mitbih_data["Tranche âge"],
        mitbih_data["label_name"],
        normalize="index"
    )

    ax = prop_age.plot(kind="bar", stacked=True, figsize=(9,5))
    ax.set_title("Répartition des types de battements par tranche d'âge", fontsize=14)
    ax.set_xlabel("Tranche d'âge")
    ax.set_ylabel("Proportion")
    ax.legend(title="Type de battement", bbox_to_anchor=(1.05, 1))
    ax.grid(axis="y", alpha=0.3)
    savefig("repartition_labels_par_age")

# ---------------- HEATMAP ÂGE / SEXE / LABEL (corrélations) ----
df_corr = mitbih_data.copy()
if "sexe" in df_corr.columns:
    df_corr["sexe_num"] = df_corr["sexe"].map({"F":0, "M":1})

if {"age","label","sexe_num"}.issubset(df_corr.columns):
    corr = df_corr[["age","label","sexe_num"]].astype(float).corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5
    )
    plt.title("Corrélation entre âge, sexe et label", fontsize=13)
    savefig("heatmap_corr_age_sexe_label")

# ----------------- COURBES DE BATTEMENTS TYPIQUES ---------------
# Détection des colonnes de signal "0".."186"
signal_cols = [c for c in mitbih_data.columns if c.isdigit()]
signal_cols = sorted(signal_cols, key=lambda x: int(x))
signal_cols = [c for c in signal_cols if 0 <= int(c) <= 186]

if signal_cols and "label_name" in mitbih_data.columns:
    # Par classe : 20 courbes (gris) + moyenne (bleu)
    for name, grp in mitbih_data.groupby("label_name"):
        plt.figure(figsize=(9,4))
        n = min(20, len(grp))
        for i in range(n):
            plt.plot(grp.iloc[i][signal_cols].to_numpy(), color="gray", alpha=0.3)
        plt.plot(grp[signal_cols].mean().to_numpy(), color="blue", linewidth=2)
        plt.axhline(0, ls="--", color="gray")
        plt.title(f"Signal moyen – {name}")
        savefig(f"signal_moyen_{name.replace(' ','_').replace('/','-')}")

    # Toutes les moyennes superposées
    plt.figure(figsize=(9,4))
    for name, grp in mitbih_data.groupby("label_name"):
        plt.plot(grp[signal_cols].mean().to_numpy(), label=name)
    plt.axhline(0, ls="--", color="gray")
    plt.title("Signal moyen par type de battement")
    plt.xlabel("Échantillons"); plt.ylabel("Amplitude")
    plt.legend()
    savefig("signal_moyen_toutes_classes")

# --------- RÉPARTITION PAR TYPE ET PAR ENREGISTREMENT ----------
if {"record_num","label"}.issubset(mitbih_data.columns):
    counts = mitbih_data.groupby(["record_num", "label"]).size()
    df_plot = counts.unstack(fill_value=0)
    df_plot = df_plot.loc[df_plot.sum(axis=1).sort_values(ascending=False).index]

    ax = df_plot.plot(kind="bar", stacked=True, figsize=(15, 9))
    ax.set_title("Nombre de battements par type et par patient")
    ax.set_xlabel("Patient (record)")
    ax.set_ylabel("Nombre de battements")
    plt.xticks(rotation=90)
    ax.legend(title="Label")
    plt.grid(axis="y", alpha=0.3)
    savefig("nb_battements_par_type_et_patient")

# ---- Khi² d'indépendance : LABEL ~ SEXE, LABEL ~ CLASSES D'ÂGE --
ALPHA = 0.05
if {"label","sexe","age"}.issubset(mitbih_data.columns):
    data = mitbih_data[["label","sexe","age"]].dropna()
    AGE_BINS   = [0, 30, 45, 60, 75, 120]
    AGE_LABELS = ["<30", "30-45", "45-60", "60-75", "75+"]
    data["age_class"] = pd.cut(data["age"].astype(float), bins=AGE_BINS, labels=AGE_LABELS,
                               right=False, include_lowest=True)

    ct_sexe = pd.crosstab(data["sexe"], data["label"])
    ct_age  = pd.crosstab(data["age_class"], data["label"])

    if ct_sexe.size > 0:
        chi2_s, p_s, ddl_s, _ = chi2_contingency(ct_sexe)
        print("=== χ² : LABEL ~ SEXE ===")
        print(ct_sexe, "\n")
        print(f"χ² = {chi2_s:.3f} | ddl = {ddl_s} | p-value = {p_s:.6f}")
        print("⇒", "Dépendance significative (rejette H0)" if p_s < ALPHA else
                    "Pas de dépendance significative (ne rejette pas H0)")

    if ct_age.size > 0:
        chi2_a, p_a, ddl_a, _ = chi2_contingency(ct_age)
        print("\n=== χ² : LABEL ~ CLASSES D'ÂGE ===")
        print(ct_age, "\n")
        print(f"χ² = {chi2_a:.3f} | ddl = {ddl_a} | p-value = {p_a:.6f}")
        print("⇒", "Dépendance significative (rejette H0)" if p_a < ALPHA else
                    "Pas de dépendance significative (ne rejette pas H0)")

# --------- Corrélation ÂGE / BPM & dispersion colorée par SEXE ---
if {"age","bpm_moyen"}.issubset(mitbih_data.columns):
    plt.figure(figsize=(6.5,4.5))
    sns.scatterplot(data=mitbih_data, x="age", y="bpm_moyen", hue="sexe")
    plt.title("Relation entre âge et fréquence cardiaque moyenne")
    savefig("age_vs_bpm_scatter")

# --------------------- DISTRIBUTION DE L’ÂGE ---------------------
if "age" in mitbih_data.columns:
    plt.figure(figsize=(5, 6))
    plt.boxplot(mitbih_data["age"].dropna())
    plt.title("Distribution de l'âge")
    plt.ylabel("Âge")
    plt.grid(True, axis='y', alpha=0.4)
    savefig("age_box")

    plt.figure(figsize=(8,5))
    pd.to_numeric(mitbih_data["age"], errors="coerce").dropna().hist(bins=20)
    plt.title("Distribution des âges (MIT-BIH)")
    plt.xlabel("Âge"); plt.ylabel("Fréquence")
    savefig("age_hist")

    print("Statistiques des âges :")
    print(mitbih_data["age"].describe())

# ------------------- DISTRIBUTION DES GENRES --------------------
if "sexe" in mitbih_data.columns:
    plt.figure(figsize=(5,4))
    mitbih_data["sexe"].value_counts(dropna=False).plot(kind="bar")
    plt.title("Distribution des genres")
    plt.xlabel("Genre"); plt.ylabel("Nombre de patients")
    plt.xticks(rotation=0)
    savefig("sexe_bar")

    print("Comptage des genres :")
    print(mitbih_data["sexe"].value_counts(dropna=False))

# ------------------- DISTRIBUTION DES LEADS ---------------------
lead_col = "lead" if "lead" in mitbih_data.columns else ("leads" if "leads" in mitbih_data.columns else None)
if lead_col:
    plt.figure(figsize=(5,4))
    mitbih_data[lead_col].value_counts(dropna=False).plot(kind="bar")
    plt.title("Distribution des leads")
    plt.xlabel("Lead(s)"); plt.ylabel("Nombre d'échantillons")
    plt.xticks(rotation=0)
    savefig("leads_bar")

    print("Comptage des leads :")
    print(mitbih_data[lead_col].value_counts(dropna=False))

print("\n[OK] Mini EDA terminé. Figures sauvegardées dans :", OUT_DIR)
