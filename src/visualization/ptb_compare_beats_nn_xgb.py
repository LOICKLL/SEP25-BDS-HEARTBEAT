# -*- coding: utf-8 -*-
"""
Comparaison des prédictions de battements PTB entre :
- Réseau de neurones (NN) -> CSV1
- XGBoost (XGB)          -> CSV2

+ Figures :
    - Barres globales % battements malades NN / XGB
    - Nuage % malades par patient (NN vs XGB)
    - Heatmap d'accord NN / XGB (0/1)
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1) PATHS
# =========================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "PTB_NN_XGB"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_NN  = DATA_DIR / "ptb_beats_nn_minimal_labels_only.csv"
CSV_XGB = DATA_DIR / "ptb_beats_xgb_minimal_labels_only.csv"

OUT_PATIENT_SUMMARY = DATA_DIR / "ptb_beats_nn_xgb_comparison_by_patient.csv"


# =========================
# 2) Lecture + fusion
# =========================
def load_and_merge():
    print("Lecture NN  :", CSV_NN)
    df_nn = pd.read_csv(CSV_NN)

    print("Lecture XGB :", CSV_XGB)
    df_xgb = pd.read_csv(CSV_XGB)

    print("\nColonnes NN :", df_nn.columns.tolist())
    print("Colonnes XGB:", df_xgb.columns.tolist())

    # Clés de jointure
    key_cols = ["patient", "record", "idx_local", "healthy_label", "gt_malade"]

    # On supprime les colonnes XGB FAUSSES du CSV NN
    cols_xgb_to_drop = ["xgb_proba_malade", "xgb_pred_malade", "correct", "threshold_used"]
    for col in cols_xgb_to_drop:
        if col in df_nn.columns:
            print(f"Suppression de la colonne XGB obsolète dans df_nn : {col}")
            df_nn = df_nn.drop(columns=[col])

    # Fusion : nn_* depuis df_nn, xgb_* depuis df_xgb
    df = pd.merge(df_nn, df_xgb, on=key_cols, how="inner", suffixes=("_nn", "_xgb"))

    print("\nTaille après fusion :", df.shape)
    print("Colonnes fusionnées :", df.columns.tolist())
    print("\nAperçu :")
    print(df.head())

    # Vérification des colonnes nécessaires
    needed = [
        "nn_pred_malade",        # vient du CSV1
        "xgb_proba_malade",      # vient du CSV2
        "xgb_pred_malade",       # vient du CSV2
    ]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le DataFrame fusionné : {col}")

    return df


# =========================
# 3) Statistiques globales
# =========================
def global_stats(df):
    print("\n=======================")
    print("Statistiques globales :")
    print("=======================")

    stats = {}
    for model, col in [("NN", "nn_pred_malade"), ("XGB", "xgb_pred_malade")]:
        print(f"\n--- {model} : répartition des prédictions (0 = sain, 1 = malade) ---")
        vc_abs = df[col].value_counts().sort_index()
        vc = df[col].value_counts(normalize=True).sort_index()
        res = pd.concat(
            [vc_abs.rename("count"), (100 * vc).rename("percent")],
            axis=1,
        )
        print(res)
        # on garde le % de malades pour le graphique global
        stats[model] = res.loc[1, "percent"] if 1 in res.index else 0.0

    return stats


# =========================
# 4) Statistiques par patient
# =========================
def per_patient_stats(df):
    print("\n=======================")
    print("Statistiques par patient :")
    print("=======================")

    grp = df.groupby("patient")

    patient_summary = grp.agg(
        nb_beats=("idx_local", "count"),
        nn_pct_malade=("nn_pred_malade", "mean"),
        xgb_pct_malade=("xgb_pred_malade", "mean"),
    ).reset_index()

    patient_summary["nn_pct_malade"] *= 100.0
    patient_summary["xgb_pct_malade"] *= 100.0

    print("\nAperçu du résumé par patient :")
    print(patient_summary.head())

    print("\nTop 5 patients avec le plus fort % de battements malades (NN) :")
    print(patient_summary.sort_values("nn_pct_malade", ascending=False).head())

    print("\nTop 5 patients avec le plus fort % de battements malades (XGB) :")
    print(patient_summary.sort_values("xgb_pct_malade", ascending=False).head())

    return patient_summary


# =========================
# 5) Accord NN / XGB
# =========================
def agreement_stats(df):
    print("\n=======================")
    print("Accord NN / XGB (battement) :")
    print("=======================")

    df["agree"] = (df["nn_pred_malade"] == df["xgb_pred_malade"]).astype(int)

    agree_rate = df["agree"].mean()
    print(f"Taux d'accord global NN/XGB : {agree_rate:.3f} ({agree_rate*100:.1f} %)")

    print("\nTable de contingence NN vs XGB (normalisée) :")
    ct_norm = pd.crosstab(df["nn_pred_malade"], df["xgb_pred_malade"], normalize="all")
    print(ct_norm)

    # table brute (pour la heatmap aussi, si besoin)
    ct_raw = pd.crosstab(df["nn_pred_malade"], df["xgb_pred_malade"])

    return agree_rate, ct_norm, ct_raw


# =========================
# 6) FIGURES
# =========================
def plot_global_bar(stats, out_path):
    """
    stats = {'NN': pct_malade, 'XGB': pct_malade}
    """
    models = list(stats.keys())
    values = [stats[m] for m in models]

    plt.figure(figsize=(5, 4))
    plt.bar(models, values)
    plt.ylabel("% de battements prédits malades")
    plt.title("Pourcentage global de battements malades\nNN vs XGBoost")
    plt.ylim(0, 100)
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Sauvé :", out_path)


def plot_patient_scatter(patient_summary, out_path):
    """
    Nuage % malades NN vs XGB par patient.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(patient_summary["nn_pct_malade"],
                patient_summary["xgb_pct_malade"],
                alpha=0.5)
    max_val = max(patient_summary["nn_pct_malade"].max(),
                  patient_summary["xgb_pct_malade"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("% battements malades NN")
    plt.ylabel("% battements malades XGB")
    plt.title("% de battements malades par patient\nNN vs XGBoost")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Sauvé :", out_path)


def plot_agreement_heatmap(ct_norm, out_path):
    """
    Heatmap de la table de contingence normalisée (NN vs XGB).
    """
    plt.figure(figsize=(4, 4))
    sns.heatmap(ct_norm, annot=True, fmt=".3f",
                cmap="Blues", cbar=False,
                xticklabels=["XGB 0 (sain)", "XGB 1 (malade)"],
                yticklabels=["NN 0 (sain)", "NN 1 (malade)"])
    plt.title("Accord NN / XGB au niveau battement\n(fréquence relative)")
    plt.xlabel("Prédiction XGB")
    plt.ylabel("Prédiction NN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Sauvé :", out_path)


# =========================
# 7) Main
# =========================
def main():
    df = load_and_merge()

    # Stat globales + figure barres
    stats = global_stats(df)
    plot_global_bar(stats, FIG_DIR / "ptb_nn_xgb_global_pct_malade.png")

    # Stat par patient + figure scatter
    patient_summary = per_patient_stats(df)
    plot_patient_scatter(patient_summary, FIG_DIR / "ptb_nn_xgb_patient_pct_scatter.png")

    # Accord + heatmap
    agree_rate, ct_norm, ct_raw = agreement_stats(df)
    plot_agreement_heatmap(ct_norm, FIG_DIR / "ptb_nn_xgb_agreement_heatmap.png")

    # Sauvegarde du résumé patient
    print(f"\nSauvegarde du résumé par patient dans : {OUT_PATIENT_SUMMARY}")
    patient_summary.to_csv(OUT_PATIENT_SUMMARY, index=False)
    print("OK.")

if __name__ == "__main__":
    main()
