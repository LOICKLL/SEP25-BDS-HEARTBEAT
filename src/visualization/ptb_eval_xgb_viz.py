# -*- coding: utf-8 -*-
"""
Évaluation & DataViz d'un fichier de battements PTB scoré par XGBoost.

Entrée attendue (colonnes tolérées) :
- patient, record, idx_local, lead, ...
- healthy_label (0=malade, 1=sain)  OU  gt_malade (0/1)
- xgb_proba_malade  (proba classe 1 = "malade")
- xgb_pred_malade   (optionnelle ; 0/1)

Sorties :
- Confusion matrix (norm/true)
- Courbes Precision/Recall/F1 vs seuil
- Histogrammes des proba par classe
- Accuracy par patient (top N)
- CSV des métriques par patient

Corrige l’erreur pandas.fillna(ndarray) en ne remplaçant les NaN
de xgb_pred_malade que positionnellement avec le seuillage des proba.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# ================== PATHS ==================
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

# ⚠️ Ajuste si besoin :
CSV_PATH = DATA_DIR / "ptb_beats_xgb_minimal_labels_only.csv"

OUT_DIR = PROJECT_ROOT / "reports" / "figures" / "ptb_eval_xgb_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================== HELPERS ==================
def savefig(name: str, dpi: int = 150):
    path = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Sauvé : {path}")


def get_col(df: pd.DataFrame, *candidates, required=False):
    """Retourne la première colonne existante parmi candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Aucune des colonnes {candidates} n'existe dans le CSV.")
    return None


# ================== LOAD ==================
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df.loc[:, ~df.columns.duplicated()].copy()  # sécurité

print("\nAperçu :")
print(df.head())
print("\nColonnes :", list(df.columns))

# ================== CIBLES & PROBA ==================
# Ground truth (préférence : gt_malade, sinon dérivé de healthy_label)
gt_col = get_col(df, "gt_malade")
if gt_col is None:
    # healthy_label (1=healthy, 0=sick) -> y_true = 1 - healthy
    hcol = get_col(df, "healthy_label", required=True)
    y_true = (1 - pd.to_numeric(df[hcol], errors="coerce").fillna(0).astype(int)).to_numpy()
else:
    y_true = pd.to_numeric(df[gt_col], errors="coerce").fillna(0).astype(int).to_numpy()

# Proba prédite
proba_col = get_col(df, "xgb_proba_malade", "proba_malade", required=True)
p_hat = pd.to_numeric(df[proba_col], errors="coerce").to_numpy()
# fallback éventuel si tout NaN
if np.isnan(p_hat).all():
    raise ValueError(f"La colonne {proba_col} est entièrement NaN (impossible de tracer).")

# Prédiction 0/1
pred_col = get_col(df, "xgb_pred_malade", "pred_malade", required=False)
if pred_col is not None:
    # convertir proprement + remplir seulement les NaN par le seuillage des proba à 0.5
    pred_series = pd.to_numeric(df[pred_col], errors="coerce")
    mask_na = pred_series.isna()
    filled = pred_series.copy()
    # NB: p_hat[mask_na] garde l'alignement positionnel
    filled.loc[mask_na] = (p_hat[mask_na] >= 0.5).astype(int)
    y_pred = filled.astype(int).to_numpy()
else:
    y_pred = (p_hat >= 0.5).astype(int)

# ================== METRICS GLOBALES ==================
labels = [0, 1]
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

print("\nMatrice de confusion (brute) :\n", cm)
print("\nClassification report :")
print(classification_report(y_true, y_pred, digits=3))
try:
    auc = roc_auc_score(y_true, p_hat)
    print(f"AUC ROC (proba) : {auc:.3f}")
except Exception:
    auc = np.nan

# ================== FIGURES ==================
# 1) Confusion matrix (normalisée)
plt.figure(figsize=(4.2, 4))
sns.heatmap(
    cm_norm,
    annot=True, fmt=".2f",
    cmap="Blues", vmin=0, vmax=1, cbar=False,
    xticklabels=["Sain (0)", "Malade (1)"],
    yticklabels=["Sain (0)", "Malade (1)"],
)
plt.title("Matrice de confusion normalisée (test)")
plt.xlabel("Prédit"); plt.ylabel("Vrai")
savefig("xgb_cm_norm")

# 2) Histogrammes des proba par classe
plt.figure(figsize=(6.5, 4))
sns.histplot(p_hat[y_true == 0], bins=40, stat="density", alpha=0.5, label="Vrai 0", kde=True)
sns.histplot(p_hat[y_true == 1], bins=40, stat="density", alpha=0.5, label="Vrai 1", kde=True)
plt.axvline(0.5, color="k", ls="--", lw=1, label="Seuil 0.5")
plt.xlabel("Proba 'malade'"); plt.ylabel("Densité")
plt.title("Distribution des probabilités par classe vraie")
plt.legend()
savefig("xgb_proba_hist_by_class")

# 3) Courbes Precision/Recall/F1 vs seuil
thr_grid = np.linspace(0.01, 0.99, 99)
prec_list, rec_list, f1_list = [], [], []
for thr in thr_grid:
    y_thr = (p_hat >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_thr, average="binary", pos_label=1, zero_division=0
    )
    prec_list.append(pr); rec_list.append(rc); f1_list.append(f1)
best_idx = int(np.nanargmax(f1_list))
best_thr = float(thr_grid[best_idx])
best_f1  = float(f1_list[best_idx])

plt.figure(figsize=(7, 4))
plt.plot(thr_grid, prec_list, label="Precision")
plt.plot(thr_grid, rec_list,  label="Recall")
plt.plot(thr_grid, f1_list,   label="F1")
plt.axvline(0.5, color="k", ls="--", lw=1, label="Seuil 0.5")
plt.axvline(best_thr, color="r", ls="--", lw=1, label=f"Seuil* {best_thr:.2f}")
plt.ylim(0, 1.05)
plt.xlabel("Seuil"); plt.ylabel("Score")
plt.title("Precision / Recall / F1 en fonction du seuil")
plt.legend(loc="lower left")
savefig("xgb_prf1_vs_threshold")

# 4) Courbe ROC (si possible)
try:
    fpr, tpr, _ = roc_curve(y_true, p_hat)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR (1-Spécificité)")
    plt.ylabel("TPR (Sensibilité)")
    plt.title("ROC curve")
    plt.legend()
    savefig("xgb_roc_curve")
except Exception:
    pass

# 5) Accuracy par patient (Top 30)
if "patient" in df.columns:
    # si df contient déjà xgb_pred_malade, on s'assure de la cohérence avec y_pred
    pred_for_group = pd.Series(y_pred, index=df.index, name="pred")
    true_for_group = pd.Series(y_true, index=df.index, name="true")
    acc_by_pat = (pred_for_group == true_for_group).groupby(df["patient"]).mean().sort_values(ascending=False)
    acc_by_pat_top = acc_by_pat.head(30)

    plt.figure(figsize=(10, 4))
    acc_by_pat_top.plot(kind="bar")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Accuracy par patient (Top 30)")
    savefig("xgb_accuracy_by_patient_top30")

    # Sauvegarde CSV des metrics par patient
    pat_df = acc_by_pat.rename("accuracy").to_frame()
    pat_df["n_beats"] = df.groupby("patient").size()
    pat_df.to_csv(OUT_DIR / "xgb_metrics_by_patient.csv", index=True)
    print("Sauvé :", OUT_DIR / "xgb_metrics_by_patient.csv")

# 6) Table (image) du classification_report
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
rep_df = pd.DataFrame(report).T[["precision", "recall", "f1-score", "support"]]
plt.figure(figsize=(7, 2.2 + 0.28 * len(rep_df)))
plt.axis("off")
plt.title("Classification report (seuil 0.5 si pas de prédictions)")
tbl = plt.table(
    cellText=np.round(rep_df.values, 3),
    rowLabels=rep_df.index,
    colLabels=rep_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.1, 1.25)
savefig("xgb_classification_report")

print("\n[OK] Évaluation & figures générées dans :", OUT_DIR)
print(f"Seuil F1* optimal ≈ {best_thr:.2f} (F1={best_f1:.3f})")
