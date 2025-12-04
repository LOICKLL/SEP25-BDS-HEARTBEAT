# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# ============================== PATHS ==============================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "Preprocess_MIT"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_IN = DATA_DIR / "mitbih_187pts_MLII.csv"   # fichier simplifié (capture)
print("Lecture :", CSV_IN)

# =============== Helper : enregistrer un classification_report en image ===============
def save_classif_report_as_table(y_true, y_pred, title, outpath):
    """
    Crée une image PNG avec un tableau (precision, recall, f1, support)
    à partir de classification_report.
    """
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Ordonner les lignes
    classes = sorted([k for k in rep.keys() if str(k).isdigit()], key=lambda x: int(x))
    row_order = classes + ["accuracy", "macro avg", "weighted avg"]

    rows = []
    for k in row_order:
        if isinstance(rep.get(k), dict):
            d = rep[k]
            rows.append([
                d.get("precision", np.nan),
                d.get("recall", np.nan),
                d.get("f1-score", np.nan),
                int(d.get("support", 0)),
            ])
        else:
            # "accuracy" est un float, on l’adapte
            rows.append([np.nan, np.nan, rep.get(k, np.nan), 0])

    df_show = pd.DataFrame(
        rows, index=row_order, columns=["precision", "recall", "f1-score", "support"]
    )

    fig = plt.figure(figsize=(9, 3.8))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.title(title, fontsize=16, pad=12)

    tbl = plt.table(
        cellText=np.round(df_show.values, 3),
        rowLabels=df_show.index,
        colLabels=df_show.columns,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
        if col == -1:
            cell.set_text_props(fontweight="bold")

    outpath = Path(outpath)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

# =============== Helper : deux matrices de confusion (counts & normalized) ===============
def save_confusions(y_true, y_pred, title_prefix, basename, fig_dir: Path):
    labels_sorted = sorted(np.unique(np.concatenate([y_true, y_pred])))
    ticklabels = [str(x) for x in labels_sorted]

    # 1) Brute (counts)
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_counts, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=ticklabels, yticklabels=ticklabels
    )
    plt.title(f"{title_prefix} – (comptes)")
    plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout()
    plt.savefig(fig_dir / f"cm_{basename}_counts.png", dpi=150)
    plt.close()

    # 2) Normalisée (par ligne)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels_sorted, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
        xticklabels=ticklabels, yticklabels=ticklabels, vmin=0, vmax=1
    )
    plt.title(f"{title_prefix} – (normalisée)")
    plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout()
    plt.savefig(fig_dir / f"cm_{basename}_norm.png", dpi=150)
    plt.close()

# ============================== LOAD ==============================
df = pd.read_csv(CSV_IN, low_memory=False)

# reconstruire record_num depuis record_x (ex: 'x_100' -> 100)
df["record_num"] = (
    df["record_x"].astype(str).str.extract(r"x_(\d+)", expand=False).astype(int)
)

# garder MLII et labels valides
df = df[df["lead"].str.upper() == "MLII"].copy()
df = df[df["label"] != -1].copy()

# colonnes signal (0..186) en float32
signal_cols = [str(i) for i in range(187)]
df[signal_cols] = df[signal_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
df["label"] = df["label"].astype(int)

df = df.sort_values("record_num").reset_index(drop=True)
print("Records restants :", df["record_num"].nunique())
print("Labels restants  :", df["label"].nunique())

# ======================= SPLIT TRAIN / TEST PAR PATIENTS =======================
print("... séparation train et test suivant les patients ... ")

def has_label4(g):
    return "class4" if (g == 4).any() else "other"

patient_labels = df.groupby("record_num")["label"].apply(has_label4)

train_pat, test_pat = train_test_split(
    patient_labels.index,
    test_size=0.2,
    random_state=42,
    stratify=patient_labels
)

df_train = df[df["record_num"].isin(train_pat)].reset_index(drop=True)
df_test  = df[df["record_num"].isin(test_pat)].reset_index(drop=True)

print("train counts:\n", df_train["label"].value_counts().sort_index())
print("test  counts:\n", df_test["label"].value_counts().sort_index())

# ======================= 1) UNDERSAMPLING PROPORTIONNEL =======================
print("\n### UNDERSAMPLING PROPORTIONNEL ###")
N_TARGET = 20000
f = min(1.0, N_TARGET / len(df_train))

train_under = (
    df_train.groupby("label", group_keys=False)
            .sample(frac=f, random_state=42)
            .reset_index(drop=True)
)
print("Répartition train sous-échantillonné:\n", train_under["label"].value_counts().sort_index())

X_train = train_under[signal_cols].values
y_train = train_under["label"].values
X_test  = df_test[signal_cols].values
y_test  = df_test["label"].values

model = xgb.XGBClassifier(
    objective="multi:softprob",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 weighted:", f1_score(y_test, y_pred, average="weighted"))

save_confusions(y_test, y_pred,
                "Matrice de confusion – undersampling proportionnel",
                "under_prop_mlii", FIG_DIR)

save_classif_report_as_table(
    y_test, y_pred,
    "Classification report XGBOOST (Under-sampling proportionnel)",
    FIG_DIR / "report_under_prop_mlii.png"
)

# =================== 2) UNDERSAMPLING ÉQUILIBRÉ ===================
print("\n### UNDERSAMPLING ÉQUILIBRÉ ###")
mini = df_train["label"].value_counts().min()
train_bal = (
    df_train.groupby("label", group_keys=False).sample(n=mini, random_state=42).reset_index(drop=True)
)
print("Répartition train équilibré:\n", train_bal["label"].value_counts().sort_index())

X_train = train_bal[signal_cols].values
y_train = train_bal["label"].values

model = xgb.XGBClassifier(
    objective="multi:softprob",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 weighted:", f1_score(y_test, y_pred, average="weighted"))

save_confusions(y_test, y_pred,
                "Matrice de confusion – undersampling équilibré",
                "under_bal_mlii", FIG_DIR)

save_classif_report_as_table(
    y_test, y_pred,
    "Classification report XGBOOST (Under-sampling équilibré)",
    FIG_DIR / "report_under_bal_mlii.png"
)

# ============ 3) UNDERSAMPLING RAISONNÉ (label 0) ============
print("\n### UNDERSAMPLING RAISONNÉ (classe 0) ###")
MAJ_LABEL = 0
df_N = df_train[df_train["label"] == MAJ_LABEL]
df_others = df_train[df_train["label"] != MAJ_LABEL]

n_keep_N = min(10000, len(df_N))  # à ajuster si besoin
df_N_sampled = df_N.sample(n=n_keep_N, random_state=42)
train_underN = pd.concat([df_others, df_N_sampled], ignore_index=True)
train_underN = train_underN.sample(frac=1, random_state=42).reset_index(drop=True)
print("Répartition train raisonné:\n", train_underN["label"].value_counts().sort_index())

X_train = train_underN[signal_cols].values
y_train = train_underN["label"].values

model = xgb.XGBClassifier(
    objective="multi:softprob",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 weighted:", f1_score(y_test, y_pred, average="weighted"))

save_confusions(y_test, y_pred,
                "Matrice de confusion – undersampling raisonné (N)",
                "underN_mlii", FIG_DIR)

save_classif_report_as_table(
    y_test, y_pred,
    "Classification report XGBOOST (Under-sampling raisonné N)",
    FIG_DIR / "report_underN_mlii.png"
)

# =================== 4) AAMI (3 classes: N/S/V) ===================
print("\n### AAMI (3 classes) ###")
df_aami = df.copy()
merge_map = {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}
df_aami["label"] = df_aami["label"].map(merge_map).astype(int)

patients = df_aami["record_num"].unique()
train_pat, test_pat = train_test_split(patients, test_size=0.2, random_state=42)
df_train_aami = df_aami[df_aami["record_num"].isin(train_pat)].reset_index(drop=True)
df_test_aami  = df_aami[df_aami["record_num"].isin(test_pat)].reset_index(drop=True)

X_train = df_train_aami[signal_cols].values
y_train = df_train_aami["label"].values
X_test  = df_test_aami[signal_cols].values
y_test  = df_test_aami["label"].values

model = xgb.XGBClassifier(
    objective="multi:softprob",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 macro:", f1_score(y_test, y_pred, average="macro"))

save_confusions(y_test, y_pred,
                "Matrice de confusion – AAMI (3 classes)",
                "aami_mlii", FIG_DIR)

save_classif_report_as_table(
    y_test, y_pred,
    "Classification report XGBOOST (AAMI 3 classes)",
    FIG_DIR / "report_aami_mlii.png"
)

# =================== 5) BINAIRE (Normal vs Anormal) ===================
print("\n### Binaire (Normal vs Anormal) ###")
df_bin = df.copy()
df_bin["binary"] = (df_bin["label"] != 0).astype(int)

patients = df_bin["record_num"].unique()
train_pat, test_pat = train_test_split(patients, test_size=0.2, random_state=42)
df_train_bin = df_bin[df_bin["record_num"].isin(train_pat)].reset_index(drop=True)
df_test_bin  = df_bin[df_bin["record_num"].isin(test_pat)].reset_index(drop=True)

# undersampling léger : max 3:1 pour les normaux
normal  = df_train_bin[df_train_bin["binary"] == 0]
abnorm  = df_train_bin[df_train_bin["binary"] == 1]
keep_norm = min(len(normal), 3 * len(abnorm))
normal_sampled = normal.sample(n=keep_norm, random_state=42)
df_train_bin_bal = pd.concat([normal_sampled, abnorm], ignore_index=True).sample(frac=1, random_state=42)

X_train = df_train_bin_bal[signal_cols].values
y_train = df_train_bin_bal["binary"].values
X_test  = df_test_bin[signal_cols].values
y_test  = df_test_bin["binary"].values

model = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 weighted:", f1_score(y_test, y_pred, average="weighted"))

save_confusions(y_test, y_pred,
                "Matrice de confusion – Binaire",
                "binary_mlii", FIG_DIR)

save_classif_report_as_table(
    y_test, y_pred,
    "Classification report XGBOOST Binaire",
    FIG_DIR / "report_binary_mlii.png"
)

print("\n[OK] Terminé. Figures dans :", FIG_DIR)
