# -*- coding: utf-8 -*-
"""
Exploration d'un CSV PTB (métadonnées) : infos, stats, graphes.
- Nettoyage (sexe, fumeur, pressions "syst/diast", dates)
- Répartition par sexe, tabagisme, localisation d'infarctus, # vaisseaux...
- Corrélations hémodynamiques
- Délais temporels, sténoses coronaires, rest vs load
- Manquants : barplot % et heatmap colonnes clés
- Médicaments : co-occurrences
- Timelines mini
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ====== PARAMÈTRES ======
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ptb_metadata_all.cleaned.csv"
OUT_DIR  = PROJECT_ROOT / "reports" / "figures" / "exploration_PTB"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FIGS = True  

# ====== OUTIL SAUVEGARDE/SHOW ======
def save_or_show(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    if SAVE_FIGS:
        out = OUT_DIR / f"{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("Sauvé :", out)
    else:
        plt.show()

# ====== HELPERS NETTOYAGE ======
def to_sex(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ("m", "male", "masculin", "man", "homme"): return "M"
    if s in ("f", "female", "feminin", "woman", "femme"): return "F"
    return np.nan

def to_bool(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ("yes", "y", "true", "1", "oui"): return True
    if s in ("no", "n", "false", "0", "non"): return False
    return np.nan

_re_pair = re.compile(r"^\s*(\d+)\s*[/]\s*(\d+)")
_re_num  = re.compile(r"(-?\d+(\.\d+)?)")

def parse_bp_pair(text):
    if pd.isna(text): return np.nan, np.nan
    m = _re_pair.match(str(text))
    if m:
        try: return float(m.group(1)), float(m.group(2))
        except: return np.nan, np.nan
    return np.nan, np.nan

def parse_numeric(text):
    if pd.isna(text): return np.nan
    m = _re_num.search(str(text))
    return float(m.group(1)) if m else np.nan

def try_parse_date(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    fmts = ["%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%y"]
    for f in fmts:
        try: return pd.to_datetime(s, format=f, dayfirst=True, errors="raise")
        except: pass
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def parse_time_hhmm_to_minutes(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace(".", ":")
    try:
        h, m = s.split(":")[0:2]
        return 60*int(h) + int(m)
    except: return np.nan

def percent_from_text(x):
    if pd.isna(x): return np.nan
    m = re.search(r"(\d{1,3})\s*%", str(x))
    return float(m.group(1)) if m else np.nan

def parse_pressure_with_units(x):
    if pd.isna(x): return np.nan
    s = str(x)
    val = parse_numeric(s)
    if pd.isna(val): return np.nan
    if "cmh2o" in s.lower():  # conversion -> mmHg
        return val * 0.7356
    return val

# ====== CHARGEMENT ======
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df.loc[:, ~df.columns.duplicated()].copy()

print("\nAperçu (head) :")
print(df.head())
print("\n.info() :")
df.info()

# ====== NETTOYAGES LÉGERS ======
if "sex" in df.columns:
    df["sex"] = df["sex"].apply(to_sex)

if "smoker" in df.columns:
    df["smoker"] = df["smoker"].apply(to_bool)

if "peripheral_blood_pressure_syst_diast" in df.columns:
    syst, diast = [], []
    for x in df["peripheral_blood_pressure_syst_diast"]:
        s, d = parse_bp_pair(x); syst.append(s); diast.append(d)
    df["pbp_sys"] = syst; df["pbp_dia"] = diast

if "aorta_at_rest_syst_diast" in df.columns:
    a_sys, a_dia = [], []
    for x in df["aorta_at_rest_syst_diast"]:
        s, d = parse_bp_pair(x); a_sys.append(s); a_dia.append(d)
    df["aorta_sys"] = a_sys; df["aorta_dia"] = a_dia

if "aorta_at_rest_mean" in df.columns:
    df["aorta_mean"] = df["aorta_at_rest_mean"].apply(parse_pressure_with_units)

for col in ["cardiac_output_at_rest", "cardiac_index_at_rest",
            "cardiac_output_load", "cardiac_index_load",
            "stroke_volume_index_at_rest", "stroke_volume_index_load",
            "left_ventricular_enddiastolic_pressure",
            "pulmonary_artery_pressure_at_rest_mean",
            "pulmonary_capillary_wedge_pressure_at_rest",
            "pulmonary_capillary_wedge_pressure_load",
            "pulmonary_artery_pressure_laod_mean"]:
    if col in df.columns:
        df[col] = df[col].apply(parse_numeric)

for dcol in ["ecg_date", "infarction_date", "admission_date",
             "infarction_date_acute", "previous_infarction_1_date",
             "previous_infarction_2_date", "catheterization_date"]:
    if dcol in df.columns:
        df[dcol + "_dt"] = df[dcol].apply(try_parse_date)

if "start_lysis_therapy_hh.mm" in df.columns:
    df["lysis_minutes"] = df["start_lysis_therapy_hh.mm"].apply(parse_time_hhmm_to_minutes)

if "number_of_coronary_vessels_involved" in df.columns:
    df["n_vessels"] = pd.to_numeric(df["number_of_coronary_vessels_involved"], errors="coerce")

# ====== RÉSUMÉS ======
print("\nRésumé numérique (describe) :")
print(df.select_dtypes(include=[np.number]).describe())

print("\nManquants (total) :", df.isna().sum().sum())
print("\nManquants par colonne (top 20) :")
print(df.isna().sum().sort_values(ascending=False).head(20))

# ====== GRAPHIQUES NA (placés tôt pour garantir la sortie) ======
def plot_na_percent(df_, top=None, exclude_prefix=None, title_suffix=""):
    cols = df_.columns
    if exclude_prefix:
        keep = []
        for c in cols:
            c_str = str(c)
            keep.append(not any(c_str.startswith(p) for p in exclude_prefix))
        sub = df_.loc[:, keep]
    else:
        sub = df_

    na_pct = (sub.isna().mean() * 100).sort_values(ascending=True)  # barh croissant
    if top is not None:
        na_pct = na_pct.tail(top)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25*len(na_pct))))
    na_pct.plot(kind="barh", ax=ax)
    ax.set_xlabel("% NA")
    ax.set_title(f"Pourcentage de valeurs manquantes par colonne{title_suffix}")
    save_or_show("ptb_na_percent_all" if top is None else f"ptb_na_percent_top{top}", fig=fig)

plot_na_percent(df, top=None)
plot_na_percent(df, top=30)

# ====== GRAPHIQUES DE BASE ======
if "sex" in df.columns:
    fig, ax = plt.subplots(figsize=(5,5))
    df["sex"].value_counts(dropna=False).plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title("Répartition par sexe")
    save_or_show("ptb_repartition_sexe", fig=fig)

if "age" in df.columns:
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(pd.to_numeric(df["age"], errors="coerce").dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Distribution de l'âge (PTB)")
    ax.set_xlabel("Âge"); ax.set_ylabel("Effectif")
    save_or_show("ptb_age_hist", fig=fig)

    fig, ax = plt.subplots(figsize=(4,5))
    ax.boxplot(pd.to_numeric(df["age"], errors="coerce").dropna())
    ax.set_title("Boxplot âge"); ax.set_ylabel("Âge")
    save_or_show("ptb_age_box", fig=fig)

if "smoker" in df.columns:
    fig, ax = plt.subplots(figsize=(5,5))
    df["smoker"].value_counts(dropna=False).plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title("Répartition fumeur (PTB)")
    save_or_show("ptb_smoker_pie", fig=fig)

if "acute_infarction_localization" in df.columns:
    top_loc = (df["acute_infarction_localization"].fillna("n/a").str.lower().value_counts().head(12))
    fig, ax = plt.subplots(figsize=(8,4))
    top_loc.plot(kind="bar", ax=ax)
    ax.set_title("Localisations d'infarctus (aigu) – Top")
    ax.set_ylabel("Comptes")
    save_or_show("ptb_infarct_localization_top", fig=fig)

if "n_vessels" in df.columns:
    cats = pd.cut(df["n_vessels"], bins=[-0.5,0.5,1.5,2.5,10], labels=["0","1","2","3+"])
    vc = cats.value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    vc.plot(kind="bar", ax=ax)
    ax.set_title("Nombre de vaisseaux coronaires impliqués")
    ax.set_xlabel("# vaisseaux"); ax.set_ylabel("Comptes")
    save_or_show("ptb_nb_vessels_bar", fig=fig)

if {"pbp_sys","pbp_dia"}.issubset(df.columns):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.kdeplot(df["pbp_sys"].dropna(), label="Systolique", ax=ax)
    sns.kdeplot(df["pbp_dia"].dropna(), label="Diastolique", ax=ax)
    ax.set_title("Distribution des pressions artérielles périphériques")
    ax.set_xlabel("mmHg"); ax.legend()
    save_or_show("ptb_pbp_kde", fig=fig)

    if "sex" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.violinplot(data=df, x="sex", y="pbp_sys", ax=ax)
        ax.set_title("PA systolique par sexe")
        save_or_show("ptb_pbp_sys_by_sex", fig=fig)

if {"aorta_sys","aorta_dia","aorta_mean"}.issubset(df.columns):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.kdeplot(df["aorta_sys"].dropna(), label="Aorte syst.", ax=ax)
    sns.kdeplot(df["aorta_dia"].dropna(), label="Aorte diast.", ax=ax)
    sns.kdeplot(df["aorta_mean"].dropna(), label="Aorte mean (mmHg)", ax=ax)
    ax.set_title("Distribution pressions aortiques (repos)")
    ax.set_xlabel("mmHg"); ax.legend()
    save_or_show("ptb_aorta_kde", fig=fig)

hemo_cols = [c for c in [
    "cardiac_output_at_rest","cardiac_index_at_rest",
    "pulmonary_artery_pressure_at_rest_mean",
    "pulmonary_capillary_wedge_pressure_at_rest",
    "left_ventricular_enddiastolic_pressure"
] if c in df.columns]
if len(hemo_cols) >= 2:
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df[hemo_cols].astype(float).corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Corrélations hémodynamiques (repos)")
    save_or_show("ptb_hemodynamics_corr", fig=fig)

# ====== BONUS (1) Délais temporels ======
if {"infarction_date_dt","admission_date_dt","ecg_date_dt"}.issubset(df.columns):
    df["delay_infx_to_adm_d"] = (df["admission_date_dt"] - df["infarction_date_dt"]).dt.days
    df["delay_infx_to_ecg_d"] = (df["ecg_date_dt"] - df["infarction_date_dt"]).dt.days

    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df["delay_infx_to_adm_d"].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Délai infarctus → admission (jours)")
    ax.set_xlabel("jours"); ax.set_ylabel("patients")
    save_or_show("ptb_delay_infarctus_to_admission_hist", fig=fig)

    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df["delay_infx_to_ecg_d"].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Délai infarctus → ECG (jours)")
    ax.set_xlabel("jours"); ax.set_ylabel("patients")
    save_or_show("ptb_delay_infarctus_to_ecg_hist", fig=fig)

if "lysis_minutes" in df.columns:
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df["lysis_minutes"].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Délai jusqu'à la thrombolyse (minutes depuis l'admission)")
    ax.set_xlabel("minutes"); ax.set_ylabel("patients")
    save_or_show("ptb_lysis_delay_minutes_hist", fig=fig)

# ====== BONUS (2) Sténoses coronaires (%) + heatmap patient×artère ======
sten_cols_raw = {
    "left_coronary_artery_stenoses_riva": "RIVA_pct",
    "left_coronary_artery_stenoses_rcx":  "RCX_pct",
    "right_coronary_artery_stenoses_rca": "RCA_pct",
}
for raw, newc in sten_cols_raw.items():
    if raw in df.columns:
        df[newc] = df[raw].apply(percent_from_text)

id_col = "patient_dir" if "patient_dir" in df.columns else ("record_name" if "record_name" in df.columns else None)
needed = list(sten_cols_raw.values())
if id_col and set(needed).issubset(df.columns):
    heat_df = df[[id_col] + needed].copy()
    heat_df = heat_df.groupby(id_col, as_index=False).mean(numeric_only=True)
    heat_df = heat_df.sort_values(by=needed, ascending=False, na_position="last").head(50)
    fig, ax = plt.subplots(figsize=(8, max(4, min(12, 0.25*len(heat_df)))))
    sns.heatmap(heat_df.set_index(id_col), annot=False, cmap="Reds", vmin=0, vmax=100, ax=ax)
    ax.set_title("Sténoses coronaires (%) – Top 50 patients")
    save_or_show("ptb_stenosis_patient_artery_heatmap", fig=fig)

# ====== BONUS (3) Rest vs Load : scatter avec diagonale ======
def rest_vs_load_scatter(xcol_rest, xcol_load, title, fname):
    if {xcol_rest, xcol_load}.issubset(df.columns):
        sub = df[[xcol_rest, xcol_load]].dropna()
        if len(sub) >= 5:
            fig, ax = plt.subplots(figsize=(5.5,5))
            sns.scatterplot(x=sub[xcol_rest], y=sub[xcol_load], s=20, ax=ax)
            lo = float(np.nanmin(sub.values)); hi = float(np.nanmax(sub.values))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
            ax.set_title(title); ax.set_xlabel(xcol_rest); ax.set_ylabel(xcol_load)
            save_or_show(fname, fig=fig)

rest_vs_load_scatter("cardiac_output_at_rest", "cardiac_output_load",
                     "Débit cardiaque : repos vs charge", "ptb_CO_rest_vs_load")
rest_vs_load_scatter("cardiac_index_at_rest", "cardiac_index_load",
                     "Index cardiaque : repos vs charge", "ptb_CI_rest_vs_load")
rest_vs_load_scatter("pulmonary_capillary_wedge_pressure_at_rest", "pulmonary_capillary_wedge_pressure_load",
                     "Wedge pressure : repos vs charge", "ptb_wedge_rest_vs_load")

# ====== BONUS (5) Heatmap de complétude (colonnes clés) ======
key_cols = [c for c in [
    "age", "sex", "smoker", "n_vessels",
    "pbp_sys", "pbp_dia", "aorta_mean",
    "cardiac_output_at_rest", "cardiac_output_load",
    "cardiac_index_at_rest",  "cardiac_index_load",
    "left_ventricular_enddiastolic_pressure",
    "pulmonary_artery_pressure_at_rest_mean",
    "pulmonary_capillary_wedge_pressure_at_rest",
    "pulmonary_capillary_wedge_pressure_load",
    "RIVA_pct", "RCX_pct", "RCA_pct"
] if c in df.columns]
if key_cols:
    miss_mat = df[key_cols].isna().astype(int)
    samp = miss_mat.sample(n=min(100, len(miss_mat)), random_state=42)
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(key_cols)), 6))
    sns.heatmap(samp.T, cmap="Greys", cbar=False, ax=ax)
    ax.set_title("Heatmap de complétude (1=NaN) – échantillon patients")
    save_or_show("ptb_missing_heatmap_keys", fig=fig)

# ====== BONUS (6) Médicaments : co-occurrence ======
def meds_cooccurrence_heatmap(columns, title, fname, top_k=20):
    pools = []
    for col in columns:
        if col in df.columns:
            s = (df[col].dropna().astype(str)
                 .str.lower().str.replace("|", ",").str.replace(";", ",").str.split(","))
            pools += [x.strip() for sub in s for x in sub if x and x.strip()]
    if not pools: return
    vc = pd.Series(pools).value_counts().head(top_k)
    meds = vc.index.tolist()
    M = pd.DataFrame(0, index=np.arange(len(df)), columns=meds, dtype=int)
    for col in columns:
        if col in df.columns:
            lists = (df[col].fillna("").astype(str)
                     .str.lower().str.replace("|", ",").str.replace(";", ",").str.split(","))
            for i, lst in enumerate(lists):
                for x in lst:
                    x = x.strip()
                    if x in meds:
                        M.loc[i, x] = 1
    co = M.T.dot(M)
    if co.values.sum() > 0:
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(meds)), max(6, 0.4*len(meds))))
        sns.heatmap(co, cmap="Blues", square=True, ax=ax)
        ax.set_title(title)
        save_or_show(fname, fig=fig)

meds_cooccurrence_heatmap(
    ["medication_pre_admission", "in_hospital_medication", "medication_after_discharge"],
    "Co-occurrences médicamenteuses (top 20, toutes phases)",
    "ptb_meds_cooccurrence_allphases"
)

# ====== BONUS (7) Timelines mini ======
if {"admission_date_dt","infarction_date_dt","ecg_date_dt","catheterization_date_dt"}.issubset(df.columns):
    id_col = "patient_dir" if "patient_dir" in df.columns else ("record_name" if "record_name" in df.columns else None)
    if id_col:
        mini = df[[id_col, "admission_date_dt","infarction_date_dt","ecg_date_dt","catheterization_date_dt"]].copy()
        melt = mini.melt(id_vars=[id_col], var_name="event", value_name="date").dropna()
        top_pat = melt[id_col].value_counts().head(20).index
        melt = melt[melt[id_col].isin(top_pat)].copy()
        t0 = melt.groupby(id_col)["date"].transform("min")
        melt["days_from_first"] = (melt["date"] - t0).dt.days
        fig, ax = plt.subplots(figsize=(10,6))
        sns.stripplot(data=melt, x="days_from_first", y=id_col, hue="event", dodge=True, ax=ax)
        ax.set_title("Mini-timelines : événements par patient (J depuis premier événement)")
        ax.set_xlabel("Jours depuis premier événement"); ax.set_ylabel("Patient")
        ax.legend(title="Événement", bbox_to_anchor=(1.02,1), loc="upper left")
        save_or_show("ptb_timelines_patients", fig=fig)

# ====== KHI² : quelques tests simples ======
def chi2_print(ct, title):
    chi2, p, ddl, _ = chi2_contingency(ct)
    print(f"\n=== χ² : {title} ===")
    print(ct, "\n")
    print(f"χ² = {chi2:.3f} | ddl = {ddl} | p-value = {p:.6f}")
    print("⇒", "Dépendance significative (rejette H0)" if p < 0.05 else "Pas de dépendance significative (ne rejette pas H0)")

if {"sex","n_vessels"}.issubset(df.columns):
    cats = pd.cut(df["n_vessels"], bins=[-0.5,0.5,1.5,2.5,10], labels=["0","1","2","3+"])
    ct = pd.crosstab(df["sex"].dropna(), cats.dropna())
    if ct.size > 0: chi2_print(ct, "SEXE ~ NB_VAISSEAUX")

if {"smoker","n_vessels"}.issubset(df.columns):
    cats = pd.cut(df["n_vessels"], bins=[-0.5,0.5,1.5,2.5,10], labels=["0","1","2","3+"])
    ct = pd.crosstab(df["smoker"].dropna(), cats.dropna())
    if ct.size > 0: chi2_print(ct, "SMOKER ~ NB_VAISSEAUX")

if {"sex","acute_infarction_localization"}.issubset(df.columns):
    loc = df["acute_infarction_localization"].fillna("n/a").str.lower()
    top5 = loc.value_counts().head(5).index
    loc_cat = loc.where(loc.isin(top5), other="autres")
    ct = pd.crosstab(df["sex"].dropna(), loc_cat)
    if ct.size > 0: chi2_print(ct, "SEXE ~ LOCALISATION_INFARCTUS(top5+autres)")

if {"smoker","acute_infarction_localization"}.issubset(df.columns):
    loc = df["acute_infarction_localization"].fillna("n/a").str.lower()
    top5 = loc.value_counts().head(5).index
    loc_cat = loc.where(loc.isin(top5), other="autres")
    ct = pd.crosstab(df["smoker"].dropna(), loc_cat)
    if ct.size > 0: chi2_print(ct, "SMOKER ~ LOCALISATION_INFARCTUS(top5+autres)")

print("\n[OK] EDA PTB terminé. Figures sauvegardées dans :", OUT_DIR if SAVE_FIGS else "(affichées)")
