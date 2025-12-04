# apply_nn_on_ptb_beats_with_label_minimal_anti_fp.py
# -*- coding: utf-8 -*-
"""
Application d'un modèle Keras (binaire) sur les battements PTB avec logique anti-FP :
- Normalisation via scaler sauvegardé si dispo, sinon standardisation par battement
- Calibration isotone optionnelle (sur 20% des patients)
- Choix de seuil visant une précision cible (réduit les FP)
- Lissage k-sur-m par patient optionnel
- Sauvegarde d'un CSV minimal (et compatibilité noms xgb_*)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

# ============================== PARAMS ==============================
TARGET_PRECISION = 0.90     # vise une précision mini (réduit les FP)
USE_ISOTONIC_CAL = True     # calibration des proba sur 20% des patients PTB
USE_SMOOTHING    = False    # lissage patient k-sur-m pour réduire les FP isolés
SMOOTH_WIN       = 5        # fenêtre glissante
SMOOTH_K         = 3        # au moins K proba>=seuil dans la fenêtre -> 1
BATCH_SIZE_PRED  = 2048     # batch pour predict()
RANDOM_STATE     = 42
# ===================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

# Model & data
MODEL_PATH = PROJECT_ROOT / "models" / "nn_mit_mlii_binary.keras"  # adapte si besoin
SCALER_PATH = PROJECT_ROOT / "models" / "scaler_mit_mlii_binary.pkl"     # si tu l'as sauvegardé
PTB_BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ptb_beats_all_with_healthy_label.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "ptb_beats_nn_minimal_labels_only.csv"

# ============================== UTILS ==============================
def load_scaler(path: Path):
    if path.exists():
        try:
            sc = joblib.load(path)
            print(f"Scaler chargé : {path}")
            return sc
        except Exception as e:
            print(f"Impossible de charger le scaler ({e}), on passera en fallback par battement.")
    else:
        print("Aucun scaler trouvé, fallback : standardisation par battement.")
    return None

def per_beat_standardize(X: np.ndarray) -> np.ndarray:
    # standardisation ligne par ligne : (x - mean)/std
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - m) / s

def prepare_X_for_model(X: np.ndarray, model: tf.keras.Model, scaler: StandardScaler | None):
    # 1) scale
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = per_beat_standardize(X)
    Xs = Xs.astype("float32")

    # 2) reshape selon l'input du modèle
    # - Dense: (None, 187) -> rank 2
    # - CNN1D: (None, 187, 1) -> rank 3
    in_shape = model.input_shape
    rank = len(in_shape) if isinstance(in_shape, tuple) else len(in_shape[0])
    if rank == 3:
        # (batch, 187, channels)
        if Xs.ndim == 2:
            Xs = Xs[..., np.newaxis]
    elif rank == 2:
        # (batch, 187)
        if Xs.ndim == 3:
            Xs = Xs.reshape((Xs.shape[0], Xs.shape[1]))
    else:
        # fallback : tenter (N, 187) -> (N, 187, 1)
        if Xs.ndim == 2:
            Xs = Xs[..., np.newaxis]
    return Xs

def make_calibration_mask(df: pd.DataFrame, frac=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    if "patient" in df.columns:
        patients = df["patient"].astype(str).unique()
        n_cal = max(1, int(len(patients) * frac))
        cal_patients = set(rng.choice(patients, size=n_cal, replace=False))
        mask = df["patient"].astype(str).isin(cal_patients).to_numpy()
    else:
        n = len(df)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(n * frac)
        cal_idx = set(idx[:cut])
        mask = np.array([(i in cal_idx) for i in range(n)], dtype=bool)
    return mask

def pick_threshold_precision(y_true, p_hat, target_precision=0.90):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1  = -1.0
    best_triplet = (np.nan, np.nan, np.nan)
    for t in thr_grid:
        pred = (p_hat >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if pr >= target_precision and f1 > best_f1:
            best_f1  = f1
            best_thr = float(t)
            best_triplet = (pr, rc, f1)
    if best_f1 < 0:  # fallback : max F1
        f1s = []
        for t in thr_grid:
            pred = (p_hat >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, pred, average="binary", zero_division=0
            )
            f1s.append(f1)
        idx = int(np.nanargmax(f1s))
        best_thr = float(thr_grid[idx])
        pr, rc, best_f1, _ = precision_recall_fscore_support(
            y_true, (p_hat >= best_thr).astype(int), average="binary", zero_division=0
        )
        best_triplet = (pr, rc, best_f1)
    print(f"[Seuil choisi] thr={best_thr:.2f} | precision={best_triplet[0]:.3f} | "
          f"recall={best_triplet[1]:.3f} | f1={best_triplet[2]:.3f}")
    return best_thr

def smooth_k_on_m_by_patient(df: pd.DataFrame, proba_col: str, thr: float,
                             win: int = 5, k: int = 3) -> np.ndarray:
    # on ordonne par idx_local si présent, sinon par index
    if "patient" in df.columns:
        order_cols = ["patient"]
        if "idx_local" in df.columns:
            order_cols.append("idx_local")
        df_tmp = df[["patient", proba_col] + (["idx_local"] if "idx_local" in df.columns else [])].copy()
        df_tmp["_order"] = np.arange(len(df_tmp))
        if "idx_local" in df_tmp.columns:
            df_tmp = df_tmp.sort_values(["patient", "idx_local"])
        else:
            df_tmp = df_tmp.sort_values(["patient", "_order"])

        def roll_func(s):
            return (s.rolling(win, min_periods=1)
                      .apply(lambda w: (w >= thr).sum() >= k)
                      .astype(int))

        sm_series = (df_tmp.groupby("patient")[proba_col]
                           .apply(roll_func)
                           .reset_index(level=0, drop=True))
        # réaligne
        sm_aligned = sm_series.loc[df_tmp.index]
        # revenir à l'ordre initial
        sm_final = sm_aligned.reindex(df_tmp.sort_values("_order").index)
        return sm_final.sort_index().to_numpy()
    else:
        # pas de colonne patient -> simple rolling global
        s = df[proba_col]
        sm = s.rolling(win, min_periods=1).apply(lambda w: (w >= thr).sum() >= k).astype(int)
        return sm.to_numpy()

# ============================== MAIN ==============================
def main():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # -------- 1) Modèle
    print(f"Chargement du modèle Keras : {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé.")

    scaler = load_scaler(SCALER_PATH)

    # -------- 2) Données
    print(f"\nLecture des battements PTB : {PTB_BEATS_PATH}")
    if not PTB_BEATS_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {PTB_BEATS_PATH}")
    beats = pd.read_csv(PTB_BEATS_PATH)
    print("Nombre total de lignes :", len(beats))

    if "healthy_label" not in beats.columns:
        raise ValueError("La colonne 'healthy_label' est requise (1=sain, 0=malade).")

    # features b0..b186 en float32, NaN->0
    feat_cols = sorted([c for c in beats.columns if c.startswith("b")],
                       key=lambda x: int(x[1:]))
    if not feat_cols:
        raise KeyError("Aucune colonne 'b0..b186' trouvée.")
    X = (beats[feat_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0.0)
         .astype(np.float32)
         .values)

    y_true = (1 - beats["healthy_label"].astype(int).values)  # 1=malade, 0=sain

    # -------- 3) Préparation + prédiction
    X_ready = prepare_X_for_model(X, model, scaler)
    print("\nPrédiction des probabilités (NN)...")
    p_hat = model.predict(X_ready, batch_size=BATCH_SIZE_PRED, verbose=0).ravel()

    # -------- 4) Calibration isotone optionnelle
    if USE_ISOTONIC_CAL:
        print("Calibration isotone (20% des patients si dispo)...")
        cal_mask = make_calibration_mask(beats, frac=0.2, random_state=RANDOM_STATE)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_hat[cal_mask], y_true[cal_mask])
        p_hat = iso.transform(p_hat)
        print("Calibration isotone : OK")

    # -------- 5) Choix de seuil (anti-FP)
    thr = pick_threshold_precision(y_true, p_hat, target_precision=TARGET_PRECISION)
    y_pred = (p_hat >= thr).astype(int)

    # -------- 6) Lissage optionnel
    if USE_SMOOTHING:
        tmp = beats.copy()
        tmp["__proba__"] = p_hat
        y_pred = smooth_k_on_m_by_patient(tmp, "__proba__", thr, win=SMOOTH_WIN, k=SMOOTH_K)
        print(f"Lissage {SMOOTH_K}-sur-{SMOOTH_WIN} appliqué.")

    # -------- 7) Sortie minimaliste (et compatibilité xgb_*)
    beats["gt_malade"] = y_true
    beats["nn_proba_malade"] = p_hat
    beats["nn_pred_malade"] = y_pred
    beats["correct"] = (y_pred == y_true).astype(int)
    beats["threshold_used"] = thr

    # alias pour compatibilité scripts existants
    beats["xgb_proba_malade"] = beats["nn_proba_malade"]
    beats["xgb_pred_malade"]  = beats["nn_pred_malade"]

    cols_final = [
        "patient","record","idx_local",
        "healthy_label","gt_malade",
        "nn_proba_malade","nn_pred_malade",
        "xgb_proba_malade","xgb_pred_malade",
        "correct","threshold_used"
    ]
    cols_final = [c for c in cols_final if c in beats.columns]
    beats_min = beats[cols_final].copy()
    beats_min.to_csv(OUT_PATH, index=False)

    # -------- 8) Petit résumé
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print("\n=== Résumé (NN) ===")
    print(f"precision={pr:.3f} | recall={rc:.3f} | f1={f1:.3f} | seuil={thr:.2f}")
    print("Fichier sauvegardé :", OUT_PATH)

if __name__ == "__main__":
    main()
