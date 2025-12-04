# apply_xgb_on_ptb_beats_with_label_minimal.py (version anti-FP)

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.isotonic import IsotonicRegression

# ============================== PARAMS ==============================
TARGET_PRECISION = 0.90     # vise une précision mini (réduit les FP)
USE_ISOTONIC_CAL = True     # calibration des proba sur 20% des patients PTB
USE_SMOOTHING = False       # lissage patient k-sur-m pour réduire les FP isolés
SMOOTH_WIN = 5              # fenêtre glissante
SMOOTH_K   = 3              # au moins K prédictions "1" dans la fenêtre pour sortir 1
# ===================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT/"models"/"XGB_mit_binary.pkl"
PTB_BEATS_PATH = PROJECT_ROOT/"data"/"processed"/"ptb_beats_all_with_healthy_label.csv"
OUT_PATH = PROJECT_ROOT/"data"/"processed"/"ptb_beats_xgb_minimal_labels_only.csv"

def pick_threshold(y_true, p_hat, target_precision=0.90):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1  = -1.0
    # 1) d'abord, garder les seuils qui respectent la précision cible
    for t in thr_grid:
        pred = (p_hat >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if pr >= target_precision and f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
            best_triplet = (pr, rc, f1)
    # 2) fallback : max F1 si aucun seuil n’atteint la précision cible
    if best_f1 < 0:
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

def main():
    # ------------------- 1) Modèle -------------------
    print(f"Chargement du modèle XGBoost : {MODEL_PATH}")
    xgb_clf = joblib.load(MODEL_PATH)

    # ------------------- 2) Données -------------------
    print(f"\nLecture des battements PTB : {PTB_BEATS_PATH}")
    beats = pd.read_csv(PTB_BEATS_PATH)
    print("Nombre total de lignes :", len(beats))

    # (Optionnel) ne garder que MLII si présent
    if "lead" in beats.columns:
        n0 = len(beats)
        mask_mlii = beats["lead"].astype(str).str.upper().eq("MLII")
        if mask_mlii.any():
            beats = beats[mask_mlii].copy()
            print(f"Filtre MLII : {n0} → {len(beats)}")

    # features b0..b186 en float32, NaN→0
    feat_cols = sorted([c for c in beats.columns if c.startswith("b")],
                       key=lambda x: int(x[1:]))
    X = (beats[feat_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0.0)
         .astype(np.float32)
         .values)

    if "healthy_label" not in beats.columns:
        raise ValueError("La colonne 'healthy_label' est requise (1=sain, 0=malade).")

    y_true = (1 - beats["healthy_label"].astype(int).values)  # 1=malade, 0=sain

    # ------------------- 3) Probas + calibration -------------------
    print("\nPrédiction des probabilités...")
    p_hat = xgb_clf.predict_proba(X)[:, 1]

    if USE_ISOTONIC_CAL:
        if "patient" in beats.columns:
            # split par patients pour éviter la fuite
            pats = beats["patient"].astype(str).unique()
            tr_p, cal_p = train_test_split(pats, test_size=0.2, random_state=42)
            cal_mask = beats["patient"].astype(str).isin(cal_p)
        else:
            # fallback : split simple par lignes (moins propre, mais mieux que rien)
            _, cal_mask = train_test_split(
                np.arange(len(beats)), test_size=0.2, random_state=42
            )
            cal_mask = beats.index.isin(cal_mask)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_hat[cal_mask], y_true[cal_mask])
        p_hat = iso.transform(p_hat)
        print("Calibration isotonic : OK")

    # ------------------- 4) Seuil anti-FP -------------------
    thr = pick_threshold(y_true, p_hat, target_precision=TARGET_PRECISION)
    y_pred = (p_hat >= thr).astype(int)

    # (Optionnel) lissage patient k-sur-m pour réduire les FP isolés
    if USE_SMOOTHING and "patient" in beats.columns:
        def smooth_probs(s):
            # 1 si au moins K proba>=thr dans une fenêtre de M
            return (s.rolling(SMOOTH_WIN, min_periods=1)
                      .apply(lambda w: (w >= thr).sum() >= SMOOTH_K)
                      .astype(int))
        y_pred_sm = (beats.groupby("patient")["xgb_proba_tmp"]
                         .apply(lambda s: smooth_probs(s))
                         if "xgb_proba_tmp" in beats.columns else
                     beats.assign(xgb_proba_tmp=p_hat)
                          .groupby("patient")["xgb_proba_tmp"]
                          .apply(lambda s: smooth_probs(s)))
        # réaligne et remplace
        y_pred = y_pred_sm.reindex(beats.index).to_numpy()
        print(f"Lissage {SMOOTH_K}-sur-{SMOOTH_WIN} appliqué.")

    # ------------------- 5) Sortie minimaliste -------------------
    beats["gt_malade"] = y_true
    beats["xgb_proba_malade"] = p_hat
    beats["xgb_pred_malade"] = y_pred
    beats["correct"] = (y_pred == y_true).astype(int)
    beats["threshold_used"] = thr

    cols_final = [
        "patient","record","idx_local",
        "healthy_label","gt_malade",
        "xgb_proba_malade","xgb_pred_malade","correct","threshold_used"
    ]
    cols_final = [c for c in cols_final if c in beats.columns]
    beats_min = beats[cols_final].copy()
    beats_min.to_csv(OUT_PATH, index=False)

    # petit résumé
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print("\n=== Résumé ===")
    print(f"precision={pr:.3f} | recall={rc:.3f} | f1={f1:.3f} | seuil={thr:.2f}")
    print("Fichier sauvegardé :", OUT_PATH)

if __name__ == "__main__":
    main()
