#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# 1) Dossiers du projet
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2) Paramètres de la fenêtre
WIN_SIZE = 187
HALF_WIN = WIN_SIZE // 2  # 93 de part et d'autre

# 3) Mapping AAMI -> label
symbol_to_label = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,     # N
    "A": 1, "a": 1, "J": 1, "S": 1,            # S
    "V": 2, "E": 2,                             # V
    "F": 3,                                     # F
    "/": 4, "f": 4, "Q": 4, "?": 4,             # Q
}
KEEP_UNKNOWN_IN_FULLMETA = True  # conserve les symboles inconnus avec label=-1 dans fullmeta

def main():
    records = sorted({p.stem for p in DATA_DIR.glob("*.dat")})
    print("Enregistrements trouvés :", records)

    rows_full = []  # fullmeta (toutes les métadonnées)
    rows_min  = []  # minimal (pour MLII, sans -1)
    unknown_sym_counter = Counter()

    for rec in records:
        print(f"\n===== Record {rec} =====")
        record = wfdb.rdrecord(str(DATA_DIR / rec))
        sig = record.p_signal
        fs = float(record.fs)
        sig_names = list(record.sig_name)
        n_samples, n_leads = sig.shape

        duree_sec = n_samples / fs if fs > 0 else np.nan
        duree_min = duree_sec / 60 if pd.notna(duree_sec) else np.nan
        nb_signaux = int(n_leads)
        nb_echantillons = int(n_samples)

        comments = record.comments if record.comments is not None else []
        comment_full = " ".join(comments)

        # Age / sexe (approximatif selon headers)
        age = np.nan
        sexe = None
        if comment_full:
            parts = comment_full.split()
            try:
                age = float(parts[0])
            except Exception:
                age = np.nan
            if len(parts) > 1:
                sexe = parts[1]

        leads_str = ",".join(sig_names)

        record_x = f"x_{rec}"
        record_num = int(rec)
        record_y = record_x

        meds = ""
        if "|" in comment_full:
            pipe_parts = [s.strip() for s in comment_full.split("|")]
            if len(pipe_parts) >= 3:
                meds = ", ".join(pipe_parts[1:-1])
        notes = comment_full

        ann = wfdb.rdann(str(DATA_DIR / rec), "atr")
        r_locs = ann.sample
        symbols = ann.symbol
        nb_beats = len(r_locs)
        bpm_moyen = 60 * nb_beats / duree_sec if duree_sec and duree_sec > 0 else np.nan

        for r_index, sym in zip(r_locs, symbols):
            # label = mapping ou -1 si inconnu
            label = symbol_to_label.get(sym, -1)
            if label == -1:
                unknown_sym_counter[sym] += 1

            # Fenêtre autour du battement
            start = r_index - HALF_WIN
            end   = r_index + HALF_WIN + 1
            if start < 0 or end > n_samples:
                continue

            t_sec = float(r_index / fs) if fs > 0 else np.nan
            sample = int(r_index)

            for lead_idx in range(n_leads):
                lead_name = sig_names[lead_idx]
                segment = sig[start:end, lead_idx]
                if len(segment) != WIN_SIZE:
                    continue

                # LIGNE "FULLMETA": TOUTES LES METADONNEES (inclut -1 si KEEP_UNKNOWN_IN_FULLMETA) 
                if (label != -1) or (label == -1 and KEEP_UNKNOWN_IN_FULLMETA):
                    row_full = {
                        "record_x": record_x,
                        "record_num": record_num,
                        "sample": sample,
                        "t_sec": t_sec,
                        "lead": lead_name,
                        "symbol": sym,
                        "label": int(label),
                        "record_y": record_y,
                        "age": float(age) if pd.notna(age) else np.nan,
                        "sexe": sexe,
                        "leads": leads_str,
                        "fs": fs,
                        "nb_signaux": nb_signaux,
                        "nb_echantillons": nb_echantillons,
                        "duree_sec": duree_sec,
                        "duree_min": duree_min,
                        "nb_beats": nb_beats,
                        "bpm_moyen": bpm_moyen,
                        "medications": meds,
                        "notes": notes,
                    }
                    for i in range(WIN_SIZE):
                        row_full[str(i)] = float(segment[i])
                    rows_full.append(row_full)

                #  LIGNE "MINIMALE": seulement pour MLII et uniquement si label != -1 
                if (lead_name == "MLII") and (label != -1):
                    row_min = {
                        "record_x": record_x,
                        "sample": sample,
                        "t_sec": t_sec,
                        "lead": lead_name,
                        "symbol": sym,
                        "label": int(label),
                    }
                    for i in range(WIN_SIZE):
                        row_min[str(i)] = float(segment[i])
                    rows_min.append(row_min)

    # ===== DataFrames =====
    df_full = pd.DataFrame(rows_full)
    df_mlii = pd.DataFrame(rows_min)

    # Colonnes signal à la fin (0..186)
    cols_points = [str(i) for i in range(WIN_SIZE)]

    # --- FULLMETA : toutes les métadonnées d’abord, puis 0..186
    cols_full_meta = [
        "record_x", "record_num", "sample", "t_sec", "lead", "symbol", "label",
        "record_y", "age", "sexe", "leads", "fs",
        "nb_signaux", "nb_echantillons", "duree_sec", "duree_min",
        "nb_beats", "bpm_moyen", "medications", "notes",
    ]
    if not df_full.empty:
        # garde uniquement les colonnes présentes (au cas où certaines metas manquent)
        cols_full_meta_present = [c for c in cols_full_meta if c in df_full.columns]
        cols_points_present = [c for c in cols_points if c in df_full.columns]
        df_full = df_full[cols_full_meta_present + cols_points_present]

    fullmeta_path = OUT_DIR / "mitbih_187pts_fullmeta.csv"
    df_full.to_csv(fullmeta_path, index=False)
    print("\nCSV fullmeta créé :", fullmeta_path, "->", df_full.shape)

    # --- MLII (minimal, sans -1) : record_x, sample, t_sec, lead, symbol, label, puis 0..186
    cols_min_meta = ["record_x", "sample", "t_sec", "lead", "symbol", "label"]
    if not df_mlii.empty:
        cols_min_meta_present = [c for c in cols_min_meta if c in df_mlii.columns]
        cols_points_present = [c for c in cols_points if c in df_mlii.columns]
        df_mlii = df_mlii[cols_min_meta_present + cols_points_present]

    mlii_path = OUT_DIR / "mitbih_187pts_MLII.csv"
    df_mlii.to_csv(mlii_path, index=False)
    print("CSV MLII (minimal, sans -1) créé :", mlii_path, "->", df_mlii.shape)


    #  Récap des symboles inconnus
    if unknown_sym_counter:
        total_unknown = sum(unknown_sym_counter.values())
        print("\n[INFO] Symboles non mappés (inclus dans fullmeta avec label=-1, exclus de MLII) :")
        for k, v in unknown_sym_counter.most_common():
            print(f"  {k!r}: {v}")
        print(f"Total non mappés : {total_unknown}")

if __name__ == "__main__":
    main()
