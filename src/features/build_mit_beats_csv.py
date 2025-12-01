import wfdb
import numpy as np
import pandas as pd
from pathlib import Path

# 1) Dossiers du projet

# Dossier où se trouve ce script
THIS_DIR = Path(__file__).resolve().parent
# Racine du projet
PROJECT_ROOT = THIS_DIR.parent.parent 
# Dossier qui contient les fichiers .dat/.hea/.atr
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
# Dossier de sortie pour les CSV
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# 2) Paramètres de la fenêtre
WIN_SIZE = 187            # nombre de points par battement
HALF_WIN = WIN_SIZE // 2  # 93 points avant / après



# 3) Mapping label ->
symbol_to_label = {
    # N : battements normaux
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S : battements supraventriculaires
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V : battements ventriculaires
    "V": 2, "E": 2,
    # F : fusion
    "F": 3,
    # Q : rythmes "unknown"
    "/": 4, "f": 4, "Q": 4,
}



def main():
    # 4) Liste des enregistrements
    records = sorted({p.stem for p in DATA_DIR.glob("*.dat")})
    print("Enregistrements trouvés :", records)
    rows = []

    # 5) Boucle sur les enregistrements
    for rec in records:
        print(f"\n===== Record {rec} =====")

        # Lecture du signal
        record = wfdb.rdrecord(str(DATA_DIR / rec))
        sig = record.p_signal
        fs = record.fs
        sig_names = record.sig_name
        n_samples, n_leads = sig.shape
        duree_sec = n_samples / fs
        duree_min = duree_sec / 60
        nb_signaux = n_leads
        nb_echantillons = n_samples


        # Récupération des metadata depuis le .hea
        comments = record.comments if record.comments is not None else []
        comment_full = " ".join(comments)


        # Age + sexe 
        age = np.nan
        sexe = None

        if comment_full:
            parts_space = comment_full.split()
            try:
                age = float(parts_space[0])
            except Exception:
                age = np.nan
            if len(parts_space) > 1:
                sexe = parts_space[1]

        # noms des leads (ex. "MLII,V1")
        leads_str = ",".join(sig_names)

        # Identifiants type Kaggle
        record_x = f"x_{rec}"
        record_num = int(rec)
        record_y = record_x

        # Médicaments 
        meds = ""
        if "|" in comment_full:
            pipe_parts = [s.strip() for s in comment_full.split("|")]
            if len(pipe_parts) >= 3:
                meds = ", ".join(pipe_parts[1:-1])
        notes = comment_full

        # Annotations (positions des battements)
        ann = wfdb.rdann(str(DATA_DIR / rec), "atr")
        r_locs = ann.sample
        symbols = ann.symbol
        nb_beats = len(r_locs)
        bpm_moyen = 60 * nb_beats / duree_sec if duree_sec > 0 else np.nan


        # 6) Boucle sur chaque battement
        for idx_local, (r_index, sym) in enumerate(zip(r_locs, symbols)):

            # On garde seulement les symboles mappés dans nos 5 classes
            if sym not in symbol_to_label:
                continue
            label = symbol_to_label[sym]

            # Fenêtre autour du battement
            start = r_index - HALF_WIN
            end = r_index + HALF_WIN + 1  

            # On saute les battements trop proches du début/fin
            if start < 0 or end > n_samples:
                continue

            t_sec = r_index / fs
            sample = int(r_index)

            # Une ligne par lead (MLII et V1)
            for lead_idx in range(n_leads):
                lead_name = sig_names[lead_idx]
                segment = sig[start:end, lead_idx]

                if len(segment) != WIN_SIZE:
                    continue

                # Colonnes "meta"
                row = {
                    "record_x": record_x,
                    "record_num": record_num,
                    "sample": sample,
                    "t_sec": float(t_sec),
                    "lead": lead_name,
                    "symbol": sym,
                    "label": int(label),
                    "record_y": record_y,
                    "age": float(age),
                    "sexe": sexe,
                    "leads": leads_str,
                    "fs": float(fs),
                    "nb_signaux": int(nb_signaux),
                    "nb_echantillons": int(nb_echantillons),
                    "duree_sec": float(duree_sec),
                    "duree_min": float(duree_min),
                    "nb_beats": int(nb_beats),
                    "bpm_moyen": float(bpm_moyen),
                    "medications": meds,
                    "notes": notes,
                }

                # Colonnes 0..186 (les 187 points du battement)
                for i in range(WIN_SIZE):
                    row[str(i)] = float(segment[i])

                rows.append(row)

    # 7) DataFrame final et sauvegarde fullmeta
    df = pd.DataFrame(rows)

    # Ordre des colonnes 
    cols_meta_debut = ["record_x", "record_num", "sample", "t_sec", "lead","symbol", "label", "record_y", "age", "sexe", "leads", "fs",
        "nb_signaux", "nb_echantillons", "duree_sec", "duree_min",
        "nb_beats", "bpm_moyen", "medications", "notes",
    ]
    cols_points = [str(i) for i in range(WIN_SIZE)]

    df = df[cols_meta_debut + cols_points ]

    # CSV complet
    fullmeta_path = OUT_DIR / "mitbih_187pts_fullmeta.csv"
    df.to_csv(fullmeta_path, index=False)

    print("\nCSV complet créé :", fullmeta_path)
    print("Shape du DataFrame complet :", df.shape)
    print(df.head())

    # 8) Filtrage uniquement du lead MLII et nouveau csv

    df_mlii = df.loc[df["lead"] == "MLII"].copy()
    mlii_path = OUT_DIR / "mitbih_187pts_MLII.csv"
    df_mlii.to_csv(mlii_path, index=False)

    print("\nCSV MLII créé :", mlii_path)
    print("Shape du DataFrame MLII :", df_mlii.shape)
    print(df_mlii.head())

if __name__ == "__main__":
    main()
