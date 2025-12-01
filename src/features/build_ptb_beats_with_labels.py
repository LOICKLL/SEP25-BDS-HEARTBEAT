# build_ptb_beats_with_labels.py

import numpy as np
import pandas as pd
from pathlib import Path

import wfdb
from wfdb import processing


# ======================================================================
# Paramètres généraux
# ======================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent 
# Dossier racine de la base PTB (adapté à ta structure)
PTB_ROOT = PROJECT_ROOT/"data"/"raw"/ "ptb-diagnostic-ecg-database-1.0.0" / "ptb-diagnostic-ecg-database-1.0.0"

# Fichier de métadonnées nettoyé
PTB_META_PATH = PROJECT_ROOT/"data"/"processed"/ "ptb_metadata_all.cleaned.csv"

# Fichier de sortie des battements
OUT_BEATS_PATH = PROJECT_ROOT/"data"/"processed"/"ptb_beats_all_with_healthy_label.csv"

# Paramètres des segments
TARGET_FS = 360         # fréquence cible (comme MIT)
WIN_SIZE = 187          # nb de points par battement
LEAD_NAME = "ii"        # dérivation à extraire (équivalent MLII)


# ======================================================================
# 1) Construction du CSV de métadonnées à partir des .hea
# ======================================================================

def build_ptb_metadata_from_headers():
    """
    Construit ptb_metadata_all.cleaned.csv en parcourant tous les fichiers
    .hea de la base PTB et en parsant les 'comments' WFDB.

    Colonnes minimum garanties :
      - patient_dir (ex: 'patient001')
      - record_name (ex: 's0010_re')
      - reason_for_admission (si trouvée dans les commentaires)

    + plein d'autres colonnes auto-générées à partir des 'clé: valeur'.
    """


    print("\n=== Construction de ptb_metadata_all.cleaned.csv à partir des .hea ===")
    rows = []
    n_records = 0

    for patient_dir in sorted(PTB_ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith("patient"):
            continue

        patient_id = patient_dir.name  # ex: 'patient001'

        for hea_path in sorted(patient_dir.glob("*.hea")):
            record_name = hea_path.stem  # ex: 's0010_re'
            rec_basepath = hea_path.with_suffix("")  # sans extension

            try:
                header = wfdb.rdheader(str(rec_basepath))
                comments = header.comments or []
            except Exception as e:
                print(f"!! Erreur lecture header pour {patient_id}/{record_name} : {e}")
                comments = []

            row = {
                "patient_dir": patient_id,
                "record_name": record_name,
            }

            # Parsing des commentaires 'clé: valeur'
            last_key = None
            for com in comments:
                line = com.strip()
                # certains headers ont encore un '#'
                if line.startswith("#"):
                    line = line[1:].strip()

                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()

                    # normalisation du nom de colonne
                    key_norm = (
                        key.lower()
                        .replace(" ", "_")
                        .replace("/", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("-", "_")
                    )

                    # ex: "Reason for admission" -> "reason_for_admission"
                    if key_norm in row and row[key_norm]:
                        # si la clé apparaît plusieurs fois, on concatène
                        row[key_norm] = row[key_norm] + " | " + val
                    else:
                        row[key_norm] = val

                    last_key = key_norm
                else:
                    # ligne sans ":", on la rattache au dernier champ si possible
                    if last_key is not None:
                        if row.get(last_key):
                            row[last_key] = row[last_key] + " " + line
                        else:
                            row[last_key] = line
                    else:
                        # sinon, on met ça dans un champ fourre-tout
                        if "free_text" in row:
                            row["free_text"] = row["free_text"] + " " + line
                        else:
                            row["free_text"] = line

            rows.append(row)
            n_records += 1

    meta_df = pd.DataFrame(rows)
    print(f"Nombre total d'enregistrements trouvés : {n_records}")
    print("Aperçu des métadonnées construites :")
    print(meta_df.head())

    # Sauvegarde du CSV
    PTB_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(PTB_META_PATH, index=False)
    print("Fichier de métadonnées nettoyé créé :", PTB_META_PATH)


# ======================================================================
# Fonction : extraction des battements pour UN enregistrement
# ======================================================================

def extract_beats_for_record(record_basepath: Path,
                             patient_id: str,
                             record_name: str,
                             beats_list: list) -> None:
    """
    Lit un enregistrement WFDB (basepath sans extension),
    détecte les R-peaks sur la dérivation LEAD_NAME,
    extrait des fenêtres de WIN_SIZE points autour de R,
    et ajoute les battements dans beats_list (liste de dict).
    """

    rec_basename = f"{patient_id}/{record_name}"
    try:
        # wfdb.rdsamp retourne (signal, champs)
        sig, fields = wfdb.rdsamp(str(record_basepath))
    except Exception as e:
        print(f"!! Erreur lecture WFDB pour {rec_basename}: {e}")
        return

    # sig : array (n_samples, n_leads)
    sig = np.asarray(sig, dtype=float)
    sig_names = [s.lower() for s in fields["sig_name"]]

    if LEAD_NAME not in sig_names:
        print(f"   -> Lead '{LEAD_NAME}' absent pour {rec_basename}, on saute.")
        return

    idx_lead = sig_names.index(LEAD_NAME)
    sig_lead = sig[:, idx_lead]

    # ------------------------------------------------------------------
    # Resampling éventuel vers 360 Hz
    # ------------------------------------------------------------------
    try:
        fs_orig = float(fields["fs"])
    except Exception as e:
        print(f"!! Erreur lecture fs pour {rec_basename}: {e}")
        return

    if fs_orig != TARGET_FS:
        try:
            sig_res, _ = processing.resample_sig(sig_lead, fs_orig, TARGET_FS)
            fs_res = TARGET_FS
        except Exception as e:
            print(f"!! Erreur resampling pour {rec_basename}: {e}")
            return
    else:
        sig_res = sig_lead
        fs_res = fs_orig

    sig_res = np.asarray(sig_res, dtype=float).flatten()
    fs_res = float(fs_res)

    # ------------------------------------------------------------------
    # Détection des R-peaks
    # ------------------------------------------------------------------
    try:
        qrs_inds = processing.xqrs_detect(sig_res, fs=fs_res)
    except Exception as e:
        print(f"!! Erreur détection R-peaks pour {record_basepath}: {e}")
        return

    if qrs_inds is None or len(qrs_inds) == 0:
        print(f"   -> Aucun R-peak détecté pour {rec_basename}")
        return

    half_win = WIN_SIZE // 2
    n_samples = len(sig_res)

    idx_local = 0
    n_kept = 0  # pour debug

    for r_idx in qrs_inds:
        r_idx = int(r_idx)

        # Fenêtre de 187 points centrée sur R
        start = r_idx - half_win
        end = r_idx + half_win + 1  # +1 pour avoir 187 points

        if start < 0 or end > n_samples:
            continue

        beat = sig_res[start:end]
        if len(beat) != WIN_SIZE:
            continue

        row = {
            "patient": patient_id,
            "record": record_name,
            "idx_local": idx_local,
            "lead": LEAD_NAME,
            "fs": TARGET_FS,
            "n_samples": WIN_SIZE,
            "r_index_360": r_idx,
        }

        for i in range(WIN_SIZE):
            row[f"b{i}"] = float(beat[i])

        beats_list.append(row)
        idx_local += 1
        n_kept += 1

    print(f"   -> Battements conservés pour {rec_basename} : {n_kept}")


# ======================================================================
# Programme principal
# ======================================================================

def main():
    print("Dossier PTB_ROOT :", PTB_ROOT)
    print("Fichier métadonnées :", PTB_META_PATH)
    print("Fichier de sortie :", OUT_BEATS_PATH)

    if not PTB_ROOT.exists():
        print("!! PTB_ROOT n'existe pas, vérifie le chemin.")
        return
    if not PTB_META_PATH.exists():
        print("!! PTB_META_PATH n'existe pas, vérifie le chemin.")
        return

    # ------------------------------------------------------------------
    # Lecture des métadonnées + construction du label healthy
    # ------------------------------------------------------------------
    meta = pd.read_csv(PTB_META_PATH)

    # Normalisation patient / record
    raw_patient = meta["patient_dir"].astype(str).str.strip()
    # Si c'est juste "227", "15", ... on construit "patient227", "patient015", etc.
    meta["patient"] = np.where(
        raw_patient.str.startswith("patient"),
        raw_patient,
        "patient" + raw_patient.str.zfill(3),
    )
    meta["record"] = meta["record_name"].astype(str).str.strip()

    reason = meta["reason_for_admission"].fillna("").str.lower()

    # 1 = Healthy control, 0 = tout le reste
    meta["healthy_label"] = np.where(
        reason.str.contains("healthy control"),
        1,
        0,
    )

    meta_small = meta[["patient", "record", "healthy_label"]].drop_duplicates()

    print("\nAperçu meta_small :")
    print(meta_small.head())

    print("\nRépartition healthy_label (métadonnées) :")
    print(meta_small["healthy_label"].value_counts(dropna=False))

    # ------------------------------------------------------------------
    # Parcours de tous les dossiers patientXXX
    # ------------------------------------------------------------------
    beats_list = []
    n_pat_dirs = 0

    for patient_dir in sorted(PTB_ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith("patient"):
            continue

        n_pat_dirs += 1
        patient_id = patient_dir.name  # ex: "patient001"

        for dat_path in sorted(patient_dir.glob("*.dat")):
            record_name = dat_path.stem  # ex: "s0010_re"
            rec_basepath = dat_path.with_suffix("")  # sans extension

            print(f"-> Lecture {patient_id}/{record_name}")

            extract_beats_for_record(
                record_basepath=rec_basepath,
                patient_id=patient_id,
                record_name=record_name,
                beats_list=beats_list,
            )

    print(f"\nNombre de dossiers patient trouvés : {n_pat_dirs}")
    print(f"Total de battements extraits (sans label) : {len(beats_list)}")

    if not beats_list:
        print("Aucun battement extrait... vérifier les chemins / librairies.")
        return

    beats_df = pd.DataFrame(beats_list)

    print("\nAperçu des couples patient/record dans beats_df :")
    print(beats_df[["patient", "record"]].drop_duplicates().head())

    # ------------------------------------------------------------------
    # Jointure avec les métadonnées pour récupérer healthy_label
    # ------------------------------------------------------------------
    beats_df = beats_df.merge(
        meta_small,
        on=["patient", "record"],
        how="left",
    )

    print("\nRépartition healthy_label dans les battements :")
    print(beats_df["healthy_label"].value_counts(dropna=False))

    n_nan = beats_df["healthy_label"].isna().sum()
    if n_nan > 0:
        print(f"\nATTENTION : {n_nan} battements sans label (NaN).")
        print("Exemples de couples patient/record sans méta :")
        print(
            beats_df.loc[beats_df["healthy_label"].isna(), ["patient", "record"]]
            .drop_duplicates()
            .head()
        )

    print("\nAperçu du DataFrame final :")
    print(beats_df.head())

    beats_df.to_csv(OUT_BEATS_PATH, index=False)
    print("\nFichier sauvegardé dans :", OUT_BEATS_PATH)


if __name__ == "__main__":
    main()
