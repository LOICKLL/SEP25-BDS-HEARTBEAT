import pandas as pd
from pathlib import Path
import wfdb


# 1) Dossiers du projet
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent  # src/data -> racine projet

PTB_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptb-diagnostic-ecg-database-1.0.0"
    / "ptb-diagnostic-ecg-database-1.0.0"
)

PTB_META_PATH = PROJECT_ROOT / "data" / "processed" / "ptb_metadata_all.cleaned.csv"


def build_ptb_metadata_from_headers():
    """
    Parcourt tous les fichiers .hea de la base PTB et construit un CSV
    'ptb_metadata_all.cleaned.csv' avec au moins :

      - patient_dir  (ex: 'patient001')
      - record_name  (ex: 's0010_re')
      - reason_for_admission (si trouvée)
      - + toutes les autres paires 'clé: valeur' trouvées dans comments.
    """

    if not PTB_ROOT.exists():
        raise FileNotFoundError(f"PTB_ROOT introuvable : {PTB_ROOT}")

    rows = []
    n_records = 0

    print("=== Construction de ptb_metadata_all.cleaned.csv à partir des .hea ===")
    print("PTB_ROOT :", PTB_ROOT)

    for patient_dir in sorted(PTB_ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith("patient"):
            continue

        patient_id = patient_dir.name  # ex: 'patient001'

        for hea_path in sorted(patient_dir.glob("*.hea")):
            record_name = hea_path.stem  # ex: 's0010_re'
            rec_basepath = hea_path.with_suffix("")  # chemin sans extension

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

            # --- Parsing des commentaires 'clé: valeur' ---
            last_key = None
            for com in comments:
                line = com.strip()

                # Les lignes commencent parfois par '#'
                if line.startswith("#"):
                    line = line[1:].strip()

                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()

                    # on normalise le nom de la clé -> snake_case
                    key_norm = (
                        key.lower()
                        .replace(" ", "_")
                        .replace("/", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("-", "_")
                    )

                    # si la même clé apparaît plusieurs fois, on concatène
                    if key_norm in row and row[key_norm]:
                        row[key_norm] = row[key_norm] + " | " + val
                    else:
                        row[key_norm] = val

                    last_key = key_norm
                else:
                    # ligne sans ":", on la rattache au champ précédent si possible
                    if last_key is not None:
                        if row.get(last_key):
                            row[last_key] = row[last_key] + " " + line
                        else:
                            row[last_key] = line
                    else:
                        # sinon, on met ça dans un champ fourre-tout
                            # 'free_text'
                        if "free_text" in row:
                            row["free_text"] = row["free_text"] + " " + line
                        else:
                            row["free_text"] = line

            rows.append(row)
            n_records += 1

    meta_df = pd.DataFrame(rows)
    print(f"Nombre total d'enregistrements trouvés : {n_records}")
    print("\nAperçu du DataFrame de métadonnées :")
    print(meta_df.head())


    # Forcer  les colonnes 
    
    desired_cols = [
        "patient_dir",
        "record_name",
        "age",
        "sex",
        "ecg_date",
        "reason_for_admission",
        "acute_infarction_localization",
        "former_infarction_localization",
        "additional_diagnoses",
        "smoker",
        "number_of_coronary_vessels_involved",
        "infarction_date_acute",
        "previous_infarction_1_date",
        "previous_infarction_2_date",
        "catheterization_date",
        "ventriculography",
        "chest_x_ray",
        "peripheral_blood_pressure_syst_diast",
        "pulmonary_artery_pressure_at_rest_syst_diast",
        "pulmonary_artery_pressure_at_rest_mean",
        "pulmonary_capillary_wedge_pressure_at_rest",
        "cardiac_output_at_rest",
        "cardiac_index_at_rest",
        "stroke_volume_index_at_rest",
        "pulmonary_capillary_wedge_pressure_load",
        "cardiac_output_load",
        "cardiac_index_load",
        "stroke_volume_index_load",
        "aorta_at_rest_syst_diast",
        "aorta_at_rest_mean",
        "left_ventricular_enddiastolic_pressure",
        "left_coronary_artery_stenoses_riva",
        "left_coronary_artery_stenoses_rcx",
        "right_coronary_artery_stenoses_rca",
        "echocardiography",
        "infarction_date",
        "admission_date",
        "medication_pre_admission",
        "start_lysis_therapy_hh.mm",
        "lytic_agent",
        "dosage_lytic_agent",
        "additional_medication",
        "in_hospital_medication",
        "medication_after_discharge",
    ]

    # Ajouter les colonnes manquantes, remplies avec NaN
    for col in desired_cols:
        if col not in meta_df.columns:
            meta_df[col] = pd.NA

    # On remet les colonnes dans l'ordre souhaité (en gardant les autres à la fin)
    other_cols = [c for c in meta_df.columns if c not in desired_cols]
    meta_df = meta_df[desired_cols + other_cols]


    PTB_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(PTB_META_PATH, index=False)
    print("\nFichier de métadonnées nettoyé créé :", PTB_META_PATH)


if __name__ == "__main__":
    build_ptb_metadata_from_headers()
