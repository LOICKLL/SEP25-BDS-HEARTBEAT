import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier

from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

import joblib


########################################################################################################
# DOSSIERS DU PROJET
########################################################################################################

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "Modelisation_XGBoost"
FIG_DIR.mkdir(parents=True, exist_ok=True)

########################################################################################################
# LECTURE DES FICHIERS
########################################################################################################

train_path = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
test_path  = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"


train=pd.read_csv(train_path, low_memory=False)
test=pd.read_csv(test_path, low_memory=False)


X_train= train.iloc[:, 7:194]
y_train=train['binary']


X_test = test.iloc[:, 7:194]
y_test=test['binary']


########################################################################################################
## CALCUL PRELIMINAIRE ##
########################################################################################################


groups_train = train["record_x"] # Important pour que le CV ne choississe pas deux fois le même patient!!

n_positive = np.sum(y_train == 1)
n_negative = np.sum(y_train == 0)

scale_pos_weight = n_negative / n_positive
print("scale_pos_weight =", scale_pos_weight)


#########################################################################################################
## FONCTION DE SAUVEGARDE DES MATRICES DE CONFUSIONS ET CLASSIFICATION REPORT ##
#########################################################################################################

out_path = FIG_DIR


def plot_confusion_matrix_norm(y_true, y_pred, filename, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    full_path = out_path / filename

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,
        cbar=False,
        square=True,
    )
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Matrice de confusion sauvegardée : {full_path}")



def save_classification_report(y_true, y_pred, filename, out_path):
    """
    Sauvegarde un classification report sklearn dans un fichier texte.
    """
    report = classification_report(y_true, y_pred)
    
    full_path = out_path / filename
    
    with open(full_path, "w") as f:
        f.write(report)
    
    print(f"Classification report sauvegardé : {full_path}")


def save_cv_score(cv_score, filename, out_path):
    """
    Sauvegarde uniquement le score de cross-validation (ex: recall CV) dans un fichier texte.
    """
    full_path = out_path / filename

    with open(full_path, "w") as f:
        f.write("### SCORE CROSS-VALIDATION ###\n")
        f.write(f"CV score : {cv_score:.6f}\n")

    print(f"Score CV sauvegardé : {full_path}")


#########################################################################################################
## BASELINE ##
#########################################################################################################

print("""
#################################################################################################
## BASELINE ##
#################################################################################################
""")

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)


save_classification_report(y_test, y_pred, "report_XGB_baseline.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred, "cm_XGB_baseline.png", out_path)


#########################################################################################################
## GRIDSEARCH CV ##
#########################################################################################################

print("""
#################################################################################################
## GRIDSEARCH CV ##
#################################################################################################
""")



model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic", 
    tree_method="hist",          
    n_jobs=-1,                   
    random_state=42
)


param_grid = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.1, 0.05],
    "max_depth": [4, 6, 8],
}

cv = GroupKFold(n_splits=3)


grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='recall',  
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid.fit(X_train, y_train, groups=groups_train)

print("Meilleurs hyperparamètres :", grid.best_params_)
print("Score CV:", grid.best_score_)

best_model = grid.best_estimator_


y_pred = best_model.predict(X_test)


save_cv_score(grid.best_score_,"cv_score_XGB_gridsearchCV.txt", out_path)

save_classification_report(y_test, y_pred, "report_XGB_gridsearchCV.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred, "cm_XGB_gridsearchCV.png", out_path)


#################################################################################################
## SMOTE ##
#################################################################################################

print("""
#################################################################################################
## SMOTE ##
#################################################################################################
""")


xgb_smote_pipeline = Pipeline(steps=[
    ('smote', SMOTE(
        sampling_strategy='not majority',  
        k_neighbors=3,                     
        random_state=42
    )),
    ('xgb', XGBClassifier(
        objective="binary:logistic",
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        
    ))
])


param_grid_smote = {
    "xgb__n_estimators": [100, 200, 400],
    "xgb__learning_rate": [0.1, 0.05],
    "xgb__max_depth": [4, 6, 8],
}

cv = GroupKFold(n_splits=3)


grid_smote = GridSearchCV(
    estimator=xgb_smote_pipeline,
    param_grid=param_grid_smote,
    scoring='recall',  
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_smote.fit(X_train, y_train, groups=groups_train)

print("Meilleurs hyperparamètres :", grid_smote.best_params_)
print("Score CV:", grid_smote.best_score_)

best_model_smote = grid_smote.best_estimator_


y_pred = best_model_smote.predict(X_test)



save_cv_score(grid_smote.best_score_,"cv_score_XGB_SMOTE.txt", out_path)

save_classification_report(y_test, y_pred, "report_XGB_SMOTE.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred, "cm_XGB_SMOTE.png", out_path)



#################################################################################################
## ADASYN ##
#################################################################################################

print("""
#################################################################################################
## ADASYN ##
#################################################################################################
""")

xgb_adasyn_pipeline = Pipeline(steps=[
    ('adasyn', ADASYN(
        sampling_strategy='not majority',  # augmente toutes les classes minoritaires
        n_neighbors=3,                     # voisinage pour générer les samples
        random_state=42
    )),
    ('xgb', XGBClassifier(
        objective="binary:logistic",
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    ))
])

param_grid_adasyn = {
    "xgb__n_estimators": [100, 200, 400],
    "xgb__learning_rate": [0.1, 0.05],
    "xgb__max_depth": [4, 6, 8],
}

cv = GroupKFold(n_splits=3)

grid_adasyn = GridSearchCV(
    estimator=xgb_adasyn_pipeline,
    param_grid=param_grid_adasyn,
    scoring='recall',
    cv=cv,
    n_jobs=1,               
    verbose=1
)

grid_adasyn.fit(X_train, y_train, groups=groups_train)

print("Meilleurs hyperparamètres :", grid_adasyn.best_params_)
print("Score CV:", grid_adasyn.best_score_)

best_model_adasyn = grid_adasyn.best_estimator_

y_pred = best_model_adasyn.predict(X_test)


save_cv_score(grid_adasyn.best_score_,"cv_score_XGB_adasyn.txt", out_path)

save_classification_report(y_test, y_pred, "report_XGB_adasyn.txt", out_path)


plot_confusion_matrix_norm(y_test, y_pred, "cm_XGB_adasyn.png", out_path)


## PRÉDICTION AVEC SEUIL 0.3


print("""
#################################################################################################
## PREDICTION AVEC SEUIL 0.3 ##
#################################################################################################
""")

# Probabilité d'appartenir à la classe 1
y_proba = best_model_adasyn.predict_proba(X_test)[:, 1]

# Application du seuil 0.3
threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)



save_classification_report(y_test, y_pred_custom, "report_XGB_seuil.txt", out_path)

# Matrice de confusion

plot_confusion_matrix_norm(y_test, y_pred_custom, "cm_XGB_seuil.png", out_path)



## SAUVEGARDE MEILLEUR MODELE


print("""
#################################################################################################
## SAUVEGARDE MEILLEUR MODELE
#################################################################################################
""")
model_path = PROJECT_ROOT / "models" / "XGB_mit_binary.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model_adasyn, model_path)
print("Modèle XGB binaire sauvegardé dans :", model_path)
