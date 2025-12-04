import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt

import joblib






########################################################################################################
# DOSSIERS DU PROJET
########################################################################################################

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "Modelisation_LogisticRegression"
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
## FONCTION DE SAUVEGARDE DES MATRICES DE CONFUSIONS ##
#########################################################################################################

out_path = FIG_DIR


def plot_confusion_matrix_norm(y_true, y_pred, filename, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    full_path=out_path/filename

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



scaler=StandardScaler()

X_train_baseline=scaler.fit_transform(X_train)
X_test_baseline=scaler.transform(X_test)

lr=LogisticRegression(max_iter=1000)

lr.fit(X_train_baseline, y_train)

y_pred=lr.predict(X_test_baseline)
 


save_classification_report(y_test, y_pred, "report_LogReg_baseline.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred, "cm_LogReg_baseline.png", out_path)



#########################################################################################################
## GRIDSEARCH CV LBFGS GroupKFold ##
#########################################################################################################

print("""
#################################################################################################
## GRIDSEARCH CV LBFGS GroupKFold ##
#################################################################################################
""")



pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000))
])

param_grid = {'clf__C': np.logspace(-3, 3, 20)}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring='recall',
    cv = GroupKFold(n_splits=5),
    n_jobs=-1
)

grid.fit(X_train, y_train, groups=groups_train)

print("Meilleur C :", grid.best_params_['clf__C'])
print("Meilleur score (CV):", grid.best_score_)

y_pred = grid.best_estimator_.predict(X_test)    



save_cv_score(grid.best_score_,"cv_score_LogReg_lbfgs_gridsearchCV.txt", out_path)

save_classification_report(y_test, y_pred, "report_LogReg_lbfgs_gridsearchCV.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred, "cm_LogReg_lbfgs_gridsearchCV.png", out_path)



#########################################################################################################
## GRIDSEARCH CV SAGA ##
#########################################################################################################

print("""
#################################################################################################
## GRIDSEARCH CV SAGA ##
#################################################################################################
""")

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        solver='saga',
        penalty='l2',            
        class_weight='balanced', 
        max_iter=50000,          
        tol=1e-2,                
        warm_start=True,         
        n_jobs=-1,
        random_state=42
    ))
])


param_grid2 = {
    'clf__C': np.logspace(-3, 1, 8)  
}

cv = GroupKFold(n_splits=5)

grid2 = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid2,
    scoring='recall',          
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid2.fit(X_train, y_train, groups=groups_train)

print("Meilleur C :", grid2.best_params_['clf__C'])
print("Meilleur score (CV):", grid2.best_score_)

y_pred = grid2.best_estimator_.predict(X_test)


save_cv_score(grid2.best_score_,"cv_score_LogReg_saga_gridsearchCV.txt", out_path)

save_classification_report(y_test, y_pred, "report_LogReg_saga_gridsearchCV.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred, "cm_LogReg_saga_gridsearchCV.png", out_path)



#################################################################################################
## CORRECTEUR D'ELASTICITE ##
#################################################################################################

print("""
#################################################################################################
## CORRECTEUR D'ELASTICITE ##
#################################################################################################
""")



pipe_en = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        class_weight='balanced',    
        max_iter=50000,             
        tol=1e-2,                    
        random_state=42,
        n_jobs=-1
    ))
])


param_grid_en = {
    'clf__C': np.logspace(-2, 1, 6),          
    'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7]     

}

cv = GroupKFold(n_splits=5)

grid_en = GridSearchCV(
    estimator=pipe_en,
    param_grid=param_grid_en,
    scoring='recall',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_en.fit(X_train, y_train, groups=groups_train)

print("Meilleur C :", grid_en.best_params_['clf__C'])
print("Meilleur l1_ratio :", grid_en.best_params_['clf__l1_ratio'])
print("Meilleur score (CV):", grid_en.best_score_)

y_pred_en = grid_en.best_estimator_.predict(X_test)



save_cv_score(grid_en.best_score_,"cv_score_LogReg_elastic.txt", out_path)

save_classification_report(y_test, y_pred_en, "report_LogReg_elastic.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred_en, "cm_LogReg_elastic.png", out_path)




#################################################################################################
## PRÉDICTION AVEC SEUIL 0.3
#################################################################################################

print("""
#################################################################################################
## PREDICTION AVEC SEUIL 0.3 ##
#################################################################################################
""")

# Probabilité d'appartenir à la classe 1
y_proba = grid_en.best_estimator_.predict_proba(X_test)[:, 1]

# Application du seuil 0.3
threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)


save_classification_report(y_test, y_pred_custom, "report_LogReg_seuil.txt", out_path)

plot_confusion_matrix_norm(y_test, y_pred_custom, "cm_LogReg_seuil.png", out_path)


#################################################################################################
## SAUVEGARDE MEILLEUR MODELE
#################################################################################################

print("""
#################################################################################################
## SAUVEGARDE MEILLEUR MODELE
#################################################################################################
""")


model_path = PROJECT_ROOT / "models" / "LogReg_mit_binary.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(grid_en.best_estimator_, model_path)
print("Modèle Logistic Regression binaire sauvegardé dans :", model_path)
