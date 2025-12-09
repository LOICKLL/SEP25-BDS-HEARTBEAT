# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# ================== PATHS GLOBAUX ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_TRAIN_PATH = DATA_DIR / "mitbih_187pts_MLII_train_patients.csv"
DATA_TEST_PATH = DATA_DIR / "mitbih_187pts_MLII_test_patients.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES_PATH = MODELS_DIR / "nn_mit_mlii_binary_features.json"
SCALER_PATH = MODELS_DIR / "scaler_mit_mlii_binary.pkl"

# modèles Keras entraînés 
NN_MODEL_PATH = MODELS_DIR / "nn_mit_mlii_binary.keras"
CNN_MODEL_PATH = MODELS_DIR / "cnn_mit_mlii_binary_best.keras"

SCORES_PATH = REPORTS_DIR / "mit_models_scores.csv"
TARGET_COL = "binary"  

# ================== CONFIG MODELES (Galerie) ==================
MODEL_CONFIG = {
    "LogReg (MIT MLII binaire)": {
        "filename": "LogReg_mit_binary.pkl",
        "framework": "sklearn",
        "task": "Classification binaire battement normal / anormal sur MIT-BIH.",
        "id": "logreg_mit_binary",
    },
    "SVM (MIT MLII binaire)": {
        "filename": "svm_mit_mlii_binary.pkl",
        "framework": "sklearn",
        "task": "SVM RBF optimisé sur MIT-BIH (binaire).",
        "id": "svm_mit_binary",
    },
    "Random Forest (MIT MLII binaire)": {
        "filename": "rf_mit_mlii_binary.pkl",
        "framework": "sklearn",
        "task": "Random Forest pour la classification binaire.",
        "id": "rf_mit_binary",
    },
    "XGBoost (MIT MLII binaire)": {
        "filename": "XGB_mit_binary.pkl",
        "framework": "sklearn",
        "task": "XGBoost sur MIT-BIH (binaire).",
        "id": "xgb_mit_binary",
    },
    "NN dense (MIT MLII binaire)": {
        "filename": "nn_mit_mlii_binary.keras",
        "framework": "keras_dense",
        "task": "Réseau de neurones fully-connected.",
        "id": "nn_mit_binary",
    },
    "CNN (MIT MLII binaire)": {
        "filename": "cnn_mit_mlii_binary_best.keras",
        "framework": "keras_cnn",
        "task": "Réseau de neurones convolutionnel 1D sur les signaux.",
        "id": "cnn_mit_binary",
    },
}

# ================== CONFIG APP ==================
st.set_page_config(
    page_title="SEP25-BDS-HEARTBEAT - MIT-BIH App",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page :",
    ["Playground entraînement", "Galerie de modèles", "Comparaison des modèles"],
)

# ================== HELPERS COMMUNS ==================
@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path)


@st.cache_data
def load_feature_cols():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_sklearn_model(path: Path):
    return joblib.load(path)


@st.cache_resource
def load_keras_model(path: Path):
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_scaler_cached(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_scores():
    if SCORES_PATH.exists():
        return pd.read_csv(SCORES_PATH)
    return None


def show_metrics(y_true, y_pred, threshold_used, title="Modèle"):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1]
    )
    report = classification_report(y_true, y_pred, digits=2)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    col_report, col_matrix = st.columns([1, 1])

    with col_report:
        st.markdown(f"### {title} — Classification report")
        st.code(report, language="text")

        st.markdown("#### Résumé")
        st.write(f"Seuil : {threshold_used:.2f}")
        st.write(f"Accuracy globale : {acc:.3f}")
        st.write(f"Precision classe 1 : {precision[1]:.3f}")
        st.write(f"Recall classe 1 : {recall[1]:.3f}")
        st.write(f"F1 classe 1 : {f1[1]:.3f}")

    with col_matrix:
        st.markdown(f"### {title} — Matrice de confusion")
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Prédit 0", "Prédit 1"],
            yticklabels=["Vrai 0", "Vrai 1"],
            ax=ax,
        )
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vrai")
        st.pyplot(fig)


def build_dense_nn(input_dim, hidden_units=(128, 64), dropout=0.3, l2_reg=1e-4):
    """Réseau de neurones fully-connected pour classification binaire."""
    inputs = keras.Input(shape=(input_dim,), name="ecg_features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f"dense_{i+1}",
        )(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mit_dense_nn")
    return model


def build_cnn_1d(input_length, filters=(32, 64), kernel_size=5, pool_size=2, dropout=0.3, l2_reg=1e-4):
    """CNN 1D pour signaux ECG (classification binaire)."""
    inputs = keras.Input(shape=(input_length, 1), name="ecg_signal")
    x = inputs
    for i, f in enumerate(filters):
        x = layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f"conv1d_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.MaxPooling1D(pool_size=pool_size, name=f"pool_{i+1}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_final")(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout, name="dropout_final")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mit_cnn_1d")
    return model


def plot_loss(history, title):
    """Affiche la courbe de loss train/val dans Streamlit."""
    hist = history.history
    fig, ax = plt.subplots()
    ax.plot(hist["loss"], label="loss")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)



# PAGE 1 : PLAYGROUND ENTRAÎNEMENT

if page == "Playground entraînement":
    st.title("MIT-BIH — Playground modèles (entraînement)")

    train = load_data(DATA_TRAIN_PATH)
    test = load_data(DATA_TEST_PATH)
    feature_cols = load_feature_cols()

    st.write("### Aperçu du dataset train :")
    st.dataframe(train.head())

    # Features / Target
    X_train = train[feature_cols]
    y_train = train["binary"]

    X_test = test[feature_cols]
    y_test = test["binary"]

    # Info déséquilibre
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    scale_pos_weight = n_negative / n_positive

    st.write(
        f"**Négatifs (0)** : {n_negative} — "
        f"**Positifs (1)** : {n_positive} — "
        f"ratio ≈ {scale_pos_weight:.2f}"
    )

    # ----- Sidebar config spécifiques playground -----
    st.sidebar.header("Configuration modèles (Playground)")

    model_choice = st.sidebar.selectbox(
        "Choix du modèle",
        [
            "LogReg (ElasticNet)",
            "RandomForest",
            "XGBoost",
            "SVM",
            "NN (Keras dense)",
            "CNN (Keras 1D)",
        ],
    )

    random_state = st.sidebar.number_input(
        "random_state",
        value=42,
        step=1,
        format="%d",
    )

    # ================== LOGREG ELASTICNET ==================
    if model_choice == "LogReg (ElasticNet)":
        st.subheader("Logistic Regression ElasticNet — solver SAGA")

        log10_C = st.slider("log10(C)", -2.0, 0.5, 0.0, step=0.1, key="logreg_en_C")
        C = 10 ** log10_C

        l1_ratio = st.select_slider(
            "l1_ratio (0=L2, 1=L1)",
            options=[0.2, 0.5, 0.8],
            value=0.5,
            key="logreg_en_l1ratio",
        )

        class_weight_option = st.selectbox(
            "class_weight",
            ["balanced", "None"],
            index=0,
            key="cw_logreg_en",
        )
        class_weight = None if class_weight_option == "None" else "balanced"

        max_iter = st.slider(
            "max_iter", 1000, 10000, 4000, step=1000, key="logreg_en_maxiter"
        )

        threshold = st.slider(
            "Seuil (proba classe 1)", 0.1, 0.9, 0.5, step=0.01, key="th_logreg_en"
        )

        compare_baseline = st.checkbox(
            "Comparer avec la baseline LogReg", value=False, key="baseline_logreg_en"
        )

        if st.button("Entraîner LogReg ElasticNet", key="btn_logreg_en"):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight=class_weight,
                            C=C,
                            l1_ratio=l1_ratio,
                            max_iter=max_iter,
                            tol=1e-3,
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            show_metrics(y_test, y_pred, threshold, title="LogReg ElasticNet")

            if compare_baseline:
                st.markdown("---")
                st.subheader("Baseline LogReg")

                baseline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                penalty="l2",
                                solver="lbfgs",
                                class_weight=None,
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                baseline.fit(X_train, y_train)
                y_proba_base = baseline.predict_proba(X_test)[:, 1]
                y_pred_base = (y_proba_base >= 0.5).astype(int)

                show_metrics(y_test, y_pred_base, 0.5, title="Baseline LogReg")

    # ================== RANDOM FOREST ==================
    elif model_choice == "RandomForest":
        st.subheader("Random Forest")

        n_estimators = st.slider("n_estimators", 50, 400, 200, step=50, key="rf_n_estimators")
        max_depth = st.slider("max_depth", 3, 20, 10, step=1, key="rf_max_depth")
        threshold = st.slider(
            "Seuil (proba classe 1)", 0.1, 0.9, 0.5, step=0.01, key="th_rf"
        )

        compare_baseline_rf = st.checkbox(
            "Comparer avec baseline RandomForest", value=False, key="baseline_rf"
        )

        if st.button("Entraîner RandomForest", key="btn_rf"):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=-1,
                random_state=random_state,
            )

            rf.fit(X_train, y_train)
            y_proba = rf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            show_metrics(y_test, y_pred, threshold, title="RandomForest")

            if compare_baseline_rf:
                st.markdown("---")
                st.subheader("Baseline RandomForest")

                baseline_rf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=random_state,
                )
                baseline_rf.fit(X_train, y_train)
                y_proba_base = baseline_rf.predict_proba(X_test)[:, 1]
                y_pred_base = (y_proba_base >= 0.5).astype(int)

                show_metrics(y_test, y_pred_base, 0.5, title="Baseline RandomForest")

    # ================== XGBOOST + ADASYN ==================
    elif model_choice == "XGBoost":
        st.subheader("XGBoost + ADASYN")

        n_estimators = st.slider("n_estimators", 100, 400, 200, step=50)
        learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01)
        max_depth = st.slider("max_depth", 3, 9, 6, step=1)

        st.markdown("#### ADASYN")
        sampling_strategy = st.selectbox(
            "sampling_strategy",
            ["not majority", "minority"],
            index=0,
            key="adasyn_sampling",
        )
        n_neighbors = st.slider("n_neighbors (ADASYN)", 1, 10, 3, step=1)

        threshold = st.slider(
            "Seuil (proba classe 1)", 0.1, 0.9, 0.5, step=0.01, key="th_xgb_adasyn"
        )

        compare_baseline_xgb = st.checkbox(
            "Comparer avec baseline XGBoost", value=False, key="baseline_xgb"
        )

        if st.button("Entraîner XGBoost + ADASYN", key="btn_xgb_adasyn"):
            xgb_adasyn_pipeline = ImbPipeline(
                steps=[
                    (
                        "adasyn",
                        ADASYN(
                            sampling_strategy=sampling_strategy,
                            n_neighbors=n_neighbors,
                            random_state=random_state,
                        ),
                    ),
                    (
                        "xgb",
                        XGBClassifier(
                            objective="binary:logistic",
                            tree_method="hist",
                            random_state=random_state,
                            n_jobs=-1,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                        ),
                    ),
                ]
            )

            xgb_adasyn_pipeline.fit(X_train, y_train)

            y_proba = xgb_adasyn_pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            show_metrics(y_test, y_pred, threshold, title="XGBoost + ADASYN")

            if compare_baseline_xgb:
                st.markdown("---")
                st.subheader("Baseline XGBoost")

                baseline_xgb = XGBClassifier(
                    objective="binary:logistic",
                    tree_method="hist",
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=random_state,
                )

                baseline_xgb.fit(X_train, y_train)
                y_proba_base = baseline_xgb.predict_proba(X_test)[:, 1]
                y_pred_base = (y_proba_base >= 0.5).astype(int)

                show_metrics(y_test, y_pred_base, 0.5, title="Baseline XGBoost")

    # ================== SVM ==================
    elif model_choice == "SVM":
        st.subheader("SVM (SVC RBF)")

        C = st.slider("C", 0.1, 10.0, 4.0, step=0.1, key="svm_C")
        gamma = st.slider("gamma", 0.0001, 0.1, 0.01, step=0.0005, key="svm_gamma")
        threshold = st.slider(
            "Seuil (proba classe 1)", 0.1, 0.9, 0.5, step=0.01, key="th_svm"
        )

        compare_baseline_svm = st.checkbox(
            "Comparer avec baseline SVM", value=False, key="baseline_svm"
        )

        if st.button("Entraîner SVM", key="btn_svm"):
            svm_clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            C=C,
                            gamma=gamma,
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            )

            svm_clf.fit(X_train, y_train)
            y_proba = svm_clf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            show_metrics(y_test, y_pred, threshold, title="SVM RBF")

            if compare_baseline_svm:
                st.markdown("---")
                st.subheader("Baseline SVM RBF")

                baseline_svm = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "svc",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                probability=True,
                                random_state=random_state,
                            ),
                        ),
                    ]
                )
                baseline_svm.fit(X_train, y_train)
                y_proba_base = baseline_svm.predict_proba(X_test)[:, 1]
                y_pred_base = (y_proba_base >= 0.5).astype(int)

                show_metrics(y_test, y_pred_base, 0.5, title="Baseline SVM RBF")

    # ================== NN KERAS DENSE ==================
    elif model_choice == "NN (Keras dense)":
        st.subheader("Réseau de neurones dense (Keras) — Entraînement")

        col_hyp1, col_hyp2 = st.columns(2)

        with col_hyp1:
            hidden_choice = st.selectbox(
                "Architecture (couches cachées)",
                options=["64,32", "128,64", "256,128,64"],
                index=1,
                key="nn_hidden_choice",
            )
            dropout = st.slider("Dropout", 0.0, 0.7, 0.3, step=0.05, key="nn_dropout")
            l2_reg = st.select_slider(
                "L2 (régularisation)",
                options=[1e-5, 1e-4, 1e-3],
                value=1e-4,
                key="nn_l2",
            )

        with col_hyp2:
            lr = st.select_slider(
                "Learning rate (Adam)",
                options=[1e-4, 5e-4, 1e-3],
                value=1e-3,
                key="nn_lr",
            )
            batch_size = st.selectbox("Batch size", options=[128, 256, 512], index=1, key="nn_bs")
            epochs = st.slider("Epochs max", 5, 100, 30, step=5, key="nn_epochs")
            patience = st.slider("Patience (EarlyStopping)", 2, 10, 5, step=1, key="nn_patience")

        threshold = st.slider(
            "Seuil (proba classe 1)", 0.1, 0.9, 0.5, step=0.01, key="th_nn_keras"
        )

        compare_baseline_nn = st.checkbox(
            "Comparer avec baseline NN dense", value=False, key="baseline_nn"
        )

        if st.button("Entraîner le NN Keras", key="btn_nn_train"):
            hidden_units = tuple(int(x.strip()) for x in hidden_choice.split(","))

            X_all = X_train.values.astype("float32")
            y_all = y_train.values.astype("int32")

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all,
                y_all,
                test_size=0.2,
                random_state=random_state,
                stratify=y_all,
            )

            scaler = StandardScaler()
            scaler.fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test.values.astype("float32"))

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),
                y=y_tr,
            )
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

            # ---- Modèle "playground" paramétrable ----
            model = build_dense_nn(
                input_dim=X_tr_s.shape[1],
                hidden_units=hidden_units,
                dropout=dropout,
                l2_reg=l2_reg,
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=[
                    "accuracy",
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                ],
            )

            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )

            st.write("Entraînement du NN Keras en cours...")
            history = model.fit(
                X_tr_s,
                y_tr,
                validation_data=(X_val_s, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=[es],   
                verbose=0,
            )

            st.success("Entraînement terminé ✅")

            st.markdown("### Courbe de loss (train / val)")
            plot_loss(history, title="NN dense — loss")

            y_proba = model.predict(X_test_s, verbose=0).ravel()
            y_pred = (y_proba >= threshold).astype(int)

            show_metrics(y_test, y_pred, threshold, title="NN Keras dense (retrained)")

            # ---- Baseline NN dense ----
            if compare_baseline_nn:
                st.markdown("---")
                st.subheader("Baseline NN dense")

                baseline_hidden = (128, 64)
                baseline_dropout = 0.3
                baseline_l2 = 1e-4
                baseline_lr = 1e-3

                baseline_model = build_dense_nn(
                    input_dim=X_tr_s.shape[1],
                    hidden_units=baseline_hidden,
                    dropout=baseline_dropout,
                    l2_reg=baseline_l2,
                )
                baseline_optimizer = keras.optimizers.Adam(learning_rate=baseline_lr)
                baseline_model.compile(
                    optimizer=baseline_optimizer,
                    loss="binary_crossentropy",
                    metrics=[
                        "accuracy",
                        keras.metrics.Precision(name="precision"),
                        keras.metrics.Recall(name="recall"),
                    ],
                )

                es_base = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )

                history_base = baseline_model.fit(
                    X_tr_s,
                    y_tr,
                    validation_data=(X_val_s, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight_dict,
                    callbacks=[es_base],
                    verbose=0,
                )

                st.markdown("### Courbe de loss (baseline NN dense)")
                plot_loss(history_base, title="Baseline NN dense — loss")

                y_proba_base = baseline_model.predict(X_test_s, verbose=0).ravel()
                y_pred_base = (y_proba_base >= 0.5).astype(int)

                show_metrics(
                    y_test, y_pred_base, 0.5, title="Baseline NN Keras dense"
                )

    # ================== CNN KERAS 1D ==================
    elif model_choice == "CNN (Keras 1D)":
        st.subheader("CNN 1D (Keras) — Entraînement")

        col_hyp1, col_hyp2 = st.columns(2)

        with col_hyp1:
            filters_choice = st.selectbox(
                "Filtres par bloc (Conv1D)",
                options=["32,64", "16,32,64", "64,128"],
                index=0,
                key="cnn_filters_choice",
            )
            kernel_size = st.slider("kernel_size (Conv1D)", 3, 11, 5, step=2, key="cnn_kernel")
            pool_size = st.selectbox("pool_size (MaxPooling1D)", [2, 4], index=0, key="cnn_pool")
            dropout = st.slider("Dropout", 0.0, 0.7, 0.3, step=0.05, key="cnn_dropout")
            l2_reg = st.select_slider(
                "L2 (régularisation)",
                options=[1e-5, 1e-4, 1e-3],
                value=1e-4,
                key="cnn_l2",
            )

        with col_hyp2:
            lr = st.select_slider(
                "Learning rate (Adam)",
                options=[1e-4, 5e-4, 1e-3],
                value=1e-3,
                key="cnn_lr",
            )
            batch_size = st.selectbox("Batch size", options=[128, 256, 512], index=1, key="cnn_bs")
            epochs = st.slider("Epochs max", 5, 100, 30, step=5, key="cnn_epochs")
            patience = st.slider("Patience (EarlyStopping)", 2, 10, 5, step=1, key="cnn_patience")

        threshold = st.slider(
            "Seuil (proba classe 1)", 0.1, 0.9, 0.5, step=0.01, key="th_cnn_keras"
        )

        compare_baseline_cnn = st.checkbox(
            "Comparer avec baseline CNN 1D", value=False, key="baseline_cnn"
        )

        if st.button("Entraîner le CNN Keras", key="btn_cnn_train"):
            filters = tuple(int(x.strip()) for x in filters_choice.split(","))

            X_all = X_train.values.astype("float32")
            y_all = y_train.values.astype("int32")

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all,
                y_all,
                test_size=0.2,
                random_state=random_state,
                stratify=y_all,
            )

            scaler = StandardScaler()
            scaler.fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test.values.astype("float32"))

            X_tr_cnn = X_tr_s.reshape(-1, X_tr_s.shape[1], 1)
            X_val_cnn = X_val_s.reshape(-1, X_val_s.shape[1], 1)
            X_test_cnn = X_test_s.reshape(-1, X_test_s.shape[1], 1)

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),
                y=y_tr,
            )
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

            # ---- Modèle CNN paramétrable ----
            model = build_cnn_1d(
                input_length=X_tr_cnn.shape[1],
                filters=filters,
                kernel_size=kernel_size,
                pool_size=pool_size,
                dropout=dropout,
                l2_reg=l2_reg,
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=[
                    "accuracy",
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                ],
            )

            es = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )

            st.write("Entraînement du CNN Keras en cours...")
            history = model.fit(
                X_tr_cnn,
                y_tr,
                validation_data=(X_val_cnn, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=[es],   
                verbose=0,
            )

            st.success("Entraînement terminé ✅")

            st.markdown("### Courbe de loss (train / val)")
            plot_loss(history, title="CNN 1D — loss")

            y_proba = model.predict(X_test_cnn, verbose=0).ravel()
            y_pred = (y_proba >= threshold).astype(int)

            show_metrics(y_test, y_pred, threshold, title="CNN Keras 1D (retrained)")

            # ---- Baseline CNN 1D ----
            if compare_baseline_cnn:
                st.markdown("---")
                st.subheader("Baseline CNN 1D")

                baseline_filters = (32, 64)
                baseline_kernel = 5
                baseline_pool = 2
                baseline_dropout = 0.3
                baseline_l2 = 1e-4
                baseline_lr = 1e-3

                baseline_model = build_cnn_1d(
                    input_length=X_tr_cnn.shape[1],
                    filters=baseline_filters,
                    kernel_size=baseline_kernel,
                    pool_size=baseline_pool,
                    dropout=baseline_dropout,
                    l2_reg=baseline_l2,
                )
                baseline_optimizer = keras.optimizers.Adam(learning_rate=baseline_lr)
                baseline_model.compile(
                    optimizer=baseline_optimizer,
                    loss="binary_crossentropy",
                    metrics=[
                        "accuracy",
                        keras.metrics.Precision(name="precision"),
                        keras.metrics.Recall(name="recall"),
                    ],
                )

                es_base = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )

                history_base = baseline_model.fit(
                    X_tr_cnn,
                    y_tr,
                    validation_data=(X_val_cnn, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight_dict,
                    callbacks=[es_base],
                    verbose=0,
                )

                st.markdown("### Courbe de loss (baseline CNN 1D)")
                plot_loss(history_base, title="Baseline CNN 1D — loss")

                y_proba_base = baseline_model.predict(X_test_cnn, verbose=0).ravel()
                y_pred_base = (y_proba_base >= 0.5).astype(int)

                show_metrics(
                    y_test, y_pred_base, 0.5, title="Baseline CNN Keras 1D"
                )



# PAGE 2 : GALERIE DE MODÈLES

elif page == "Galerie de modèles":
    st.title(" SEP25-BDS-HEARTBEAT – Galerie de modèles MIT-BIH")
    st.markdown(
        "Cette page présente les modèles **sauvegardés** (fichiers .pkl / .keras) "
        "et permet de charger un CSV de test pour obtenir une matrice de confusion."
    )

    st.sidebar.header("Galerie")
    model_name = st.sidebar.selectbox(
        "Choisis un modèle à explorer :", list(MODEL_CONFIG.keys())
    )
    config = MODEL_CONFIG[model_name]

    st.sidebar.write("**Fichier :**", config["filename"])
    st.sidebar.write("**Framework :**", config["framework"])

    col_info, col_detail = st.columns([1, 2])

    with col_info:
        st.subheader("Informations générales")
        st.write(f"**Modèle sélectionné :** {model_name}")
        st.write(f"**Tâche :** {config['task']}")
        st.write(f"**Chemin complet :** `{(MODELS_DIR / config['filename']).as_posix()}`")

        scores = load_scores()
        if scores is not None and "model_id" in scores.columns:
            this_score = scores[scores["model_id"] == config["id"]]
            if not this_score.empty:
                st.markdown("### 📊 Performances (test)")
                st.dataframe(this_score)
            else:
                st.info(
                    "Pas de ligne de scores pour ce modèle dans `mit_models_scores.csv`."
                )
        else:
            st.info(
                "Aucun fichier de scores trouvé. "
                "Tu peux générer ce CSV avec le script `mit_eval_models.py`."
            )

    with col_detail:
        st.subheader(" Modèle")

        model_path = MODELS_DIR / config["filename"]
        if not model_path.exists():
            st.error(f"Fichier modèle introuvable : {model_path}")
        else:
            if st.button("Charger et afficher un aperçu du modèle"):
                try:
                    if config["framework"].startswith("sklearn"):
                        model = load_sklearn_model(model_path)
                    else:
                        model = load_keras_model(model_path)
                    st.success("Modèle chargé avec succès ✅")
                    st.markdown("**Aperçu (repr du modèle)**")
                    st.code(repr(model), language="python")
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle : {e}")

        st.markdown("---")
        st.subheader(" Prédiction & matrice de confusion")

        st.markdown(
            "Charge un CSV contenant :\n"
            "- les colonnes de **features** (b0, b1, ..., b186) comme à l'entraînement,\n"
            "- la colonne cible `binary` (0=normal, 1=anormal) pour calculer la matrice de confusion."
        )

        uploaded_file = st.file_uploader(
            "Choisis un fichier CSV (ex : `mitbih_187pts_MLII_test_patients.csv` ou un sous-ensemble)",
            type="csv",
        )

        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Erreur de lecture du CSV : {e}")
                df_input = None

            if df_input is not None:
                st.markdown("**Aperçu des premières lignes :**")
                st.dataframe(df_input.head())

                if TARGET_COL not in df_input.columns:
                    st.error(
                        f"La colonne cible `{TARGET_COL}` n'est pas présente dans le fichier. "
                        "Impossible de calculer une matrice de confusion."
                    )
                else:
                    feature_cols = load_feature_cols()
                    missing = [c for c in feature_cols if c not in df_input.columns]

                    if missing:
                        st.error(
                            "Certaines colonnes de features sont manquantes dans le CSV uploadé : "
                            + ", ".join(missing)
                        )
                    else:
                        X = df_input[feature_cols].astype("float32").values
                        y_true = df_input[TARGET_COL].values

                        #  on ne scale QUE pour les modèles Keras.
                        if config["framework"] in ["keras_dense", "keras_cnn"]:
                            scaler = load_scaler_cached(SCALER_PATH)
                            if scaler is not None:
                                X_input = scaler.transform(X)
                            else:
                                X_input = X
                                st.warning(
                                    "Aucun scaler chargé : les features ne sont pas standardisées."
                                )
                        else:
                            # Modèles sklearn : les pipelines gèrent eux-mêmes le scaling.
                            X_input = X

                        if st.button("Lancer la prédiction et afficher la matrice de confusion"):
                            try:
                                if config["framework"].startswith("sklearn"):
                                    model = load_sklearn_model(model_path)
                                    proba = None
                                    if hasattr(model, "predict_proba"):
                                        proba = model.predict_proba(X_input)[:, 1]
                                        y_pred = (proba >= 0.5).astype(int)
                                    else:
                                        y_pred = model.predict(X_input)
                                elif config["framework"] == "keras_dense":
                                    model = load_keras_model(model_path)
                                    proba = model.predict(X_input, verbose=0).ravel()
                                    y_pred = (proba >= 0.5).astype(int)
                                elif config["framework"] == "keras_cnn":
                                    model = load_keras_model(model_path)
                                    X_cnn = X_input.reshape(-1, X_input.shape[1], 1)
                                    proba = model.predict(X_cnn, verbose=0).ravel()
                                    y_pred = (proba >= 0.5).astype(int)
                                else:
                                    st.error("Framework non géré pour la prédiction.")
                                    y_pred = None

                                if y_pred is not None:
                                    st.success("Prédiction terminée ✅")

                                    cm = confusion_matrix(y_true, y_pred)
                                    cm_norm = confusion_matrix(
                                        y_true, y_pred, normalize="true"
                                    )
                                    labels = ["normal (0)", "anormal (1)"]

                                    # ---------- Matrice brute ----------
                                    st.markdown("### 🧮 Matrice de confusion")
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        cm,
                                        annot=True,
                                        fmt="d",
                                        cmap="Blues",
                                        xticklabels=labels,
                                        yticklabels=labels,
                                        ax=ax,
                                    )
                                    ax.set_xlabel("Prédit")
                                    ax.set_ylabel("Réel")
                                    st.pyplot(fig)

                                    # ---------- Matrice normalisée ----------
                                    st.markdown("### 🧮 Matrice de confusion (normalisée)")
                                    fig2, ax2 = plt.subplots()
                                    sns.heatmap(
                                        cm_norm,
                                        annot=True,
                                        fmt=".2f",
                                        cmap="Blues",
                                        xticklabels=labels,
                                        yticklabels=labels,
                                        ax=ax2,
                                    )
                                    ax2.set_xlabel("Prédit")
                                    ax2.set_ylabel("Réel")
                                    st.pyplot(fig2)

                                    # ---------- Rapport de classification ----------
                                    st.markdown("### 📋 Rapport de classification")
                                    report = classification_report(
                                        y_true, y_pred, target_names=["normal", "anormal"]
                                    )
                                    st.text(report)

                            except Exception as e:
                                st.error(f"Erreur pendant la prédiction : {e}")
        else:
            st.info(
                "Charge un CSV pour activer la prédiction et la matrice de confusion.\n\n"
                "Le fichier doit contenir au minimum :\n"
                "- les features (b0...b186),\n"
                "- la colonne `binary` avec la vraie étiquette."
            )


# PAGE 3 : COMPARAISON DES MODÈLES

elif page == "Comparaison des modèles":
    st.title("📊 Comparaison des modèles MIT-BIH")

    st.markdown(
        "Cette page affiche le fichier de scores global "
        "`mit_models_scores.csv` généré par le script d'évaluation "
        "afin de comparer les modèles entre eux."
    )

    scores = load_scores()

    if scores is None:
        st.error(
            f"Aucun fichier de scores trouvé à l'emplacement : `{SCORES_PATH.as_posix()}`.\n\n"
            "Lance d'abord le script qui génère `mit_models_scores.csv`."
        )
    else:
        st.markdown("### Tableau complet des scores")
        st.dataframe(scores)

        # Option simple de tri 
        numeric_cols = scores.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            sort_col = st.selectbox(
                "Trier les modèles par :", numeric_cols, index=0
            )
            sort_ascending = st.checkbox("Ordre croissant", value=False)
            st.markdown(f"### Scores triés par `{sort_col}`")
            st.dataframe(scores.sort_values(by=sort_col, ascending=sort_ascending))
