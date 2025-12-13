
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# preprocessing pipeline

def build_preprocessor(cfg: dict):
    # Numeric pipeline
    numeric_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg["preprocessing"]["numeric_imputer_strategy"])),
        ("scaler", StandardScaler() if cfg["preprocessing"]["scale_numeric"] else "passthrough")
    ])

    # Categorical pipeline
    categorical_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg["preprocessing"]["categorical_imputer_strategy"])),
        ("onehot", OneHotEncoder(
            drop=cfg["preprocessing"]["onehot_drop"],
            handle_unknown=cfg["preprocessing"]["onehot_handle_unknown"],
            sparse_output=False
        ))
    ])

    # Combine both numeric and categorical using the columntransfer
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_processor, cfg["preprocessing"]["numeric_cols"]),
        ("categorical", categorical_processor, cfg["preprocessing"]["categorical_cols"])
    ])

    return preprocessor


#  model pipeline

def build_model(cfg: dict):
    preprocessor = build_preprocessor(cfg)

    # choose estimator whether randomforest or gradient boosting
    if cfg["training"]["estimator"] == "RandomForest":
        model = RandomForestClassifier(**cfg["training"]["random_forest_params"])
    else:
        model = GradientBoostingClassifier(**cfg["training"]["gradient_boosting_params"])

    # full pipeline with SMOTE
    steps = [("preprocessor", preprocessor)]
    if cfg["training"]["use_smote"]:
        steps.append(("smote", SMOTE(random_state=cfg["project"]["seed"])))
    steps.append(("classifier", model))

    pipe = ImbPipeline(steps=steps)
    return pipe


# Train the model

def train_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe


# Evaluate  for the model

def evaluate_model(pipe, X_test, y_test, cfg: dict):
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # AUC
    auc = roc_auc_score(y_test, y_proba)

    # Precision-Recall threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    

    f1_eps = float(cfg["evaluation"].get("f1_eps", 1e-6))

    f1_scores = 2 * precision * recall / (precision + recall + f1_eps)
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_proba >= best_threshold).astype(int)




    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {auc:.3f}, Best Threshold: {best_threshold:.2f}")
    print(f"F1: {f1_score(y_test, y_pred):.2f}, Recall: {recall_score(y_test, y_pred):.2f}, Precision: {precision_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Plots
    if cfg["evaluation"]["plot_roc"]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.show()

    if cfg["evaluation"]["plot_pr"]:
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.show()

    return {
        "auc": auc,
        "best_threshold": best_threshold,
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred)
    }


# Save and load model

def save_model(pipe, cfg: dict):
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models_dir"], cfg["serialization"]["model_filename"])
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")

def load_model(cfg: dict):
    model_path = os.path.join(cfg["paths"]["models_dir"], cfg["serialization"]["model_filename"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)
print('models are over')
# model building over