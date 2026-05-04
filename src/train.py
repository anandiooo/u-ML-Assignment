import pandas as pd
import json
import os
import joblib
import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_make import load_data, clean_data
from featurize import Featurizer, save_featurizer

MODEL_CLASSES = {
    "XGBoost": XGBClassifier,
    "RandomForest": RandomForestClassifier,
}

DEFAULT_PARAMS = {
    "XGBoost": {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.1,
        "random_state": 42,
        "verbosity": 0,
    },
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": 42,
    },
}

# main train function
def train_model(
    model_name="XGBoost",
    model_params=None,
    data_path="data/ecommerce_shipping_data.csv",
    preprocess_params=None,
    test_size=0.2,
    cv_folds=5,
):
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model name: {model_name}")

    if model_params is None:
        model_params = DEFAULT_PARAMS[model_name]

    if preprocess_params is None:
        preprocess_params = {
            "drop_duplicates": True,
            "cat_fill": "mode",
            "num_fill": "median",
            "outlier_method": "None",
            "fe_toggles": [],
            "imbalance": "None"
        }

    df = load_data(data_path)

    # clean and fe
    imb_strategy = preprocess_params.get("imbalance", "None")
    clean_params = {k: v for k, v in preprocess_params.items() if k not in ["imbalance", "mode_name"]}
    df = clean_data(df, **clean_params)

    X = df.drop(columns=['Reached.on.Time_Y.N'])
    y = df['Reached.on.Time_Y.N']

    # handle imbalance
    if imb_strategy == "Undersampling":
        df_temp = pd.concat([X, y], axis=1)
        count_min = y.value_counts().min()
        df_balanced = pd.concat([
            df_temp[df_temp['Reached.on.Time_Y.N'] == 1].sample(count_min, random_state=42),
            df_temp[df_temp['Reached.on.Time_Y.N'] == 0].sample(count_min, random_state=42)
        ]).sample(frac=1, random_state=42)
        X = df_balanced.drop(columns=['Reached.on.Time_Y.N'])
        y = df_balanced['Reached.on.Time_Y.N']

    # transform data
    featurizer = Featurizer()
    X_transformed = featurizer.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=test_size, random_state=42, stratify=y
    )

    # run training
    model_cls = MODEL_CLASSES[model_name]
    model = model_cls(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    cv_scores = cross_val_score(model, X_transformed, y, cv=cv_folds, scoring='f1')

    # get metrics
    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "recall": round(recall_score(y_val, y_pred), 4),
        "f1_score": round(f1_score(y_val, y_pred), 4),
        "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
        "cv_f1_std": round(float(np.std(cv_scores)), 4),
        "preprocess_params": preprocess_params,
        "train_settings": {"test_size": test_size, "cv_folds": cv_folds},
    }

    cm = confusion_matrix(y_val, y_pred)
    metrics["confusion_matrix"] = {
        "true_negatives": int(cm[0, 0]), "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]), "true_positives": int(cm[1, 1])
    }

    importance_df = pd.DataFrame({
        "feature": featurizer.get_feature_names(),
        "importance": model.feature_importances_.tolist()
    }).sort_values("importance", ascending=False)
    metrics["feature_importance"] = importance_df.to_dict("records")

    metrics["dataset_info"] = {
        "total_rows": len(df), "train_size": len(X_train), "val_size": len(X_val),
        "n_features": X_transformed.shape[1]
    }

    # save results
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name.lower().replace(' ', '_')}_model.pkl")
    save_featurizer(featurizer, "models/preprocessor.pkl")

    with open(f"models/metrics_{model_name.lower().replace(' ', '_')}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

# run test
if __name__ == "__main__":
    train_model()
