import pandas as pd
import json
import os
import joblib
import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_make import load_data, clean_data
from featurize import Featurizer, save_featurizer

def train_model():
    # load and prep data
    df = load_data("data/ecommerce_shipping_data.csv")
    df = clean_data(df)

    # split features and target
    X = df.drop(columns=['Reached.on.Time_Y.N'])
    y = df['Reached.on.Time_Y.N']

    # encode features
    featurizer = Featurizer()
    X_transformed = featurizer.fit_transform(X)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )

    # calculate scale_pos_weight
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    # train xgboost with tuned hyperparams
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        gamma=0.1, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    # evaluate on validation set
    y_pred = model.predict(X_val)

    # cross validation score
    cv_scores = cross_val_score(model, X_transformed, y, cv=5, scoring='f1')

    # measure single prediction latency
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(X_val[:1])
        latencies.append((time.perf_counter() - start) * 1000)
    avg_latency_ms = sum(latencies) / len(latencies)

    # compute metrics
    metrics = {
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "recall": round(recall_score(y_val, y_pred), 4),
        "f1_score": round(f1_score(y_val, y_pred), 4),
        "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
        "cv_f1_std": round(float(np.std(cv_scores)), 4),
        "avg_latency_ms": round(float(avg_latency_ms), 3)
    }

    # confusion matrix breakdown
    cm = confusion_matrix(y_val, y_pred)
    metrics["confusion_matrix"] = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }

    # feature importance ranking
    importance_df = pd.DataFrame({
        "feature": featurizer.get_feature_names(),
        "importance": model.feature_importances_.tolist()
    }).sort_values("importance", ascending=False)
    metrics["feature_importance"] = importance_df.to_dict("records")

    # record dataset stats
    metrics["dataset_info"] = {
        "total_rows": len(df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "n_features": X_transformed.shape[1]
    }

    # make sure output dir exists
    os.makedirs("models", exist_ok=True)

    # persist model as pickle
    joblib.dump(model, "models/xgboost_model.pkl")

    # persist featurizer
    save_featurizer(featurizer, "models/preprocessor.pkl")

    # write feature names as json
    with open("models/features.json", "w") as f:
        json.dump(featurizer.get_feature_names(), f, indent=4)

    # dump metrics to json
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("training completed and model artifacts saved successfully in /models.")

if __name__ == "__main__":
    train_model()
