import pandas as pd
import joblib
import os
import json
from data_make import clean_data
from featurize import load_featurizer

_cached_models = {}
_cached_featurizers = {}
_cached_params = {}

def get_model_and_featurizer(model_name="XGBoost"):
    model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
    metrics_path = f"models/metrics_{model_name.lower().replace(' ', '_')}.json"
    featurizer_path = "models/preprocessor.pkl"

    if model_path not in _cached_models:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model '{model_name}' not found. please train it first.")
        _cached_models[model_path] = joblib.load(model_path)

    if featurizer_path not in _cached_featurizers:
        if not os.path.exists(featurizer_path):
            raise FileNotFoundError("preprocessor not found. please train a model first.")
        _cached_featurizers[featurizer_path] = load_featurizer(featurizer_path)

    if metrics_path not in _cached_params:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                _cached_params[metrics_path] = json.load(f).get("preprocess_params", {})
        else:
            _cached_params[metrics_path] = {}

    return _cached_models[model_path], _cached_featurizers[featurizer_path], _cached_params[metrics_path]

def predict_risk(
    warehouse_block, mode_of_shipment, product_importance, gender,
    customer_care_calls, customer_rating, prior_purchases, discount_offered,
    cost_of_product, weight_in_gms, model_name="XGBoost"
):
    model, featurizer, preprocess_params = get_model_and_featurizer(model_name=model_name)

    # create input df
    input_data = pd.DataFrame({
        'Warehouse_block': [warehouse_block],
        'Mode_of_Shipment': [mode_of_shipment],
        'Product_importance': [product_importance],
        'Gender': [gender],
        'Customer_care_calls': [int(customer_care_calls)],
        'Customer_rating': [int(customer_rating)],
        'Cost_of_the_Product': [float(cost_of_product)],
        'Prior_purchases': [int(prior_purchases)],
        'Discount_offered': [int(discount_offered)],
        'Weight_in_gms': [float(weight_in_gms)]
    })

    # apply same feature engineering used in trained
    fe_toggles = preprocess_params.get("fe_toggles", [])
    input_data = clean_data(input_data, fe_toggles=fe_toggles, outlier_method="None")

    # transform n predict
    X_transformed = featurizer.transform(input_data)

    risk_class = int(model.predict(X_transformed)[0])
    risk_probability = float(model.predict_proba(X_transformed)[0, 1])

    feature_names = featurizer.get_feature_names()
    importances = getattr(model, "feature_importances_", None)
    contributions = []

    if importances is not None:
        for i, name in enumerate(feature_names):
            val = X_transformed[0, i]
            if val > 0:
                contributions.append({"feature": name, "importance": float(importances[i])})
        contributions = sorted(contributions, key=lambda x: x["importance"], reverse=True)[:5]

    return {
        'model_name': model_name,
        'risk_class': risk_class,
        'risk_label': "High Risk (Delayed)" if risk_class == 1 else "Low Risk (On Time)",
        'probability': risk_probability,
        'confidence': risk_probability if risk_class == 1 else (1 - risk_probability),
        'top_factors': contributions
    }
