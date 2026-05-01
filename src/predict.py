import pandas as pd
import joblib
import os

from featurize import load_featurizer

# cached so we only load once
_cached_model = None
_cached_featurizer = None

def get_model_and_featurizer(model_path="models/xgboost_model.pkl", featurizer_path="models/preprocessor.pkl"):
    # grab model and featurizer, loading from disk if needed
    global _cached_model, _cached_featurizer
    
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError("model not found. please run train.py first.")
        _cached_model = joblib.load(model_path)
        
    if _cached_featurizer is None:
        if not os.path.exists(featurizer_path):
            raise FileNotFoundError("featurizer not found. please run train.py first.")
        _cached_featurizer = load_featurizer(featurizer_path)
        
    return _cached_model, _cached_featurizer

def predict_risk(
    warehouse_block, mode_of_shipment, product_importance, gender,
    customer_care_calls, customer_rating, prior_purchases, discount_offered,
    cost_of_product, weight_in_gms
):
    # quick sanity check on numbers
    if cost_of_product < 0 or weight_in_gms < 0 or discount_offered < 0:
        raise ValueError("cost, weight, and discount must be non-negative")
        
    # get model and featurizer
    model, featurizer = get_model_and_featurizer()
    
    # build a single-row dataframe from the inputs
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
    
    # run through the same encoding pipeline as training
    X_transformed = featurizer.transform(input_data)
    
    # get prediction and probability
    risk_class = int(model.predict(X_transformed)[0])
    risk_probability = float(model.predict_proba(X_transformed)[0, 1])
    
    # calculate feature contributions (simple proxy via feature importance * input value)
    feature_names = featurizer.get_feature_names()
    importances = model.feature_importances_
    contributions = []
    
    for i, name in enumerate(feature_names):
        # Only non-zero encoded values contribute for this specific prediction
        val = X_transformed[0, i]
        if val > 0:
            contributions.append({"feature": name, "importance": importances[i]})
            
    # sort by importance
    contributions = sorted(contributions, key=lambda x: x["importance"], reverse=True)[:5]
    
    # build the result dict
    if risk_class == 0:
        risk_label = "Low Risk (On Time)"
        confidence = 1 - risk_probability
    else:
        risk_label = "High Risk (Delayed)"
        confidence = risk_probability
        
    return {
        'risk_class': risk_class,
        'risk_label': risk_label,
        'probability': risk_probability,
        'confidence': confidence,
        'top_factors': contributions
    }
