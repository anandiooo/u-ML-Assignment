import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class Featurizer:
    def __init__(self):
        # nothing fitted yet
        self.preprocessor = None
        self.feature_names = None

    def fit_transform(self, X):
        # set up categorical and numeric columns
        cat_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
        num_cols = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product", "Prior_purchases", "Discount_offered", "Weight_in_gms"]

        # one-hot encode categoricals, pass through numerics
        self.preprocessor = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("numeric", "passthrough", num_cols)
        ])

        # fit and transform
        transformed = self.preprocessor.fit_transform(X)

        # keep track of the final feature names
        cat_features = self.preprocessor.named_transformers_["onehot"].get_feature_names_out(cat_cols)
        self.feature_names = list(cat_features) + num_cols

        return transformed

    def transform(self, X):
        # encode new data using the fitted pipeline
        if self.preprocessor is None:
            raise ValueError("featurizer is not fitted yet")
        return self.preprocessor.transform(X)

    def get_feature_names(self):
        # return feature names after encoding
        return self.feature_names

def save_featurizer(featurizer, filepath="models/preprocessor.pkl"):
    # write featurizer to disk
    joblib.dump(featurizer, filepath)

def load_featurizer(filepath="models/preprocessor.pkl"):
    # read featurizer from disk
    return joblib.load(filepath)
