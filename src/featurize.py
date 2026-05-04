import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class Featurizer:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    # fit n transform the data
    def fit_transform(self, X):
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(include=['number']).columns.tolist()

        self.preprocessor = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("numeric", "passthrough", num_cols)
        ])

        transformed = self.preprocessor.fit_transform(X)

        if len(cat_cols) > 0:
            cat_features = self.preprocessor.named_transformers_["onehot"].get_feature_names_out(cat_cols)
            self.feature_names = list(cat_features) + num_cols
        else:
            self.feature_names = num_cols

        return transformed

    # transform only
    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("featurizer is not fitted yet")
        return self.preprocessor.transform(X)

    # get column names
    def get_feature_names(self):
        return self.feature_names

# save state
def save_featurizer(featurizer, filepath="models/preprocessor.pkl"):
    joblib.dump(featurizer, filepath)

# load state
def load_featurizer(filepath="models/preprocessor.pkl"):
    return joblib.load(filepath)
