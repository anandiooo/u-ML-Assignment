import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

df = pd.read_csv('data/ecommerce_shipping_data.csv')
df = df.drop(columns=['ID'])

X = df.drop(columns=['Reached.on.Time_Y.N'])
y = df['Reached.on.Time_Y.N']

cat_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
            'Prior_purchases', 'Discount_offered', 'Weight_in_gms']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('numeric', 'passthrough', num_cols)
])

X_transformed = preprocessor.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)

configs = [
    ("baseline (current)", dict(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)),
    ("deeper trees", dict(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42, verbosity=0)),
    ("shallow+regularized", dict(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, gamma=0.2, random_state=42, verbosity=0)),
    ("medium depth+reg", dict(n_estimators=200, learning_rate=0.08, max_depth=5, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, gamma=0.1, random_state=42, verbosity=0)),
    ("n500 lr0.03 md5", dict(n_estimators=500, learning_rate=0.03, max_depth=5, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, gamma=0.05, random_state=42, verbosity=0)),
]

for name, params in configs:
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    cv = cross_val_score(model, X_transformed, y, cv=5, scoring='f1')
    print(f'{name:30s} | val F1={f1:.4f} | CV F1={cv.mean():.4f} +/- {cv.std():.4f}')
    print(classification_report(y_val, y_pred, digits=3))
    print()
