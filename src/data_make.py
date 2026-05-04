import pandas as pd
import numpy as np
import os

# load csv
def load_data(filepath="data/ecommerce_shipping_data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    return df

# cleaning pipeline
def clean_data(
    df,
    drop_duplicates=True,
    cat_fill="mode",
    num_fill="median",
    cat_fill_value="Unknown",
    outlier_method="None",
    fe_toggles=None,
):
    if fe_toggles is None:
        fe_toggles = []

    # copy and drop id
    if 'ID' in df.columns:
        df = df.drop(columns=['ID']).copy()
    else:
        df = df.copy()

    # drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates()

    # handle categorical
    for col in df.select_dtypes(include=['object']).columns:
        if cat_fill == "constant":
            fill_value = cat_fill_value
        else:
            mode_vals = df[col].mode(dropna=True)
            fill_value = mode_vals[0] if len(mode_vals) else cat_fill_value
        df[col] = df[col].fillna(fill_value)

    # handle numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        if num_fill == "mean":
            fill_value = df[col].mean()
        else:
            fill_value = df[col].median()
        if pd.isna(fill_value):
            fill_value = 0
        df[col] = df[col].fillna(fill_value)

    # handle outliers
    if outlier_method in ["Clip (IQR)", "Remove"]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "Reached.on.Time_Y.N" in num_cols:
            num_cols.remove("Reached.on.Time_Y.N")

        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            if outlier_method == "Clip (IQR)":
                df[col] = df[col].clip(lower, upper)
            elif outlier_method == "Remove":
                df = df[(df[col] >= lower) & (df[col] <= upper)]

    # feature engineering
    if "Weight/Cost Ratio" in fe_toggles:
        df["Weight_Cost_Ratio"] = df["Weight_in_gms"] / (df["Cost_of_the_Product"] + 1e-5)

    if "High Discount" in fe_toggles:
        df["Is_High_Discount"] = (df["Discount_offered"] > 10).astype(int)

    if "Engagement Score" in fe_toggles:
        df["Engagement_Score"] = df["Customer_care_calls"] * df["Customer_rating"]

    if "Loyalty Level" in fe_toggles:
        df["Loyalty_Level"] = pd.cut(df["Prior_purchases"], bins=[0, 3, 5, 10], labels=["New", "Regular", "VIP"]).astype(str)

    if "Delivery Urgency" in fe_toggles:
        priority_map = {"low": 1, "medium": 2, "high": 3}
        df["Urgency_Score"] = df["Product_importance"].str.lower().map(priority_map).fillna(1) * df["Discount_offered"]

    if "Care Intensity" in fe_toggles:
        df["Calls_per_Purchase"] = df["Customer_care_calls"] / (df["Prior_purchases"] + 1)

    return df

# test run
if __name__ == "__main__":
    data = load_data()
    clean_df = clean_data(data, fe_toggles=["Loyalty Level", "Delivery Urgency"])
    print(f"done. columns: {clean_df.columns.tolist()}")
