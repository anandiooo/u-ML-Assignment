import pandas as pd
import numpy as np
import os

def load_data(filepath="data/ecommerce_shipping_data.csv"):
    # make sure the file is there
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"dataset not found at {filepath}")

    # read into a dataframe
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # remove ID column if present, keep everything else
    if 'ID' in df.columns:
        df = df.drop(columns=['ID']).copy()
    else:
        df = df.copy()

    # drop duplicates
    df = df.drop_duplicates()

    # fill missing categoricals with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # fill missing numerics with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df

if __name__ == "__main__":
    # quick test
    data = load_data()
    clean_df = clean_data(data)
    print(f"data cleaned successfully. shape: {clean_df.shape}")
