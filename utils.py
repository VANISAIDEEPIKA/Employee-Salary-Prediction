import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    df = df.dropna()
    df.columns = df.columns.str.strip()
    return df

def encode_features(df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def apply_encoders(df, encoders):
    df_encoded = df.copy()
    for col, le in encoders.items():
        if col in df_encoded.columns:
            try:
                df_encoded[col] = le.transform(df_encoded[col])
            except ValueError:
                df_encoded[col] = le.transform([le.classes_[0]] * len(df_encoded))
    return df_encoded
