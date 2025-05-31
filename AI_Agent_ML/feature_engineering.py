# feature_engineering.py

import numpy as np
import pandas as pd

def automated_feature_engineering(df):
    df_new = df.copy()
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns

    interaction_features = []
    ratio_features = []
    log_features = []
    poly_features = []

    # Interaction features
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1:]:
            new_col_name = f"{col1}_x_{col2}"
            interaction_features.append(
                pd.Series(df_new[col1] * df_new[col2], name=new_col_name)
            )

    # Ratio features
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1:]:
            new_col_name = f"{col1}_div_{col2}"
            ratio_features.append(
                pd.Series(df_new[col1] / (df_new[col2] + 1e-5), name=new_col_name)
            )

    # Log transform
    for col in numeric_cols:
        if (df_new[col] > 0).all():
            new_col_name = f"log_{col}"
            log_features.append(
                pd.Series(np.log(df_new[col]), name=new_col_name)
            )

    # Polynomial degree 2
    for col in numeric_cols:
        new_col_name = f"{col}_sq"
        poly_features.append(
            pd.Series(df_new[col] ** 2, name=new_col_name)
        )

    # Gabungkan semua fitur baru sekaligus
    all_new_features = interaction_features + ratio_features + log_features + poly_features
    if all_new_features:
        df_new = pd.concat([df_new] + all_new_features, axis=1)

    # Copy ulang untuk hindari fragmentasi
    df_new = df_new.copy()

    return df_new
