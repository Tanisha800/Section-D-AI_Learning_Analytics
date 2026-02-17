
import pandas as pd
import numpy as np
from app import load_data, preprocess_data

try:
    print("Loading data...")
    raw_df = load_data("./Data/raw/Student_Performance.csv")
    print("Data loaded. Shape:", raw_df.shape)
    
    print("\nRunning preprocess_data...")
    processed_df = preprocess_data(raw_df)
    
    print("\nChecking for NaNs in critical columns:")
    print("AverageScore NaNs:", processed_df["AverageScore"].isna().sum())
    print("WklyStudyHours NaNs:", processed_df["WklyStudyHours"].isna().sum())
    
    print("\nChecking dtypes:")
    print(processed_df[["AverageScore", "WklyStudyHours"]].dtypes)
    
    print("\nUnique values in WklyStudyHours:")
    print(processed_df["WklyStudyHours"].unique())
    
    # Check if any "nan" strings exist
    if processed_df["WklyStudyHours"].dtype == 'object':
        print("\nChecking for string 'nan':")
        print((processed_df["WklyStudyHours"].astype(str) == "nan").sum())

except Exception as e:
    print(f"\nError occurred: {e}")
