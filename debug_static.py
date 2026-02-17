
import pandas as pd
import numpy as np
import joblib
from app import preprocess_data, predict_new_data

# Load models and other necessary files
try:
    log_model = joblib.load("logistic_model.pkl")
    lin_model = joblib.load("linear_model.pkl")
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    print("Model files not found. Please run 'app.py' to train and save models first.")
    exit()

# Create two different input data points to see if predictions change
input_data_1 = {
    "Gender": ["female"],
    "EthnicGroup": ["group A"],
    "ParentEduc": ["bachelor's degree"],
    "LunchType": ["standard"],
    "TestPrep": ["none"],
    "ParentMaritalStatus": ["married"],
    "PracticeSport": ["regularly"],
    "IsFirstChild": ["yes"],
    "NrSiblings": [1],
    "TransportMeans": ["school_bus"],
    "WklyStudyHours": ["> 10"],
    "MathScore": [90],
    "ReadingScore": [95],
    "WritingScore": [93]
}

input_data_2 = {
    "Gender": ["male"],
    "EthnicGroup": ["group E"], # Different ethnic group (should change hot encoding)
    "ParentEduc": ["some high school"], # Lower education
    "LunchType": ["free/reduced"], # Different lunch
    "TestPrep": ["none"],
    "ParentMaritalStatus": ["single"],
    "PracticeSport": ["never"],
    "IsFirstChild": ["no"],
    "NrSiblings": [5],
    "TransportMeans": ["private"],
    "WklyStudyHours": ["< 5"], # Low study hours
    "MathScore": [40], # Low scores
    "ReadingScore": [45],
    "WritingScore": [43]
}

df_1 = pd.DataFrame(input_data_1)
df_2 = pd.DataFrame(input_data_2)

print("--- Prediction 1 (High performing inputs) ---")
try:
    results_1 = predict_new_data(df_1.copy(), log_model, lin_model, scaler, kmeans, cluster_scaler)
    print("Predicted Pass/Fail:", results_1["Predicted_PassFail"][0])
    print("Predicted Average Score:", results_1["Predicted_AverageScore"][0])
    print("Learner Category:", results_1["Learner Category"][0])
except Exception as e:
    print(f"Error in prediction 1: {e}")

print("\n--- Prediction 2 (Low performing inputs) ---")
try:
    results_2 = predict_new_data(df_2.copy(), log_model, lin_model, scaler, kmeans, cluster_scaler)
    print("Predicted Pass/Fail:", results_2["Predicted_PassFail"][0])
    print("Predicted Average Score:", results_2["Predicted_AverageScore"][0])
    print("Learner Category:", results_2["Learner Category"][0])
except Exception as e:
    print(f"Error in prediction 2: {e}")

# Check feature columns alignment
print("\n--- Feature Alignment Check ---")
df_new_1 = preprocess_data(df_1.copy())
feature_cols = [
    "EthnicGroup",
    "ParentEduc",
    "LunchType",
    "TestPrep",
    "ParentMaritalStatus",
    "PracticeSport",
    "IsFirstChild",
    "NrSiblings",
    "TransportMeans",
    "WklyStudyHours"
]
X_new_1 = df_new_1[feature_cols]
X_new_1 = pd.get_dummies(X_new_1, drop_first=True)
print("Columns before reindex:", X_new_1.columns.tolist())
X_new_1 = X_new_1.reindex(columns=feature_columns, fill_value=0) # WAIT: feature_columns loaded from pkl, not the list above
print("Columns in pkl:", feature_columns)
print("Columns after reindex:", X_new_1.columns.tolist())
print("Non-zero values in reindexed X_new_1:", X_new_1.loc[0][X_new_1.loc[0] != 0])
