import pandas as pd
import joblib
from app import predict_new_data, generate_recommendations

# Load models
log_model = joblib.load("logistic_model.pkl")
lin_model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_scaler = joblib.load("cluster_scaler.pkl")

# Load new data
df_new = pd.read_csv("new_students.csv")

# Run prediction
df_new = predict_new_data(
    df_new,
    log_model,
    lin_model,
    scaler,
    kmeans,
    cluster_scaler
)

df_new = generate_recommendations(df_new)

print(df_new[[
    "Predicted_PassFail",
    "Predicted_AverageScore",
    "Learner Category",
    "Recommendation"
]])
