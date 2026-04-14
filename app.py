import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
from sklearn.cluster import KMeans
import joblib


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])


    df["AverageScore"] = (
        df["MathScore"] +
        df["ReadingScore"] +
        df["WritingScore"]
    ) / 3

    df["ParentEduc"] = df["ParentEduc"].replace({
        "some high school": "high school",
    })

    df["WklyStudyHours"] = df["WklyStudyHours"].astype(str).str.strip()
    df["WklyStudyHours"] = df["WklyStudyHours"].str.replace(" ", "")

    df["WklyStudyHours"] = df["WklyStudyHours"].replace({
    "<5": 3,
    "5-10": 7,
    ">10": 12,
    "nan": np.nan 
})

    df["Result"] = (df["AverageScore"] >= 40).astype(int)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def get_features_targets(df):

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

    X = df[feature_cols]

    X = pd.get_dummies(X, drop_first=True)

    y_class = df["Result"]          
    y_reg = df["AverageScore"]      

    return X, y_class, y_reg 


def train_models(X, y_class, y_reg):

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train_class)

    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train_reg)

    y_pred_class = log_model.predict(X_test_scaled)
    y_pred_reg = lin_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    return log_model, lin_model, scaler, feature_columns, accuracy, precision, rmse



def train_kmeans(df):

    cluster_features = df[["AverageScore", "WklyStudyHours"]]

    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    return kmeans, cluster_scaler

def predict_new_data(df_new, log_model, lin_model, scaler, kmeans, cluster_scaler):

    df_new = preprocess_data(df_new)

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

    categories = {
        "EthnicGroup": ["group A", "group B", "group C", "group D", "group E"],
        "ParentEduc": ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
        "LunchType": ["standard", "free/reduced"],
        "TestPrep": ["none", "completed"],
        "ParentMaritalStatus": ["married", "single", "widowed", "divorced"],
        "PracticeSport": ["regularly", "sometimes", "never"],
        "IsFirstChild": ["yes", "no"],
        "TransportMeans": ["school_bus", "private"],
    }

    X_new = df_new[feature_cols]

    for col, cats in categories.items():
        if col in X_new.columns:
            X_new[col] = pd.Categorical(X_new[col], categories=cats)

    X_new = pd.get_dummies(X_new, drop_first=True)

    feature_columns = joblib.load("feature_columns.pkl")
    X_new = X_new.reindex(columns=feature_columns, fill_value=0)


    X_scaled = scaler.transform(X_new)

    df_new["Predicted_PassFail"] = log_model.predict(X_scaled)

    df_new["Predicted_AverageScore"] = lin_model.predict(X_scaled)

    cluster_features = df_new[["AverageScore", "WklyStudyHours"]]
    cluster_scaled = cluster_scaler.transform(cluster_features)

    df_new["Cluster"] = kmeans.predict(cluster_scaled)

    cluster_centers = kmeans.cluster_centers_[:, 0]
    sorted_indices = np.argsort(cluster_centers)

    cluster_labels = {
        sorted_indices[0]: "At Risk",
        sorted_indices[1]: "Average",
        sorted_indices[2]: "High Performer"
    }

    df_new["Learner Category"] = df_new["Cluster"].map(cluster_labels)

    return df_new


def generate_recommendations(df):

    recommendations = []

    for _, row in df.iterrows():

        if row["Learner Category"] == "At Risk":
            rec = "Revise fundamentals daily, increase weekly study hours, focus on weak subjects."

        elif row["Learner Category"] == "Average":
            rec = "Practice moderate to advanced problems and attempt weekly mock tests."

        else:
            rec = "Maintain performance and explore competitive or advanced-level materials."

        recommendations.append(rec)

    df["Recommendation"] = recommendations
    return df


if __name__ == "__main__":

    path = "./Data/raw/Student_Performance.csv"

    df = load_data(path)
    df = preprocess_data(df)

    X, y_class, y_reg = get_features_targets(df)

    log_model, lin_model, scaler, feature_columns, acc, prec, rmse = train_models(X, y_class, y_reg)

    kmeans, cluster_scaler = train_kmeans(df)

    print("Training Results")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("RMSE:", rmse)

    joblib.dump(log_model, "logistic_model.pkl")
    joblib.dump(lin_model, "linear_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(cluster_scaler, "cluster_scaler.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")

    print("Models Saved Successfully ✅")


def run_full_pipeline(file_path):
    df = load_data(file_path)
    df = preprocess_data(df)

    X, y_class, y_reg = get_features_targets(df)

    log_model = joblib.load("logistic_model.pkl")
    lin_model = joblib.load("linear_model.pkl")
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")

    df = predict_new_data(df, log_model, lin_model, scaler, kmeans, cluster_scaler)
    df = generate_recommendations(df)

    summary = {
    "AverageScore": round(df["AverageScore"].mean(), 2),
    "PredictedResult": int(df["Predicted_PassFail"].mode()[0]),
    "LearnerCategory": df["Learner Category"].mode()[0],
    "TopRecommendation": df["Recommendation"].mode()[0]
}

    return summary
