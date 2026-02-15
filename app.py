import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
from sklearn.cluster import KMeans


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    df = df.drop(columns=["Unnamed: 0"])

    df["AverageScore"] = (
        df["MathScore"] +
        df["ReadingScore"] +
        df["WritingScore"]
    ) / 3

    df["ParentEduc"] = df["ParentEduc"].replace({
        "some high school": "high school",
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

    y_logistic = df["Result"]          
    y_linear = df["AverageScore"]      

    return X, y_logistic, y_linear


def logistic_regression(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, precision


def linear_regression(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse


def k_means(df):

    cluster_features = df[["AverageScore", "WklyStudyHours"]]

    cluster_features = pd.get_dummies(cluster_features, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_means = df.groupby("Cluster")["AverageScore"].mean()
    sorted_clusters = cluster_means.sort_values().index

    cluster_labels = {
        sorted_clusters[0]: "At Risk",
        sorted_clusters[1]: "Average",
        sorted_clusters[2]: "High Performer"
    }

    df["Learner Category"] = df["Cluster"].map(cluster_labels)

    return df


if __name__ == "__main__":

    path = "./Data/raw/Student_Performance.csv"

    df = load_data(path)
    df = preprocess_data(df)

    X, y_logistic, y_linear = get_features_targets(df)

    accuracy, precision = logistic_regression(X, y_logistic)
    rmse = linear_regression(X, y_linear)

    df = k_means(df)

    print("Results:")
    print(f"Logistic Regression Accuracy: {accuracy}")
    print(f"Logistic Regression Precision: {precision}")
    print(f"Linear Regression RMSE: {rmse}")

    print("\nCluster Distribution:")
    print(df["Learner Category"].value_counts())
