import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, root_mean_squared_error

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):

    # Replace absent with 0
    df = df.replace(['absent', 'Absent'], 0)

    # Fix result column
    df['Result'] = df['Result'].str.lower()
    df['Result'] = df['Result'].replace({'pass': 1, 'fail': 0})

    # Convert numeric columns and handle missing values
    numeric_cols = [
        "Age", "Quiz 1", "Quiz 2", "Quiz 3",
        "Assignment Score", "Midterm",
        "Final Exam", "Time Spent (hrs/week)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

    # Drop rows without Result column
    df = df.dropna(subset=['Result'])

    # Add new features
    df["avg_quiz"] = df[["Quiz 1", "Quiz 2", "Quiz 3"]].mean(axis=1)

    df["total_score"] = df[
        ["avg_quiz", "Assignment Score", "Midterm", "Final Exam"]
    ].mean(axis=1)

    return df

def logistic_regression(df):

    features = [
        "avg_quiz",
        "Assignment Score",
        "Midterm",
        "Time Spent (hrs/week)"
    ]

    X = df[features]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, precision


