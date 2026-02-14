import pandas as pd
import numpy as np

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

    df["avg_quiz"] = df[["Quiz 1", "Quiz 2", "Quiz 3"]].mean(axis=1)

    df["total_score"] = df[
        ["avg_quiz", "Assignment Score", "Midterm", "Final Exam"]
    ].mean(axis=1)

    return df