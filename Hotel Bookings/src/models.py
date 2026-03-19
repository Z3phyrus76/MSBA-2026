"""
models.py

Purpose:
Builds and evaluates a logistic regression model to predict cancellations.

What this file does:
- Splits data into train and test sets
- Runs logistic regression
- Calculates accuracy and ROC AUC
- Returns evaluation results

Why it exists:
Keeps modeling separate from cleaning and feature engineering.
Makes the project reusable and structured.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def run_logistic(df: pd.DataFrame, feature_cols: list[str], seed: int = 206) -> dict:
    """
    Runs a logistic regression to predict cancellations.

    Inputs:
    df cleaned dataframe
    feature_cols list of column names to use as features
    seed random seed for reproducibility

    Returns:
    dict with accuracy roc_auc classification report and rows_used
    """
    df = df.copy()

    # Target
    if "iscanceled" not in df.columns:
        raise KeyError("Target column iscanceled not found. Did you run clean_hotel_data")

    # Check features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"These feature columns are missing from df: {missing}")

    # Keep only needed columns
    keep_cols = feature_cols + ["iscanceled"]
    df_model = df[keep_cols].dropna()
    rows_used = len(df_model)

    X = df_model[feature_cols].copy()
    y = df_model["iscanceled"].astype(int)

    # Convert categoricals to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    
    #Scaler fitted
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000)

    #Scaled data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Predicted and trained on scaled data
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # ROC AUC can fail if y_test has only one class
    roc_auc = None
    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(y_test, probs)

    return {
        "rows_used": rows_used,
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc,
        "report": classification_report(y_test, preds),
    }