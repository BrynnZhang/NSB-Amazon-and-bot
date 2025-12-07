#You’ll write basic unit tests for your data analysis functions and set up a 
# development environment using either Dev Container or Docker. 
# It should be in the same Github Repository you created last week.

#Test Coverage: Includes meaningful unit and system tests 
# that validate core functions such as 
# 1. data loading, 2. filtering, 3. grouping, 4. preprocessing, and 5. machine learning model behavior, 
# with clear structure and edge case handling. Make sure all tests pass.

#Dev Environment Setup: A fully functional Dev Container 
# or Docker setup (3 bonus points for both Docker and Dev Container), with requirement file, 
# devcontainer.json/Docker files, ensuring all dependencies correctly installed 
# and clear instructions for building, running, and using the environment.

import pandas as pd
import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# 1. Data Loading Tests
# ---------------------------
def test_csv_loads_correctly():
    df = pd.read_csv("amazon_cleaned.csv")
    assert not df.empty, "CSV file should not be empty"

    expected_cols = [
        "title", "rating", "number_of_reviews",
        "current/discounted_price", "listed_price"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

# ---------------------------
# 2. Filtering / Cleaning Tests
# ---------------------------
def test_no_missing_values_after_cleaning():
    df = pd.read_csv("amazon_cleaned.csv")
    assert df.notnull().all().all(), "There should be no NaN values after cleaning"

# ---------------------------
# 3. Grouping Tests
# ---------------------------
def test_grouping_by_rating():
    df = pd.read_csv("amazon_cleaned.csv")
    grouped = df.groupby("rating").agg({
        "number_of_reviews": "mean",
        "current/discounted_price": "mean",
        "listed_price": "mean"
    }).reset_index()

    assert grouped["rating"].is_unique, "Ratings should be unique in grouped dataframe"
    assert not grouped.empty, "Grouped dataframe should not be empty"

# ---------------------------
# 4. Preprocessing Tests
# ---------------------------
def test_discount_rate_calculation():
    df = pd.read_csv("amazon_cleaned.csv")
    df["discount_rate"] = (
        df["listed_price"] - df["current/discounted_price"]
    ) / df["listed_price"]

    assert ((df["discount_rate"] >= 0) & (df["discount_rate"] <= 1)).all(), \
        "Discount rate should always be between 0 and 1"

# ---------------------------
# 5. Machine Learning Model Tests
# ---------------------------
def test_models_train_and_predict():
    df = pd.read_csv("amazon_cleaned.csv")
    df["discount_rate"] = (
        df["listed_price"] - df["current/discounted_price"]
    ) / df["listed_price"]

    features = ["rating", "number_of_reviews", "listed_price", "discount_rate"]
    X = df[features]
    y = df["current/discounted_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
    rf_score = r2_score(y_test, rf.predict(X_test))

    # XGBoost
    xgb = XGBRegressor(n_estimators=10, learning_rate=0.1, random_state=42).fit(X_train, y_train)
    xgb_score = r2_score(y_test, xgb.predict(X_test))

    assert rf_score > 0, "RandomForest should explain some variance (R² > 0)"
    assert xgb_score > 0, "XGBoost should explain some variance (R² > 0)"

# ---------------------------
# 6. A/B Test Comparison Between Models
# ---------------------------
def test_ab_testing_models():
    """
    Simple A/B test: Compare RandomForest vs XGBoost on RMSE.
    Whichever has lower RMSE 'wins'. The test checks that there is
    a statistically meaningful difference (not identical).
    """
    df = pd.read_csv("amazon_cleaned.csv")
    df["discount_rate"] = (
        df["listed_price"] - df["current/discounted_price"]
    ) / df["listed_price"]

    features = ["rating", "number_of_reviews", "listed_price", "discount_rate"]
    X = df[features]
    y = df["current/discounted_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train both models
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42).fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

    print(f"RF RMSE: {rf_rmse:.4f}, XGB RMSE: {xgb_rmse:.4f}")

    # Ensure one model is strictly better
    assert rf_rmse != xgb_rmse, "A/B test inconclusive: models performed identically"
