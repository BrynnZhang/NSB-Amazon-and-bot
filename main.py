import pandas as pd
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# import using pandas
df0 = pd.read_csv("amazon_products_sales_data_uncleaned.csv")

# dispaly the first 5 rows of the dataframe
print(df0.head())

# display the dataframe info and summary statistics
print(df0.info())
print("\n")
print("The statistics summary of the dataframe is: ", df0.describe())

# check for missing values and duplicates
print("Missing values:", "\n", df0.isnull().sum())
print("\n")
print("numbers of duplicates: ", df0.duplicated().sum())


# Read the CSV with error handling
plf = pl.read_csv(
    "amazon_products_sales_data_uncleaned.csv",
    ignore_errors=True,  # Refering Copilot, Skip problematic rows
)

# Display the first 5 rows
print(plf.head())

# Show the statistical summary
print(plf.describe())
print("\n")

# Check for missing values and duplicates
print("Missing values:", "\n", plf.null_count())
# check the missing value in each column
for i in range(plf.width):
    print(
        f"Missing values in column {i + 1}:",
        plf.select(pl.col(plf.columns[i]).is_null().sum()),
    )

print("\n")
print("Number of duplicates:", plf.is_duplicated().sum())

# manipulate on the cloned dataframe
df = df0.copy()

# show the head/first row
print(df.head(0))

# remove "out of 5 stars" from the ratings column
df["rating"] = df["rating"].str.replace(" out of 5 stars", "")
# convert the ratings column to numeric
df["rating"] = pd.to_numeric(df["rating"])
print(df["rating"])

# remove "," separator in number_of_reviews and change str to float
df["number_of_reviews"] = df["number_of_reviews"].str.replace(",", "")
df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"])
print(df["number_of_reviews"])

# 1.replace cells without "$" to NA,
# and remove "basic variant price: $" part in price_on_variant
df["price_on_variant"] = df["price_on_variant"].where(
    df["price_on_variant"].str.contains(r"\$", na=False), np.nan
)
df["price_on_variant"] = df["price_on_variant"].str.replace(
    "basic variant price: $", ""
)
# print(df['price_on_variant'])

# 2. apply cleaned price_on_variant to missing current_discounted_price
df["current/discounted_price"] = df.apply(
    lambda row: (
        row["price_on_variant"]
        if pd.isnull(row["current/discounted_price"])
        else row["current/discounted_price"]
    ),
    axis=1,
)

# 3.remove all "," separator in current/discounted_price
# and convert to numeric
df["current/discounted_price"] = df[
    "current/discounted_price"].str.replace(",", "")
# print(df['current/discounted_price'])
df["current/discounted_price"] = pd.to_numeric(df["current/discounted_price"])
print(df["current/discounted_price"])
# 4. check missing value (12941 before cleaning)
print(
    "Missing values after cleaning:",
    "\n",
    df["current/discounted_price"].isnull().sum(),
)

# remove "$" and "," in listed_price
df["listed_price"] = df["listed_price"].str.replace("$", "")
df["listed_price"] = df["listed_price"].str.replace(",", "")

# replace "No Discount" with current/discounted_price and convert to numeric
df["listed_price"] = df.apply(
    lambda row: (
        row["current/discounted_price"]
        if row["listed_price"] == "No Discount"
        else row["listed_price"]
    ),
    axis=1,
)
df["listed_price"] = pd.to_numeric(df["listed_price"])

print(df["listed_price"])


# check missing value, should align with
# current/discounted_price(2062 after cleaning)
print("Missing values after cleaning:",
      "\n", df["listed_price"].isnull().sum())

# keep only title, rating, number_of_reviews,
# current/discounted_price, listed_price columns
df_filtered = df[
    ["title", "rating", "number_of_reviews",
     "current/discounted_price", "listed_price"]
]
# drop rows with missing values
df_cleaned = df_filtered.dropna()
print(df_cleaned.head())
# export cleaned dataframe to a new csv file
df_cleaned.to_csv("amazon_cleaned.csv", index=False)
# Use groupby() to group the rating and get the corresponding
# average number_of_review, current/discounted_price and listed_price.
grouped_df_cleaned = (
    df_cleaned.groupby("rating")
    .agg(
        {
            "number_of_reviews": "mean",
            "current/discounted_price": "mean",
            "listed_price": "mean",
        }
    )
    .reset_index()
)

print(grouped_df_cleaned)

# Feature engineering: discount rate
df_cleaned["discount_rate"] = (
    df_cleaned["listed_price"] - df_cleaned["current/discounted_price"]
) / df["listed_price"]

# Define features (X) and target (y)
features = ["rating", "number_of_reviews", "listed_price", "discount_rate"]
x = df_cleaned[features]
y = df_cleaned["current/discounted_price"]

# Train-test split: 20% data for testing, fixed seed = 42 for reproducibility
x_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -----------------------------
# Random Forest Regressor
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_preds = rf_model.predict(X_test)

# -----------------------------
# XGBoost Regressor
# -----------------------------
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(x_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluation RMSE(the lower the better)
# and R²(the closer to 1 the better)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)
print(f"\nRandom Forest Regressor\nRMSE: {rf_rmse:.2f}\nR²: {rf_r2:.4f}")
print(f"\nXGBoost Regressor\nRMSE: {xgb_rmse:.2f}\nR²: {xgb_r2:.4f}")


# feature importance visualization
def plot_feature_importance(model, model_name):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title(f"{model_name} - Feature Importance")
    plt.xlabel("Importance")
    plt.show()


plot_feature_importance(rf_model, "Random Forest")
plot_feature_importance(xgb_model, "XGBoost")

# Random forest model data visualization using matplotlib and seaborn
# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, alpha=0.5, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.show()
