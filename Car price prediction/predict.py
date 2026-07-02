# ============================================================
# Car Price Prediction using Machine Learning
# File Name: predict.py
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Create Output Folder
# ============================================================

os.makedirs("output", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("car data.csv")

print("=" * 60)
print("First 5 Rows")
print("=" * 60)
print(df.head())

print("\nDataset Information")
print("=" * 60)
print(df.info())

print("\nMissing Values")
print("=" * 60)
print(df.isnull().sum())

print("\nDataset Columns")
print("=" * 60)
print(df.columns.tolist())

# ============================================================
# Encode Categorical Columns
# ============================================================

fuel_encoder = LabelEncoder()
selling_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

df["Fuel_Type"] = fuel_encoder.fit_transform(df["Fuel_Type"])
df["Selling_type"] = selling_encoder.fit_transform(df["Selling_type"])
df["Transmission"] = transmission_encoder.fit_transform(df["Transmission"])

# ============================================================
# Feature Engineering
# ============================================================

CURRENT_YEAR = 2025

df["Car_Age"] = CURRENT_YEAR - df["Year"]

df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# ============================================================
# Correlation Heatmap
# ============================================================

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=300)
plt.show()
plt.close()

# ============================================================
# Prepare Data
# ============================================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Train Model
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE       : {mae:.2f}")
print(f"MSE       : {mse:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"R2 Score  : {r2:.2f}")

# ============================================================
# Actual vs Predicted Plot
# ============================================================

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/actual_vs_predicted.png", dpi=300)
plt.show()
plt.close()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("Feature Importance")
print("=" * 60)
print(importance)

plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=300)
plt.show()
plt.close()

# ============================================================
# Save Feature Importance CSV
# ============================================================

importance.to_csv(
    "output/feature_importance.csv",
    index=False
)

# ============================================================
# Predict New Car Price
# ============================================================

print("\n")
print("=" * 60)
print("Predict Selling Price")
print("=" * 60)

present_price = float(input("Present Price (Lakhs): "))
driven_kms = int(input("Driven Kilometers: "))

print("\nFuel Type")
print("0 = CNG")
print("1 = Diesel")
print("2 = Petrol")
fuel_type = int(input("Enter Fuel Type: "))

print("\nSelling Type")
print("0 = Dealer")
print("1 = Individual")
selling_type = int(input("Enter Selling Type: "))

print("\nTransmission")
print("0 = Automatic")
print("1 = Manual")
transmission = int(input("Enter Transmission: "))

owner = int(input("Previous Owners: "))
car_age = int(input("Car Age (Years): "))

sample = pd.DataFrame([{
    "Present_Price": present_price,
    "Driven_kms": driven_kms,
    "Fuel_Type": fuel_type,
    "Selling_type": selling_type,
    "Transmission": transmission,
    "Owner": owner,
    "Car_Age": car_age
}])

predicted_price = model.predict(sample)

print("\n" + "=" * 60)
print(f"Predicted Selling Price : {predicted_price[0]:.2f} Lakhs")
print("=" * 60)

# ============================================================
# Finished
# ============================================================

print("\nProject completed successfully!")
print("All graphs have been saved in the 'output' folder.")