import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Housing.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

# -----------------------------
# Convert categorical columns
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("--------------------")
print("MAE :", mae)
print("MSE :", mse)
print("R² Score :", r2)

# -----------------------------
# Coefficients
# -----------------------------
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients")
print(coef)

# -----------------------------
# Simple Regression Plot
# -----------------------------
# Only if 'area' exists

if "area" in X.columns:

    simple_X = df[["area"]]
    simple_y = df["price"]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        simple_X,
        simple_y,
        test_size=0.2,
        random_state=42
    )

    simple_model = LinearRegression()

    simple_model.fit(X_train_s, y_train_s)

    y_line = simple_model.predict(simple_X)

    plt.figure(figsize=(8,6))
    plt.scatter(simple_X, simple_y, color="blue")
    plt.plot(simple_X, y_line, color="red")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Simple Linear Regression")

    plt.savefig("regression_plot.png")
    plt.show()

print("\nRegression plot saved successfully.")