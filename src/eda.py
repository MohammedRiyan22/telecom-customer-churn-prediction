import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

DATA_PATH = "data/processed/cleaned_churn.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. NUMERICAL DATA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("1. NUMERICAL FEATURE SUMMARY")
print("=" * 60)

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

print(df[numerical_columns].describe())


# ============================================================
# 3. CHURN RATE BY NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("2. CHURN RATE BY NUMERICAL FEATURES")
print("=" * 60)

for column in ["SeniorCitizen", "tenure"]:
    print(f"\n--- {column} ---")
    print(
        df.groupby(column)["Churn"]
        .mean()
        .mul(100)
        .round(2)
    )


# ============================================================
# 4. CHURN VS CONTINUOUS NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("3. NUMERICAL FEATURES BY CHURN")
print("=" * 60)

print(
    df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]]
    .agg(["mean", "median"])
    .round(2)
)


# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("4. NUMERICAL CORRELATION MATRIX")
print("=" * 60)

correlation = df[
    ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
].corr()

print(correlation.round(3))


# ============================================================
# 6. VISUALIZATIONS
# ============================================================

figures = [
    ("tenure", "Tenure Distribution", "Tenure (months)"),
    ("MonthlyCharges", "Monthly Charges Distribution", "Monthly Charges"),
    ("TotalCharges", "Total Charges Distribution", "Total Charges")
]

for column, title, xlabel in figures:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, hue="Churn", kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. BOXPLOTS
# ============================================================

for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y=column)
    plt.title(f"{column} by Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

    # ============================================================
# 8. CATEGORICAL FEATURES VS CHURN
# ============================================================

print("\n" + "=" * 60)
print("5. CATEGORICAL FEATURES - CHURN RATES")
print("=" * 60)

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

for column in categorical_columns:
    print(f"\n--- {column} ---")

    churn_rate = (
        df.groupby(column)["Churn"]
        .agg(["count", "mean"])
        .assign(churn_rate=lambda x: x["mean"] * 100)
        [["count", "churn_rate"]]
        .round(2)
    )

    print(churn_rate)