import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_churn.csv"


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

df = pd.read_csv(RAW_DATA_PATH)

print("\n" + "=" * 60)
print("1. RAW DATA")
print("=" * 60)
print(f"Shape: {df.shape}")


# ============================================================
# 3. REMOVE IDENTIFIER
# ============================================================

df = df.drop(columns=["customerID"])


# ============================================================
# 4. CONVERT TOTALCHARGES TO NUMERIC
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill the 11 blank TotalCharges values
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)


# ============================================================
# 5. ENCODE TARGET
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 6. SAVE PROCESSED DATA
# ============================================================

df.to_csv(PROCESSED_DATA_PATH, index=False)


# ============================================================
# 7. VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("2. CLEANED DATA")
print("=" * 60)

print(f"Shape: {df.shape}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTarget distribution:")
print(df["Churn"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("3. FILE SAVED")
print("=" * 60)
print(f"Saved to: {PROCESSED_DATA_PATH}")