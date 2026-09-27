import pandas as pd

# Path to the untouched raw dataset
DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("1. DATASET SHAPE")
print("=" * 60)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 60)
print("2. COLUMN NAMES")
print("=" * 60)
for column in df.columns:
    print(column)

print("\n" + "=" * 60)
print("3. DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n" + "=" * 60)
print("5. DUPLICATE ROWS")
print("=" * 60)
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n" + "=" * 60)
print("6. UNIQUE VALUES PER COLUMN")
print("=" * 60)
print(df.nunique())

print("\n" + "=" * 60)
print("7. TARGET DISTRIBUTION")
print("=" * 60)
print(df["Churn"].value_counts())
print("\nTarget percentage:")
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))

print("\n" + "=" * 60)
print("8. SAMPLE RECORDS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("9. NUMERICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("10. TOTALCHARGES QUALITY CHECK")
print("=" * 60)

# Find values that cannot be converted to numbers
total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")

invalid_total_charges = df.loc[
    total_charges_numeric.isna(),
    "TotalCharges"
]

print(f"Non-numeric / invalid values: {len(invalid_total_charges)}")

if len(invalid_total_charges) > 0:
    print("\nInvalid values found:")
    print(invalid_total_charges.value_counts())
else:
    print("No invalid values found.")

print("\nData type after numeric conversion:")
print(total_charges_numeric.dtype)

print("\n" + "=" * 60)
print("11. INVALID TOTALCHARGES RECORDS")
print("=" * 60)

invalid_records = df.loc[
    total_charges_numeric.isna(),
    ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
]

print(invalid_records.to_string(index=False))

print("\n" + "=" * 60)
print("12. CATEGORICAL VALUE CHECK")
print("=" * 60)

categorical_columns = df.select_dtypes(include=["str"]).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))