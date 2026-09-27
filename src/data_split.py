import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/cleaned_churn.csv"

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("1. CLEANED DATA")
print("=" * 60)
print(f"Shape: {df.shape}")


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. RECREATE TRAIN AND TEST DATAFRAMES
# ============================================================

train_df = X_train.copy()
train_df["Churn"] = y_train

test_df = X_test.copy()
test_df["Churn"] = y_test


# ============================================================
# 6. BALANCE TRAINING DATA ONLY
# ============================================================

majority_class = train_df[train_df["Churn"] == 0]
minority_class = train_df[train_df["Churn"] == 1]

minority_upsampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),
    random_state=42
)

train_df = pd.concat(
    [majority_class, minority_upsampled],
    axis=0
)

train_df = train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 7. SAVE TRAIN AND TEST DATA
# ============================================================

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)


# ============================================================
# 8. VERIFY SPLIT
# ============================================================

print("\n" + "=" * 60)
print("2. SPLIT RESULTS")
print("=" * 60)

print(f"Training rows: {len(train_df)}")
print(f"Test rows    : {len(test_df)}")

print("\nTraining shape:")
print(train_df.shape)

print("\nTest shape:")
print(test_df.shape)


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("3. TARGET DISTRIBUTION")
print("=" * 60)

print("\nBalanced training target:")
print(train_df["Churn"].value_counts())

print("\nBalanced training target percentage:")
print(
    train_df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTest target:")
print(test_df["Churn"].value_counts())

print("\nTest target percentage:")
print(
    test_df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 10. VERIFY FEATURE CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("4. FEATURE CONSISTENCY CHECK")
print("=" * 60)

train_features = list(
    train_df.drop(columns=["Churn"]).columns
)

test_features = list(
    test_df.drop(columns=["Churn"]).columns
)

print(f"Same feature columns: {train_features == test_features}")

print("\nTraining features:")
print(train_features)


# ============================================================
# 11. FILES SAVED
# ============================================================

print("\n" + "=" * 60)
print("5. FILES SAVED")
print("=" * 60)

print(f"Training data: {TRAIN_PATH}")
print(f"Test data    : {TEST_PATH}")