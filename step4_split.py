import pandas as pd
from sklearn.model_selection import train_test_split


# Load the single CSV dataset
df = pd.read_csv(
    "VUA_formatted_train.csv",
    encoding="latin1"
)


# ------------------------------------------------------------
# Train/Test Split
# ------------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


print("===== Train/Test Split =====")

print(f"Full dataset: {len(df)}")
print(f"Training set: {len(train_df)}")
print(f"Testing set:  {len(test_df)}")


print("\n===== Training Class Distribution =====")

print(
    train_df["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print("\n===== Testing Class Distribution =====")

print(
    test_df["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# Save the split for later modeling
train_df.to_csv(
    "train_split.csv",
    index=False
)

test_df.to_csv(
    "test_split.csv",
    index=False
)


print("\nSaved:")
print("train_split.csv")
print("test_split.csv")