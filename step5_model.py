from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


 
# Load train/test data
 

train = pd.read_csv("train_split.csv")
test = pd.read_csv("test_split.csv")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


 
# Feature engineering
 

def add_features(df, verb_counts):

    df = df.copy()

    df["sentence_length"] = (
        df["sentence"].str.split().str.len()
    )

    df["verb_length"] = (
        df["verb"].str.len()
    )

    df["relative_position"] = (
        df["verb_idx"]
        / (df["sentence_length"] - 1).clip(lower=1)
    )

    df["verb_frequency"] = (
        df["verb"]
        .map(verb_counts)
        .fillna(0)
    )

    return df


# Frequency is learned ONLY from training data
verb_counts = train["verb"].value_counts()

train = add_features(train, verb_counts)
test = add_features(test, verb_counts)


# Keep only the 50 most common training verbs
top_verbs = (
    train["verb"]
    .value_counts()
    .head(50)
    .index
)

train["verb_category"] = train["verb"].where(
    train["verb"].isin(top_verbs),
    "OTHER"
)

test["verb_category"] = test["verb"].where(
    test["verb"].isin(top_verbs),
    "OTHER"
)


 
# Select features and target
 

features = [
    "sentence_length",
    "verb_length",
    "relative_position",
    "verb_frequency",
    "verb_category",
]

X_train = train[features]
X_test = test[features]

y_train = train["label"]
y_test = test["label"]



# Preprocessing


numeric_features = [
    "sentence_length",
    "verb_length",
    "relative_position",
    "verb_frequency",
]

categorical_features = [
    "verb_category"
]


preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "verb",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)



# Step 5-6 — Classification model and training


model = Pipeline(
    [
        ("preprocessing", preprocessor),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


print("Training Logistic Regression...")

model.fit(
    X_train,
    y_train,
)


# Step 7 — Test model


predictions = model.predict(
    X_test
)

# Step 8 — Model measurements


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


print("\n===== Test Results =====")

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")


# Confusion Matrix


ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=[
        "Literal",
        "Metaphorical",
    ],
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
)

plt.close()


print(
    "\nSaved: outputs/confusion_matrix.png"
)