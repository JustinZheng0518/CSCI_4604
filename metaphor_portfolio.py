from pathlib import Path

import matplotlib

# Use a non-interactive backend so figures can be saved
# without opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


 
# File paths
 

DATA_PATH = Path("VUA_formatted_train.csv")

OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


 
# Load dataset
 

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1",
)


print("===== Dataset Overview =====")

print(
    f"Number of rows: {len(df)}"
)

print(
    f"Number of columns: {len(df.columns)}"
)

print("\nColumns:")

print(
    df.columns.tolist()
)


 
# Step 2.1 — Check class balance
 

class_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

class_percentages = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)


print("\n===== Class Counts =====")

print(
    f"Literal (0):      {class_counts.get(0, 0)}"
)

print(
    f"Metaphorical (1): {class_counts.get(1, 0)}"
)


print("\n===== Class Percentages =====")

print(
    f"Literal:      {class_percentages.get(0, 0):.2f}%"
)

print(
    f"Metaphorical: {class_percentages.get(1, 0):.2f}%"
)


 
# Create class-distribution chart
 

labels = [
    "Literal",
    "Metaphorical",
]

values = [
    class_counts.get(0, 0),
    class_counts.get(1, 0),
]


plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    labels,
    values,
)


plt.title(
    "Class Distribution in the VUA Dataset"
)

plt.xlabel(
    "Verb Usage"
)

plt.ylabel(
    "Number of Examples"
)


# Add counts above bars
for bar, value in zip(
    bars,
    values,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        str(value),

        ha="center",

        va="bottom",
    )


plt.tight_layout()


 
# Save chart
 

OUTPUT_PATH = (
    OUTPUT_DIR
    / "class_distribution.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print("\nSaved chart to:")

print(
    OUTPUT_PATH
)

 
# Step 2.2 — Create numerical features
 

print("\n===== Step 2.2: Feature Engineering =====")


# Sentence length:
# Number of word tokens in each sentence.
df["sentence_length"] = (
    df["sentence"]
    .fillna("")
    .str.split()
    .str.len()
)


# Verb length:
# Number of characters in the target verb.
df["verb_length"] = (
    df["verb"]
    .fillna("")
    .astype(str)
    .str.len()
)


# Relative verb position:
# 0 means near the beginning of the sentence,
# while 1 means near the end.
df["relative_position"] = (
    df["verb_idx"]
    / (df["sentence_length"] - 1).clip(lower=1)
)


# Verb frequency:
# Number of times the target verb appears in the dataset.
verb_frequency = df["verb"].value_counts()

df["verb_frequency"] = (
    df["verb"]
    .map(verb_frequency)
)


FEATURES = [
    "sentence_length",
    "verb_length",
    "relative_position",
    "verb_frequency",
]


 
# Show feature statistics
 

print("\n===== Feature Summary =====")

print(
    df[FEATURES].describe().round(3)
)


 
# Check potential outliers using the IQR rule
 

print("\n===== Potential Outlier Check =====")

for feature in FEATURES:

    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[feature] < lower_bound)
        | (df[feature] > upper_bound)
    ]

    print(f"\n{feature}:")
    print(f"  Q1: {q1:.3f}")
    print(f"  Q3: {q3:.3f}")
    print(f"  IQR: {iqr:.3f}")
    print(
        f"  Potential outliers: "
        f"{len(outliers)} "
        f"({len(outliers) / len(df) * 100:.2f}%)"
    )


 
# Feature histograms
 

histogram_settings = {
    "sentence_length": {
        "title": "Sentence Length Distribution",
        "xlabel": "Number of Words",
        "filename": "sentence_length_histogram.png",
    },

    "verb_length": {
        "title": "Target Verb Length Distribution",
        "xlabel": "Number of Characters",
        "filename": "verb_length_histogram.png",
    },

    "relative_position": {
        "title": "Relative Position of Target Verb",
        "xlabel": "Relative Position (0 = Start, 1 = End)",
        "filename": "relative_position_histogram.png",
    },

    "verb_frequency": {
        "title": "Target Verb Frequency Distribution",
        "xlabel": "Number of Occurrences",
        "filename": "verb_frequency_histogram.png",
    },
}


for feature, settings in histogram_settings.items():

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        df[feature],
        bins=30,
        edgecolor="black",
    )

    plt.title(
        settings["title"]
    )

    plt.xlabel(
        settings["xlabel"]
    )

    plt.ylabel(
        "Number of Examples"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / settings["filename"]
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


print(
    "\nStep 2.2 complete."
)

 
# Step 2.3 — Check Missing Values
 

print("\n===== Step 2.3: Missing Values =====")

missing_values = df.isnull().sum()

print("\nMissing values by column:")

print(missing_values)


total_missing = missing_values.sum()

print(
    f"\nTotal missing values: {total_missing}"
)


 
# Imputation
 

if total_missing == 0:

    print(
        "\nNo missing values were found. "
        "No imputation is required."
    )

else:

    print(
        "\nMissing values were found. "
        "Applying simple imputation."
    )

    # Text columns:
    # Replace missing text with "Unknown".
    text_columns = [
        "text_idx",
        "sentence_idx",
        "verb",
        "sentence",
    ]

    for column in text_columns:

        if df[column].isnull().any():

            df[column] = (
                df[column]
                .fillna("Unknown")
            )


    # Numerical columns:
    # Replace missing values with the median.
    numeric_columns = [
        "verb_idx",
        "sentence_length",
        "verb_length",
        "relative_position",
        "verb_frequency",
    ]

    for column in numeric_columns:

        if df[column].isnull().any():

            median_value = (
                df[column].median()
            )

            df[column] = (
                df[column]
                .fillna(median_value)
            )


    print(
        "\nMissing values after imputation:"
    )

    print(
        df.isnull().sum()
    )


print(
    "\nStep 2.3 complete."
)

 
# Step 2.4 — One-Hot Encode the Target Verb
 

print("\n===== Step 2.4: One-Hot Encoding =====")


# The target verb is a categorical feature.
# Verbs do not have a natural numerical order, so one-hot
# encoding is more appropriate than ordinal encoding.

unique_verbs = df["verb"].nunique()

print(
    f"\nNumber of unique verbs: {unique_verbs}"
)


 
# Keep the 50 most common verbs
 

TOP_N = 50

top_verbs = (
    df["verb"]
    .value_counts()
    .head(TOP_N)
    .index
)


# Less common verbs are grouped into one category called OTHER.
# This prevents the feature space from becoming unnecessarily large.

df["verb_category"] = df["verb"].where(
    df["verb"].isin(top_verbs),
    "OTHER"
)


print(
    f"Number of verb categories after grouping: "
    f"{df['verb_category'].nunique()}"
)


 
# Check how much of the dataset is covered by the top 50 verbs
 

top_verb_rows = (
    df["verb"]
    .isin(top_verbs)
    .sum()
)

top_verb_percentage = (
    top_verb_rows
    / len(df)
    * 100
)

print(
    f"Rows represented by top {TOP_N} verbs: "
    f"{top_verb_rows} "
    f"({top_verb_percentage:.2f}%)"
)


 
# Apply one-hot encoding
 

verb_encoded = pd.get_dummies(
    df["verb_category"],
    prefix="verb",
    dtype=int,
)


print(
    f"\nOne-hot encoded feature shape: "
    f"{verb_encoded.shape}"
)


print(
    "\nExample encoded columns:"
)

print(
    verb_encoded.columns[:10].tolist()
)


 
# Combine encoded verb features with numerical features
 

numerical_features = df[
    [
        "sentence_length",
        "verb_length",
        "relative_position",
        "verb_frequency",
    ]
]


X_engineered = pd.concat(
    [
        numerical_features,
        verb_encoded,
    ],
    axis=1,
)


print(
    f"\nFinal engineered feature matrix shape: "
    f"{X_engineered.shape}"
)


print(
    "\nFirst few feature columns:"
)

print(
    X_engineered.columns[:15].tolist()
)


print("\nStep 2.4 complete.")

 
# Step 2.5 — Check Correlated Numerical Features
 

print("\n===== Step 2.5: Correlation Check =====")


correlation_features = [
    "sentence_length",
    "verb_length",
    "relative_position",
    "verb_frequency",
]


# Calculate Pearson correlation between numerical features
correlation_matrix = (
    df[correlation_features]
    .corr()
)


print("\nCorrelation Matrix:")

print(
    correlation_matrix.round(3)
)


 
# Check for highly correlated feature pairs
 

HIGH_CORRELATION_THRESHOLD = 0.80

high_correlation_pairs = []


for i in range(len(correlation_features)):

    for j in range(i + 1, len(correlation_features)):

        feature_1 = correlation_features[i]
        feature_2 = correlation_features[j]

        correlation = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:

            high_correlation_pairs.append(
                (
                    feature_1,
                    feature_2,
                    correlation,
                )
            )


if len(high_correlation_pairs) == 0:

    print(
        "\nNo feature pairs have an absolute "
        "correlation of 0.80 or higher."
    )

    print(
        "No numerical feature needs to be removed "
        "because of high correlation."
    )

else:

    print(
        "\nHighly correlated feature pairs:"
    )

    for feature_1, feature_2, correlation in high_correlation_pairs:

        print(
            f"{feature_1} vs {feature_2}: "
            f"{correlation:.3f}"
        )


 
# Create correlation heatmap
 

fig, ax = plt.subplots(
    figsize=(8, 6)
)


heatmap = ax.imshow(
    correlation_matrix,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
)


# Axis labels
ax.set_xticks(
    range(len(correlation_features))
)

ax.set_yticks(
    range(len(correlation_features))
)


ax.set_xticklabels(
    correlation_features,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(
    correlation_features
)


# Add correlation values to each cell
for i in range(len(correlation_features)):

    for j in range(len(correlation_features)):

        value = correlation_matrix.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
        )


ax.set_title(
    "Correlation Between Numerical Features"
)


# Add color bar
fig.colorbar(
    heatmap,
    ax=ax,
    label="Pearson Correlation",
)


plt.tight_layout()


 
# Save correlation heatmap
 

CORRELATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "correlation_heatmap.png"
)


plt.savefig(
    CORRELATION_OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print(
    "\nSaved correlation heatmap to:"
)

print(
    CORRELATION_OUTPUT_PATH
)


print(
    "\nStep 2.5 complete."
)