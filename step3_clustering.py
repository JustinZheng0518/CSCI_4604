from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load data
df = pd.read_csv(
    "VUA_formatted_train.csv",
    encoding="latin1"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Create features
# ------------------------------------------------------------

df["sentence_length"] = (
    df["sentence"]
    .str.split()
    .str.len()
)

df["verb_length"] = (
    df["verb"]
    .str.len()
)

df["relative_position"] = (
    df["verb_idx"]
    / (df["sentence_length"] - 1).clip(lower=1)
)

verb_counts = df["verb"].value_counts()

df["verb_frequency"] = (
    df["verb"]
    .map(verb_counts)
)


features = [
    "sentence_length",
    "verb_length",
    "relative_position",
    "verb_frequency",
]


# ------------------------------------------------------------
# Scale features
# ------------------------------------------------------------

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# KMeans clustering
# IMPORTANT: label (y) is NOT used here
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

df["cluster"] = kmeans.fit_predict(
    X_scaled
)


print("===== Cluster Sizes =====")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Cluster summary
# ------------------------------------------------------------

summary = (
    df.groupby("cluster")[features]
    .mean()
    .round(2)
)

print("\n===== Cluster Summary =====")

print(summary)


# ------------------------------------------------------------
# PCA visualization
# ------------------------------------------------------------

pca = PCA(n_components=2)

X_2d = pca.fit_transform(
    X_scaled
)


plt.figure(figsize=(8, 6))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["cluster"],
    s=10,
    alpha=0.5,
    cmap="viridis",
)

plt.title("KMeans Clusters of Verb Usage")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cluster_visualization.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Examine y only AFTER clustering
# ------------------------------------------------------------

metaphor_rate = (
    df.groupby("cluster")["label"]
    .mean()
    * 100
)

print("\n===== Metaphorical Percentage by Cluster =====")

print(
    metaphor_rate.round(2)
)


print(
    "\nSaved: outputs/cluster_visualization.png"
)