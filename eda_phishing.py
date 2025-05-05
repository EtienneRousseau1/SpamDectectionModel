"""
eda_phishing.py

This script performs exploratory data analysis (EDA) on a phishing email dataset.
The dataset contains various features extracted from emails, such as the number of words,
unique words, links, and other characteristics. The goal is to understand the data distribution,
identify patterns, and visualize relationships between features and the target label (safe vs phishing).

Outputs:
- Summary statistics and missing value analysis
- Class distribution plot
- Histograms and boxplots for numerical features
- Correlation heatmap for numerical features and the target label
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_theme(style="whitegrid")

DATA_PATH = Path("data/email_phishing_data.csv")
FIGURE_PATH = Path("outputs/figures")
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# --- Overview ---
print(f"Shape: {df.shape}")
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# --- Class Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label")
plt.title("Phishing vs Safe Emails")
plt.xticks(ticks=[0, 1], labels=["Safe", "Phishing"])
plt.xlabel("Email Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "class_distribution.png")
plt.close()

# --- Numerical Features ---
features = [
    "num_words", "num_unique_words", "num_stopwords", "num_links",
    "num_unique_domains", "num_email_addresses", "num_spelling_errors", "num_urgent_keywords"
]

# --- Histograms ---
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=feature, hue="label", bins=50, kde=True, element="step", palette="Set2")
    plt.title(f"{feature} Distribution by Label")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(title="Label", labels=["Safe", "Phishing"])
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / f"hist_{feature}.png")
    plt.close()

# --- Boxplots ---
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="label", y=feature, palette="pastel")
    plt.title(f"{feature} by Email Type")
    plt.xticks(ticks=[0, 1], labels=["Safe", "Phishing"])
    plt.xlabel("Email Type")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / f"boxplot_{feature}.png")
    plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr = df[features + ["label"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "correlation_heatmap.png")
plt.close()
