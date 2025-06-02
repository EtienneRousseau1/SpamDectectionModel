import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="Set2")

# Load the balanced dataset
print("Loading balanced dataset...")
df = pd.read_csv('data/processed_balanced.csv')

# Show class distribution
print("\nClass distribution:")
print(df['label'].value_counts())

# Remove outliers (outside 1st and 99th percentiles) for all numeric features except 'label'
selected_features = [
    'num_words',
    'num_unique_words',
    'num_stopwords',
    'num_links',
    'num_unique_domains',
    'num_email_addresses',
    'num_spelling_errors',
    'num_urgent_keywords'
]
print("\nRemoving outliers (outside 1st and 99th percentiles) from selected features...")
filtered_df = df.copy()
for feature in selected_features:
    lower = filtered_df[feature].quantile(0.01)
    upper = filtered_df[feature].quantile(0.99)
    filtered_df = filtered_df[(filtered_df[feature] >= lower) & (filtered_df[feature] <= upper)]

print(f"Rows before outlier removal: {df.shape[0]}")
print(f"Rows after outlier removal: {filtered_df.shape[0]}")
df = filtered_df

# Create a directory for plots if it doesn't exist
if not os.path.exists('/charts'):
    os.makedirs('/charts')

# Feature distributions by class (boxplots, improved, specified features)
n_features = len(selected_features)
n_cols = 2
n_rows = (n_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for idx, feature in enumerate(selected_features):
    sns.boxplot(
        data=df, x='label', y=feature, palette='Set2', ax=axes[idx], showfliers=False, width=0.6, linewidth=2
    )
    axes[idx].set_title(f'{feature.replace("_", " ").title()} by Class', fontsize=17, pad=14)
    axes[idx].set_xlabel('Label (0: Legitimate, 1: Phishing)', fontsize=14)
    axes[idx].set_ylabel(feature.replace("_", " ").title(), fontsize=14)
    axes[idx].grid(True, linestyle='--', alpha=0.7)
    axes[idx].tick_params(axis='both', labelsize=12)
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])  # Remove unused subplots

plt.suptitle('Feature Distributions by Class (Balanced Dataset)', fontsize=22, y=1.03)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('outputs/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nBoxplots saved to outputs/plots/feature_distributions.png")

# 1. Distribution of target variable (improved)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='label', palette='Set2', edgecolor='black')
plt.title('Distribution of Email Types', fontsize=18, pad=20)
plt.xlabel('Label (0: Legitimate, 1: Phishing)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('outputs/plots/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation analysis (improved)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            annot_kws={"size": 10}, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=18, pad=20)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('outputs/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Statistical tests for feature importance (improved)
print("\nStatistical Tests for Feature Importance:")
feature_importance = {}
for feature in selected_features:
    legitimate = df[df['label'] == 0][feature]
    phishing = df[df['label'] == 1][feature]
    t_stat, p_value = stats.ttest_ind(legitimate, phishing)
    feature_importance[feature] = abs(t_stat)

sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 Most Important Features (based on t-statistic):")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance:.2f}")

# Plot feature importance (improved)
plt.figure(figsize=(10, 6))
features, importance = zip(*sorted_features[:10])
bar = plt.barh(features, importance, color=sns.color_palette('Set2'))
plt.title('Top 10 Most Important Features', fontsize=18, pad=20)
plt.xlabel('Absolute t-statistic', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Save summary statistics to a text file
with open('outputs/eda_summary.txt', 'w') as f:
    f.write("EDA Summary Report\n")
    f.write("=================\n\n")
    f.write("Rows before outlier removal: {}\n".format(df.shape[0]))
    f.write("Rows after outlier removal: {}\n".format(filtered_df.shape[0]))
    f.write("Dataset Shape: {}\n".format(df.shape))
    f.write("\nClass Distribution:\n")
    f.write(str(df['label'].value_counts()))
    f.write("\n\nTop 10 Most Important Features:\n")
    for feature, importance in sorted_features[:10]:
        f.write(f"{feature}: {importance:.2f}\n")
    f.write("\n\nSummary Statistics:\n")
    f.write(str(df.describe()))

print("\nEDA completed! Check the 'outputs' directory for visualizations and summary statistics.") 