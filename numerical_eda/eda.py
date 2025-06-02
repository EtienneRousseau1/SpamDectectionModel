import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure the directory for saving charts exists
os.makedirs('charts', exist_ok=True)

# 1. Load dataset and identify numeric columns
df = pd.read_csv('processed_balanced.csv')
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('label')
target = 'label'

# Quick sanity check: print mean of each numeric feature by label
print("Mean values by label:\n", df.groupby(target)[numeric_cols].mean(), "\n")

# 2. Bar chart of label counts (Spam vs. Not Spam)
counts = df[target].value_counts().sort_index()
plt.figure(figsize=(6, 4))
bars = plt.bar(
    ['Not Spam', 'Spam'], 
    counts, 
    color=['#4C72B0', '#55A868'], 
    edgecolor='black'
)
plt.title('Count of Emails by Label', fontsize=14, fontweight='bold')
plt.ylabel('Number of Emails', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
# Save and show
plt.savefig('charts/label_counts.png', dpi=300)
plt.show()
plt.close()


# 3. Box plots of each numeric feature, comparing Not Spam vs. Spam (no outliers)
for col in numeric_cols:
    data_not_spam = df[df[target] == 0][col]
    data_spam     = df[df[target] == 1][col]
    
    plt.figure(figsize=(6, 4))
    box = plt.boxplot(
        [data_not_spam, data_spam],
        labels=['Not Spam', 'Spam'],
        patch_artist=True,
        notch=True,
        widths=0.6,
        showfliers=False  # Hide outliers
    )
    # Color each box
    colors = ['#4C72B0', '#55A868']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title(f'{col} Distribution by Label', fontsize=14, fontweight='bold')
    plt.ylabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save and show
    filename = f'charts/{col.lower().replace(" ", "_")}_boxplot.png'
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()


# 4. Correlation matrix (all numeric features)
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(np.arange(len(numeric_cols)), numeric_cols, rotation=45, fontsize=10)
plt.yticks(np.arange(len(numeric_cols)), numeric_cols, fontsize=10)
plt.title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')

# Annotate each cell
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        value = corr.iloc[i, j]
        plt.text(
            j, i,
            f"{value:.2f}",
            ha='center', va='center',
            color='white' if abs(value) > 0.5 else 'black',
            fontsize=8
        )

plt.tight_layout()
# Save and show
plt.savefig('charts/correlation_matrix.png', dpi=300)
plt.show()
plt.close()
