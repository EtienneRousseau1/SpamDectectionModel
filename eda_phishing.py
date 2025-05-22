import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ——— Config —————————————————————————————————————————————————————————————————————————————
INPUT_PATH  = 'data/phishing_email.csv'
FIGURE_DIR = 'outputs/figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

# ——— 1) Load & Inspect —————————————————————————————————————————————————————————————————
df = pd.read_csv(INPUT_PATH)
print(f">>> Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
print(df.dtypes)
print(df.head(), '\n')
print("Missing values per column:\n", df.isna().sum(), '\n')

# Detailed check of text_combined column
print("Sample of text_combined field:")
if 'text_combined' in df.columns:
    first_non_empty = df['text_combined'].dropna().iloc[0] if not df['text_combined'].dropna().empty else "NO NON-EMPTY TEXT FOUND"
    print(f"First non-empty text: {first_non_empty[:200]}..." if len(str(first_non_empty)) > 200 else first_non_empty)
    print(f"\nText type: {type(first_non_empty)}")
    print(f"Text length: {len(str(first_non_empty))}")
    
    print(f"\nNumber of rows with empty text_combined: {(df['text_combined'].isna() | (df['text_combined'] == '')).sum()}")
    print(f"Number of rows with text_combined of type string: {df['text_combined'].apply(lambda x: isinstance(x, str)).sum()}")

    # ——— 2) Feature Engineering —————————————————————————————————————————————————————————————
    # Ensure text is string type and handle NaN values properly
    df['text_combined'] = df['text_combined'].fillna("").astype(str)
    text = df['text_combined']

    # Check if there are any empty strings after conversion
    print(f"\nNumber of empty strings after conversion: {(text == '').sum()}")

    # raw length in chars - with more detailed error checking
    df['email_length'] = text.str.len()  # Using str.len() instead of apply(len)
    print(f"Email length stats: min={df['email_length'].min()}, max={df['email_length'].max()}, avg={df['email_length'].mean():.2f}")

    # Check for zero lengths
    zero_lengths = (df['email_length'] == 0).sum()
    print(f"Number of emails with zero length: {zero_lengths}")
    if zero_lengths > 0:
        print("Warning: Some emails have zero length!")

    # number of http/https links
    df['num_links'] = text.str.count(r'http')

    # exclamation marks
    df['num_exclaims'] = text.str.count('!')

    # question marks
    df['num_questions'] = text.str.count(r'\?')

    # dollar signs
    df['num_dollar'] = text.str.count(r'\$')

    # uppercase WORDS
    df['num_uppercase_words'] = text.apply(lambda s: sum(1 for w in s.split() if w.isupper() and len(w) > 1))

    # average word length - with improved handling
    df['avg_word_len'] = text.apply(
        lambda s: np.mean([len(w) for w in s.split() if len(w) > 0]) if len(s.split()) > 0 else 0
    )
else:
    print("Warning: 'text_combined' column not found in the dataset!")
    # Create dummy features if text_combined is missing
    for col in ['email_length', 'num_links', 'num_exclaims', 
                'num_questions', 'num_dollar', 'num_uppercase_words', 'avg_word_len']:
        if col not in df.columns:
            df[col] = 0
            print(f"Created placeholder column '{col}' with zeros")

numeric_cols = [
    'email_length',
    'num_links',
    'num_exclaims',
    'num_questions',
    'num_dollar',
    'num_uppercase_words',
    'avg_word_len'
]

# Print summary of engineered features
print("\nEngineered features summary:")
print(df[numeric_cols].describe().T)

# ——— 3) Class Balance —————————————————————————————————————————————————————————————————————
counts = df['label'].value_counts().sort_index()
labels = ['Legit (0)', 'Phish (1)']

plt.figure(figsize=(6,4))
bars = plt.bar(labels, counts.values, edgecolor='black')
plt.title("Class Balance: Legit vs. Phishing", fontsize=14)
plt.ylabel("Number of Emails", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# annotate counts on bars
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             h + counts.values.max()*0.01,
             f"{int(h)}",
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/class_balance.png", dpi=300)
plt.close()  # Use plt.close() instead of plt.show() for non-interactive environments

# ——— 4) Histograms of Each Numeric Feature —————————————————————————————————————————————
for col in numeric_cols:
    # Skip if all values are 0
    if df[col].max() == 0:
        print(f"Skipping histogram for {col} - all values are zero")
        continue
        
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30, edgecolor='black', alpha=0.75)
    plt.title(f"Distribution of {col.replace('_',' ').title()}", fontsize=14)
    plt.xlabel(col.replace('_',' ').title(), fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/{col}_hist.png", dpi=300)
    plt.close()

# ——— 5) Feature Distribution by Class —————————————————————————————————————————————————————
for col in numeric_cols:
    # Skip if all values are 0
    if df[col].max() == 0:
        print(f"Skipping boxplot for {col} - all values are zero")
        continue
        
    plt.figure(figsize=(6,4))
    sns.boxplot(x='label', y=col, data=df)
    plt.title(f"{col.replace('_',' ').title()} by Class", fontsize=14)
    plt.xlabel("Class (0=Legit, 1=Phish)", fontsize=12)
    plt.ylabel(col.replace('_',' ').title(), fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/{col}_by_class.png", dpi=300)
    plt.close()

# ——— 6) Log-transformed Histograms —————————————————————————————————————————————————————
# For skewed data like email_length, a log transform helps visualize the distribution
for col in numeric_cols:
    # Skip columns with lots of zeros as log transform won't help
    if df[col].eq(0).sum() / len(df) > 0.5 or df[col].max() == 0:
        continue
        
    plt.figure(figsize=(6,4))
    # Log transform with offset to handle zeros
    log_values = np.log1p(df[col])
    plt.hist(log_values, bins=30, edgecolor='black', alpha=0.75)
    plt.title(f"Log Distribution of {col.replace('_',' ').title()}", fontsize=14)
    plt.xlabel(f"{col.replace('_',' ').title()} (log scale)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/{col}_log_hist.png", dpi=300)
    plt.close()

# ——— 7) Top 20 Words in Corpus —————————————————————————————————————————————————————————————
if 'text_combined' in df.columns:
    all_text = ' '.join(df['text_combined'].fillna("").astype(str).str.lower())
    words = re.findall(r'\b\w+\b', all_text)
    common20 = Counter(words).most_common(20)
    
    if common20:
        wc, ct = zip(*common20)
        
        plt.figure(figsize=(8,4))
        bars = plt.barh(wc[::-1], ct[::-1], edgecolor='black')
        plt.title("Top 20 Words in Email Corpus", fontsize=14)
        plt.xlabel("Count", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        # annotate counts
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + max(ct)*0.01, bar.get_y() + bar.get_height()/2,
                    str(bar.get_width()), va='center')

        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/top20_words.png", dpi=300)
        plt.close()
    else:
        print("Warning: No words found in corpus for top 20 analysis")

# ——— 8) Email Length by Label Boxplot —————————————————————————————————————————————
if df['email_length'].max() > 0:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='label', y='email_length', data=df)
    plt.title("Email Length by Label (0=Legit, 1=Phish)", fontsize=14)
    plt.xlabel("Label", fontsize=12)
    plt.ylabel("Length (chars)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/length_by_label.png", dpi=300)
    plt.close()
else:
    print("Skipping email length boxplot - all values are zero")

# ——— 9) Correlation Matrix —————————————————————————————————————————————————————————————
corr = df[numeric_cols].corr()

plt.figure(figsize=(7,6))
im = plt.imshow(corr, cmap='viridis', interpolation='none')
plt.colorbar(im, fraction=0.046, pad=0.04, label='Pearson r')
plt.xticks(range(len(numeric_cols)), [c.replace('_',' ').title() for c in numeric_cols],
           rotation=45, ha='right')
plt.yticks(range(len(numeric_cols)), [c.replace('_',' ').title() for c in numeric_cols])
plt.title("Correlation Matrix of Engineered Features", fontsize=14)

# annotate correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha='center', va='center', color='white' if abs(corr.iloc[i, j])>0.5 else 'black')

plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/correlation_matrix.png", dpi=300)
plt.close()

print("EDA completed successfully!")
