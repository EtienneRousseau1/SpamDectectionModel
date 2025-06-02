import pandas as pd
import matplotlib.pyplot as plt
import string
import os
import re

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

df = pd.read_csv('balanced_dataset.csv')

# Basic text features
df['body_length'] = df['body'].fillna('').apply(len)
df['subject_word_count'] = df['subject'].fillna('').apply(lambda txt: len(txt.split()))
df['body_word_count'] = df['body'].fillna('').apply(lambda txt: len(txt.split()))
df['punctuation_count'] = df['body'].fillna('').apply(lambda txt: sum(1 for ch in txt if ch in string.punctuation))

# New features
df['capital_ratio'] = df['body'].fillna('').apply(lambda txt: sum(1 for ch in txt if ch.isupper()) / len(txt) if len(txt) > 0 else 0)
df['url_count'] = df['body'].fillna('').apply(lambda txt: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)))
df['number_count'] = df['body'].fillna('').apply(lambda txt: sum(1 for ch in txt if ch.isdigit()))
df['special_char_count'] = df['body'].fillna('').apply(lambda txt: sum(1 for ch in txt if ch not in string.punctuation and ch not in string.ascii_letters and ch not in string.digits and ch != ' '))
df['avg_word_length'] = df['body'].fillna('').apply(lambda txt: sum(len(word) for word in txt.split()) / len(txt.split()) if len(txt.split()) > 0 else 0)

legitimate = df[df['label'] == 0]
spam = df[df['label'] == 1]

# 1. Bar chart: Count of samples by class
plt.figure(figsize=(6, 5))
class_counts = df['label'].value_counts().reindex([0, 1])
plt.bar(['Legitimate', 'Spam'], class_counts.values, edgecolor='black')
plt.title('Number of Samples by Email Class')
plt.ylabel('Number of Samples')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('charts/sample_count_bar.png')
plt.close()

# 2. Box plot: Capitalization ratio by class
plt.figure(figsize=(6, 5))
plt.boxplot([legitimate['capital_ratio'], spam['capital_ratio']], labels=['Legitimate', 'Spam'], showfliers=False)
plt.title('Capitalization Ratio by Email Class')
plt.ylabel('Ratio of Capital Letters')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('charts/capitalization_ratio_boxplot.png')
plt.close()

# 3. Box plot: URL count by class
plt.figure(figsize=(6, 5))
plt.boxplot([legitimate['url_count'], spam['url_count']], labels=['Legitimate', 'Spam'], showfliers=False)
plt.title('URL Count by Email Class')
plt.ylabel('Number of URLs')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('charts/url_count_boxplot.png')
plt.close()

# 4. Box plot: Number count by class
plt.figure(figsize=(6, 5))
plt.boxplot([legitimate['number_count'], spam['number_count']], labels=['Legitimate', 'Spam'], showfliers=False)
plt.title('Number Count by Email Class')
plt.ylabel('Count of Digits')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('charts/number_count_boxplot.png')
plt.close()

# 5. Box plot: Special character count by class
plt.figure(figsize=(6, 5))
plt.boxplot([legitimate['special_char_count'], spam['special_char_count']], labels=['Legitimate', 'Spam'], showfliers=False)
plt.title('Special Character Count by Email Class')
plt.ylabel('Count of Special Characters')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('charts/special_char_count_boxplot.png')
plt.close()

# 6. Box plot: Average word length by class
plt.figure(figsize=(6, 5))
plt.boxplot([legitimate['avg_word_length'], spam['avg_word_length']], labels=['Legitimate', 'Spam'], showfliers=False)
plt.title('Average Word Length by Email Class')
plt.ylabel('Average Word Length (characters)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('charts/avg_word_length_boxplot.png')
plt.close()
