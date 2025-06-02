import pandas as pd
import matplotlib.pyplot as plt
import string
import os
import re
import seaborn as sns
import numpy as np

# Set the style for all plots
plt.style.use('seaborn')
sns.set_palette("husl")

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

def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def create_boxplot(data1, data2, title, ylabel, filename):
    # Remove outliers from both datasets
    data1_clean = remove_outliers(data1)
    data2_clean = remove_outliers(data2)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([data1_clean, data2_clean], 
                labels=['Legitimate', 'Spam'],
                showfliers=False,  # Remove outliers
                patch_artist=True)  # Fill boxes with color
    
    # Customize the plot
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some padding
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig(f'charts/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def create_barplot(data, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Legitimate', 'Spam'], data, edgecolor='black', width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'charts/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# 1. Bar chart: Count of samples by class
class_counts = df['label'].value_counts().reindex([0, 1])
create_barplot(class_counts.values, 
              'Distribution of Email Classes',
              'Number of Samples',
              'sample_count_bar.png')

# 2. Box plot: Capitalization ratio by class
create_boxplot(legitimate['capital_ratio'], 
              spam['capital_ratio'],
              'Capitalization Ratio Distribution',
              'Ratio of Capital Letters',
              'capitalization_ratio_boxplot.png')

# 3. Box plot: URL count by class
create_boxplot(legitimate['url_count'], 
              spam['url_count'],
              'URL Count Distribution',
              'Number of URLs',
              'url_count_boxplot.png')

# 4. Box plot: Number count by class
create_boxplot(legitimate['number_count'], 
              spam['number_count'],
              'Number Count Distribution',
              'Count of Digits',
              'number_count_boxplot.png')

# 5. Box plot: Special character count by class
create_boxplot(legitimate['special_char_count'], 
              spam['special_char_count'],
              'Special Character Count Distribution',
              'Count of Special Characters',
              'special_char_count_boxplot.png')

# 6. Box plot: Average word length by class
create_boxplot(legitimate['avg_word_length'], 
              spam['avg_word_length'],
              'Average Word Length Distribution',
              'Average Word Length (characters)',
              'avg_word_length_boxplot.png') 