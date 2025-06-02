import pandas as pd

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/email_phishing_data.csv')

# Display initial dataset info
print("\nInitial dataset shape:", df.shape)
print("\nClass distribution:")
print(df['label'].value_counts())

# Separate legitimate and phishing emails
legitimate = df[df['label'] == 0]
phishing = df[df['label'] == 1]

print("Legitimate count:", len(legitimate))
print("Phishing count:", len(phishing))

# Determine the maximum possible balanced sample size
max_sample = min(len(legitimate), len(phishing), 10000)
print(f"Sampling {max_sample} from each class.")

legitimate_sampled = legitimate.sample(n=max_sample, random_state=42)
phishing_sampled = phishing.sample(n=max_sample, random_state=42)

# Combine the sampled data
balanced_df = pd.concat([legitimate_sampled, phishing_sampled])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the processed balanced dataset
print("\nSaving processed balanced dataset...")
balanced_df.to_csv('data/processed_balanced.csv', index=False)

print("\nFinal balanced dataset shape:", balanced_df.shape)
print("\nClass distribution:")
print(balanced_df['label'].value_counts()) 