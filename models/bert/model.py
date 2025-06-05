import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# 1) Load & preprocess the full CSV
df = pd.read_csv("nlp_dataset.csv")[["body", "label"]].dropna()
df["label"] = df["label"].astype(int)

texts = df["body"].tolist()
labels = df["label"].tolist()

# 2) Split nlp_dataset.csv into train (90%) and validation (10%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels,
    test_size=0.10,
    random_state=42,
    stratify=labels,
)

# 3) Load validation_dataset.csv as test set
test_df = pd.read_csv("validation_dataset.csv")[["body", "label"]].dropna()
test_df["label"] = test_df["label"].astype(int)
test_texts = test_df["body"].tolist()
test_labels = test_df["label"].tolist()

# 4) Tokenizer setup
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
max_length = 256

# Tokenize train / val / test
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=max_length,
)
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding="max_length",
    max_length=max_length,
)
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=max_length,
)

# 5) Build HF Dataset objects
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels,
})
val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels,
})
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels,
})

# 6) Convert to PyTorch format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7) Load a pretrained DistilBERT for binary classification
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

# 8) Define compute_metrics for Trainer
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 9) Training arguments: enable per‐epoch evaluation
training_args = TrainingArguments(
    output_dir="./phishing_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="steps",
    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 10) Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 11) Train
trainer.train()

# 12) After training, run final evaluation on the held‐out TEST set
print("\n***** Final evaluation on TEST set *****")
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)

# Create output directory for plots
os.makedirs("charts", exist_ok=True)

# Get predictions
test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=1)
test_true = test_predictions.label_ids

# Create confusion matrix
cm = confusion_matrix(test_true, test_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Phishing', 'Phishing'],
            yticklabels=['Not Phishing', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png")
plt.close()

# 13) Plotting: extract from trainer.state.log_history
log_history = trainer.state.log_history

train_losses = [
    entry["loss"]
    for entry in log_history
    if ("loss" in entry and "eval_loss" not in entry)
]

eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
eval_accuracies = [entry["eval_accuracy"] for entry in log_history if "eval_accuracy" in entry]

# 14) Plot training loss over time
if train_losses:
    plt.figure()
    plt.plot(train_losses, marker="o")
    plt.xlabel("Logging Step")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Over Time")
    plt.tight_layout()
    plt.savefig("charts/train_loss.png")
    plt.close()

# 15) Plot validation loss over time
if eval_losses:
    plt.figure()
    plt.plot(eval_losses, marker="o", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title("Validation Loss Over Time")
    plt.tight_layout()
    plt.savefig("charts/eval_loss.png")
    plt.close()

# 16) Plot validation accuracy over time
if eval_accuracies:
    plt.figure()
    plt.plot(eval_accuracies, marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Accuracy")
    plt.title("Validation Accuracy Over Time")
    plt.tight_layout()
    plt.savefig("charts/eval_accuracy.png")
    plt.close()

print("Charts saved in /charts/")
print(os.listdir("charts"))

# 17) Save final model & tokenizer
trainer.save_model("./phishing_model")
tokenizer.save_pretrained("./phishing_model")
