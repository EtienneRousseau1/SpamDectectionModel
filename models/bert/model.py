import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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

# 1) Load & preprocess
df = pd.read_csv("nlp_dataset.csv")[["body", "label"]].dropna()
df["label"] = df["label"].astype(int)  # ensure labels are ints

texts = df["body"].tolist()
labels = df["label"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# 2) Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
max_length = 256

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

# 3) Build HF Dataset objects
train_dataset = Dataset.from_dict(
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels,
    }
)
val_dataset = Dataset.from_dict(
    {
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_labels,
    }
)

# 4) Convert to PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5) Load model (classification head starts randomly initialized)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# 6) Compute metrics for Trainer
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 7) Training arguments (minimal for compatibility)
training_args = TrainingArguments(
    output_dir="./phishing_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
)

# 8) Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 9) Train
trainer.train()

# 10) Run final evaluation (metrics printed and returned)
print("\n***** Running final evaluation on validation set *****")
metrics = trainer.evaluate(val_dataset)
print(metrics)

# 11) Plot metrics from trainer.state.log_history
log_history = trainer.state.log_history

# Extract training losses logged during training (exclude eval entries)
train_losses = [
    entry["loss"] 
    for entry in log_history 
    if "loss" in entry and "eval_loss" not in entry
]

# Extract validation losses and accuracies
eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
eval_accuracies = [
    entry["eval_accuracy"] for entry in log_history if "eval_accuracy" in entry
]

# Create the charts directory if it doesn't exist
os.makedirs("charts", exist_ok=True)

# Plot training loss over time
if train_losses:
    plt.figure()
    plt.plot(train_losses, marker="o")
    plt.xlabel("Logging Step")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Over Time")
    plt.tight_layout()
    plt.savefig("charts/train_loss.png")
    plt.close()

# Plot validation loss over time
if eval_losses:
    plt.figure()
    plt.plot(eval_losses, marker="o", color="orange")
    plt.xlabel("Epoch/Logging Step")
    plt.ylabel("Eval Loss")
    plt.title("Validation Loss Over Time")
    plt.tight_layout()
    plt.savefig("charts/eval_loss.png")
    plt.close()

# Plot validation accuracy over time
if eval_accuracies:
    plt.figure()
    plt.plot(eval_accuracies, marker="o", color="green")
    plt.xlabel("Epoch/Logging Step")
    plt.ylabel("Eval Accuracy")
    plt.title("Validation Accuracy Over Time")
    plt.tight_layout()
    plt.savefig("charts/eval_accuracy.png")
    plt.close()

print("Charts saved in /charts:")
print(os.listdir("charts"))

# 12) Save final model & tokenizer
trainer.save_model("./phishing_model")
tokenizer.save_pretrained("./phishing_model")