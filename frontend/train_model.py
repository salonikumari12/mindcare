import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ✅ Add compute_metrics without affecting training logic
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision_anger": precision[0],
        "precision_stress": precision[1],
        "precision_anxiety": precision[2],
        "recall_anger": recall[0],
        "recall_stress": recall[1],
        "recall_anxiety": recall[2],
        "f1_anger": f1[0],
        "f1_stress": f1[1],
        "f1_anxiety": f1[2],
        "confusion_matrix": cm.tolist()
    }

def main():
    # Load dataset
    df = pd.read_csv("mental_health_dataset.csv")
    label_mapping = {"anger": 0, "stress": 1, "anxiety": 2}
    df["label"] = df["label"].map(label_mapping)
    dataset = Dataset.from_pandas(df)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    test_dataset = split["test"]

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        dataloader_num_workers=2,
        logging_steps=10 #  Add evaluation after each epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics  # ✅ Attach metrics function
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    print("📊 Evaluation Metrics:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save model
    output_dir = os.path.join(os.getcwd(), "mindcare_model")
    os.makedirs(output_dir, exist_ok=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Model saved at: {output_dir}")

    # List files
    print("📂 Saved files:")
    for filename in os.listdir(output_dir):
        print("  -", filename)

if __name__ == "__main__":
    main()
  


