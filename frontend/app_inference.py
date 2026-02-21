import os
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Fix path format
model_dir = os.path.join(os.getcwd(), "mindcare_model").replace("\\", "/")
print(f"Loading model from: {model_dir}")

# Load model and tokenizer from local directory
model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir, local_files_only=True)

# id2label dictionary
id2label = {0: "anger", 1: "stress", 2: "anxiety"}

def classify_mental_health(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        return id2label[predicted_class_id]

if __name__ == "__main__":
    while True:
        user_input = input("Enter your text (or 'exit' or 'quit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the program. Take care!")
            break
        prediction = classify_mental_health(user_input)
        print(f"Detected mental health category: {prediction}")
