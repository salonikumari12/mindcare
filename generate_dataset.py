from datasets import load_dataset
import pandas as pd

dataset = load_dataset("go_emotions", split="train")

label_mapping = {
    "anger": "anger",
    "annoyance": "anger",
    "disgust": "anger",
    "fear": "anxiety",
    "nervousness": "anxiety",
    "embarrassment": "anxiety",
    "sadness": "stress",
    "grief": "stress",
    "disappointment": "stress"
}

filtered_samples = []
for sample in dataset:
    mapped_labels = []
    for label in sample['labels']:
        # Use `.feature.int2str` because 'labels' is a Sequence feature
        label_name = dataset.features['labels'].feature.int2str(label)
        if label_name in label_mapping:
            mapped_labels.append(label_mapping[label_name])
    if mapped_labels:
        filtered_samples.append({"text": sample["text"], "label": mapped_labels[0]})

df = pd.DataFrame(filtered_samples)
df = df.sample(n=2000, random_state=42)
df.to_csv("mental_health_dataset.csv", index=False)

print("✅ Dataset saved as 'mental_health_dataset.csv'.")

