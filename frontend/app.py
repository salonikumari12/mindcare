from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_path = "./mindcare_model"  # Folder that contains model files
print(f"Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict function
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map predicted class to label
    label_map = {0: "anger", 1: "anxiety", 2: "stress"}
    return label_map.get(predicted_class, "unknown")

# Flask API route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text field provided"}), 400

        prediction = predict_emotion(data["text"])
        return jsonify({"label": prediction})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)

