# 🧠 MindCare: A Wellness Hub for Employees

MindCare is an AI-powered web-based mental wellness platform that detects emotions such as **anger**, **stress**, and **anxiety** from user-input text and provides personalized wellness recommendations like **music**, **videos**, and **places to visit**.

This project integrates **NLP (DistilBERT)** for emotion detection, **Flask** for backend API management, **MongoDB** for data storage, and an interactive **HTML/CSS/JavaScript** frontend for seamless user experience.

---

## 🚀 Features

* 💬 **Emotion Detection:** Uses DistilBERT model to classify text into *anger, stress,* or *anxiety*.
* 🌐 **Interactive Web UI:** Clean and responsive interface built with HTML, CSS, and JS.
* 🔗 **Flask Backend:** Handles text input, model inference, and recommendation logic.
* 🎧 **Personalized Recommendations:** Suggests media (music, podcasts, articles, etc.) based on detected emotion.
* 🧾 **Database Integration:** Stores user emotion data and logs using MongoDB.
* ⚙️ **API Connectivity:** Connects to third-party APIs for real-time wellness suggestions.

---

## 🧩 System Architecture

```
User Input (Web UI)
        ↓
Flask Backend (app.py)
        ↓
Emotion Detection Model (DistilBERT)
        ↓
Recommendation Engine
        ↓
Frontend Display (Music / Video / Articles / Places)
```

---

## 🗃️ Dataset

The dataset used is a **filtered subset of the GoEmotions dataset**, containing text samples mapped to three mental health-related categories:

* **Anger**
* **Stress**
* **Anxiety**

The dataset was preprocessed and saved as `mental_health_dataset.csv` for model training and evaluation.

---

## 🧠 Model Training

* Framework: **Hugging Face Transformers (DistilBERT)**
* Fine-tuned on the filtered mental health dataset
* Trained using `Trainer API` with metrics such as accuracy, precision, recall, and F1-score

Key Scripts:

* `generate_dataset.py` – Dataset creation and preprocessing
* `train_model.py` – Model fine-tuning and evaluation
* `app_inference.py` – Handles prediction logic for deployed model

---

## 🖥️ Project Structure

```
MindCare/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py                # Flask app main file
├── app_inference.py      # Emotion detection function
├── generate_dataset.py   # Dataset creation script
├── train_model.py        # Model training script
├── mental_health_dataset.csv
├── model/
│   ├── config.json
│   ├── tokenizer/
│   └── pytorch_model.bin
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/MindCare.git
cd MindCare
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # For macOS/Linux
venv\Scripts\activate        # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

### 5. Access in Browser

Visit → `http://127.0.0.1:5000/`

---

## 🧪 Testing

* Input any sentence describing your mood or feelings.
* The system detects the emotion and provides corresponding wellness suggestions.

Example:

> “I feel like everything is going wrong.”
> → Emotion: *Stress*
> → Recommendation: Relaxing music, motivational articles, and meditation videos.

---

## 📊 Results

| Emotion | Precision | Recall | F1-Score |
|:--------|:----------:|:------:|:--------:|
| Anger   | 0.84 | 0.88 | 0.86 |
| Stress  | 0.79 | 0.72 | 0.75 |
| Anxiety | 0.65 | 0.64 | 0.64 |

**Overall Accuracy:** 80%


---

## 🧭 Future Enhancements

* Integrate real-time chat support for mental wellness
* Add sentiment intensity scoring
* Expand dataset with multilingual inputs
* Deploy on cloud (AWS / Render / Hugging Face Spaces)

---

## 👩‍💻 Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **Model:** DistilBERT (Hugging Face)
* **Database:** MongoDB
* **APIs:** Music & wellness content recommendation APIs

---

## 🙌 Contributors

**Ritika Ruhal**
🎓 B.Tech (Computer Science & Engineering)
📍 Sharda University

---

## 🪪 License

This project is licensed under the **MIT License** – feel free to modify and use it for educational or research purposes.

---

## 💡 Acknowledgments

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [GoEmotions Dataset (Google Research)](https://github.com/google-research/google-research/tree/master/goemotions)
* [Flask Documentation](https://flask.palletsprojects.com/)

---

> “MindCare helps you understand your emotions better — because mental health matters.”
