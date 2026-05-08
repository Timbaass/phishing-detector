# 🎣 Phishing Detector — MLOps Pipeline

> End-to-end phishing email detection with a Flask API and lightweight web UI.  
> Covers data ingestion, feature engineering, model training, and real-time inference.
<img width="1915" height="912" alt="image" src="https://github.com/user-attachments/assets/9b59a403-64af-4186-a32e-ff045a5e7a2f" />

---

## ✨ Highlights

| Feature | Description |
|---|---|
| 🔤 Text Processing | TF-IDF features + numeric signals (URLs, urgency, HTML, exclamations) |
| 🕐 Temporal Features | Time-of-day bucketing from email `Date` header |
| 🌐 Flask API | RESTful endpoints with web UI and CORS support |
| 🔧 Modular Design | Clean separation of training and inference pipelines |

---

## 🏗️ Architecture

```
Raw CSV  →  Data Ingestion  →  Feature Engineering  →  Model Training
                                                              ↓
HTTP Request  →  Flask API  →  Prediction Pipeline  →  JSON Response
```

1. **Data Ingestion** — splits raw data into train/test CSVs
2. **Feature Engineering** — builds text (TF-IDF) and numeric features
3. **Training** — selects best model and saves serialized artifacts
4. **Prediction** — loads artifacts, caches in memory, and serves results

---

## 📁 Project Structure

```
phishing-detector/
├── data/
│   └── raw/
│       └── TREC-05.csv
├── artifacts/
│   ├── preprocesser.pkl
│   └── model.pkl
├── flaskr/
│   ├── __init__.py
│   ├── __main__.py
│   └── templates/
│       └── index.html
├── src/
│   ├── components/
│   │   ├── data_cleaner.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── feature_extractor.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   └── config/
│       ├── constants.py
│       └── paths.py
├── exception.py
├── logger.py
├── utils.py
└── requirements.txt
```

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Download NLTK data

If you encounter a WordNet error during preprocessing:

```bash
python -m nltk.downloader wordnet
```

### 3. Place your dataset

```
data/raw/TREC-05.csv
```

---

## ▶️ Run the API

```bash
python -m flaskr
```

Then open your browser at:

```
http://localhost:5000
```

---

## 📡 API Reference

### `GET /health`
Health check endpoint.

---

### `POST /predict`
Single email prediction.

```json
{
  "sender": "alerts@bank.com",
  "receiver": "user@example.com",
  "date": "Fri, 29 Jun 2001 08:36:09 -0500",
  "subject": "Urgent: Verify your account",
  "body": "Please verify at https://secure.example.com"
}
```

---

### `POST /predict/batch`
Batch prediction for multiple emails.

```json
{
  "records": [
    {
      "sender": "alerts@bank.com",
      "receiver": "user@example.com",
      "date": "Fri, 29 Jun 2001 08:36:09 -0500",
      "subject": "Urgent: Verify your account",
      "body": "Please verify at https://secure.example.com"
    }
  ]
}
```

---

### `POST /train` *(optional)*
Trigger model retraining. Can be protected with an API key (see Configuration).

---

## ⚙️ Configuration

Control behavior via environment variables:

| Variable | Description | Default |
|---|---|---|
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |
| `ENABLE_TRAINING` | Enable/disable `/train` endpoint | `true` |
| `TRAIN_API_KEY` | If set, requires `X-API-Key` header on `/train` | *(none)* |

**Example (Windows):**

```cmd
set CORS_ORIGINS=http://localhost:3000,http://localhost:5500
set ENABLE_TRAINING=true
set TRAIN_API_KEY=dev-key
```

**Example (Unix):**

```bash
export CORS_ORIGINS=http://localhost:3000,http://localhost:5500
export ENABLE_TRAINING=true
export TRAIN_API_KEY=dev-key
```

---

## 📦 Artifacts

Trained artifacts are saved at:

```
artifacts/
├── preprocesser.pkl   ← TF-IDF + feature transformer
└── model.pkl          ← trained classifier
```

> **Note:** The prediction pipeline caches artifacts in memory after the first load.  
> Restart the API after retraining to pick up new artifacts.

---

## 🗺️ Roadmap

- [ ] Add automated tests with `pytest`
- [ ] Dockerize with `docker-compose`
- [ ] CI/CD via GitHub Actions
- [ ] MLflow experiment tracking

---

## 📄 License

MIT
