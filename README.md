# Phishing Detector (MLOps)

End-to-end phishing email detection pipeline with a Flask API and a lightweight web UI. The
project covers data ingestion, feature engineering, model training, and inference.

## Highlights

- Text cleaning + TF-IDF features with numeric signals (URLs, urgency, HTML, exclamations)
- Time-of-day feature bucketing from email date
- Flask API with web UI and CORS support
- Modular pipeline for training and inference

## Architecture (High Level)

1. Data ingestion splits raw data into train/test CSVs
2. Feature engineering builds text and numeric features
3. Training selects best model and saves artifacts
4. Prediction loads artifacts and serves results

## Project Structure

```
flaskr/
	__init__.py
	__main__.py
	templates/
		index.html
	static/
		app.js
		styles.css
src/
	components/
		data_ingestion.py
		data_transformation.py
		feature_extractor.py
		model_trainer.py
	pipeline/
		predict_pipeline.py
		train_pipeline.py
	config/
		constants.py
		paths.py
```

## Setup

### 1) Install dependencies

```
pip install -r requirements.txt
```

### 2) (Optional) NLTK data

If you see a WordNet error during preprocessing:

```
python -m nltk.downloader wordnet
```

### 3) Ensure dataset is available

Place your raw dataset at:

```
data/raw/TREC-05.csv
```

## Run the API + UI

```
python -m flaskr
```

Open:

```
http://localhost:5000
```

## API Endpoints

### Health

```
GET /health
```

### Single prediction

```
POST /predict
Content-Type: application/json

{
	"sender": "alerts@bank.com",
	"receiver": "user@example.com",
	"date": "Fri, 29 Jun 2001 08:36:09 -0500",
	"subject": "Urgent: Verify your account",
	"body": "Please verify at https://secure.example.com"
}
```

### Batch prediction

```
POST /predict/batch
Content-Type: application/json

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

### Train (optional)

```
POST /train
```

## Configuration

Environment variables:

- `CORS_ORIGINS`: Comma-separated list of allowed origins
- `ENABLE_TRAINING`: Set to `false` to disable `/train`
- `TRAIN_API_KEY`: If set, requires `X-API-Key` header for `/train`

Example:

```
set CORS_ORIGINS=http://localhost:3000,http://localhost:5500
set ENABLE_TRAINING=true
set TRAIN_API_KEY=dev-key
```

## Artifacts

Artifacts are stored under:

```
artifacts/
	preprocesser.pkl
	model.pkl
```

## Notes

- The prediction pipeline caches model artifacts in memory after the first load.
- Restart the API if you retrain and want the latest artifacts loaded.

## Next Steps (Roadmap)

- Add automated tests (pytest)
- Add Docker and docker-compose
- Add CI (GitHub Actions)
- Add MLflow tracking