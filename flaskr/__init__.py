import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.components import DataIngestion, DataTransformation, ModelTrainer
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


def create_app(test_config=None):
    app = Flask(__name__)

    enable_training = os.environ.get("ENABLE_TRAINING", "true").lower() in {"1", "true", "yes"}
    train_api_key = os.environ.get("TRAIN_API_KEY")
    is_debug = app.debug or os.environ.get("APP_ENV", "").lower() == "development"

    allowed_origins = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500",
    )
    cors_origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]

    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            return "", 204

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        allow_origin = None

        if "*" in cors_origins:
            allow_origin = "*"
        elif origin and origin in cors_origins:
            allow_origin = origin

        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

        return response

    @app.route("/")
    def hello():
        return render_template("index.html")

    @app.route("/health", methods=["GET", "OPTIONS"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        try:
            if request.method == "OPTIONS":
                return "", 204

            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify({"error": "Invalid or missing JSON body"}), 400

            required_fields = ["sender", "receiver", "date", "subject", "body"]
            missing_fields = [field for field in required_fields if field not in payload]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

            input_data = CustomData(
                sender=payload["sender"],
                receiver=payload["receiver"],
                date=payload["date"],
                subject=payload["subject"],
                body=payload["body"],
            )

            input_df = input_data.get_data_as_dataframe()
            prediction = PredictPipeline().predict(input_df)
            predicted_label = int(prediction[0])

            return jsonify(
                {
                    "prediction": predicted_label,
                    "label": "phishing" if predicted_label == 1 else "normal",
                }
            )

        except Exception as e:
            raise CustomException(e, sys)

    @app.route("/predict/batch", methods=["POST", "OPTIONS"])
    def predict_batch():
        try:
            if request.method == "OPTIONS":
                return "", 204
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify({"error": "Invalid or missing JSON body"}), 400

            records = payload.get("records", [])

            if not records:
                return jsonify({"error": "records field is required and cannot be empty"}), 400

            if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                return jsonify({"error": "records must be a list of objects"}), 400

            required_fields = ["sender", "receiver", "date", "subject", "body"]
            invalid_records = []
            for index, record in enumerate(records):
                missing = [field for field in required_fields if field not in record]
                if missing:
                    invalid_records.append({"index": index, "missing": missing})

            if invalid_records:
                return jsonify({"error": "records missing required fields", "details": invalid_records}), 400

            input_df = pd.DataFrame(records)
            prediction = PredictPipeline().predict(input_df)

            results = []
            for record, predicted_label in zip(records, prediction):
                results.append(
                    {
                        "input": record,
                        "prediction": int(predicted_label),
                        "label": "phishing" if int(predicted_label) == 1 else "normal",
                    }
                )

            return jsonify({"results": results})

        except Exception as e:
            raise CustomException(e, sys)

    @app.route("/train", methods=["POST", "OPTIONS"])
    def train():
        try:
            if request.method == "OPTIONS":
                return "", 204

            if not enable_training:
                return jsonify({"error": "Training endpoint is disabled"}), 403

            if train_api_key:
                provided_key = request.headers.get("X-API-Key")
                if provided_key != train_api_key:
                    return jsonify({"error": "Unauthorized"}), 401

            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(
                train_path, test_path
            )

            model_trainer = ModelTrainer()
            model_report = model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )

            return jsonify({"status": "training completed", "model_report": model_report})

        except Exception as e:
            raise CustomException(e, sys)

    @app.errorhandler(CustomException)
    def handle_custom_exception(error):
        message = str(error) if is_debug else "Internal server error"
        return jsonify({"error": message}), 500

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error):
        if is_debug:
            return jsonify({"error": "Unexpected server error", "details": str(error)}), 500
        return jsonify({"error": "Unexpected server error"}), 500

    return app