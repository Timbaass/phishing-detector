import os
import sys

from flask import Flask, jsonify, render_template, request

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.security.url_analyzer import UrlAnalyzer
from src.security.url_extractor import extract_urls
from src.services.llm_service import LlmService

def create_app(test_config=None):
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
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

            prediction, proba = PredictPipeline().predict(input_df)

            model_prediction = int(prediction[0])

            if proba is not None and len(proba) > 0:
                model_probability = round(float(proba[0]) * 100, 2)
            else:
                model_probability = None

            vt_probability = None
            vt_urls = []
            vt_error = None

            try:
                text_for_urls = f"{payload['subject']} {payload['body']}"
                urls = extract_urls(text_for_urls)
                if urls:
                    analyzer = UrlAnalyzer()
                    max_urls = int(os.getenv("VT_MAX_URLS", "3"))
                    for url in urls[:max_urls]:
                        report = analyzer.analyze_url_with_report(url)
                        score = analyzer.calculate_vt_score(report)
                        stats = (
                            report.get("data", {})
                            .get("attributes", {})
                            .get("stats", {})
                        )
                        analysis_url = (
                            report.get("data", {})
                            .get("links", {})
                            .get("self")
                        )
                        vt_urls.append(
                            {
                                "url": url,
                                "score": score,
                                "stats": stats,
                                "analysis_url": analysis_url,
                            }
                        )
                        if score is not None:
                            if vt_probability is None or score > vt_probability:
                                vt_probability = score
            except Exception as vt_exc:
                vt_error = str(vt_exc)

            combined_probability = model_probability
            if model_probability is not None and vt_probability is not None:
                combined_probability = round(
                    (model_probability * 0.65) + (vt_probability * 0.35),
                    2,
                )
            elif vt_probability is not None:
                combined_probability = vt_probability

            final_prediction = model_prediction
            if combined_probability is not None:
                final_prediction = 1 if combined_probability >= 50 else 0
                            
            security_report = {
                "final_prediction": int(final_prediction),
                "final_label": "phishing" if final_prediction == 1 else "normal",
                "final_probability": float(combined_probability),

                "ml_model": {
                    "prediction": int(model_prediction),
                    "label": "phishing" if model_prediction == 1 else "normal",
                    "probability": float(model_probability),
                },

                "virustotal": {
                    "probability": float(vt_probability) if vt_probability is not None else None,
                    "urls": vt_urls,
                    "error": vt_error,
                }
            }
            
            llm_service = LlmService()
            ai_explanation = llm_service.generate_explanation(security_report)

            return jsonify(
                {
                    "prediction": final_prediction,
                    "label": "phishing" if final_prediction == 1 else "normal",
                    "probability": combined_probability,

                    "ai_explanation": ai_explanation,

                    "technical_details": {
                        "model_prediction": model_prediction,
                        "model_probability": model_probability,
                        "vt_probability": vt_probability,
                        "vt_urls": vt_urls,
                        "vt_error": vt_error,
                    }
                }
            )

        except Exception as e:
            raise CustomException(e, sys)

    @app.errorhandler(CustomException)
    def handle_custom_exception(error):
        return jsonify({"error": str(error)}), 500

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error):
        return jsonify({"error": "Unexpected server error", "details": str(error)}), 500

    return app