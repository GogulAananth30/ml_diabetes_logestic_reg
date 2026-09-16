from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("standard_scaler.joblib")

# Feature columns
feature_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


@app.route("/")
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running",
        "endpoint": "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        input_df = pd.DataFrame([json_data])

        missing_cols = [
            col for col in feature_columns
            if col not in input_df.columns
        ]

        if missing_cols:
            return jsonify({
                "error": f"Missing columns: {missing_cols}"
            }), 400

        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        result = (
            "Diabetic"
            if prediction[0] == 1
            else "Non-Diabetic"
        )

        return jsonify({
            "prediction": result,
            "probability_non_diabetic": float(prediction_proba[0][0]),
            "probability_diabetic": float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
