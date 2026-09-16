from flask import Flask, request, jsonify
from pyngrok import ngrok
import joblib
import pandas as pd
import threading
import time

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("standard_scaler.joblib")

# Define feature columns used during training
# Replace these with your actual column names
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
    return {
        "message": "Diabetes Prediction API is running",
        "endpoint": "/predict"
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([json_data])

        # Check for missing columns
        missing_cols = [
            col for col in feature_columns
            if col not in input_df.columns
        ]

        if missing_cols:
            return jsonify(
                {"error": f"Missing columns: {missing_cols}"}
            ), 400

        # Arrange columns in correct order
        input_df = input_df[feature_columns]

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
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


def run_ngrok():
    """
    Starts ngrok tunnel and keeps it alive.
    """

    # Uncomment and add your auth token if required
    # ngrok.set_auth_token("YOUR_AUTHTOKEN")

    public_url = ngrok.connect(addr=5000, proto="http")

    print(f"\n* ngrok tunnel available at: {public_url}")
    print(f"* API Endpoint: {public_url}/predict\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ngrok.kill()


if __name__ == "__main__":

    # Start ngrok in background
    ngrok_thread = threading.Thread(target=run_ngrok)
    ngrok_thread.daemon = True
    ngrok_thread.start()

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
