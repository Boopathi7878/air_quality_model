# deployment/app.py

from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load trained models
rf_model = joblib.load("../models/random_forest_model.pkl")
nn_model = load_model("../models/neural_network_model.keras")

@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = rf_model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predict/nn', methods=['POST'])
def predict_nn():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = nn_model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
