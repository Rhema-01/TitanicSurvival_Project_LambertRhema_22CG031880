import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION (Absolute Pathing) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'titanic_survival_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'titanic_scaler.pkl')

# Load real artifacts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None
    print(f"Error loading model artifacts: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Server-side model files missing.'}), 500
    
    try:
        data = request.json.get('data') # [Pclass, Sex, Age, SibSp, Fare]
        
        # Mapping Sex string back to numeric
        sex_map = 0 if data[1].lower() == 'male' else 1
        
        # Construct input array in correct feature order
        # Pclass, Sex, Age, SibSp, Fare
        features = np.array([[float(data[0]), sex_map, float(data[2]), float(data[3]), float(data[4])]])
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        result = "Survived" if prediction[0] == 1 else "Did Not Survive"
        color = "#059669" if prediction[0] == 1 else "#1e293b" # Green vs Deep Navy
        
        return jsonify({'prediction': result, 'color': color})
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    # Recommended for production standard
    app.run(debug=False)