from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__, template_folder="../templates")
CORS(app)

print("🚀 Starting AgriSmart AI...")

# Initialize models
models_loaded = False
regressor = None
classifier = None
scaler_reg = None
scaler_clf = None
le = None

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    print("📁 Loading AI models...")
    regressor = joblib.load(os.path.join(MODEL_DIR, "soil_moisture_model.pkl"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "crop_classifier_model.pkl"))
    scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler_reg.pkl"))
    scaler_clf = joblib.load(os.path.join(MODEL_DIR, "scaler_clf.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    models_loaded = True
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    models_loaded = False

# Crop conditions
crop_conditions = {
    'Wheat': {'N': (60, 120), 'P': (30, 60), 'K': (70, 120), 'Temp': (15, 25), 'Humidity': (40, 70), 'Moisture': (20, 35)},
    'Rice': {'N': (80, 150), 'P': (40, 80), 'K': (100, 180), 'Temp': (20, 35), 'Humidity': (60, 90), 'Moisture': (30, 50)},
    'Cotton': {'N': (50, 100), 'P': (25, 55), 'K': (80, 140), 'Temp': (25, 40), 'Humidity': (30, 60), 'Moisture': (15, 30)},
    'Maize': {'N': (80, 150), 'P': (30, 70), 'K': (60, 120), 'Temp': (18, 32), 'Humidity': (50, 80), 'Moisture': (20, 40)},
    'Soybean': {'N': (40, 100), 'P': (20, 50), 'K': (50, 100), 'Temp': (20, 30), 'Humidity': (60, 85), 'Moisture': (25, 45)}
}

def analyze_nutrient(value, nutrient_name, optimal_range, crop_name):
    low, high = optimal_range
    if value < low:
        return {
            "nutrient": nutrient_name,
            "status": "Low",
            "level": "danger",
            "current_value": f"{value} kg/ha",
            "optimal_range": f"{low}-{high} kg/ha",
            "recommendation": f"Increase {nutrient_name} application for {crop_name}"
        }
    elif value > high:
        return {
            "nutrient": nutrient_name,
            "status": "High",
            "level": "warning", 
            "current_value": f"{value} kg/ha",
            "optimal_range": f"{low}-{high} kg/ha",
            "recommendation": f"Reduce {nutrient_name} application"
        }
    else:
        return {
            "nutrient": nutrient_name,
            "status": "Optimal",
            "level": "success",
            "current_value": f"{value} kg/ha",
            "optimal_range": f"{low}-{high} kg/ha",
            "recommendation": f"{nutrient_name} levels are perfect for {crop_name}"
        }

def calculate_confidence(N, P, K, temp, humidity):
    """Calculate prediction confidence score"""
    score = 0.8  # Base confidence
    
    # Adjust based on input validity
    if 0 <= N <= 200: score += 0.05
    if 0 <= P <= 200: score += 0.05  
    if 0 <= K <= 200: score += 0.05
    if 10 <= temp <= 45: score += 0.05
    if 20 <= humidity <= 90: score += 0.05
    
    return min(0.95, score)

def categorize_moisture(moisture):
    """Categorize moisture level"""
    if moisture < 20: return "Critical"
    elif moisture < 35: return "Low"
    elif moisture < 60: return "Optimal"
    elif moisture < 75: return "High"
    else: return "Excessive"

def generate_business_intelligence(crop):
    """Generate business intelligence insights"""
    crop_market_data = {
        'Wheat': {"demand": "High", "profit_margin": "Medium", "market_trend": "Stable"},
        'Rice': {"demand": "Very High", "profit_margin": "Medium", "market_trend": "Growing"},
        'Cotton': {"demand": "High", "profit_margin": "High", "market_trend": "Volatile"},
        'Maize': {"demand": "Medium", "profit_margin": "Low", "market_trend": "Stable"},
        'Soybean': {"demand": "High", "profit_margin": "Medium", "market_trend": "Growing"}
    }
    
    data = crop_market_data.get(crop, crop_market_data['Wheat'])
    
    return {
        "market_demand": data["demand"],
        "profit_potential": data["profit_margin"],
        "market_trend": data["market_trend"],
        "estimated_yield_increase": "15-25%",
        "cost_savings_potential": "20-30%",
        "roi_timeline": "1-2 seasons"
    }

def calculate_sustainability_score(N, P, K, moisture):
    """Calculate environmental sustainability score"""
    score = 100
    
    # Penalize excessive fertilizer use
    if N > 150: score -= 20
    if P > 80: score -= 15  
    if K > 200: score -= 15
    
    # Reward optimal moisture
    if 30 <= moisture <= 60: score += 10
    
    return max(0, min(100, score))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/health')
def health_check():
    return jsonify({
        "status": "operational",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📊 Received data:", data)
        
        # --- Extract raw input ---
        N = float(data.get('N', 50))
        P = float(data.get('P', 40))
        K = float(data.get('K', 45))
        temperature = float(data.get('Temperature', 25))
        humidity = float(data.get('Humidity', 60))
        
        print(f"🔢 Processing: N={N}, P={P}, K={K}, Temp={temperature}, Humidity={humidity}")
        
        # --- Compute engineered features ---
        NPK_Total = N + P + K
        NP_Ratio = N / (P + 1e-5)  # Avoid divide by zero
        NK_Ratio = N / (K + 1e-5)

        if models_loaded:
            # --- Prepare input for moisture regressor (8 features) ---
            moisture_input = np.array([[N, P, K, temperature, humidity, NPK_Total, NP_Ratio, NK_Ratio]])
            moisture_input_scaled = scaler_reg.transform(moisture_input)
            moisture_pred = regressor.predict(moisture_input_scaled)[0]

            # --- Prepare input for crop classifier (9 features: includes moisture) ---
            crop_input = np.array([[N, P, K, temperature, humidity, NPK_Total, NP_Ratio, NK_Ratio, moisture_pred]])
            crop_input_scaled = scaler_clf.transform(crop_input)
            crop_encoded = classifier.predict(crop_input_scaled)[0]
            crop_pred = le.inverse_transform([crop_encoded])[0]

            print(f"🤖 AI Prediction: {crop_pred} with {moisture_pred:.1f}% moisture")
            prediction_engine = "ai_models"
        else:
            # --- Fallback logic if models not loaded ---
            moisture_pred = 30 + (humidity * 0.3) + ((N + P + K) / 30)
            moisture_pred = max(10, min(80, moisture_pred))
            
            if N > 100 and humidity > 70:
                crop_pred = "Rice"
            elif temperature > 30:
                crop_pred = "Cotton"
            elif N < 60:
                crop_pred = "Soybean"
            elif temperature < 20:
                crop_pred = "Wheat"
            else:
                crop_pred = "Maize"
            
            prediction_engine = "fallback_logic"
            print(f"🔄 Fallback Prediction: {crop_pred} with {moisture_pred:.1f}% moisture")

        # --- Get optimal conditions ---
        cond = crop_conditions.get(crop_pred, crop_conditions['Wheat'])
        
        # --- Nutrient analysis ---
        nutrient_analysis = [
            analyze_nutrient(N, "Nitrogen", cond['N'], crop_pred),
            analyze_nutrient(P, "Phosphorus", cond['P'], crop_pred),
            analyze_nutrient(K, "Potassium", cond['K'], crop_pred)
        ]
        
        # --- Build response ---
        response = {
            "success": True,
            "prediction_engine": prediction_engine,
            "confidence_score": calculate_confidence(N, P, K, temperature, humidity),
            "moisture": round(moisture_pred, 2),
            "moisture_level": categorize_moisture(moisture_pred),
            "recommended_crop": crop_pred,
            "crop_suitability_score": 85,
            "nutrient_analysis": nutrient_analysis,
            "business_intelligence": generate_business_intelligence(crop_pred),
            "sustainability_score": calculate_sustainability_score(N, P, K, moisture_pred),
            "risk_assessment": "Low" if moisture_pred > 30 else "Medium",
            "timestamp": datetime.now().isoformat(),
            "prediction_id": f"AGRISMART-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "overview": {
                "temperature": f"{temperature}°C",
                "humidity": f"{humidity}%",
                "nitrogen": f"{N} kg/ha",
                "phosphorus": f"{P} kg/ha",
                "potassium": f"{K} kg/ha"
            }
        }

        print(f"✅ Prediction successful: {crop_pred}")
        return jsonify(response)

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            "success": False,
            "error": "Internal server error. Please try again.",
            "error_code": "PREDICTION_FAILED"
        }), 500


if __name__ == "__main__":
    print(f"🌐 AgriSmart AI Server Ready!")
    print(f"📊 Models Loaded: {models_loaded}")
    print(f"🚀 Starting server at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)