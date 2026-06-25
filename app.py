from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from irrigation import get_irrigation_suggestion

app = Flask(__name__)
CORS(app)

model    = joblib.load("model_artifacts/model.pkl")
encoders = joblib.load("model_artifacts/label_encoders.pkl")

feature_cols = ["Crop_enc","Season_enc","State_enc","Area","Annual_Rainfall","Fertilizer","Pesticide"]

def encode_input(crop, season, state, area, rainfall, fertilizer, pesticide):
    return {
        "Crop_enc": int(encoders["Crop"].transform([crop])[0]),
        "Season_enc": int(encoders["Season"].transform([season])[0]),
        "State_enc": int(encoders["State"].transform([state])[0]),
        "Area": float(area), "Annual_Rainfall": float(rainfall),
        "Fertilizer": float(fertilizer), "Pesticide": float(pesticide),
    }

def predict_yield(encoded):
    df   = pd.DataFrame([encoded])[feature_cols]
    pred = model.predict(df)[0]
    return float(np.expm1(pred))

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Crop Yield API is running!"})

@app.route("/options", methods=["GET"])
def options():
    return jsonify({
        "crops"  : sorted(list(encoders["Crop"].classes_)),
        "seasons": sorted(list(encoders["Season"].classes_)),
        "states" : sorted(list(encoders["State"].classes_)),
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        crop, season, state = data["crop"], data["season"], data["state"]
        encoded    = encode_input(crop, season, state, data["area"], data["rainfall"], data["fertilizer"], data["pesticide"])
        pred_yield = predict_yield(encoded)
        return jsonify({"predicted_yield": round(pred_yield, 4), "unit": "tonnes/hectare", "crop": crop, "season": season, "state": state})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        data = request.get_json()
        crop, season, state = data["crop"], data["season"], data["state"]
        fertilizer, pesticide = float(data["fertilizer"]), float(data["pesticide"])
        base_encoded = encode_input(crop, season, state, data["area"], data["rainfall"], fertilizer, pesticide)
        base_yield   = predict_yield(base_encoded)
        best_yield, best_fert, best_pest = base_yield, fertilizer, pesticide
        for fm in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
            for pm in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
                trial_enc   = encode_input(crop, season, state, data["area"], data["rainfall"], fertilizer*fm, pesticide*pm)
                trial_yield = predict_yield(trial_enc)
                if trial_yield > best_yield:
                    best_yield, best_fert, best_pest = trial_yield, fertilizer*fm, pesticide*pm
        improvement = ((best_yield - base_yield) / base_yield * 100) if base_yield > 0 else 0
        return jsonify({
            "current"  : {"fertilizer": round(fertilizer,2), "pesticide": round(pesticide,2), "predicted_yield": round(base_yield,4)},
            "optimized": {"fertilizer": round(best_fert,2),  "pesticide": round(best_pest,2),  "predicted_yield": round(best_yield,4)},
            "improvement_percent": round(improvement, 2), "unit": "tonnes/hectare",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/irrigate", methods=["POST"])
def irrigate():
    try:
        data   = request.get_json()
        result = get_irrigation_suggestion(data["crop"], data["season"], float(data["rainfall"]))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Crop Yield API...")
    print("Open http://127.0.0.1:5000 in your browser to check it's running")
    app.run(debug=True, port=5001)