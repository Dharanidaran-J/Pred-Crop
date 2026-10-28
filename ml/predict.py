"""
AgriPredict AI - Prediction Engine
Loads the trained Random Forest Pipeline trained on 10,000 agricultural observations.
Handles inference, input feature mapping (NPK, soil chemistry, weather, irrigation),
input validation, production aggregation, and agronomic factor insights.
"""

from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"

# Benchmarks for comparison derived from 10,000-row empirical dataset (Tonnes per Hectare)
CROP_BENCHMARKS = {
    "Rice": {"avg_yield": 9.40, "min": 4.0, "max": 14.5, "unit": "tonnes/ha"},
    "Wheat": {"avg_yield": 8.72, "min": 3.5, "max": 14.5, "unit": "tonnes/ha"},
    "Maize": {"avg_yield": 10.08, "min": 5.0, "max": 16.0, "unit": "tonnes/ha"},
    "Cotton": {"avg_yield": 7.43, "min": 1.5, "max": 13.0, "unit": "tonnes/ha"},
    "Sugarcane": {"avg_yield": 74.83, "min": 65.0, "max": 85.0, "unit": "tonnes/ha"},
    "Potato": {"avg_yield": 24.89, "min": 20.0, "max": 31.0, "unit": "tonnes/ha"},
}

STATE_TO_REGION = {
    "punjab": "North", "haryana": "North", "uttar pradesh": "North", "himachal pradesh": "North",
    "delhi": "North", "uttarakhand": "North", "jammu & kashmir": "North",
    "tamil nadu": "South", "karnataka": "South", "andhra pradesh": "South", "telangana": "South", "kerala": "South",
    "maharashtra": "West", "gujarat": "West", "rajasthan": "West", "goa": "West",
    "west bengal": "East", "bihar": "East", "odisha": "East", "jharkhand": "East", "assam": "East",
    "madhya pradesh": "Central", "chhattisgarh": "Central"
}

SOIL_MAPPING = {
    "loamy": "Loamy", "sandy": "Sandy", "clay": "Clay", "silt": "Silt",
    "alluvial": "Loamy", "black": "Clay", "red": "Loamy"
}

SEASON_MAPPING = {
    "kharif": "Kharif", "rabi": "Rabi", "zaid": "Zaid", "summer": "Zaid", "whole year": "Kharif"
}

class CropYieldPredictor:
    """Manages model loading and prediction inference."""
    _model = None

    @classmethod
    def load_model(cls):
        """Lazy load the Scikit-learn Pipeline."""
        if cls._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found at {MODEL_PATH}. Run 'python ml/train_model.py' to generate."
                )
            cls._model = joblib.load(MODEL_PATH)
        return cls._model

    @classmethod
    def predict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates input features, predicts yield per hectare, and calculates total production.
        """
        model = cls.load_model()

        # Validate crop type
        raw_crop = str(data.get("crop_type", "")).strip().capitalize()
        # Normalization
        matched_crop = None
        for c in CROP_BENCHMARKS:
            if c.lower() == raw_crop.lower():
                matched_crop = c
                break
        
        if not matched_crop:
            raise ValueError(f"Invalid crop type '{raw_crop}'. Supported crops: {list(CROP_BENCHMARKS.keys())}")

        crop = matched_crop

        # Land area validation
        try:
            land_area = float(data.get("land_area", 1.0))
        except (ValueError, TypeError):
            raise ValueError("Invalid land area provided.")

        if land_area <= 0:
            raise ValueError("Land area must be a positive number greater than 0.")

        # Map state/region
        state_input = str(data.get("state", "North")).strip().lower()
        region = STATE_TO_REGION.get(state_input, data.get("region", "North"))
        if region not in ["North", "South", "East", "West", "Central"]:
            region = "North"

        # Map season
        season_input = str(data.get("season", "Kharif")).strip().lower()
        season = SEASON_MAPPING.get(season_input, "Kharif")

        # Map soil
        soil_input = str(data.get("soil_type", "Loamy")).strip().lower()
        soil_type = SOIL_MAPPING.get(soil_input, "Loamy")

        # Irrigation
        irrigation = str(data.get("irrigation_type", "Canal")).strip().capitalize()
        if irrigation not in ["Drip", "Canal", "Rainfed", "Sprinkler"]:
            irrigation = "Canal"

        # Meteorological and Soil features with agronomic fallbacks
        temp = float(data.get("temperature", 25.0))
        rainfall = float(data.get("rainfall", 750.0))
        humidity = float(data.get("humidity", 60.0))
        soil_moisture = float(data.get("soil_moisture", 40.0)) if data.get("soil_moisture") else 40.0
        
        # Chemical & farm practices (with sensible agricultural defaults if not provided)
        n = int(float(data.get("n", 100))) if data.get("n") is not None else 100
        p = int(float(data.get("p", 50))) if data.get("p") is not None else 50
        k = int(float(data.get("k", 50))) if data.get("k") is not None else 50
        soil_ph = float(data.get("soil_ph", 6.5)) if data.get("soil_ph") is not None else 6.5
        organic_carbon = float(data.get("organic_carbon", 0.75)) if data.get("organic_carbon") is not None else 0.75
        sunlight_hours = float(data.get("sunlight_hours", 8.0)) if data.get("sunlight_hours") is not None else 8.0
        wind_speed = float(data.get("wind_speed", 12.0)) if data.get("wind_speed") is not None else 12.0
        altitude = int(float(data.get("altitude", 250))) if data.get("altitude") is not None else 250
        fertilizer_used = float(data.get("fertilizer_used", 180.0)) if data.get("fertilizer_used") is not None else 180.0
        pesticide_used = float(data.get("pesticide_used", 12.0)) if data.get("pesticide_used") is not None else 12.0
        
        # Satellite NDVI
        ndvi = float(data.get("ndvi", 0.65)) if data.get("ndvi") is not None else 0.65

        # Build feature DataFrame matching trained pipeline
        feature_dict = {
            "Crop_Type": [crop],
            "Region": [region],
            "Season": [season],
            "Soil_Type": [soil_type],
            "Irrigation_Type": [irrigation],
            "N": [n],
            "P": [p],
            "K": [k],
            "Soil_pH": [soil_ph],
            "Soil_Moisture": [soil_moisture],
            "Organic_Carbon": [organic_carbon],
            "Temperature": [temp],
            "Humidity": [humidity],
            "Rainfall": [rainfall],
            "Sunlight_Hours": [sunlight_hours],
            "Wind_Speed": [wind_speed],
            "Altitude": [altitude],
            "Fertilizer_Used": [fertilizer_used],
            "Pesticide_Used": [pesticide_used]
        }
        input_df = pd.DataFrame(feature_dict)

        # Model inference
        predicted_yield = float(model.predict(input_df)[0])
        predicted_yield = max(0.1, round(predicted_yield, 2))

        # Calculate estimated total harvest
        total_production = round(predicted_yield * land_area, 2)

        # Performance vs Benchmark
        benchmark = CROP_BENCHMARKS[crop]["avg_yield"]
        delta_pct = round(((predicted_yield - benchmark) / benchmark) * 100, 1)

        # NDVI health classification
        if ndvi >= 0.70:
            ndvi_status = "Very Healthy / Peak Canopy Vigour"
            ndvi_badge = "success"
        elif ndvi >= 0.50:
            ndvi_status = "Normal Vegetative Vigor"
            ndvi_badge = "primary"
        elif ndvi >= 0.30:
            ndvi_status = "Moderate / Early Growth or Moisture Stress"
            ndvi_badge = "warning"
        else:
            ndvi_status = "Low Vegetative Vigor / Sparse Canopy"
            ndvi_badge = "danger"

        return {
            "crop_type": crop,
            "state": data.get("state", region),
            "region": region,
            "district": data.get("district", "General Farmland Region"),
            "season": season,
            "soil_type": soil_type,
            "irrigation_type": irrigation,
            "land_area": land_area,
            "temperature": temp,
            "rainfall": rainfall,
            "humidity": humidity,
            "soil_moisture": soil_moisture,
            "n": n,
            "p": p,
            "k": k,
            "soil_ph": soil_ph,
            "organic_carbon": organic_carbon,
            "sunlight_hours": sunlight_hours,
            "fertilizer_used": fertilizer_used,
            "ndvi": ndvi,
            "ndvi_status": ndvi_status,
            "ndvi_badge": ndvi_badge,
            "predicted_yield": predicted_yield,
            "estimated_total_production": total_production,
            "benchmark_yield": benchmark,
            "yield_delta_pct": delta_pct,
            "unit": "tonnes/hectare",
            "total_unit": "tonnes"
        }
