"""
AgriPredict AI - Prediction Engine
Loads the trained Random Forest Pipeline and handles inference, input validation,
production aggregation, and agronomic factor insights.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"

# National/Historical average benchmarks for comparison (Tonnes per Hectare)
CROP_BENCHMARKS = {
    "Rice": {"avg_yield": 3.60, "min": 1.5, "max": 6.2, "unit": "tonnes/ha"},
    "Wheat": {"avg_yield": 4.10, "min": 2.0, "max": 5.9, "unit": "tonnes/ha"},
    "Maize": {"avg_yield": 3.85, "min": 1.8, "max": 7.2, "unit": "tonnes/ha"},
    "Cotton": {"avg_yield": 2.10, "min": 0.9, "max": 3.6, "unit": "tonnes/ha"},
    "Sugarcane": {"avg_yield": 78.5, "min": 45.0, "max": 115.0, "unit": "tonnes/ha"},
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
        crop = data.get("crop_type", "").strip()
        if crop not in CROP_BENCHMARKS:
            raise ValueError(f"Invalid crop type '{crop}'. Supported crops: {list(CROP_BENCHMARKS.keys())}")

        # Land area validation
        try:
            land_area = float(data.get("land_area", 1.0))
        except (ValueError, TypeError):
            raise ValueError("Invalid land area provided.")

        if land_area <= 0:
            raise ValueError("Land area must be a positive number greater than 0.")

        # Default fallbacks for optional or missing sensors
        temp = float(data.get("temperature", 25.0))
        rainfall = float(data.get("rainfall", 750.0))
        humidity = float(data.get("humidity", 65.0))
        soil_moisture = float(data.get("soil_moisture", 35.0)) if data.get("soil_moisture") else 35.0
        ndvi = float(data.get("ndvi", 0.60)) if data.get("ndvi") else 0.60

        # Build feature DataFrame matching trained pipeline
        feature_dict = {
            "crop_type": [crop],
            "state": [data.get("state", "Punjab").strip()],
            "season": [data.get("season", "Kharif").strip()],
            "soil_type": [data.get("soil_type", "Alluvial").strip()],
            "temperature": [temp],
            "rainfall": [rainfall],
            "humidity": [humidity],
            "soil_moisture": [soil_moisture],
            "ndvi": [ndvi]
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
            ndvi_status = "Very Healthy / High Vigour"
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
            "state": feature_dict["state"][0],
            "district": data.get("district", "General Region"),
            "season": feature_dict["season"][0],
            "soil_type": feature_dict["soil_type"][0],
            "land_area": land_area,
            "temperature": temp,
            "rainfall": rainfall,
            "humidity": humidity,
            "soil_moisture": soil_moisture,
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
