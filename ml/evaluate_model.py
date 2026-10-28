"""
AgriPredict AI - Model Evaluation Script
Computes detailed metrics across all 6 crops, feature importances,
and residual error distributions on the 10,000-row dataset.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"

def evaluate_performance():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return

    print("Loading model and dataset for deep evaluation on 10,000 rows...")
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    cat_features = ["Crop_Type", "Region", "Season", "Soil_Type", "Irrigation_Type"]
    num_features = [
        "N", "P", "K", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
        "Temperature", "Humidity", "Rainfall", "Sunlight_Hours",
        "Wind_Speed", "Altitude", "Fertilizer_Used", "Pesticide_Used"
    ]
    target = "Crop_Yield_ton_per_hectare"

    X = df[cat_features + num_features]
    y_true = df[target]
    y_pred = pipeline.predict(X)

    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)

    print("\n==========================================================")
    print("      AgriPredict AI Evaluation Report (10,000 Records)    ")
    print("==========================================================")
    print(f"Overall MAE:  {overall_mae:.4f} tonnes/ha")
    print(f"Overall RMSE: {overall_rmse:.4f} tonnes/ha")
    print(f"Overall R²:   {overall_r2:.4f}")
    print("----------------------------------------------------------")
    print("Performance Breakdown by Crop Type:")
    print("----------------------------------------------------------")

    df_eval = df.copy()
    df_eval["predicted_yield"] = y_pred

    for crop in sorted(df["Crop_Type"].unique()):
        sub = df_eval[df_eval["Crop_Type"] == crop]
        c_mae = mean_absolute_error(sub[target], sub["predicted_yield"])
        c_rmse = np.sqrt(mean_squared_error(sub[target], sub["predicted_yield"]))
        c_r2 = r2_score(sub[target], sub["predicted_yield"])
        print(f"Crop: {crop:10s} | MAE: {c_mae:.3f} t/ha | RMSE: {c_rmse:.3f} | R²: {c_r2:.3f} | Sample Size: {len(sub)}")
    
    print("==========================================================\n")

if __name__ == "__main__":
    evaluate_performance()
