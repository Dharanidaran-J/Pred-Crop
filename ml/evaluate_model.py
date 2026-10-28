"""
AgriPredict AI - Model Evaluation Script
Computes detailed metrics across all crops, feature importances,
and residual error distributions.
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

    print("Loading model and dataset for deep evaluation...")
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    features = ["crop_type", "state", "season", "soil_type", "temperature", "rainfall", "humidity", "soil_moisture", "ndvi"]
    X = df[features]
    y_true = df["yield"]
    y_pred = pipeline.predict(X)

    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)

    print("\n==========================================")
    print("      AgriPredict AI Evaluation Report     ")
    print("==========================================")
    print(f"Overall MAE:  {overall_mae:.4f} t/ha")
    print(f"Overall RMSE: {overall_rmse:.4f} t/ha")
    print(f"Overall R²:   {overall_r2:.4f}")
    print("------------------------------------------")
    print("Performance Breakdown by Crop Type:")
    print("------------------------------------------")

    df_eval = df.copy()
    df_eval["predicted_yield"] = y_pred
    df_eval["error"] = df_eval["predicted_yield"] - df_eval["yield"]

    for crop in df["crop_type"].unique():
        sub = df_eval[df_eval["crop_type"] == crop]
        c_mae = mean_absolute_error(sub["yield"], sub["predicted_yield"])
        c_rmse = np.sqrt(mean_squared_error(sub["yield"], sub["predicted_yield"]))
        c_r2 = r2_score(sub["yield"], sub["predicted_yield"])
        print(f"Crop: {crop:10s} | MAE: {c_mae:.3f} | RMSE: {c_rmse:.3f} | R²: {c_r2:.3f} | Sample Size: {len(sub)}")
    
    print("==========================================\n")

if __name__ == "__main__":
    evaluate_performance()
