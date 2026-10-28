"""
AgriPredict AI - Machine Learning Pipeline Training Script
Builds a robust, production-grade Random Forest Regressor pipeline with:
- Scikit-learn Pipeline
- ColumnTransformer for OneHotEncoding and StandardScaler
- Cross-validation and test set evaluation (MAE, RMSE, R²)
- Joblib serialization
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "crop_yield_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

CATEGORICAL_FEATURES = ["crop_type", "state", "season", "soil_type"]
NUMERICAL_FEATURES = ["temperature", "rainfall", "humidity", "soil_moisture", "ndvi"]
TARGET_FEATURE = "yield"

def train_crop_yield_pipeline():
    """Trains and exports the Random Forest Crop Yield Prediction Pipeline."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run download_or_generate_dataset.py first.")

    print(f"Loading agricultural dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Drop any rows where target is missing
    df = df.dropna(subset=[TARGET_FEATURE])

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET_FEATURE]

    # Split dataset into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=df["crop_type"]
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 1. Numerical preprocessor
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Categorical preprocessor
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    # 4. Estimator: RandomForestRegressor
    rf_regressor = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # 5. Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_regressor)
    ])

    print("Fitting Random Forest pipeline...")
    pipeline.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))

    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))

    # Cross validation score
    print("Computing 5-fold cross validation R2...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std = float(cv_scores.std())

    metrics = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 120,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_mae": round(test_mae, 4),
        "test_rmse": round(test_rmse, 4),
        "test_r2": round(test_r2, 4),
        "train_r2": round(train_r2, 4),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_r2_std": round(cv_r2_std, 4),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": TARGET_FEATURE
    }

    print("\n--- Model Evaluation Results ---")
    print(f"Test MAE:  {test_mae:.4f} tonnes/ha")
    print(f"Test RMSE: {test_rmse:.4f} tonnes/ha")
    print(f"Test R²:   {test_r2:.4f}")
    print(f"CV 5-Fold R² Mean: {cv_r2_mean:.4f} (± {cv_r2_std:.4f})")
    print("--------------------------------\n")

    # Save model pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to: {MODEL_PATH}")

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {METRICS_PATH}")

    return pipeline, metrics

if __name__ == "__main__":
    train_crop_yield_pipeline()
