"""
AgriPredict AI - Automated Test Suite
Validates Machine Learning Pipeline, Database ORM, Auth Flows,
Weather and Satellite Service fallbacks, and API endpoints.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import _patch_cython  # Windows App Control / WDAC Cython compatibility hook
import pytest
from app import app, db
from models_db import User, Prediction
from ml.predict import CropYieldPredictor
from services.weather_service import WeatherService
from services.satellite_service import SatelliteService

@pytest.fixture
def client():
    """Configures an in-memory testing client."""
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


# -------------------------------------------------------------------------
# 1. Machine Learning Model & Validation Tests
# -------------------------------------------------------------------------
def test_model_loading_and_prediction():
    """Validates that the Random Forest model loads and predicts valid agronomic ranges."""
    payload = {
        "crop_type": "Wheat",
        "state": "Punjab",
        "district": "Ludhiana",
        "land_area": 3.0,
        "season": "Rabi",
        "soil_type": "Alluvial",
        "temperature": 18.5,
        "rainfall": 450.0,
        "humidity": 60.0,
        "soil_moisture": 32.0,
        "ndvi": 0.72
    }
    result = CropYieldPredictor.predict(payload)
    assert result["crop_type"] == "Wheat"
    assert 2.0 <= result["predicted_yield"] <= 15.0
    assert result["estimated_total_production"] == round(result["predicted_yield"] * 3.0, 2)
    assert "benchmark_yield" in result
    assert result["ndvi_status"] is not None


def test_input_validation_invalid_crop():
    """Verifies that invalid crop names are rejected with ValueError."""
    payload = {
        "crop_type": "Pineapple",
        "state": "Punjab",
        "land_area": 1.0,
        "season": "Kharif",
        "soil_type": "Alluvial"
    }
    with pytest.raises(ValueError) as exc:
        CropYieldPredictor.predict(payload)
    assert "Invalid crop type" in str(exc.value)


def test_input_validation_negative_area():
    """Verifies that negative land areas are rejected."""
    payload = {
        "crop_type": "Rice",
        "state": "Punjab",
        "land_area": -5.0
    }
    with pytest.raises(ValueError) as exc:
        CropYieldPredictor.predict(payload)
    assert "Land area must be a positive number" in str(exc.value)


# -------------------------------------------------------------------------
# 2. Authentication & Security Tests
# -------------------------------------------------------------------------
def test_user_registration_and_login(client):
    """Tests successful user registration, secure password hashing, and login."""
    # Register
    reg_response = client.post("/register", data={
        "name": "Kisan Patel",
        "email": "kisan@example.com",
        "password": "FarmerPassword123",
        "confirm_password": "FarmerPassword123"
    }, follow_redirects=True)
    assert reg_response.status_code == 200
    assert b"Account registered successfully" in reg_response.data

    # Verify user in database
    with app.app_context():
        user = User.query.filter_by(email="kisan@example.com").first()
        assert user is not None
        assert user.name == "Kisan Patel"
        # Password must NOT be plaintext
        assert user.password_hash != "FarmerPassword123"
        assert user.check_password("FarmerPassword123") is True
        assert user.check_password("WrongPassword") is False

    # Login
    login_response = client.post("/login", data={
        "email": "kisan@example.com",
        "password": "FarmerPassword123"
    }, follow_redirects=True)
    assert login_response.status_code == 200
    assert b"Welcome back, Kisan Patel" in login_response.data


def test_unauthorized_dashboard_access(client):
    """Verifies unauthenticated users are redirected from protected routes."""
    res = client.get("/dashboard", follow_redirects=False)
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]


# -------------------------------------------------------------------------
# 3. Database Operations & Prediction History Tests
# -------------------------------------------------------------------------
def test_prediction_creation_and_deletion(client):
    """Tests saving predictions to the database and subsequent deletion."""
    # Register and log in
    client.post("/register", data={
        "name": "Aman Verma",
        "email": "aman@example.com",
        "password": "SecurePassword99",
        "confirm_password": "SecurePassword99"
    })
    client.post("/login", data={
        "email": "aman@example.com",
        "password": "SecurePassword99"
    })

    # Submit prediction form
    pred_res = client.post("/predict", data={
        "crop_type": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "land_area": 4.5,
        "season": "Kharif",
        "soil_type": "Clay",
        "temperature": 27.5,
        "rainfall": 1150.0,
        "humidity": 78.0,
        "soil_moisture": 42.0,
        "ndvi": 0.78
    }, follow_redirects=True)
    assert pred_res.status_code == 200
    assert b"Crop Yield &amp; Production Report" in pred_res.data or b"Crop Yield" in pred_res.data

    # Check history page
    hist_res = client.get("/history")
    assert b"Rice" in hist_res.data
    assert b"Ludhiana" in hist_res.data

    # Delete prediction
    with app.app_context():
        user = User.query.filter_by(email="aman@example.com").first()
        pred = Prediction.query.filter_by(user_id=user.id).first()
        assert pred is not None
        pred_id = pred.id

    del_res = client.post(f"/history/delete/{pred_id}", follow_redirects=True)
    assert del_res.status_code == 200
    assert b"Prediction entry successfully deleted" in del_res.data


# -------------------------------------------------------------------------
# 4. External Services & Fallback Tests
# -------------------------------------------------------------------------
def test_weather_service_graceful_fallback():
    """Ensures weather service returns transparent demo data without API keys."""
    res = WeatherService.get_weather("Ludhiana", api_key="")
    assert res is not None
    assert "temp" in res
    assert "humidity" in res
    assert res["is_demo"] is True
    assert "Simulation Mode" in res["data_source"] or "Demonstration" in res["data_source"]


def test_satellite_service_graceful_fallback():
    """Ensures satellite service returns valid Sentinel-2 demo telemetry."""
    res = SatelliteService.get_satellite_health("Ludhiana")
    assert res is not None
    assert 0.0 <= res["ndvi"] <= 1.0
    assert "Sentinel-2" in res["satellite"]
    assert len(res["trends"]) > 0
    assert res["is_demo"] is True


# -------------------------------------------------------------------------
# 5. REST API Endpoints Tests
# -------------------------------------------------------------------------
def test_api_predict_endpoint(client):
    """Tests the /api/predict JSON endpoint."""
    res = client.post("/api/predict", json={
        "crop_type": "Cotton",
        "state": "Gujarat",
        "land_area": 5.0,
        "season": "Kharif",
        "soil_type": "Black",
        "temperature": 28.0,
        "rainfall": 800.0,
        "humidity": 60.0
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["data"]["crop_type"] == "Cotton"
    assert data["data"]["predicted_yield"] > 0
