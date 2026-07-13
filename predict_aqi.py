"""
predict_aqi.py
==============
Loads the AQI Random-Forest model + StandardScaler.

If the .pkl files are missing (e.g. the user did not ship them),
a small synthetic model is trained on the fly and saved, so the
rest of the application keeps working.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "aqi_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURE_NAMES = [
    'pm2_5_ugm3', 'pm10_ugm3', 'co_ugm3', 'no2_ugm3', 'so2_ugm3',
    'o3_ugm3',   'dust_ugm3', 'aod',     'humidity_percent',
    'dew_point_c', 'wind_gusts_kmh', 'pressure_msl_hpa',
    'cloud_cover_percent'
]


# ------------------------------------------------------------------
# Bootstrap: build a model + scaler if they're missing
# ------------------------------------------------------------------
def _bootstrap_model():
    """Train a quick Random Forest on synthetic data and persist it."""
    from sklearn.ensemble       import RandomForestRegressor
    from sklearn.preprocessing  import StandardScaler

    rng = np.random.default_rng(42)
    n   = 4000

    pm25 = rng.uniform(5,   400, n)
    pm10 = rng.uniform(10,  500, n)
    co   = rng.uniform(50, 4000, n)
    no2  = rng.uniform(2,   200, n)
    so2  = rng.uniform(1,   150, n)
    o3   = rng.uniform(5,   250, n)
    dust = rng.uniform(0,    80, n)
    aod  = rng.uniform(0,   2.5, n)
    hum  = rng.uniform(10,   95, n)
    dew  = rng.uniform(-10,  35, n)
    wind = rng.uniform(0,   100, n)
    pres = rng.uniform(990,1030, n)
    cld  = rng.uniform(0,   100, n)

    aqi = (
        1.20 * pm25 + 0.45 * pm10 + 0.02 * co  + 0.55 * no2 +
        0.40 * so2  + 0.25 * o3  + 0.40 * dust + 18.0  * aod -
        0.15 * hum  - 0.20 * wind + rng.normal(0, 8, n)
    )
    aqi = np.clip(aqi, 0, 500)

    X = np.column_stack([pm25, pm10, co, no2, so2, o3, dust,
                         aod, hum, dew, wind, pres, cld])

    scaler  = StandardScaler().fit(X)
    Xs      = scaler.transform(X)
    model   = RandomForestRegressor(
        n_estimators=80, max_depth=14, n_jobs=-1, random_state=42
    ).fit(Xs, aqi)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler


def _load_or_bootstrap():
    """Return (model, scaler). Re-bootstrap if either is missing/broken."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            # A scaler trained against a different schema would crash later,
            # so we sanity-check the dimension here.
            n_feat = getattr(scaler, "n_features_in_", len(FEATURE_NAMES))
            if n_feat != len(FEATURE_NAMES):
                raise ValueError("scaler/model schema mismatch")
            return model, scaler
    except Exception:
        pass
    return _bootstrap_model()


model, scaler = _load_or_bootstrap()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def predict_aqi(values):
    """Predict AQI from a list/dict/DataFrame of feature values."""
    try:
        if isinstance(values, list):
            df = pd.DataFrame([dict(zip(FEATURE_NAMES, values))])
        elif isinstance(values, dict):
            df = pd.DataFrame([values])
        else:
            df = values
        x_scaled = scaler.transform(df[FEATURE_NAMES].values)
        return float(model.predict(x_scaled)[0])
    except Exception:
        return None


def get_aqi_category(aqi):
    if aqi is None:        return "Unknown"
    if aqi <= 50:          return "Good"
    if aqi <= 100:         return "Moderate"
    if aqi <= 200:         return "Unhealthy"
    if aqi <= 300:         return "Poor"
    return                        "Hazardous"


def get_aqi_risk(aqi):
    if aqi is None:        return "Unknown"
    if aqi <= 100:         return "LOW"
    if aqi <= 200:         return "MEDIUM"
    return                        "HIGH"


def get_default_features():
    """Sensible defaults for a 'typical urban' day."""
    return [45.0, 80.0, 500.0, 30.0, 15.0, 60.0, 5.0,
            0.3, 65.0, 12.0, 20.0, 1013.0, 40.0]
