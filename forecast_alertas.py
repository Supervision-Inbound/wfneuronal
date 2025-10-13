# =======================================================================
# forecast_alertas.py
# Inferencia IA de alertas climáticas por comuna usando clima futuro
# Genera un JSON con predicción por comuna, % incremento, y flag alerta
# =======================================================================

import os, json
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from dateutil import tz

# ---------------- CONFIG ----------------
MODEL_NAME = "modelo_alertas_clima.h5"
SCALER_NAME = "scaler_alertas_clima.pkl"
COLS_NAME = "training_columns_alertas_clima.json"
LOC_CSV = "data/Comunas_Cordenadas.csv"
OUT_JSON = "public/alertas_clima.json"

CLIMA_API_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["temperature_2m", "precipitation", "rain", "wind_speed_10m", "wind_gusts_10m"]
TIMEZONE = "America/Santiago"
ALERTA_THRESHOLD = 0.40  # 40% de aumento esperado
FORECAST_HOURS = 72

# -------------- Funciones --------------
def fetch_forecast(lat, lon):
    import requests
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": 3,
        "timezone": TIMEZONE
    }
    r = requests.get(CLIMA_API_URL, params=params)
    r.raise_for_status()
    return r.json()

def process_clima_json(data, comuna):
    times = pd.to_datetime(data["hourly"]["time"])
    df = pd.DataFrame({"ts": times})
    for var in HOURLY_VARS:
        df[var] = data["hourly"].get(var, np.nan)
    df["comuna"] = comuna
    return df

def add_time_features(df):
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.month
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df

# -------------- Main ------------------
def main():
    print("📦 Cargando modelo y artefactos...")
    model = tf.keras.models.load_model(f"models/{MODEL_NAME}")
    scaler = joblib.load(f"models/{SCALER_NAME}")
    with open(f"models/{COLS_NAME}", "r") as f:
        training_cols = json.load(f)

    print("📍 Cargando coordenadas de comunas...")
    df_loc = pd.read_csv(LOC_CSV)
    df_loc.columns = df_loc.columns.str.strip().str.lower()
    df_loc = df_loc.drop_duplicates(subset=["comuna"])
    all_preds = []

    for _, row in df_loc.iterrows():
        comuna, lat, lon = row["comuna"], row["lat"], row["lon"]
        print(f"🌦️ Consultando clima para: {comuna}...")
        try:
            raw = fetch_forecast(lat, lon)
            df_c = process_clima_json(raw, comuna)
            df_c = add_time_features(df_c)
            X = df_c[["temperature_2m", "precipitation", "rain",
                      "wind_speed_10m", "wind_gusts_10m",
                      "sin_hour", "cos_hour", "sin_dow", "cos_dow",
                      "dow", "hour", "month"]]
            X = pd.get_dummies(X, columns=["dow", "hour", "month"], drop_first=False)
            for col in training_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[training_cols].fillna(0)
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled, verbose=0).flatten()
            df_c["llamadas_estimadas"] = y_pred
            df_c["porcentaje_incremento"] = (df_c["llamadas_estimadas"] / y_pred.mean()) - 1
            df_c["alerta"] = df_c["porcentaje_incremento"] > ALERTA_THRESHOLD
            all_preds.append(df_c)
        except Exception as e:
            print(f"⚠️ Error en {comuna}: {e}")

    if not all_preds:
        print("❌ No se pudieron generar predicciones.")
        return

    df_all = pd.concat(all_preds, ignore_index=True)
    df_all = df_all[["comuna", "ts", "llamadas_estimadas", "porcentaje_incremento", "alerta"]]
    df_all["ts"] = df_all["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_all["llamadas_estimadas"] = df_all["llamadas_estimadas"].round(1)
    df_all["porcentaje_incremento"] = (df_all["porcentaje_incremento"] * 100).round(1)
    df_all["alerta"] = df_all["alerta"].astype(bool)

    os.makedirs("public", exist_ok=True)
    df_all.to_json(OUT_JSON, orient="records", indent=2)
    print(f"✅ JSON de alertas generado: {OUT_JSON}")

if __name__ == "__main__":
    main()
