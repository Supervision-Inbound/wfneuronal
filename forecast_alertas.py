# =======================================================================
# forecast_alertas.py
# Inferencia IA de alertas climáticas por comuna usando clima futuro
# Genera JSON con: comuna, ts, llamadas_estimadas, porcentaje_incremento, alerta
# =======================================================================

import os, json, time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import requests
import requests_cache
from retry_requests import retry
from datetime import datetime

# ---------------- CONFIG (alineado a tu histórico) ----------------
MODEL_NAME = "modelo_alertas_clima.h5"
SCALER_NAME = "scaler_alertas_clima.pkl"
COLS_NAME = "training_columns_alertas_clima.json"
LOC_CSV = "data/Comunas_Cordenadas.csv"
OUT_JSON = "public/alertas_clima.json"

CLIMA_API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "America/Santiago"  # ⚙️ igual que histórico

# ⚙️ mismas variables que usaste en histórico
HOURLY_VARS = ["temperature_2m", "precipitation", "rain", "wind_speed_10m", "wind_gusts_10m"]

# ⚙️ mismas unidades que histórico
UNITS = {
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm"
}

# Horizonte de pronóstico (ajústalo libremente)
FORECAST_DAYS = 8  # ⚙️ antes eran 3; ahora default 8 (puedes llevarlo a 14)
ALERTA_THRESHOLD = 0.40  # 40% de incremento vs baseline del propio horizonte
BASE_SLEEP_S = 1.5       # ⚙️ pausa amigable entre comunas
MAX_RETRIES = 3          # ⚙️ reintentos
BACKOFF_FACTOR = 1.8     # ⚙️ backoff progresivo
COOL_DOWN_429 = 60       # ⚙️ enfriamiento si la API responde 429

# ---------------- Utilidades ----------------
def add_time_features(df):
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.month
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df

def process_clima_json(data, comuna):
    times = pd.to_datetime(data["hourly"]["time"])
    df = pd.DataFrame({"ts": times})
    for var in HOURLY_VARS:
        df[var] = data["hourly"].get(var, np.nan)
    df["comuna"] = comuna
    return df

# ⚙️ Cliente HTTP con cache + retry/backoff (similar espíritu a tu histórico)
def build_http_client():
    cache_session = requests_cache.CachedSession(
        ".openmeteo_cache_forecast", expire_after=3600  # 1h de caché
    )
    return retry(cache_session, retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR)

def fetch_forecast(client, lat, lon):
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": int(FORECAST_DAYS),     # ⚙️ horizonte
        "timezone": TIMEZONE,
        **UNITS                                   # ⚙️ unidades
    }
    try:
        r = client.get(CLIMA_API_URL, params=params)
        if r.status_code == 429:
            # ⚙️ enfriamiento si rate-limited
            time.sleep(COOL_DOWN_429)
            r = client.get(CLIMA_API_URL, params=params)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Fallo al consultar Open-Meteo: {e}")

# ---------------- Main ----------------
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

    client = build_http_client()
    all_preds = []

    for _, row in df_loc.iterrows():
        comuna, lat, lon = row["comuna"], row["lat"], row["lon"]
        print(f"🌦️ Pronóstico para: {comuna} (lat={lat}, lon={lon})")
        try:
            raw = fetch_forecast(client, lat, lon)
            df_c = process_clima_json(raw, comuna)
            df_c = add_time_features(df_c)

            # Ensamble de features como en entrenamiento
            X = df_c[[
                "temperature_2m", "precipitation", "rain",
                "wind_speed_10m", "wind_gusts_10m",
                "sin_hour", "cos_hour", "sin_dow", "cos_dow",
                "dow", "hour", "month"
            ]]
            X = pd.get_dummies(X, columns=["dow", "hour", "month"], drop_first=False)

            # Alinear columnas con entrenamiento
            for col in training_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[training_cols].fillna(0)

            # Predicción
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled, verbose=0).flatten()

            # Baseline del propio horizonte de esa comuna (media simple del pronóstico)
            # Nota: si más adelante quieres usar otro baseline (p. ej. media de “clima normal”),
            # aquí es donde cambiaríamos la referencia.
            baseline = np.nanmean(y_pred) if np.isfinite(y_pred).any() else 0.0
            if baseline <= 0:
                baseline = 1e-6  # evita división por cero

            df_c["llamadas_estimadas"] = y_pred
            df_c["porcentaje_incremento"] = (y_pred / baseline) - 1.0
            df_c["alerta"] = df_c["porcentaje_incremento"] > ALERTA_THRESHOLD

            all_preds.append(df_c)

            time.sleep(BASE_SLEEP_S)  # ⚙️ pausa amistosa entre comunas
        except Exception as e:
            print(f"⚠️ {comuna} falló: {e}")

    if not all_preds:
        print("❌ Sin predicciones válidas.")
        return

    df_all = pd.concat(all_preds, ignore_index=True)
    df_all = df_all[["comuna", "ts", "llamadas_estimadas", "porcentaje_incremento", "alerta"]]
    df_all["ts"] = pd.to_datetime(df_all["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df_all["llamadas_estimadas"] = pd.to_numeric(df_all["llamadas_estimadas"], errors="coerce").round(1)
    df_all["porcentaje_incremento"] = (pd.to_numeric(df_all["porcentaje_incremento"], errors="coerce") * 100).round(1)
    df_all["alerta"] = df_all["alerta"].astype(bool)

    os.makedirs("public", exist_ok=True)
    df_all.to_json(OUT_JSON, orient="records", indent=2, force_ascii=False)
    print("✅ JSON generado:", OUT_JSON)

if __name__ == "__main__":
    main()

