# =======================================================================
# forecast_alertas.py  (con DEBUG)
# =======================================================================

import os, json, time, sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import requests
import requests_cache
from retry_requests import retry
from datetime import datetime

# ---------------- CONFIG ----------------
MODEL_NAME = "modelo_alertas_clima.h5"
SCALER_NAME = "scaler_alertas_clima.pkl"
COLS_NAME = "training_columns_alertas_clima.json"
LOC_CSV = "data/Comunas_Cordenadas.csv"
OUT_JSON = "public/alertas_clima.json"
DEBUG_LOG = "public/alertas_debug.txt"

CLIMA_API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "America/Santiago"

HOURLY_VARS = ["temperature_2m", "precipitation", "rain", "wind_speed_10m", "wind_gusts_10m"]
UNITS = {"temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm"}

FORECAST_DAYS = 8
ALERTA_THRESHOLD = 0.40
BASE_SLEEP_S = 1.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.8
COOL_DOWN_429 = 60

DEBUG = True

# ---------------- Debug helpers ----------------
def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def dbg(*args):
    if not DEBUG:
        return
    msg = " ".join(str(a) for a in args)
    print(msg)
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{_now()}] {msg}\n")
    except Exception:
        pass

def section(title):
    line = "=" * 60
    dbg("\n" + line)
    dbg(title)
    dbg(line)

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
    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError("Respuesta de Open-Meteo sin 'hourly.time'")
    times = pd.to_datetime(data["hourly"]["time"])
    df = pd.DataFrame({"ts": times})
    for var in HOURLY_VARS:
        df[var] = data["hourly"].get(var, np.nan)
    df["comuna"] = comuna
    return df

def build_http_client():
    cache_session = requests_cache.CachedSession(".openmeteo_cache_forecast", expire_after=3600)
    return retry(cache_session, retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR)

def fetch_forecast(client, lat, lon):
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": int(FORECAST_DAYS),
        "timezone": TIMEZONE,
        **UNITS
    }
    r = client.get(CLIMA_API_URL, params=params)
    dbg("GET", r.url, "->", r.status_code)
    if r.status_code == 429:
        dbg("429 rate limited. Sleeping", COOL_DOWN_429, "s")
        time.sleep(COOL_DOWN_429)
        r = client.get(CLIMA_API_URL, params=params)
        dbg("Retry GET", r.url, "->", r.status_code)
    r.raise_for_status()
    return r.json()

# --- lector robusto para CSV de coordenadas ---
def read_csv_smart(path):
    encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    seps = (None, ",", ";", "\t", "|")
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(path, encoding=enc, engine="python")
                else:
                    df = pd.read_csv(path, encoding=enc, sep=sep)
                dbg(f"CSV leído con encoding={enc} sep={sep or 'auto'} shape={df.shape}")
                return df
            except Exception as e:
                last_err = e
                continue
    raise last_err or ValueError(f"No pude leer {path} con encodings/separadores estándar.")

def _pick_col(cols, candidates):
    cols_map = {c.lower().strip(): c for c in cols}
    for c in candidates:
        key = c.lower().strip()
        if key in cols_map:
            return cols_map[key]
    return None

def normalize_location_columns(df):
    comuna_cands = ["comuna", "municipio", "localidad", "ciudad", "name", "nombre"]
    lat_cands    = ["lat", "latitude", "latitud", "y"]
    lon_cands    = ["lon", "lng", "long", "longitude", "longitud", "x"]

    comuna_col = _pick_col(df.columns, comuna_cands)
    lat_col    = _pick_col(df.columns, lat_cands)
    lon_col    = _pick_col(df.columns, lon_cands)

    if comuna_col is None:
        raise ValueError(f"No encuentro columna comuna en {list(df.columns)}")
    if lat_col is None or lon_col is None:
        raise ValueError(f"No encuentro columnas lat/lon en {list(df.columns)}")

    df = df.rename(columns={comuna_col: "comuna", lat_col: "lat", lon_col: "lon"}).copy()

    for c in ["lat", "lon"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df = df.drop_duplicates(subset=["comuna"]).reset_index(drop=True)
    dbg(f"Normalizado coords: {before} -> {len(df)} filas válidas")
    return df

# ---------------- Main ----------------
def main():
    section("LOAD ARTIFACTS")
    print("📦 Cargando modelo y artefactos...")
    model = tf.keras.models.load_model(f"models/{MODEL_NAME}", compile=False)
    scaler = joblib.load(f"models/{SCALER_NAME}")
    with open(f"models/{COLS_NAME}", "r") as f:
        training_cols = json.load(f)
    dbg(f"training_cols: {len(training_cols)}")

    section("LOAD LOCATIONS CSV")
    print("📍 Cargando coordenadas de comunas...")
    df_loc_raw = read_csv_smart(LOC_CSV)
    df_loc_raw.columns = [c.strip() for c in df_loc_raw.columns]
    dbg("Cols originales:", df_loc_raw.columns.tolist())
    df_loc = normalize_location_columns(df_loc_raw)
    dbg("Ejemplo coords:", df_loc.head(3).to_dict(orient="records"))

    client = build_http_client()
    all_preds = []
    total_alerts = 0
    errores = []

    section("LOOP COMUNAS")
    for idx, row in df_loc.iterrows():
        comuna, lat, lon = row["comuna"], row["lat"], row["lon"]
        dbg(f"[{idx+1}/{len(df_loc)}] {comuna} lat={lat} lon={lon}")
        try:
            raw = fetch_forecast(client, lat, lon)
            if "hourly" not in raw:
                raise ValueError("Respuesta sin 'hourly'")
            df_c = process_clima_json(raw, comuna)
            dbg(f"{comuna}: horas pronóstico={len(df_c)}")
            if df_c.empty:
                dbg(f"{comuna}: df_c vacío, salto.")
                continue

            df_c = add_time_features(df_c)
            X = df_c[[
                "temperature_2m", "precipitation", "rain",
                "wind_speed_10m", "wind_gusts_10m",
                "sin_hour", "cos_hour", "sin_dow", "cos_dow",
                "dow", "hour", "month"
            ]]
            X = pd.get_dummies(X, columns=["dow", "hour", "month"], drop_first=False)

            # Column alignment debug
            missing = [c for c in training_cols if c not in X.columns]
            extra   = [c for c in X.columns if c not in training_cols]
            if missing:
                dbg(f"{comuna}: faltan {len(missing)} cols respecto a training (se rellenan con 0). Ej: {missing[:5]}")
            if extra:
                dbg(f"{comuna}: sobran {len(extra)} cols respecto a training (se ignoran). Ej: {extra[:5]}")

            for col in training_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[training_cols].fillna(0)

            dbg(f"{comuna}: X shape={X.shape}")
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled, verbose=0).flatten()

            if not np.isfinite(y_pred).any():
                dbg(f"{comuna}: pred no finita, salto.")
                continue

            baseline = float(np.nanmean(y_pred)) if np.isfinite(y_pred).any() else 1e-6
            if baseline <= 0:
                baseline = 1e-6

            pct = (y_pred / baseline) - 1.0
            alerts = (pct > ALERTA_THRESHOLD)
            n_alerts = int(alerts.sum())
            total_alerts += n_alerts

            dbg(f"{comuna}: pred[min={y_pred.min():.4f} mean={np.mean(y_pred):.4f} max={y_pred.max():.4f}] "
                f"baseline={baseline:.4f} alertas={n_alerts}/{len(y_pred)}")

            df_c["llamadas_estimadas"] = y_pred
            df_c["porcentaje_incremento"] = pct
            df_c["alerta"] = alerts

            all_preds.append(df_c)
            time.sleep(BASE_SLEEP_S)

        except Exception as e:
            err = f"{comuna}: {e}"
            errores.append(err)
            dbg("⚠️", err)

    section("FINALIZE")
    if errores:
        dbg("Errores ocurridos:", len(errores))
        for e in errores[:10]:
            dbg(" -", e)

    if not all_preds:
        dbg("No se generaron predicciones. Escribo JSON vacío para no romper el flujo.")
        os.makedirs("public", exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✅ JSON generado (vacío):", OUT_JSON)
        print("ℹ️ Revisa", DEBUG_LOG, "para el detalle de depuración.")
        return

    df_all = pd.concat(all_preds, ignore_index=True)
    dbg("Total filas pronóstico:", len(df_all), "Total alertas TRUE:", int(df_all["alerta"].sum()))

    df_all = df_all[["comuna", "ts", "llamadas_estimadas", "porcentaje_incremento", "alerta"]]
    df_all["ts"] = pd.to_datetime(df_all["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df_all["llamadas_estimadas"] = pd.to_numeric(df_all["llamadas_estimadas"], errors="coerce").round(1)
    df_all["porcentaje_incremento"] = (pd.to_numeric(df_all["porcentaje_incremento"], errors="coerce") * 100).round(1)
    df_all["alerta"] = df_all["alerta"].astype(bool)

    os.makedirs("public", exist_ok=True)
    df_all.to_json(OUT_JSON, orient="records", indent=2, force_ascii=False)
    print("✅ JSON generado:", OUT_JSON)
    print("ℹ️ Log de depuración en:", DEBUG_LOG)

if __name__ == "__main__":
    main()
