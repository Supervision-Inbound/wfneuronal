# =======================================================================
# forecast_alertas.py — Failsafe
# - Siempre escribe public/alertas_clima.json
# - Log en public/alertas_debug.txt
# - Ordena comunas: con alertas primero
# - Agrupa horas consecutivas mismo día en rangos
# =======================================================================

import os, json, time, traceback
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
DEBUG = True  # deja True para ver logs; luego puedes poner False

# ---------------- Debug helpers ----------------
def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def dbg(*args):
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

# ---------------- Utilidades de features ----------------
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

# ---------------- HTTP con cache + retry ----------------
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

# ---------------- Lector robusto de CSV coords ----------------
def read_csv_smart(path):
    encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    candidate_delims = [";", "|", "\t", ","]
    last_err = None

    for enc in encodings:
        # 1) Autodetect con engine='python'
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            if df.shape[1] == 1 and isinstance(df.columns[0], str):
                header = df.columns[0]
                for d in candidate_delims:
                    if d in header:
                        try:
                            df2 = pd.read_csv(path, encoding=enc, sep=d)
                            if df2.shape[1] > 1:
                                dbg(f"CSV leído con encoding={enc} sep='{d}' shape={df2.shape}")
                                return df2
                        except Exception as e2:
                            last_err = e2
                            continue
            else:
                dbg(f"CSV leído con encoding={enc} sep=auto shape={df.shape}")
                return df
        except Exception as e:
            last_err = e

        # 2) Intentos explícitos por separador
        for sep in candidate_delims:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    dbg(f"CSV leído con encoding={enc} sep='{sep}' shape={df.shape}")
                    return df
            except Exception as e:
                last_err = e
                continue

    if last_err:
        raise last_err
    raise ValueError(f"No pude leer {path} con encodings/separadores estándar.")

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

# ---------------- Agrupación a rangos de horas ----------------
def merge_consecutive_hours_same_day(df_alerts):
    if df_alerts.empty:
        return []

    df_alerts = df_alerts.copy()
    df_alerts["fecha"] = df_alerts["ts"].dt.date
    df_alerts["hora"] = df_alerts["ts"].dt.hour
    df_alerts = df_alerts.sort_values(["fecha", "hora"])

    rangos = []
    cur_fecha = None
    start_h = None
    prev_h = None
    vals = []

    def push_range(fecha, h0, h1, vals_pct):
        if fecha is None or h0 is None or h1 is None or not vals_pct:
            return
        n = (h1 - h0 + 1)
        rangos.append({
            "fecha": fecha.strftime("%Y-%m-%d"),
            "desde": f"{h0:02d}:00",
            "hasta": f"{h1:02d}:00",
            "n_horas": int(n),
            "incremento_promedio_pct": round(np.mean(vals_pct) * 100, 1),
            "incremento_max_pct": round(np.max(vals_pct) * 100, 1)
        })

    for _, r in df_alerts.iterrows():
        f = r["fecha"]
        h = int(r["hora"])
        pct = float(r["porcentaje_incremento"])
        if cur_fecha is None:
            cur_fecha, start_h, prev_h, vals = f, h, h, [pct]
            continue
        if f == cur_fecha and h == prev_h + 1:
            prev_h = h
            vals.append(pct)
        else:
            push_range(cur_fecha, start_h, prev_h, vals)
            cur_fecha, start_h, prev_h, vals = f, h, h, [pct]
    push_range(cur_fecha, start_h, prev_h, vals)
    return rangos

# ---------------- MAIN ----------------
def main():
    os.makedirs("public", exist_ok=True)
    # reset debug log
    try:
        if os.path.exists(DEBUG_LOG):
            os.remove(DEBUG_LOG)
    except Exception:
        pass

    # contadores para impresión final
    n_comunas_total = 0
    n_comunas_ok = 0
    n_comunas_con_alerta = 0
    errores = []

    try:
        section("LOAD ARTIFACTS")
        dbg("📦 Cargando modelo y artefactos...")
        model = tf.keras.models.load_model(f"models/{MODEL_NAME}", compile=False)
        scaler = joblib.load(f"models/{SCALER_NAME}")
        with open(f"models/{COLS_NAME}", "r") as f:
            training_cols = json.load(f)
        dbg(f"training_cols: {len(training_cols)}")

        section("LOAD LOCATIONS CSV")
        dbg("📍 Cargando coordenadas de comunas...")
        df_loc_raw = read_csv_smart(LOC_CSV)
        df_loc_raw.columns = [c.strip() for c in df_loc_raw.columns]
        dbg("Cols originales:", df_loc_raw.columns.tolist())
        df_loc = normalize_location_columns(df_loc_raw)
        n_comunas_total = len(df_loc)
        dbg("Ejemplo coords:", df_loc.head(3).to_dict(orient="records"))

        client = build_http_client()
        resumen = []

        section("LOOP COMUNAS")
        for idx, row in df_loc.iterrows():
            comuna, lat, lon = row["comuna"], row["lat"], row["lon"]
            dbg(f"[{idx+1}/{len(df_loc)}] {comuna} lat={lat} lon={lon}")
            try:
                raw = fetch_forecast(client, lat, lon)
                df_c = process_clima_json(raw, comuna)
                if df_c.empty:
                    dbg(f"{comuna}: df_c vacío.")
                    resumen.append({"comuna": comuna, "rango_alertas": [], "detalles": []})
                    continue

                df_c = add_time_features(df_c)
                X = df_c[[
                    "temperature_2m", "precipitation", "rain",
                    "wind_speed_10m", "wind_gusts_10m",
                    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
                    "dow", "hour", "month"
                ]]
                X = pd.get_dummies(X, columns=["dow", "hour", "month"], drop_first=False)

                for col in training_cols:
                    if col not in X.columns:
                        X[col] = 0
                X = X[training_cols].fillna(0)

                y_pred = model.predict(scaler.transform(X), verbose=0).flatten()
                if not np.isfinite(y_pred).any():
                    dbg(f"{comuna}: pred no finita.")
                    resumen.append({"comuna": comuna, "rango_alertas": [], "detalles": []})
                    continue

                baseline = float(np.nanmean(y_pred)) if np.isfinite(y_pred).any() else 1e-6
                if baseline <= 0:
                    baseline = 1e-6

                df_c["llamadas_estimadas"] = y_pred
                df_c["porcentaje_incremento"] = (y_pred / baseline) - 1.0
                df_c["alerta"] = df_c["porcentaje_incremento"] > ALERTA_THRESHOLD

                # construir salida por comuna
                dfg = df_c.sort_values("ts").copy()
                dfg_alerts = dfg[dfg["alerta"] == True][["ts", "porcentaje_incremento"]]
                rangos = merge_consecutive_hours_same_day(dfg_alerts)

                detalles = (dfg.assign(
                                llamadas_estimadas=lambda x: x["llamadas_estimadas"].astype(float).round(1),
                                porcentaje_incremento=lambda x: (x["porcentaje_incremento"].astype(float) * 100).round(1),
                                alerta=lambda x: x["alerta"].astype(bool)
                            )[["ts", "llamadas_estimadas", "porcentaje_incremento", "alerta"]]
                            .assign(ts=lambda x: x["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
                            .to_dict(orient="records"))

                resumen.append({
                    "comuna": comuna,
                    "rango_alertas": rangos,
                    "detalles": detalles
                })

                n_comunas_ok += 1
                if len(rangos) > 0:
                    n_comunas_con_alerta += 1

                time.sleep(BASE_SLEEP_S)

            except Exception as e:
                err = f"{comuna}: {e}"
                errores.append(err)
                dbg("⚠️", err)
                dbg(traceback.format_exc())
                # aun así agregamos la comuna con estructura vacía
                resumen.append({"comuna": comuna, "rango_alertas": [], "detalles": []})

        # ordenar: comunas con alertas primero
        resumen.sort(key=lambda x: (len(x.get("rango_alertas", [])) == 0, x["comuna"]))

        # escribir salida
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)

        print("✅ JSON generado:", OUT_JSON)
        print(f"📊 Comunas totales: {n_comunas_total} | OK: {n_comunas_ok} | Con alerta: {n_comunas_con_alerta} | Con error: {len(errores)}")
        if errores:
            print(f"ℹ️ Revisa {DEBUG_LOG} para detalles de errores.")

    except Exception as e:
        # Falla global: aún así escribimos un JSON mínimo
        dbg("❌ Falla global:", e)
        dbg(traceback.format_exc())
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✅ JSON generado (mínimo por failsafe):", OUT_JSON)
        print(f"ℹ️ Revisa {DEBUG_LOG} para detalles.")
        raise

if __name__ == "__main__":
    main()
