# =======================================================================
# forecast3m.py (VERSIÓN FINAL CON LÓGICA DE FERIADOS CORREGIDA Y EFICIENTE)
# =======================================================================

import os, json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from utils_release import download_asset_from_latest

# --- Parámetros generales ---
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"
MODELS_DIR = "models"
DATA_FILE = "data/historical_data.csv"
FERIADOS_FILE = "data/Feriados_Chile.csv"
OUT_CSV_DATAOUT = "public/predicciones.csv"
OUT_JSON_PUBLIC = "public/predicciones.json"
OUT_CSV_DAILY = "public/llamadas_por_dia.csv"
STAMP_JSON = "public/last_update.json"

ASSET_LLAMADAS = "modelo_llamadas_nn.h5"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_TMO = "modelo_tmo_nn.h5"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

TIMEZONE = "America/Santiago"
FREQ = "H"
TARGET_LLAMADAS = "recibidos"
TARGET_TMO = "tmo_seg"

MAD_K = 3.5
SUAVIZADO = "cap"

# --- Funciones de preprocesamiento ---
def ensure_datetime(df, col_fecha="fecha", col_hora="hora"):
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
    df["hora_str"] = df[col_hora].astype(str).str.slice(0, 5)
    df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str) + " " + df["hora_str"], errors="coerce", format='%Y-%m-%d %H:%M')
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df['ts'] = df['ts'].dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
    return df.set_index("ts")

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(',', '.')
    if s.replace('.','',1).isdigit(): return float(s)
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(s)
    except: return np.nan

# <<< CAMBIO: Se crea una función específica para generar features para UN solo paso futuro >>>
def create_features_for_step(timestamp, df_history, target_col, set_feriados):
    """Crea un DataFrame de una sola fila con todas las features para un timestamp futuro."""
    features = {}
    features['dow'] = timestamp.dayofweek
    features['month'] = timestamp.month
    features['hour'] = timestamp.hour
    features['feriados'] = 1 if timestamp.date() in set_feriados else 0
    
    features["sin_hour"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["cos_hour"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["sin_dow"]  = np.sin(2 * np.pi * features["dow"]  / 7)
    features["cos_dow"]  = np.cos(2 * np.pi * features["dow"]  / 7)
    
    # Calcular features dependientes del tiempo (lag, rolling)
    features[f"{target_col}_lag24"] = df_history[target_col].asof(timestamp - pd.Timedelta(hours=24))
    
    start_ma24 = timestamp - pd.Timedelta(hours=25)
    end_ma24 = timestamp - pd.Timedelta(hours=1)
    features[f"{target_col}_ma24"] = df_history[target_col].loc[start_ma24:end_ma24].mean()

    start_ma168 = timestamp - pd.Timedelta(hours=169)
    end_ma168 = timestamp - pd.Timedelta(hours=1)
    features[f"{target_col}_ma168"] = df_history[target_col].loc[start_ma168:end_ma168].mean()
    
    return pd.DataFrame([features], index=[timestamp])

# <<< CAMBIO: Esta función ahora se usa solo para el suavizado final, no en el bucle >>>
def add_time_features_for_smoothing(df, set_feriados):
    df_copy = df.copy()
    df_copy["dow"] = df_copy.index.dayofweek
    df_copy["hour"] = df_copy.index.hour
    df_copy['feriados'] = df_copy.index.to_series().dt.date.isin(set_feriados).astype(int)
    return df_copy

def build_feature_matrix_nn(df_step, training_columns):
    df_dummies = pd.get_dummies(df_step[["dow", "month"]], drop_first=False, dtype=int)
    
    # Une las features base con los dummies
    X = pd.concat([df_step, df_dummies], axis=1)

    # Asegura que todas las columnas de entrenamiento existan
    for c in set(training_columns) - set(X.columns):
        X[c] = 0
    
    return X[training_columns].fillna(0)

def robust_baseline_by_dow_hour(df, col):
    grouped = df.groupby(["dow", "hour"])[col].agg(["median"])
    grouped.rename(columns={"median": "med"}, inplace=True)
    grouped["mad"] = df.groupby(["dow", "hour"])[col].apply(lambda x: np.median(np.abs(x - np.median(x)))).values
    return grouped

def apply_peak_smoothing(df, col, mad_k=1.5, method="cap"):
    baseline = robust_baseline_by_dow_hour(df, col)
    df = df.merge(baseline, left_on=["dow", "hour"], right_index=True, how="left")
    df["upper_cap"] = df["med"] + mad_k * df["mad"].replace(0, df["mad"].median())
    df["is_peak"] = (df[col] > df["upper_cap"]).astype(int)
    if method == "cap":
        df[col] = np.where(df["is_peak"] == 1, df["upper_cap"], df[col])
    elif method == "med":
        df[col] = np.where(df["is_peak"] == 1, df["med"], df[col])
    return df.drop(columns=["med", "mad", "upper_cap", "is_peak"], errors='ignore')

# --- Lógica principal de predicción (Reestructurada) ---
def predecir_futuro_iterativo(df_hist, modelo, scaler, target_col, future_timestamps, set_feriados):
    training_columns = scaler.get_feature_names_out()
    df_prediccion = df_hist.copy() # df_prediccion ahora crece con cada predicción

    for ts in future_timestamps:
        # 1. Crear features SOLO para el paso actual, usando el historial disponible
        df_step_features = create_features_for_step(ts, df_prediccion, target_col, set_feriados)
        
        # 2. Preparar la matriz de features para el modelo
        X_step = build_feature_matrix_nn(df_step_features, training_columns)
        
        # 3. Escalar y Predecir
        X_step_scaled = scaler.transform(X_step)
        prediccion = modelo.predict(X_step_scaled, verbose=0).flatten()[0]
        
        # 4. Añadir la predicción al historial para el siguiente paso
        df_prediccion.loc[ts, target_col] = prediccion

    return df_prediccion.loc[future_timestamps, target_col]

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # --- Descargar modelos y scalers ---
    print("Descargando modelos desde el release de GitHub...")
    for asset in [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]:
        download_asset_from_latest(OWNER, REPO, asset, MODELS_DIR)

    # --- Cargar modelos y scalers ---
    print("Cargando modelos y scalers...")
    model_ll = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_LLAMADAS}")
    scaler_ll = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_LLAMADAS}")
    model_tmo = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_TMO}")
    scaler_tmo = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_TMO}")

    # --- Cargar y procesar la lista de feriados ---
    print(f"Cargando feriados desde {FERIADOS_FILE}...")
    df_feriados = pd.read_csv(FERIADOS_FILE, delimiter=';', encoding='latin-1')
    df_feriados.columns = df_feriados.columns.str.strip()
    set_feriados = set(pd.to_datetime(df_feriados['Fecha']).dt.date)

    # --- Cargar y procesar datos históricos ---
    print(f"Cargando datos históricos desde {DATA_FILE}...")
    df_hist_raw = pd.read_csv(DATA_FILE, delimiter=';')
    df_hist_raw.columns = df_hist_raw.columns.str.strip()
    df_hist_raw['tmo_seg'] = df_hist_raw['tmo (segundos)'].apply(parse_tmo_to_seconds)
    
    # <<< CAMBIO: Se usa la columna 'feriados' original del archivo histórico. No se sobreescribe. >>>
    df_hist = ensure_datetime(df_hist_raw)
    df_hist = df_hist[[TARGET_LLAMADAS, TARGET_TMO, 'feriados']].dropna(subset=[TARGET_LLAMADAS])

    # --- Definir el rango de fechas futuras a predecir ---
    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (last_known_date.to_period('M') + 3).to_timestamp(how='end').tz_localize(TIMEZONE)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred}.")

    # --- Predicción iterativa ---
    print("Realizando predicción iterativa de llamadas...")
    pred_ll = predecir_futuro_iterativo(df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts, set_feriados)
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    print("Realizando predicción iterativa de TMO...")
    pred_tmo = predecir_futuro_iterativo(df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts, set_feriados)
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(int)

    # --- Aplicar suavizado de picos ---
    print("Aplicando suavizado de picos...")
    df_final_with_features = add_time_features_for_smoothing(df_final, set_feriados)
    df_final_smoothed = apply_peak_smoothing(df_final_with_features, "pred_llamadas", mad_k=MAD_K, method=SUAVIZADO)
    df_final_smoothed["pred_llamadas"] = df_final_smoothed["pred_llamadas"].round().astype(int)
    
    # --- Guardar outputs ---
    print("Guardando archivos de salida...")
    out = df_final_smoothed[["pred_llamadas", "pred_tmo_seg"]].reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    # --- Diarios ---
    daily = (out.assign(date=pd.to_datetime(out["ts"]).dt.date)
               .groupby("date", as_index=False)["pred_llamadas"]
               .sum()
               .rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False)

    # --- Timestamp ---
    json.dump(
        {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        open(STAMP_JSON, "w")
    )
    print("✔ Inferencia completada con éxito.")

if __name__ == "__main__":
    main()
