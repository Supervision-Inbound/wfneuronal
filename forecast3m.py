# =======================================================================
# forecast3m.py (VERSIÓN FINAL CON LÓGICA DE FERIADOS FUTUROS)
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
# <<< NUEVO: Ruta al archivo de feriados >>>
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

# --- Funciones de preprocesamiento (del script de entrenamiento) ---
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

# <<< MODIFICADO: La función ahora acepta la lista de feriados >>>
def add_time_features(df, set_feriados):
    df_copy = df.copy()
    df_copy["dow"] = df_copy.index.dayofweek
    df_copy["month"] = df_copy.index.month
    df_copy["hour"] = df_copy.index.hour
    # <<< NUEVO: Crea la columna 'feriados' a partir de la lista >>>
    df_copy['feriados'] = df_copy.index.date_isin(set_feriados).astype(int)
    df_copy["sin_hour"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["cos_hour"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["sin_dow"]  = np.sin(2 * np.pi * df_copy["dow"]  / 7)
    df_copy["cos_dow"]  = np.cos(2 * np.pi * df_copy["dow"]  / 7)
    return df_copy

def rolling_features(df, target_col):
    df_copy = df.copy()
    df_copy[f"{target_col}_lag24"]  = df_copy[target_col].shift(24)
    df_copy[f"{target_col}_ma24"]   = df_copy[target_col].shift(1).rolling(24, min_periods=1).mean()
    df_copy[f"{target_col}_ma168"]  = df_copy[target_col].shift(1).rolling(24*7, min_periods=1).mean()
    return df_copy

# --- Funciones auxiliares de inferencia (Modificadas) ---
# <<< MODIFICADO: Se añade 'feriados' a las características base >>>
def build_feature_matrix_nn(df, training_columns, target_col):
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow", "feriados",
        f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"
    ]
    existing_feats = [feat for feat in base_feats if feat in df.columns]
    X = pd.concat([df[existing_feats], df_dummies], axis=1)
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

# --- Lógica principal de predicción iterativa ---
# <<< MODIFICADO: La función ahora necesita la lista de feriados >>>
def predecir_futuro_iterativo(df_hist, modelo, scaler, target_col, future_timestamps, set_feriados):
    training_columns = scaler.get_feature_names_out()
    df_prediccion = df_hist.copy()
    for ts in future_timestamps:
        temp_df = pd.DataFrame(index=[ts])
        df_completo = pd.concat([df_prediccion, temp_df])
        # <<< MODIFICADO: Pasa la lista de feriados para crear la característica >>>
        df_completo = add_time_features(df_completo, set_feriados)
        df_completo = rolling_features(df_completo, target_col)
        X_step = build_feature_matrix_nn(df_completo.tail(1), training_columns, target_col)
        X_step_scaled = scaler.transform(X_step)
        prediccion = modelo.predict(X_step_scaled, verbose=0).flatten()[0]
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

    # <<< NUEVO: Cargar y procesar la lista de feriados >>>
    print(f"Cargando feriados desde {FERIADOS_FILE}...")
    df_feriados = pd.read_csv(FERIADOS_FILE, delimiter=';')
    df_feriados.columns = df_feriados.columns.str.strip()
    # Asegúrate de que el nombre de la columna sea 'Fecha'
    set_feriados = set(pd.to_datetime(df_feriados['Fecha'], dayfirst=True).dt.date)

    # --- Cargar y procesar datos históricos ---
    print(f"Cargando datos históricos desde {DATA_FILE}...")
    df_hist_raw = pd.read_csv(DATA_FILE, delimiter=';')
    df_hist_raw.columns = df_hist_raw.columns.str.strip()
    df_hist_raw['tmo_seg'] = df_hist_raw['tmo (segundos)'].apply(parse_tmo_to_seconds)
    df_hist = ensure_datetime(df_hist_raw)
    # <<< MODIFICADO: Asegurarse de que la columna 'feriados' del historial se mantenga >>>
    df_hist = df_hist[[TARGET_LLAMADAS, TARGET_TMO, 'feriados']].dropna(subset=[TARGET_LLAMADAS])

    # --- Definir el rango de fechas futuras a predecir ---
    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (last_known_date.to_period('M') + 3).to_timestamp(how='end').tz_localize(TIMEZONE)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred}.")

    # --- Predicción iterativa ---
    print("Realizando predicción iterativa de llamadas...")
    # <<< MODIFICADO: Pasa la lista de feriados a la función de predicción >>>
    pred_ll = predecir_futuro_iterativo(df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts, set_feriados)
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    print("Realizando predicción iterativa de TMO...")
    # <<< MODIFICADO: Pasa la lista de feriados a la función de predicción >>>
    pred_tmo = predecir_futuro_iterativo(df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts, set_feriados)
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(int)

    # --- Aplicar suavizado de picos ---
    print("Aplicando suavizado de picos...")
    # <<< MODIFICADO: Pasa la lista de feriados para crear las características necesarias para el suavizado >>>
    df_final_with_features = add_time_features(df_final, set_feriados)
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
