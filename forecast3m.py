# =======================================================================
# forecast3m.py (VERSIÓN OPTIMIZADA)
# =======================================================================

import os, json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path  # Mejor manejo de rutas
from typing import List, Dict, Set # Para type hints

# Importar la función de descarga (asumiendo que está en un archivo utils_release.py)
from utils_release import download_asset_from_latest

# --- Parámetros generales ---
OWNER = "Supervision-Inbound"
REPO = "wfneuronal"
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
PUBLIC_DIR = Path("public")

DATA_FILE = DATA_DIR / "historical_data.csv"
HOLIDAYS_FILE = DATA_DIR / "Feriados_Chilev2.csv"
OUT_CSV_DATAOUT = PUBLIC_DIR / "predicciones.csv"
OUT_JSON_PUBLIC = PUBLIC_DIR / "predicciones.json"
OUT_CSV_DAILY = PUBLIC_DIR / "llamadas_por_dia.csv"
STAMP_JSON = PUBLIC_DIR / "last_update.json"

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

# --- Funciones de preprocesamiento (sin cambios) ---
def ensure_datetime(df, col_fecha="fecha", col_hora="hora"):
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
    df["hora_str"] = df[col_hora].astype(str).str.zfill(5).str.slice(0, 5) # zfill para horas como 9:00
    df["ts"] = pd.to_datetime(
        df["fecha_dt"].astype(str) + " " + df["hora_str"],
        errors="coerce", format="%Y-%m-%d %H:%M"
    )
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["ts"] = df["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["ts"])
    return df.set_index("ts")

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if s.replace(".", "", 1).isdigit():
        try: return float(s)
        except: return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(s)
    except:
        return np.nan

def add_time_features(df):
    df_copy = df.copy()
    idx = df_copy.index
    df_copy["dow"] = idx.dayofweek
    df_copy["month"] = idx.month
    df_copy["hour"] = idx.hour
    df_copy["sin_hour"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["cos_hour"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["sin_dow"]  = np.sin(2 * np.pi * df_copy["dow"]  / 7)
    df_copy["cos_dow"]  = np.cos(2 * np.pi * df_copy["dow"]  / 7)
    return df_copy

def rolling_features(df, target_col):
    df_copy = df.copy()
    # shift(1) es crucial para evitar data leakage. El valor actual no debe influir en sus propias features.
    shifted_target = df_copy[target_col].shift(1)
    df_copy[f"{target_col}_lag24"]  = df_copy[target_col].shift(24)
    df_copy[f"{target_col}_ma24"]   = shifted_target.rolling(24, min_periods=1).mean()
    df_copy[f"{target_col}_ma168"] = shifted_target.rolling(24*7, min_periods=1).mean()
    return df_copy

# --- Funciones auxiliares de inferencia (sin cambios) ---
def build_feature_matrix_nn(df, training_columns, target_col):
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"
    ]
    existing_feats = [feat for feat in base_feats if feat in df.columns]
    X = pd.concat([df[existing_feats], df_dummies], axis=1)
    # Reindex para asegurar que todas las columnas de entrenamiento estén presentes
    X = X.reindex(columns=training_columns, fill_value=0)
    return X.fillna(0)

def robust_baseline_by_dow_hour(df, col):
    grouped = df.groupby(["dow", "hour"])[col].agg(["median"])
    grouped.rename(columns={"median": "med"}, inplace=True)
    grouped["mad"] = df.groupby(["dow", "hour"])[col].apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    ).values
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
    return df.drop(columns=["med", "mad", "upper_cap", "is_peak"], errors="ignore")

# --- FERIADOS (sin cambios) ---
def load_holidays(csv_path, tz=TIMEZONE) -> Set:
    if not os.path.exists(csv_path):
        print(f"ADVERTENCIA: No se encontró archivo de feriados en {csv_path}. No se aplicarán ajustes.")
        return set()
    fer = pd.read_csv(csv_path)
    if "Fecha" not in fer.columns:
        print("ADVERTENCIA: El CSV de feriados no tiene columna 'Fecha'. No se aplicarán ajustes.")
        return set()
    fechas = pd.to_datetime(fer["Fecha"].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(index, holidays_set):
    idx_dates = index.tz_convert(TIMEZONE).date if index.tz is not None else index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=index, dtype=bool, name="is_holiday")

def _safe_ratio(num, den, fallback=1.0):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return fallback
    return num / den

def compute_holiday_factors(df_hist, holidays_set, col_calls=TARGET_LLAMADAS, col_tmo=TARGET_TMO):
    dfh = df_hist.copy()
    dfh = add_time_features(dfh)
    dfh["is_holiday"] = mark_holidays_index(dfh.index, holidays_set).values

    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_hol_tmo   = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
    med_nor_tmo   = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()

    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    g_hol_tmo   = dfh[dfh["is_holiday"]][col_tmo].median()
    g_nor_tmo   = dfh[~dfh["is_holiday"]][col_tmo].median()

    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)
    global_tmo_factor   = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)

    factors_calls_by_hour = {h: _safe_ratio(med_hol_calls.get(h), med_nor_calls.get(h), fallback=global_calls_factor) for h in range(24)}
    factors_tmo_by_hour   = {h: _safe_ratio(med_hol_tmo.get(h),   med_nor_tmo.get(h),   fallback=global_tmo_factor)   for h in range(24)}

    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.20)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    return factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor

def apply_holiday_adjustment(df_future, holidays_set, factors_calls_by_hour, factors_tmo_by_hour):
    df = df_future.copy()
    df = add_time_features(df)
    is_hol = mark_holidays_index(df.index, holidays_set)
    if not is_hol.any():
        return df[["pred_llamadas", "pred_tmo_seg"]]
        
    hours = df["hour"].values
    call_factors = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_factors  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    df.loc[is_hol, "pred_llamadas"] = (df.loc[is_hol, "pred_llamadas"] * call_factors[is_hol]).round().astype(int)
    df.loc[is_hol, "pred_tmo_seg"]  = (df.loc[is_hol, "pred_tmo_seg"]  * tmo_factors[is_hol]).round().astype(int)
    return df[["pred_llamadas", "pred_tmo_seg"]]

# --- Lógica de predicción OPTIMIZADA ---
def predecir_futuro_optimizado(df_hist, modelo, scaler, target_col, future_timestamps):
    """
    Realiza predicciones iterativas de forma eficiente usando una ventana deslizante.
    """
    training_columns = scaler.get_feature_names_out()
    predictions = []
    
    # La ventana de datos debe ser tan grande como el mayor 'lookback' necesario.
    # En este caso, es 24*7 = 168 para 'ma168'. Damos un poco de margen.
    max_lookback = 24 * 7 
    
    # Tomamos la última porción del historial como punto de partida.
    # Usamos .copy() para evitar warnings.
    df_window = df_hist[[target_col]].tail(max_lookback).copy()

    for ts in future_timestamps:
        # 1. Crear features solo para la ventana actual
        df_features = add_time_features(df_window)
        df_features = rolling_features(df_features, target_col)
        
        # 2. Construir la matriz de features para el último punto (el que queremos predecir)
        # El índice de la ventana ahora es ts
        current_step_features = df_features.tail(1)
        X_step = build_feature_matrix_nn(current_step_features, training_columns, target_col)
        
        # 3. Escalar y predecir
        X_step_scaled = scaler.transform(X_step)
        prediccion = modelo.predict(X_step_scaled, verbose=0).flatten()[0]
        
        # 4. Guardar la predicción
        predictions.append(prediccion)
        
        # 5. Actualizar la ventana para la siguiente iteración:
        # Añade la nueva predicción y elimina el valor más antiguo.
        new_row = pd.DataFrame({target_col: [prediccion]}, index=[ts])
        df_window = pd.concat([df_window, new_row]).iloc[1:]

    return pd.Series(predictions, index=future_timestamps)

# --- Main (refactorizado para claridad) ---
def main():
    # Crear directorios si no existen
    for d in [MODELS_DIR, DATA_DIR, PUBLIC_DIR]:
        d.mkdir(exist_ok=True)

    # 1. Descargar y cargar modelos
    print("Descargando y cargando modelos...")
    for asset in [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]:
        download_asset_from_latest(OWNER, REPO, asset, str(MODELS_DIR))
    
    model_ll = tf.keras.models.load_model(MODELS_DIR / ASSET_LLAMADAS)
    scaler_ll = joblib.load(MODELS_DIR / ASSET_SCALER_LLAMADAS)
    model_tmo = tf.keras.models.load_model(MODELS_DIR / ASSET_TMO)
    scaler_tmo = joblib.load(MODELS_DIR / ASSET_SCALER_TMO)

    # 2. Cargar y procesar datos
    print(f"Cargando datos históricos desde {DATA_FILE}...")
    df_hist_raw = pd.read_csv(DATA_FILE, delimiter=';')
    df_hist_raw.columns = df_hist_raw.columns.str.strip()
    df_hist_raw['tmo_seg'] = df_hist_raw['tmo (segundos)'].apply(parse_tmo_to_seconds)
    df_hist = ensure_datetime(df_hist_raw)
    df_hist = df_hist[[TARGET_LLAMADAS, TARGET_TMO]].dropna(subset=[TARGET_LLAMADAS])

    # 3. Cargar feriados y calcular factores
    print(f"Cargando feriados desde {HOLIDAYS_FILE}...")
    holidays_set = load_holidays(HOLIDAYS_FILE)
    if holidays_set:
        print("Calculando factores de ajuste por feriados...")
        f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo = compute_holiday_factors(df_hist, holidays_set)
        print(f"Factor global llamadas feriado: {g_calls:.3f} | TMO: {g_tmo:.3f}")

    # 4. Definir horizonte de predicción
    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (last_known_date.to_period('M') + 3).to_timestamp(how='end').tz_localize(TIMEZONE)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ, tz=TIMEZONE)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred}.")

    # 5. Realizar predicciones
    df_final = pd.DataFrame(index=future_ts)
    
    print("Realizando predicción optimizada de llamadas...")
    pred_ll = predecir_futuro_optimizado(df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    print("Realizando predicción optimizada de TMO...")
    pred_tmo = predecir_futuro_optimizado(df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts)
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(int)

    # 6. Post-procesamiento
    print("Aplicando suavizado de picos...")
    df_final_with_features = add_time_features(df_final)
    df_final_smoothed = apply_peak_smoothing(df_final_with_features, "pred_llamadas", mad_k=MAD_K, method=SUAVIZADO)
    df_final_smoothed["pred_llamadas"] = df_final_smoothed["pred_llamadas"].round().astype(int)

    if holidays_set:
        print("Aplicando ajuste por feriados...")
        df_final_adj = apply_holiday_adjustment(
            df_final_smoothed[["pred_llamadas", "pred_tmo_seg"]],
            holidays_set, f_calls_by_hour, f_tmo_by_hour
        )
    else:
        df_final_adj = df_final_smoothed[["pred_llamadas", "pred_tmo_seg"]]

    # 7. Guardar resultados
    print("Guardando archivos de salida...")
    out = df_final_adj.reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    daily = (out.assign(date=pd.to_datetime(out["ts"]).dt.date)
               .groupby("date", as_index=False)["pred_llamadas"]
               .sum()
               .rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False)

    with open(STAMP_JSON, "w") as f:
        json.dump(
            {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}, f
        )
    
    print("✔ Inferencia completada con éxito.")

if __name__ == "__main__":
    main()

