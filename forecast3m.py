# =======================================================================
# forecast3m.py
# VERSIÓN FINAL CORREGIDA: Sin comandos de instalación y con inferencia estable.
# ADAPTADO PARA MODELO UNIFICADO (LLAMADAS + TMO).
# DESCARGA DE RELEASE, PREDICCIÓN ITERATIVA, RECALIBRACIÓN, AJUSTES Y ERLANG-C
# =======================================================================

import os
import json
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
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"

OUT_CSV_DATAOUT = "public/predicciones.csv"
OUT_JSON_PUBLIC = "public/predicciones.json"
OUT_CSV_DAILY = "public/llamadas_por_dia.csv"
STAMP_JSON = "public/last_update.json"
OUT_JSON_ERLANG = "public/erlang_forecast.json"

ASSET_MODELO_UNIFICADO = "modelo_unificado_nn.h5"
ASSET_SCALER_UNIFICADO = "scaler_unificado.pkl"
ASSET_COLUMNAS = "training_columns_unificado.json"

TIMEZONE = "America/Santiago"
FREQ = "H"
TARGET_LLAMADAS = "recibidos"
TARGET_TMO = "tmo_seg"

HORIZON_DAYS = 120
MAD_K = 5.0
MAD_K_WEEKEND = 6.5

# --- Parámetros Erlang / Dimensionamiento ---
SLA_TARGET        = 0.90
ASA_TARGET_S      = 22
INTERVAL_S        = 3600
MAX_OCC           = 0.85
SHRINKAGE         = 0.30
ABSENTEEISM_RATE  = 0.23

# =======================================================================
# Utilidades
# =======================================================================
def ensure_datetime(df, col_fecha="fecha", col_hora="hora"):
    df = df.copy()
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
    df["hora_str"] = df[col_hora].astype(str).str.slice(0, 5)
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
    df_copy["dow"] = df_copy.index.dayofweek
    df_copy["month"] = df_copy.index.month
    df_copy["hour"] = df_copy.index.hour
    df_copy["sin_hour"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["cos_hour"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["sin_dow"]  = np.sin(2 * np.pi * df_copy["dow"]  / 7)
    df_copy["cos_dow"]  = np.cos(2 * np.pi * df_copy["dow"]  / 7)
    return df_copy

def rolling_features(df, target_col):
    df_copy = df.copy()
    df_copy[f"{target_col}_lag24"]  = df_copy[target_col].shift(24)
    df_copy[f"{target_col}_lag48"]  = df_copy[target_col].shift(48)
    df_copy[f"{target_col}_lag72"]  = df_copy[target_col].shift(72)
    df_copy[f"{target_col}_ma24"]   = df_copy[target_col].shift(1).rolling(24, min_periods=1).mean()
    df_copy[f"{target_col}_ma72"]   = df_copy[target_col].shift(1).rolling(72, min_periods=1).mean()
    df_copy[f"{target_col}_ma168"]  = df_copy[target_col].shift(1).rolling(168, min_periods=1).mean()
    return df_copy

def build_feature_matrix_nn(df, training_columns, target_cols):
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    base_feats = ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]
    rolling_feats = []
    for target_col in target_cols:
        rolling_feats.extend([
            f"{target_col}_lag24", f"{target_col}_lag48", f"{target_col}_lag72",
            f"{target_col}_ma24", f"{target_col}_ma72", f"{target_col}_ma168"
        ])
    existing_feats = [feat for feat in base_feats + rolling_feats if feat in df.columns]
    X = pd.concat([df[existing_feats], df_dummies], axis=1)
    return X.reindex(columns=training_columns, fill_value=0)

# =======================================================================
# FUNCIÓN DE PREDICCIÓN CORREGIDA Y ESTABLE
# =======================================================================
def predecir_futuro_unificado(df_hist, modelo, scaler, training_columns, future_timestamps):
    """
    Realiza la predicción iterativa usando el modelo unificado.
    CORRECCIÓN CLAVE: No retroalimenta la predicción de TMO para evitar inestabilidad.
    En su lugar, usa un valor de relleno estable (mediana histórica) para el TMO.
    """
    df_prediccion = df_hist.copy()
    target_cols = [TARGET_LLAMADAS, TARGET_TMO]

    print("Calculando TMO histórico estable para la inferencia...")
    df_hist_con_tiempo = add_time_features(df_hist)
    tmo_historico_estable = df_hist_con_tiempo.groupby(['dow', 'hour'])[TARGET_TMO].median()
    tmo_global_fallback = df_hist[TARGET_TMO].median()

    df_prediccion['pred_tmo_final'] = np.nan

    for ts in future_timestamps:
        temp_df = pd.DataFrame(index=[ts])
        df_completo = pd.concat([df_prediccion, temp_df])

        df_features = add_time_features(df_completo)
        for col in target_cols:
            df_features = rolling_features(df_features, col)

        X_step = build_feature_matrix_nn(df_features.tail(1), training_columns, target_cols)
        X_step_scaled = scaler.transform(X_step)
        
        prediccion = modelo.predict(X_step_scaled, verbose=0)[0]
        
        pred_llamadas = prediccion[0]
        df_prediccion.loc[ts, TARGET_LLAMADAS] = pred_llamadas
        
        dow_actual = ts.dayofweek
        hour_actual = ts.hour
        tmo_estable_para_paso = tmo_historico_estable.get((dow_actual, hour_actual), tmo_global_fallback)
        df_prediccion.loc[ts, TARGET_TMO] = tmo_estable_para_paso
        
        df_prediccion.loc[ts, 'pred_tmo_final'] = prediccion[1]

    df_resultado = df_prediccion.loc[future_timestamps, [TARGET_LLAMADAS, 'pred_tmo_final']]
    df_resultado = df_resultado.rename(columns={'pred_tmo_final': TARGET_TMO})
    
    return df_resultado

# =======================================================================
# Funciones de post-procesamiento
# =======================================================================
def compute_seasonal_weights(df_hist, col, weeks=8, clip_min=0.75, clip_max=1.30):
    d = df_hist.copy()
    if len(d) == 0:
        return { (dow,h): 1.0 for dow in range(7) for h in range(24) }
    end = d.index.max()
    start = end - pd.Timedelta(weeks=weeks)
    d = d.loc[d.index >= start]
    d = add_time_features(d[[col]])
    med_dow_hour = d.groupby(["dow","hour"])[col].median()
    med_hour     = d.groupby("hour")[col].median()
    weights = {}
    for dow in range(7):
        for h in range(24):
            num = med_dow_hour.get((dow,h), np.nan)
            den = med_hour.get(h, np.nan)
            w = 1.0
            if not np.isnan(num) and not np.isnan(den) and den != 0:
                w = float(num / den)
            weights[(dow,h)] = float(np.clip(w, clip_min, clip_max))
    return weights

def apply_seasonal_weights(df_future, weights):
    df = add_time_features(df_future.copy())
    idx = list(zip(df["dow"].values, df["hour"].values))
    w = np.array([weights.get(key, 1.0) for key in idx], dtype=float)
    df["pred_llamadas"] = (df["pred_llamadas"].astype(float) * w).round().astype(int)
    return df.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

def baseline_from_history(df_hist, col):
    d = add_time_features(df_hist[[col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    mad = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    q95 = g.quantile(0.95).rename("q95")
    base = base.join([mad, q95])
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(
