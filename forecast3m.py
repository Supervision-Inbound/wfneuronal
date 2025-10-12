# =======================================================================
# forecast3m.py
# VERSIÓN FINAL: Corregido el NameError por un error de tipeo.
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

ASSET_MODELO_UNIFICADO = "modelo_unificado.keras"
ASSET_SCALER_UNIFICADO = "scaler_unificado.pkl"
ASSET_COLUMNAS = "training_columns_unificado.json"

TIMEZONE = "America/Santiago"
FREQ = "h"
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

def build_feature_matrix_nn(df, training_columns):
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    feature_cols = [col for col in training_columns if col not in df_dummies.columns and col in df.columns]
    X = pd.concat([df[feature_cols], df_dummies], axis=1)
    return X.reindex(columns=training_columns, fill_value=0)

def predecir_futuro_unificado(df_hist, modelo, scaler, training_columns, future_timestamps, target_calls_col, target_tmo_col):
    df_prediccion = df_hist.copy()
    df_hist_con_tiempo = add_time_features(df_hist)
    tmo_historico_estable = df_hist_con_tiempo.groupby(['dow', 'hour'])[target_tmo_col].median()
    tmo_global_fallback = df_hist[target_tmo_col].median()

    for ts in future_timestamps:
        temp_df = pd.DataFrame(index=[ts])
        df_completo = pd.concat([df_prediccion, temp_df])
        
        dow_actual, hour_actual = ts.dayofweek, ts.hour
        tmo_estable = tmo_historico_estable.get((dow_actual, hour_actual), tmo_global_fallback)
        df_completo.loc[ts, target_tmo_col] = tmo_estable

        df_features = add_time_features(df_completo)
        df_features = rolling_features(df_features, target_calls_col)
        df_features = rolling_features(df_features, target_tmo_col)

        X_step = build_feature_matrix_nn(df_features.tail(1), training_columns)
        X_step_scaled = scaler.transform(X_step)
        
        prediccion = modelo.predict(X_step_scaled, verbose=0)[0]
        
        df_prediccion.loc[ts, target_calls_col] = prediccion[0]
        df_prediccion.loc[ts, target_tmo_col] = prediccion[1]
        
    return df_prediccion.loc[future_timestamps, [target_calls_col, target_tmo_col]]

def compute_seasonal_weights(df_hist, col, weeks=8, clip_min=0.75, clip_max=1.30):
    d = df_hist.copy()
    if len(d) == 0: return { (dow,h): 1.0 for dow in range(7) for h in range(24) }
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
            if not np.isnan(num) and not np.isnan(den) and den != 0: w = float(num / den)
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
    if base["mad"].isna().all(): base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    base["q95"] = base["q95"].fillna(base["med"])
    return base

def apply_peak_smoothing_history(df_future, col, base, k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND):
    df = add_time_features(df_future.copy())
    keys = list(zip(df["dow"].values, df["hour"].values))
    b = base.reindex(keys)
    b = b.fillna(base.median(numeric_only=True))
    K = np.where(df["dow"].isin([5, 6]), k_weekend, k_weekday).astype(float)
    upper_cap = b["med"].values + K * b["mad"].values
    mask = (df[col].astype(float).values > upper_cap) & (df[col].astype(float).values > b["q95"].values)
    df.loc[mask, col] = upper_cap[mask]
    return df.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

def load_holidays(csv_path, tz=TIMEZONE):
    if not os.path.exists(csv_path):
        print(f"ADVERTENCIA: No se encontró archivo de feriados en {csv_path}.")
        return set()
    fer = pd.read_csv(csv_path)
    if "Fecha" not in fer.columns:
        print("ADVERTENCIA: El CSV de feriados no tiene columna 'Fecha'.")
        return set()
    return set(pd.to_datetime(fer["Fecha"], dayfirst=True, errors="coerce").dropna().dt.date)

def mark_holidays_index(index, holidays_set):
    idx_dates = index.tz_convert(TIMEZONE).date
    return pd.Series([d in holidays_set for d in idx_dates], index=index, dtype=bool, name="is_holiday")

def _safe_ratio(num, den, fallback=1.0):
    if num is None or den is None or np.isnan(num) or np.isnan(den) or den == 0: return fallback
    return num / den

def compute_holiday_factors(df_hist, holidays_set, col_calls=TARGET_LLAMADAS, col_tmo=TARGET_TMO):
    dfh = add_time_features(df_hist[[col_calls, col_tmo]].copy())
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
    df = add_time_features(df_future.copy())
    is_hol = mark_holidays_index(df.index, holidays_set).values
    hours = df["hour"].values
    call_f = np.array([factors_calls_by_hour.get(h, 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(h, 1.0) for h in hours])
    df.loc[is_hol, "pred_llamadas"] = (df.loc[is_hol, "pred_llamadas"] * call_f[is_hol]).round().astype(int)
    df.loc[is_hol, "pred_tmo_seg"]  = (df.loc[is_hol, "pred_tmo_seg"]  * tmo_f[is_hol]).round().astype(int)
    return df[["pred_llamadas", "pred_tmo_seg"]]

def erlang_c_prob_wait(agents, load_erlangs):
    if agents <= 0 or load_erlangs <= 0: return 1.0 if load_erlangs > 0 else 0.0
    rho = load_erlangs / agents
    if rho >= 1.0: return 1.0
    summation, term = 1.0, 1.0
    for n in range(1, int(agents)):
        term *= load_erlangs / n
        summation += term
    denominator = (summation * (1 - rho) + (term * load_erlangs / int(agents)))
    if denominator == 0: return 1.0
    p_wait = (term * load_erlangs / int(agents)) / denominator
    return float(np.clip(p_wait, 0.0, 1.0))

def required_agents(arrivals, aht_s, asa_target_s=ASA_TARGET_S, sla_target=SLA_TARGET, interval_s=INTERVAL_S, max_occ=MAX_OCC):
    aht, lam = max(aht_s, 1.0), max(arrivals, 0.0)
    load = lam * aht / interval_s
    agents = max(int(np.ceil(load / max_occ)), int(np.ceil(load)) + 1)
    for _ in range(200):
        p_wait = erlang_c_prob_wait(agents, load)
        if p_wait > 0 and aht > 0:
            sla = 1.0 - p_wait * np.exp(-(agents - load) * asa_target_s / aht)
            if sla >= sla_target: break
        else: break
        agents += 1
    return agents, load

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    print("Descargando artefactos del modelo unificado desde GitHub...")
    for asset in [ASSET_MODELO_UNIFICADO, ASSET_SCALER_UNIFICADO, ASSET_COLUMNAS]:
        download_asset_from_latest(OWNER, REPO, asset, MODELS_DIR)

    print("Cargando modelo unificado, scaler y columnas de entrenamiento...")
    model_unificado = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_MODELO_UNIFICADO}", safe_mode=False)
    
    scaler_unificado = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_UNIFICADO}")
    
    with open(f"{MODELS_DIR}/{ASSET_COLUMNAS}", "r") as f:
        training_artifacts = json.load(f)
        training_columns = training_artifacts["all_training_cols"]

    print(f"Cargando datos históricos desde {DATA_FILE}...")
    df_hist_raw = pd.read_csv(DATA_FILE, delimiter=';')
    df_hist_raw.columns = df_hist_raw.columns.str.strip().str.lower()
    df_hist_raw[TARGET_TMO] = df_hist_raw["tmo (segundos)"].apply(parse_tmo_to_seconds)
    df_hist = ensure_datetime(df_hist_raw)
    
    # === LÍNEA CORREGIDA ===
    df_hist = df_hist[[TARGET_LLAMADAS, TARGET_TMO]].dropna(subset=[TARGET_LLAMADAS])

    print(f"Cargando feriados desde {HOLIDAYS_FILE}...")
    holidays_set = load_holidays(HOLIDAYS_FILE)

    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (start_pred + pd.Timedelta(days=HORIZON_DAYS)) - pd.Timedelta(hours=1)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ, tz=TIMEZONE)
    print(f"Se predecirán {len(future_ts)} horas (≈ {HORIZON_DAYS} días).")

    print("Realizando predicción iterativa con modelo causal...")
    predicciones = predecir_futuro_unificado(
        df_hist, model_unificado, scaler_unificado, training_columns, future_ts, TARGET_LLAMADAS, TARGET_TMO
    )
    
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(predicciones[TARGET_LLAMADAS])).astype(int)
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(predicciones[TARGET_TMO])).astype(int)
    
    print("Aplicando recalibración estacional (dow-hour)...")
    seasonal_w = compute_seasonal_weights(df_hist, TARGET_LLAMADAS)
    df_final = apply_seasonal_weights(df_final, seasonal_w)

    print("Aplicando suavizado robusto (baseline histórico)...")
    base_hist = baseline_from_history(df_hist, TARGET_LLAMADAS)
    df_final_smoothed = apply_peak_smoothing_history(
        df_final, "pred_llamadas", base_hist,
        k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND
    )
    df_final_smoothed["pred_llamadas"] = df_final_smoothed["pred_llamadas"].round().astype(int)

    if holidays_set:
        print("Calculando y aplicando factores de ajuste por feriados...")
        f_calls_by_hour, f_tmo_by_hour, _, _ = compute_holiday_factors(df_hist, holidays_set)
        df_final_adj = apply_holiday_adjustment(
            df_final_smoothed,
            holidays_set, f_calls_by_hour, f_tmo_by_hour
        )
    else:
        df_final_adj = df_final_smoothed

    print("Guardando archivos de salida (predicciones)...")
    out = df_final_adj.reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    daily = (out.assign(date=pd.to_datetime(out["ts"]).dt.date)
               .groupby("date")["pred_llamadas"].sum()
               .reset_index().rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False)

    print("Generando erlang_forecast.json...")
    df_er = df_final_adj.rename(columns={"pred_llamadas": "calls", "pred_tmo_seg": "aht_s"})
    df_er["agents_prod"], df_er["erlangs"] = 0, 0.0
    for ts, row in df_er.iterrows():
        agents, load = required_agents(
            arrivals=row["calls"], aht_s=row["aht_s"]
        )
        df_er.loc[ts, "agents_prod"] = agents
        df_er.loc[ts, "erlangs"] = load
    df_er["agents_sched"] = np.ceil(df_er["agents_prod"] / (1 - SHRINKAGE) / (1 - ABSENTEEISM_RATE)).astype(int)
    erjson = df_er.reset_index().rename(columns={"index": "ts"})
    erjson["ts"] = erjson["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erjson.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    json.dump(
        {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        open(STAMP_JSON, "w")
    )
    print("✔ Inferencia + Erlang completadas con éxito.")

if __name__ == "__main__":
    main()
