# =======================================================================
# forecast3m.py (VERSIÓN FINAL CON DESCARGA DE RELEASE, PREDICCIÓN ITERATIVA
# y AJUSTE AUTOMÁTICO POR FERIADOS DESDE CSV)
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
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"  # <--- NUEVO
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
    df["ts"] = pd.to_datetime(
        df["fecha_dt"].astype(str) + " " + df["hora_str"],
        errors="coerce", format="%Y-%m-%d %H:%M"
    )
    df = df.dropna(subset=["ts"]).sort_values("ts")
    # Localización a timezone; maneja DST (ambiguous/nonexistent a NaT y se filtra)
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
    df_copy[f"{target_col}_ma24"]   = df_copy[target_col].shift(1).rolling(24, min_periods=1).mean()
    df_copy[f"{target_col}_ma168"]  = df_copy[target_col].shift(1).rolling(24*7, min_periods=1).mean()
    return df_copy

# --- Funciones auxiliares de inferencia (Modificadas) ---
def build_feature_matrix_nn(df, training_columns, target_col):
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
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

# --- FERIADOS: lectura y ajustes (NUEVO) ---
def load_holidays(csv_path, tz=TIMEZONE):
    """
    Lee un CSV de una columna 'Fecha' (DD-MM-YYYY) y devuelve un set de fechas timezone-aware (solo fecha).
    """
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
    """
    Dado un DatetimeIndex, devuelve un boolean Series marcando si la fecha (sin hora) está en el set de feriados.
    """
    idx_dates = index.tz_convert(TIMEZONE).date if index.tz is not None else index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=index, dtype=bool, name="is_holiday")

def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
    return num / den

def compute_holiday_factors(df_hist, holidays_set, col_calls=TARGET_LLAMADAS, col_tmo=TARGET_TMO):
    """
    Calcula factores robustos por HORA para ajustar predicciones en feriados:
      factor_hora = mediana(valor en feriados a esa hora) / mediana(valor en NO feriados a esa hora)
    Retorna dicts: factors_calls_by_hour, factors_tmo_by_hour, y factores globales de respaldo.
    """
    dfh = df_hist.copy()
    dfh = add_time_features(dfh)
    dfh["is_holiday"] = mark_holidays_index(dfh.index, holidays_set).values

    # Medianas por hora en feriados / no feriados
    med_hol_calls = dfh[dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor_calls = dfh[~dfh["is_holiday"]].groupby("hour")[col_calls].median()
    med_hol_tmo   = dfh[dfh["is_holiday"]].groupby("hour")[col_tmo].median()
    med_nor_tmo   = dfh[~dfh["is_holiday"]].groupby("hour")[col_tmo].median()

    # Globales (por si faltan horas)
    g_hol_calls = dfh[dfh["is_holiday"]][col_calls].median()
    g_nor_calls = dfh[~dfh["is_holiday"]][col_calls].median()
    g_hol_tmo   = dfh[dfh["is_holiday"]][col_tmo].median()
    g_nor_tmo   = dfh[~dfh["is_holiday"]][col_tmo].median()

    global_calls_factor = _safe_ratio(g_hol_calls, g_nor_calls, fallback=0.75)  # fallback razonable
    global_tmo_factor   = _safe_ratio(g_hol_tmo, g_nor_tmo, fallback=1.00)     # TMO puede subir/bajar; neutro si no hay datos

    # Por hora con fallback al global
    factors_calls_by_hour = {h: _safe_ratio(med_hol_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=global_calls_factor) for h in range(24)}
    factors_tmo_by_hour   = {h: _safe_ratio(med_hol_tmo.get(h, np.nan),   med_nor_tmo.get(h, np.nan),   fallback=global_tmo_factor)   for h in range(24)}

    # Limitar factores a rangos razonables para evitar explosiones por datos raros
    # Llamadas: entre 0.1x y 1.2x | TMO: entre 0.7x y 1.5x (ajustable)
    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.20)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}

    return factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor

def apply_holiday_adjustment(df_future, holidays_set, factors_calls_by_hour, factors_tmo_by_hour):
    """
    Aplica, solo en fechas feriado, un factor por HORA a pred_llamadas y pred_tmo_seg.
    """
    df = df_future.copy()
    df = add_time_features(df)
    is_hol = mark_holidays_index(df.index, holidays_set)
    # Ajuste solo en feriados
    hours = df["hour"].values
    call_factors = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_factors  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    df.loc[is_hol.values, "pred_llamadas"] = (df.loc[is_hol.values, "pred_llamadas"] * call_factors[is_hol.values]).round().astype(int)
    df.loc[is_hol.values, "pred_tmo_seg"]  = (df.loc[is_hol.values, "pred_tmo_seg"]  * tmo_factors[is_hol.values]).round().astype(int)
    return df[["pred_llamadas", "pred_tmo_seg"]]

# --- Lógica principal de predicción iterativa ---
def predecir_futuro_iterativo(df_hist, modelo, scaler, target_col, future_timestamps):
    training_columns = scaler.get_feature_names_out()
    df_prediccion = df_hist.copy()
    for ts in future_timestamps:
        temp_df = pd.DataFrame(index=[ts])
        df_completo = pd.concat([df_prediccion, temp_df])
        df_completo = add_time_features(df_completo)
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

    # --- Cargar y procesar datos históricos ---
    print(f"Cargando datos históricos desde {DATA_FILE}...")
    df_hist_raw = pd.read_csv(DATA_FILE, delimiter=';')
    df_hist_raw.columns = df_hist_raw.columns.str.strip()
    # Columna TMO a segundos
    df_hist_raw['tmo_seg'] = df_hist_raw['tmo (segundos)'].apply(parse_tmo_to_seconds)
    df_hist = ensure_datetime(df_hist_raw)
    df_hist = df_hist[[TARGET_LLAMADAS, TARGET_TMO]].dropna(subset=[TARGET_LLAMADAS])

    # --- Cargar feriados ---
    print(f"Cargando feriados desde {HOLIDAYS_FILE}...")
    holidays_set = load_holidays(HOLIDAYS_FILE)

    # --- Definir el rango de fechas futuras a predecir ---
    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (last_known_date.to_period('M') + 3).to_timestamp(how='end').tz_localize(TIMEZONE)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ, tz=TIMEZONE)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred}.")

    # --- Predicción iterativa ---
    print("Realizando predicción iterativa de llamadas...")
    pred_ll = predecir_futuro_iterativo(df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts)
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    print("Realizando predicción iterativa de TMO...")
    pred_tmo = predecir_futuro_iterativo(df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts)
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(int)

    # --- Suavizado de picos (mantiene tu lógica actual) ---
    print("Aplicando suavizado de picos...")
    df_final_with_features = add_time_features(df_final)
    df_final_smoothed = apply_peak_smoothing(df_final_with_features, "pred_llamadas", mad_k=MAD_K, method=SUAVIZADO)

    # <<< Asegurar enteros >>>
    df_final_smoothed["pred_llamadas"] = df_final_smoothed["pred_llamadas"].round().astype(int)

    # --- Ajuste por feriados (NUEVO) ---
    if holidays_set:
        print("Calculando factores de ajuste por feriados a partir del histórico...")
        f_calls_by_hour, f_tmo_by_hour, g_calls, g_tmo = compute_holiday_factors(df_hist, holidays_set)
        print(f"Factor global llamadas feriado: {g_calls:.3f} | TMO: {g_tmo:.3f}")
        print("Aplicando ajuste por feriados al horizonte futuro...")
        df_final_adj = apply_holiday_adjustment(
            df_final_smoothed[["pred_llamadas", "pred_tmo_seg"]],
            holidays_set,
            f_calls_by_hour,
            f_tmo_by_hour
        )
    else:
        print("No hay feriados cargados; se omite ajuste por feriados.")
        df_final_adj = df_final_smoothed[["pred_llamadas", "pred_tmo_seg"]]

    # --- Guardar outputs ---
    print("Guardando archivos de salida...")
    out = df_final_adj.reset_index().rename(columns={"index": "ts"})
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

