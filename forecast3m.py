# =======================================================================
# forecast3m.py (VERSIÓN FINAL CON DESCARGA DE RELEASE Y PREDICCIÓN ITERATIVA)
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
SUAVIZADO = "cap"  # "cap" o "med"
FLOOR_RATIO = 0.80  # piso mínimo = 80% de la mediana histórica por (dow,hour)

# --- Funciones de preprocesamiento (del script de entrenamiento) ---
def ensure_datetime(df, col_fecha="fecha", col_hora="hora"):
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)
    df["hora_str"] = df[col_hora].astype(str).str.slice(0, 5)
    df["ts"] = pd.to_datetime(
        df["fecha_dt"].astype(str) + " " + df["hora_str"],
        errors="coerce", format="%Y-%m-%d %H:%M"
    )
    df = df.dropna(subset=["ts"]).sort_values("ts")
    # Localizar a TZ objetivo (marcando ambigüedades como NaT)
    df["ts"] = df["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return df.set_index("ts")

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if s.replace(".", "", 1).isdigit():
        return float(s)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
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

# --- Funciones auxiliares de inferencia ---
def build_feature_matrix_nn(df, training_columns, target_col):
    """
    Construye la matriz de features en el mismo orden usado al entrenar.
    """
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"
    ]
    existing_feats = [feat for feat in base_feats if feat in df.columns]
    X = pd.concat([df[existing_feats], df_dummies], axis=1)
    # Añadir faltantes con 0 y reordenar
    for c in set(training_columns) - set(X.columns):
        X[c] = 0
    X = X[training_columns].fillna(0)
    return X

def robust_baseline_by_dow_hour_from_hist(df_hist, target_col):
    """
    Calcula baseline robusto (mediana y MAD) por (dow, hour) usando históricos.
    """
    dfh = add_time_features(df_hist)
    grouped_med = dfh.groupby(["dow", "hour"])[target_col].median().rename("med")
    grouped_mad = dfh.groupby(["dow", "hour"])[target_col].apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    ).rename("mad")
    baseline = pd.concat([grouped_med, grouped_mad], axis=1)

    # Reemplazo de MAD=0 por la mediana global de MAD (o 1.0 si todo 0/NaN)
    mad_global = baseline["mad"].replace(0, np.nan).median()
    if pd.isna(mad_global):
        mad_global = 1.0
    baseline["mad"] = baseline["mad"].replace(0, mad_global)
    return baseline

def get_training_columns_from_scaler(scaler):
    """
    Obtiene el orden de columnas usado al entrenar.
    - Pipeline/ColumnTransformer: get_feature_names_out()
    - StandardScaler clásico: feature_names_in_ (si existe) o error
    """
    # Try Pipeline/CT
    try:
        return list(scaler.get_feature_names_out())
    except Exception:
        pass
    # Try atributo sklearn
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # Sin info suficiente → levantar error explícito
    raise RuntimeError(
        "No se pudo obtener el orden de columnas del scaler. "
        "Guarda 'feature_names_in_' o usa un Pipeline con get_feature_names_out()."
    )

def predecir_futuro_iterativo(df_hist, modelo, scaler, target_col, future_timestamps, training_columns):
    """
    Predicción autoregresiva: va extendiendo df_hist con cada ts futuro.
    """
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
    # parseo TMO
    if "tmo (segundos)" in df_hist_raw.columns:
        df_hist_raw["tmo_seg"] = df_hist_raw["tmo (segundos)"].apply(parse_tmo_to_seconds)
    elif "tmo_seg" not in df_hist_raw.columns:
        # Si no viene ninguna de las dos, crea vacía para evitar KeyError
        df_hist_raw["tmo_seg"] = np.nan

    df_hist = ensure_datetime(df_hist_raw)
    df_hist = df_hist[[c for c in [TARGET_LLAMADAS, TARGET_TMO] if c in df_hist.columns]].dropna(subset=[TARGET_LLAMADAS])

    # --- Baseline robusto por (dow,hour) desde históricos ---
    print("Calculando baseline robusto histórico por (dow, hour)...")
    baseline_hist = robust_baseline_by_dow_hour_from_hist(df_hist, TARGET_LLAMADAS)

    # --- Definir el rango de fechas futuras a predecir ---
    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    # Fin de mes +3 meses, TZ robusto
    last_local = last_known_date.tz_convert(TIMEZONE)
    end_pred = (pd.Timestamp(year=last_local.year, month=last_local.month, day=1, tz=TIMEZONE)
                + pd.offsets.MonthEnd(3))
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred}.")

    # --- Obtener columnas de entrenamiento desde cada scaler ---
    training_columns_ll = get_training_columns_from_scaler(scaler_ll)
    training_columns_tmo = get_training_columns_from_scaler(scaler_tmo)

    # --- Predicción iterativa ---
    print("Realizando predicción iterativa de llamadas...")
    pred_ll = predecir_futuro_iterativo(df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts, training_columns_ll)
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(float)  # float para suavizado posterior

    print("Realizando predicción iterativa de TMO...")
    # Para TMO, si no hay histórico, rellena con NaN y luego 0
    pred_tmo = predecir_futuro_iterativo(df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts, training_columns_tmo)
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(float)

    # --- Aplicar suavizado de picos con baseline histórico ---
    print("Aplicando suavizado de picos con baseline histórico...")
    df_final_with_features = add_time_features(df_final).copy()
    df_final_smoothed = df_final_with_features.merge(
        baseline_hist, left_on=["dow", "hour"], right_index=True, how="left"
    )

    # Evitar NaN en med/mad (si faltan combos raros de dow-hour en histórico)
    df_final_smoothed["med"] = df_final_smoothed["med"].fillna(df_final_smoothed["pred_llamadas"].median())
    df_final_smoothed["mad"] = df_final_smoothed["mad"].fillna(df_final_smoothed["pred_llamadas"].mad() if hasattr(pd.Series, "mad") else 1.0)
    df_final_smoothed["mad"] = df_final_smoothed["mad"].replace(0, 1.0)

    upper_cap = df_final_smoothed["med"] + MAD_K * df_final_smoothed["mad"]

    if SUAVIZADO == "cap":
        df_final_smoothed["pred_llamadas"] = np.where(
            df_final_smoothed["pred_llamadas"] > upper_cap,
            upper_cap,
            df_final_smoothed["pred_llamadas"]
        )
    elif SUAVIZADO == "med":
        df_final_smoothed["pred_llamadas"] = np.where(
            df_final_smoothed["pred_llamadas"] > upper_cap,
            df_final_smoothed["med"],
            df_final_smoothed["pred_llamadas"]
        )
    # Piso suave para evitar quedarse demasiado abajo
    min_floor = (df_final_smoothed["med"] * FLOOR_RATIO).fillna(0)
    df_final_smoothed["pred_llamadas"] = np.maximum(df_final_smoothed["pred_llamadas"], min_floor)

    # <<< SALIDA FINAL EN ENTEROS >>>
    df_final_smoothed["pred_llamadas"] = df_final_smoothed["pred_llamadas"].round().astype(int)
    df_final_smoothed["pred_tmo_seg"] = np.maximum(0, df_final_smoothed["pred_tmo_seg"]).round().astype(int)

    # --- Guardar outputs ---
    print("Guardando archivos de salida...")
    out = df_final_smoothed[["pred_llamadas", "pred_tmo_seg"]].reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    # --- Diarios ---
    daily = (
        out.assign(date=pd.to_datetime(out["ts"]).dt.date)
           .groupby("date", as_index=False)["pred_llamadas"]
           .sum()
           .rename(columns={"pred_llamadas": "total_llamadas"})
    )
    daily.to_csv(OUT_CSV_DAILY, index=False)

    # --- Timestamp ---
    json.dump(
        {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        open(STAMP_JSON, "w")
    )
    print("✔ Inferencia completada con éxito.")

if __name__ == "__main__":
    main()

