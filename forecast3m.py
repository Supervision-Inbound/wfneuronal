# =======================================================================
# forecast3m.py (VERSIÓN CORREGIDA CON PREDICCIÓN ITERATIVA)
# =======================================================================

import os, json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
# <<< CAMBIO: Eliminada la dependencia de utils_release para que el script sea autocontenido.
# Si necesitas la función de descarga, puedes volver a añadirla.

# --- Parámetros generales ---
MODELS_DIR = "models"
# <<< CAMBIO: Añadida la ruta al archivo de datos históricos.
# Asegúrate de que esta ruta es correcta en tu entorno.
DATA_FILE = "data/Hosting ia.xlsx - Tabla1.csv" 
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
TARGET_TMO = "tmo_seg" # <<< CAMBIO: Usaremos el nombre de columna procesado

MAD_K = 3.5
SUAVIZADO = "cap"

# <<< CAMBIO: Añadidas funciones de preprocesamiento del script de entrenamiento.
# Es mejor tenerlas aquí para que el script de inferencia sea independiente.
def ensure_datetime(df, col_fecha="fecha", col_hora="hora"):
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce").dt.date
    df["hora_str"] = df[col_hora].astype(str).str.slice(0, 5)
    df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str) + " " + df["hora_str"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    return df

def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if s.isdigit(): return float(s)
    parts = s.split(":")
    try:
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(s)
    except: return np.nan

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

# --- Funciones auxiliares (Modificadas) ---
# <<< CAMBIO: Esta función ahora solo hace la parte final de la preparación de datos.
# Ya no pone a cero las features, sino que espera que se calculen correctamente.
def build_feature_matrix_nn(df, training_columns):
    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "recibidos_lag24", "recibidos_ma24", "recibidos_ma168", # Para llamadas
        "tmo_seg_lag24", "tmo_seg_ma24", "tmo_seg_ma168"      # Para TMO
    ]
    # Filtra las features base que realmente existen en el df
    existing_base_feats = [feat for feat in base_feats if feat in df.columns]
    
    df_dummies = pd.get_dummies(df[["dow", "month"]], drop_first=False, dtype=int)
    X = pd.concat([df[existing_base_feats], df_dummies], axis=1)

    # Añade las columnas que falten del entrenamiento y las rellena con 0
    for c in set(training_columns) - set(X.columns): 
        X[c] = 0
        
    # Reordena y devuelve las columnas en el orden exacto del entrenamiento
    return X[training_columns].fillna(0)

# <<< CAMBIO: La lógica de suavizado permanece igual, pero ahora se aplica a predicciones de calidad.
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
    return df.drop(columns=["med", "mad", "upper_cap", "is_peak"])

# <<< CAMBIO RADICAL: Nueva función para predicción iterativa
def predecir_futuro_iterativo(df_hist, modelo, scaler, target_col, future_timestamps):
    """
    Realiza predicciones futuras de forma iterativa, un paso a la vez.
    """
    training_columns = scaler.get_feature_names_out()
    df_prediccion = df_hist.copy()

    for ts in future_timestamps:
        # 1. Crear una fila temporal para el timestamp que queremos predecir
        temp_df = pd.DataFrame(index=[ts])
        
        # 2. Unir con el historial para poder calcular features
        df_completo = pd.concat([df_prediccion, temp_df])
        
        # 3. Calcular todas las features necesarias
        df_completo = add_time_features(df_completo)
        df_completo = rolling_features(df_completo, target_col)
        
        # 4. Extraer la última fila (la que queremos predecir)
        X_step = build_feature_matrix_nn(df_completo.tail(1), training_columns)
        
        # 5. Escalar y Predecir
        X_step_scaled = scaler.transform(X_step)
        prediccion = modelo.predict(X_step_scaled, verbose=0).flatten()[0]
        
        # 6. Actualizar el dataframe de predicción con el valor predicho
        df_prediccion.loc[ts, target_col] = prediccion

    return df_prediccion.loc[future_timestamps, target_col]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # --- Descargar modelos (si es necesario, descomenta y adapta esta parte) ---
    # print("Descargando modelos...")
    # for asset in [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]:
    #     download_asset_from_latest(OWNER, REPO, asset, MODELS_DIR)

    # --- Cargar modelos y scalers ---
    print("Cargando modelos y scalers...")
    model_ll = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_LLAMADAS}")
    scaler_ll = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_LLAMADAS}")
    model_tmo = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_TMO}")
    scaler_tmo = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_TMO}")

    # <<< CAMBIO: Cargar y procesar datos históricos
    print(f"Cargando datos históricos desde {DATA_FILE}...")
    df_hist_raw = pd.read_csv(DATA_FILE)
    df_hist = ensure_datetime(df_hist_raw)
    df_hist[TARGET_TMO] = df_hist_raw.set_index(df_hist.index)['tmo (segundos)'].apply(parse_tmo_to_seconds)
    df_hist = df_hist[[TARGET_LLAMADAS, TARGET_TMO]] # Mantener solo columnas necesarias

    # <<< CAMBIO: Definir el rango de fechas futuras a predecir
    last_known_date = df_hist.index.max()
    # Predecimos desde la siguiente hora hasta el final de 3 meses en el futuro
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (last_known_date.to_period('M') + 3).to_timestamp(how='end').tz_localize(TIMEZONE)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred}.")

    # --- Predicción iterativa de llamadas ---
    print("Realizando predicción iterativa de llamadas...")
    pred_ll = predecir_futuro_iterativo(
        df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts
    )
    
    # Crear el DataFrame final
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    # --- Predicción iterativa de TMO ---
    print("Realizando predicción iterativa de TMO...")
    pred_tmo = predecir_futuro_iterativo(
        df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts
    )
    df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(int)

    # --- Aplicar suavizado solo a llamadas ---
    print("Aplicando suavizado de picos a las predicciones de llamadas...")
    # Añadimos features de tiempo al df final para que el suavizado funcione
    df_final["dow"] = df_final.index.dayofweek
    df_final["hour"] = df_final.index.hour
    df_final = apply_peak_smoothing(df_final, "pred_llamadas", mad_k=MAD_K, method=SUAVIZADO)
    df_final["pred_llamadas"] = df_final["pred_llamadas"].round().astype(int)
    
    # --- Guardar outputs ---
    print("Guardando archivos de salida...")
    out = df_final[["pred_llamadas", "pred_tmo_seg"]].reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    # --- Diarios ---
    daily = (out.assign(date=pd.to_datetime(out["ts"]).dt.date)
               .groupby("date", as_index=False)["pred_llamadas"]
               .sum()
               .rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False)

    # --- Timestamp de actualización ---
    json.dump(
        {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        open(STAMP_JSON, "w")
    )

    print("✔ Inferencia completada con predicción iterativa y suavizado.")

if __name__ == "__main__":
    main()
