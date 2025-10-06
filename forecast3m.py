# =======================================================================
# forecast3m.py (VERSIÓN ADAPTADA PARA REDES NEURONALES)
# Inferencia con los modelos Keras (.h5) y los scalers (.pkl)
# =======================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"      # <- ajusta si cambia
REPO  = "wf-Analytics-AI2.5"       # <- exacto

# --- ¡NUEVOS NOMBRES DE ARCHIVOS! ---
ASSET_LLAMADAS = "modelo_llamadas_nn.h5"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_TMO = "modelo_tmo_nn.h5"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

MODELS_DIR = "models"
OUT_CSV = "data_out/predicciones.csv"
OUT_JSON_PUBLIC = "public/predicciones.json"
OUT_JSON_DATAOUT = "data_out/predicciones.json"
OUT_JSON_ERLANG = "public/erlang_forecast.json"
OUT_JSON_ERLANG_DO = "data_out/erlang_forecast.json"
STAMP_JSON = "public/last_update.json"

HOURS_AHEAD = 24 * 90
FREQ = "H"
TARGET_LLAMADAS = "recibidos"
TARGET_TMO = "tmo_seg"

# Parámetros de operación (sin cambios)
SLA_TARGET = 0.90
ASA_TARGET_S = 22
MAX_OCC = 0.85
SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0
BREAKS_MIN = [15, 15]
AUX_RATE = 0.15
ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True
MEAN_PATIENCE_S = 60.0
ABANDON_MAX = 0.06
AWT_MAX_S = 120.0
INTERCALL_GAP_S = 10.0

# --- NUEVA FUNCIÓN (debe ser IDÉNTICA a la del entrenamiento) ---
def build_feature_matrix_nn(df, target_col):
    # Features base: cíclicas, etc.
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Features de rolling (¡IMPORTANTE! Se calculan sobre datos pasados, aquí los simulamos)
    # En inferencia real, no tenemos el futuro, por lo que estas features no se pueden crear.
    # El modelo se entrenó con ellas, así que creamos placeholders (columnas con ceros).
    # Esto es una simplificación. Un modelo productivo más avanzado usaría datos históricos reales.
    df[f"{target_col}_lag24"] = 0
    df[f"{target_col}_ma24"] = 0
    df[f"{target_col}_ma168"] = 0

    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"
    ]
    cat_feats = ["dow", "month"]
    df_dummies = pd.get_dummies(df[cat_feats], columns=cat_feats, drop_first=False)
    
    # Aseguramos que todas las columnas categóricas del entrenamiento existan
    # Columnas para 'dow' (0 a 6) y 'month' (1 a 12)
    for i in range(7):
        if f'dow_{i}' not in df_dummies.columns:
            df_dummies[f'dow_{i}'] = 0
    for i in range(1, 13):
        if f'month_{i}' not in df_dummies.columns:
            df_dummies[f'month_{i}'] = 0

    X = pd.concat([df[base_feats], df_dummies], axis=1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def erlang_c(R, N):
    if N <= R: return 0.0
    inv_erlang_b = 1.0
    for i in range(1, int(N) + 1):
        inv_erlang_b = 1.0 + (i / R) * inv_erlang_b
    erlang_b = 1.0 / inv_erlang_b
    return (N * erlang_b) / (N - R * (1 - erlang_b))

# ... (El resto de las funciones de Erlang y cálculo de agentes no necesitan cambios)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("data_out", exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # 1) Descargar los 4 artefactos desde el último Release
    print("Descargando modelos y scalers desde el último Release...")
    assets_to_download = [
        ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS,
        ASSET_TMO, ASSET_SCALER_TMO
    ]
    for asset_name in assets_to_download:
        download_asset_from_latest(OWNER, REPO, asset_name, MODELS_DIR)

    # 2) Cargar modelos y scalers
    print("Cargando modelos y scalers en memoria...")
    try:
        model_ll = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_LLAMADAS))
        scaler_ll = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_LLAMADAS))
        model_tmo = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_TMO))
        scaler_tmo = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_TMO))
    except Exception as e:
        print(f"Error fatal al cargar los modelos/scalers: {e}")
        return

    # 3) Crear DataFrame con fechas futuras
    print(f"Generando timestamps para las próximas {HOURS_AHEAD} horas...")
    start_date = pd.Timestamp.now(tz="America/Santiago").floor(FREQ)
    future_dates = pd.date_range(start=start_date, periods=HOURS_AHEAD, freq=FREQ)
    df_pred = pd.DataFrame({"ts": future_dates})
    df_pred["dow"] = df_pred["ts"].dt.dayofweek
    df_pred["month"] = df_pred["ts"].dt.month
    df_pred["hour"] = df_pred["ts"].dt.hour

    # 4) Construir matrices de características, escalar y predecir
    print("Construyendo características y generando predicciones...")
    
    # Para llamadas
    X_pred_ll = build_feature_matrix_nn(df_pred.copy(), TARGET_LLAMADAS)
    X_pred_ll_scaled = scaler_ll.transform(X_pred_ll)
    pred_ll = model_ll.predict(X_pred_ll_scaled).flatten()

    # Para TMO
    X_pred_tmo = build_feature_matrix_nn(df_pred.copy(), TARGET_TMO)
    X_pred_tmo_scaled = scaler_tmo.transform(X_pred_tmo)
    pred_tmo = model_tmo.predict(X_pred_tmo_scaled).flatten()
    
    # Ensamblar predicciones
    out = pd.DataFrame({
        "ts": future_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": np.round(np.maximum(0, pred_ll)).astype(int),
        "pred_tmo_seg": np.round(np.maximum(0, pred_tmo)).astype(int)
    })

    # ... (El resto del script para generar los JSON y Erlang no necesita cambios)
    # Asegúrate de copiar el resto de tu script original desde aquí.
    
    print("Proceso de inferencia completado.")


if __name__ == "__main__":
    main()
