# =======================================================================
# forecast3m.py (VERSIÓN CON DETECCIÓN DE PICOS Y SUAVIZADO EN INFERENCIA)
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

# --- Funciones auxiliares ---
def build_feature_matrix_nn(df, target_col, training_columns):
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]  = np.cos(2 * np.pi * df["dow"] / 7)
    df[f"{target_col}_lag24"] = 0
    df[f"{target_col}_ma24"] = 0
    df[f"{target_col}_ma168"] = 0

    base_feats = ["sin_hour","cos_hour","sin_dow","cos_dow",
                  f"{target_col}_lag24",f"{target_col}_ma24",f"{target_col}_ma168"]
    cat_feats = ["dow","month"]
    df_dummies = pd.get_dummies(df[cat_feats], drop_first=False)
    X = pd.concat([df[base_feats], df_dummies], axis=1)
    for c in set(training_columns) - set(X.columns): X[c] = 0
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
    return df

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # --- Descargar modelos y scalers ---
    for asset in [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]:
        download_asset_from_latest(OWNER, REPO, asset, MODELS_DIR)

    model_ll = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_LLAMADAS}")
    scaler_ll = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_LLAMADAS}")
    model_tmo = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_TMO}")
    scaler_tmo = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_TMO}")
    cols_ll = scaler_ll.get_feature_names_out()
    cols_tmo = scaler_tmo.get_feature_names_out()

    # --- Fechas ---
    today = pd.Timestamp.now(tz=TIMEZONE)
    start = (today.to_period('M') - 1).to_timestamp(how='start').tz_localize(TIMEZONE)
    end = (today.to_period('M') + 2).to_timestamp(how='start').tz_localize(TIMEZONE)
    ts = pd.date_range(start=start, end=end, freq=FREQ, inclusive='left')

    df = pd.DataFrame({"ts": ts})
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour

    # --- Predicción llamadas ---
    X_ll = build_feature_matrix_nn(df.copy(), TARGET_LLAMADAS, cols_ll)
    pred_ll = model_ll.predict(scaler_ll.transform(X_ll)).flatten()

    # --- Predicción TMO ---
    X_tmo = build_feature_matrix_nn(df.copy(), TARGET_TMO, cols_tmo)
    pred_tmo = model_tmo.predict(scaler_tmo.transform(X_tmo)).flatten()

    # --- Resultado bruto ---
    df["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)
    df["pred_tmo_seg"]  = np.maximum(0, np.round(pred_tmo)).astype(int)

    # --- Aplicar suavizado solo a llamadas ---
    df = apply_peak_smoothing(df, "pred_llamadas", mad_k=MAD_K, method=SUAVIZADO)
    df["pred_llamadas"] = df["pred_llamadas"].round().astype(int)

    # --- Guardar outputs ---
    out = df[["ts", "pred_llamadas", "pred_tmo_seg"]].copy()
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    # --- Diarios ---
    daily = (df.assign(date=df["ts"].dt.date)
               .groupby("date", as_index=False)["pred_llamadas"]
               .sum()
               .rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False)

    # --- Timestamp de actualización ---
    json.dump(
        {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        open(STAMP_JSON, "w")
    )

    print("✔ Inferencia completada con detección de picos y suavizado")

if __name__ == "__main__":
    main()

