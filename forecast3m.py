# =======================================================================
# forecast3m.py (VERSIÓN CORREGIDA: con desnormalización de Y)
# =======================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"

# --- NOMBRES DE LOS ARCHIVOS DE MODELO ---
ASSET_LLAMADAS = "modelo_llamadas_nn.h5"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_Y_NORM = "y_norm_params.pkl"  # <- NUEVO
ASSET_TMO = "modelo_tmo_nn.h5"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

MODELS_DIR = "models"

# --- SALIDAS (todas en /public) ---
OUT_CSV_DATAOUT   = "public/predicciones.csv"
OUT_JSON_PUBLIC   = "public/predicciones.json"
OUT_JSON_ERLANG   = "public/erlang_forecast.json"
STAMP_JSON        = "public/last_update.json"
OUT_CSV_DAILY     = "public/llamadas_por_dia.csv"

# --- Parámetros de Fechas ---
TIMEZONE = "America/Santiago"
FREQ     = "H"
TARGET_LLAMADAS = "recibidos"
TARGET_TMO      = "tmo_seg"

# --- Parámetros de operación (Erlang) ---
SLA_TARGET = 0.90; ASA_TARGET_S = 22; MAX_OCC = 0.85; SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0; BREAKS_MIN = [15, 15]; AUX_RATE = 0.15; ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True; MEAN_PATIENCE_S = 60.0; ABANDON_MAX = 0.06
AWT_MAX_S = 120.0; INTERCALL_GAP_S = 10.0

# --- Función de Features (debe coincidir con entrenamiento) ---
def build_feature_matrix_nn(df, target_col, training_columns):
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]  = np.cos(2 * np.pi * df["dow"] / 7)
    df[f"{target_col}_lag24"]  = 0
    df[f"{target_col}_ma24"]   = 0
    df[f"{target_col}_ma168"]  = 0
    
    # NUEVO: Agregar is_peak si está en las columnas de entrenamiento
    if "is_peak" in training_columns:
        df["is_peak"] = 0

    base_feats = ["sin_hour","cos_hour","sin_dow","cos_dow",
                  f"{target_col}_lag24",f"{target_col}_ma24",f"{target_col}_ma168"]
    
    # Agregar is_peak a las features base si existe
    if "is_peak" in training_columns:
        base_feats.append("is_peak")
    
    cat_feats  = ["dow","month"]

    df_dummies = pd.get_dummies(df[cat_feats], columns=cat_feats, drop_first=False)
    X = pd.concat([df[base_feats], df_dummies], axis=1)

    missing_cols = set(training_columns) - set(X.columns)
    for c in missing_cols: X[c] = 0
    X = X[training_columns]
    return X.fillna(0)

# --- Erlang ---
def erlang_c(R, N):
    if N <= R: return 0.0
    inv_erlang_b = 1.0
    for i in range(1, int(N) + 1):
        inv_erlang_b = 1.0 + (i / R) * inv_erlang_b
    erlang_b = 1.0 / inv_erlang_b
    return (N * erlang_b) / (N - R * (1 - erlang_b))

def erlang_a(R, N, patience_s, aht_s):
    if N <= R: return 0.0, 0.0, R - N
    p_wait = erlang_c(R, N)
    asa = (p_wait * aht_s) / (N - R) if p_wait > 0 else 0
    sla = 1 - p_wait * np.exp(-(N - R) * patience_s / aht_s) if p_wait > 0 else 1.0
    abn = p_wait * (1 - (1 - np.exp(-patience_s / asa))) if asa > 0 else 0
    return sla, abn, asa

def calculate_agents(llamadas, tmo_seg, sla_target, asa_target_s):
    if llamadas == 0: return 0
    erlangs = (llamadas * tmo_seg) / 3600.0
    min_agents = int(np.ceil(erlangs)) + 1
    for agents in range(min_agents, min_agents + 100):
        if USE_ERLANG_A:
            sla, _, asa = erlang_a(erlangs, agents, asa_target_s, tmo_seg)
        else:
            p_wait = erlang_c(erlangs, agents)
            asa = (p_wait * tmo_seg) / (agents - erlangs) if p_wait > 0 else 0
            sla = 1 - p_wait * np.exp(-(agents - erlangs) * asa_target_s / tmo_seg) if p_wait > 0 else 1.0
        if sla >= sla_target and asa <= asa_target_s:
            return agents
    return min_agents + 100

def get_prod_factor(shift_h, lunch_h, breaks_m):
    total_paid = shift_h * 60
    non_prod = (lunch_h * 60) + sum(breaks_m)
    prod_minutes = total_paid - non_prod
    return (prod_minutes / total_paid) if total_paid > 0 else 0

def main():
    # carpetas necesarias
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # 1) Descargar artefactos
    print(f"Descargando modelos desde: {OWNER}/{REPO}")
    assets = [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_Y_NORM, ASSET_TMO, ASSET_SCALER_TMO]  # <- AGREGADO y_norm_params
    for asset_name in assets:
        download_asset_from_latest(OWNER, REPO, asset_name, MODELS_DIR)
        path = os.path.join(MODELS_DIR, asset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró {asset_name} en {MODELS_DIR}. "
                f"Verifica el Release de {OWNER}/{REPO} y el nombre exacto del asset."
            )
    print("✔ Modelos y scalers descargados.")

    # 2) Cargar modelos y scalers
    print("Cargando modelos y scalers…")
    model_ll = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_LLAMADAS))
    scaler_ll = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_LLAMADAS))
    y_norm_params = joblib.load(os.path.join(MODELS_DIR, ASSET_Y_NORM))  # <- NUEVO
    model_tmo = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_TMO))
    scaler_tmo = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_TMO))
    training_cols_ll  = scaler_ll.get_feature_names_out()
    training_cols_tmo = scaler_tmo.get_feature_names_out()

    # 3) Rango de fechas: 1 mes anterior hasta fin de mes siguiente (exclusivo el inicio del subsiguiente)
    print("Generando rango de fechas…")
    today = pd.Timestamp.now(tz=TIMEZONE)
    start_date = (today.to_period('M') - 1).to_timestamp(how='start').tz_localize(TIMEZONE)
    end_date_exclusive = (today.to_period('M') + 2).to_timestamp(how='start').tz_localize(TIMEZONE)
    prediction_dates = pd.date_range(start=start_date, end=end_date_exclusive, freq=FREQ, inclusive='left')

    df_pred = pd.DataFrame({"ts": prediction_dates})
    df_pred["dow"]   = df_pred["ts"].dt.dayofweek
    df_pred["month"] = df_pred["ts"].dt.month
    df_pred["hour"]  = df_pred["ts"].dt.hour

    # 4) Features, escala y predice LLAMADAS
    print("Construyendo features y prediciendo LLAMADAS…")
    X_pred_ll = build_feature_matrix_nn(df_pred.copy(), TARGET_LLAMADAS, training_cols_ll)
    X_pred_ll_scaled = scaler_ll.transform(X_pred_ll)
    pred_ll_norm = model_ll.predict(X_pred_ll_scaled).flatten()
    
    # DESNORMALIZAR las predicciones de llamadas
    pred_ll = pred_ll_norm * y_norm_params['y_std'] + y_norm_params['y_mean']
    print(f"Predicciones desnormalizadas - Min: {pred_ll.min():.0f}, Max: {pred_ll.max():.0f}, Media: {pred_ll.mean():.0f}")

    # Predice TMO
    print("Prediciendo TMO…")
    X_pred_tmo = build_feature_matrix_nn(df_pred.copy(), TARGET_TMO, training_cols_tmo)
    X_pred_tmo_scaled = scaler_tmo.transform(X_pred_tmo)
    pred_tmo = model_tmo.predict(X_pred_tmo_scaled).flatten()

    # 5) Ensamblar y guardar (en /public)
    out = pd.DataFrame({
        "ts": prediction_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "pred_llamadas": np.round(np.maximum(0, pred_ll)).astype(int),
        "pred_tmo_seg": np.round(np.maximum(0, pred_tmo)).astype(int)
    })
    print(f"Guardando {len(out)} filas en CSV/JSON (carpeta public)…")
    out.to_csv(OUT_CSV_DATAOUT, index=False, encoding="utf-8")

    out_dict = out.to_dict(orient="records")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    # 5.1) Totales diarios (CSV en /public)
    print("Generando totales diarios de llamadas…")
    tmp = out.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"])
    daily = (tmp.assign(date=tmp["ts"].dt.date)
                 .groupby("date", as_index=False)["pred_llamadas"]
                 .sum()
                 .rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False, encoding="utf-8")
    print(f"Guardado: {OUT_CSV_DAILY} ({len(daily)} filas)")
    print(f"Total llamadas predichas por día - Min: {daily['total_llamadas'].min()}, Max: {daily['total_llamadas'].max()}, Media: {daily['total_llamadas'].mean():.0f}")

    # 6) Dimensionamiento Erlang (JSON en /public)
    print("Calculando dimensionamiento de agentes…")
    erlang_rows = []
    prod_factor = get_prod_factor(SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN)
    derived_shrinkage   = 1 - prod_factor
    effective_shrinkage = 1 - (prod_factor * (1 - ABSENTEEISM_RATE))

    for row in out_dict:
        llamadas = row["pred_llamadas"]
        tmo      = row["pred_tmo_seg"]
        agentes_prod = calculate_agents(llamadas, tmo, SLA_TARGET, ASA_TARGET_S)
        agentes_agendados = int(np.ceil(agentes_prod / (1 - effective_shrinkage))) if effective_shrinkage < 1 else agentes_prod
        erlangs = (llamadas * tmo) / 3600.0
        occupancy = erlangs / agentes_prod if agentes_prod > 0 else 0
        sla, abn, asa = erlang_a(erlangs, agentes_prod, ASA_TARGET_S, tmo)
        erlang_rows.append({
            "ts": row["ts"],
            "llamadas": llamadas,
            "tmo_seg": tmo,
            "erlangs": round(erlangs, 2),
            "agentes_productivos": agentes_prod,
            "agentes_agendados": agentes_agendados,
            "occupancy": round(occupancy, 4),
            "service_level": round(sla, 4),
            "abandon_rate": round(abn, 4),
            "avg_wait_s": round(asa, 2),
            "model": "Erlang-A",
            "params": {
                "SLA_TARGET": SLA_TARGET, "ASA_TARGET_S": ASA_TARGET_S, "MAX_OCC": MAX_OCC,
                "SHIFT_HOURS": SHIFT_HOURS, "LUNCH_HOURS": LUNCH_HOURS, "BREAKS_MIN": BREAKS_MIN,
                "AUX_RATE": AUX_RATE, "DERIVED_SHRINKAGE": round(derived_shrinkage, 4),
                "PRODUCTIVITY_FACTOR": round(prod_factor, 4),
                "USE_ERLANG_A": USE_ERLANG_A, "MEAN_PATIENCE_S": MEAN_PATIENCE_S,
                "ABANDON_MAX": ABANDON_MAX, "AWT_MAX_S": AWT_MAX_S, "INTERCALL_GAP_S": INTERCALL_GAP_S,
                "ABSENTEEISM_RATE": ABSENTEEISM_RATE, "EFFECTIVE_SHRINKAGE": round(effective_shrinkage, 4)
            }
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # 7) Timestamp
    stamp = {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(STAMP_JSON, "w") as f:
        json.dump(stamp, f)

    print("Proceso de inferencia completado (salidas en /public).")

if __name__ == "__main__":
    main()
