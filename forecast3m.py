# =======================================================================
# forecast3m.py (VERSIÓN FINAL - USA DATOS REALES DE data/Hosting ia.xlsx)
# =======================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utils_release import download_asset_from_latest

# ---------- Parámetros de Modelo y Repositorio ----------
OWNER = "Supervision-Inbound"
REPO = "wfneuronal"
SEQ_LEN = 2160
HORIZON = 2160

# --- Nombres de los Archivos en el Release ---
ASSET_LLAMADAS = "modelo_llamadas.keras"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_TMO = "modelo_tmo.keras"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

MODELS_DIR = "models"
PUBLIC_DIR = "public"

# --- NUEVA FUENTE DE DATOS HISTÓRICOS ---
HISTORICAL_DATA_PATH = "data/Hosting ia.xlsx"

# --- Nombres de los Archivos de Salida ---
OUT_CSV_PRED = os.path.join(PUBLIC_DIR, "predicciones.csv")
OUT_JSON_PRED = os.path.join(PUBLIC_DIR, "predicciones.json")
OUT_JSON_ERLANG = os.path.join(PUBLIC_DIR, "erlang_forecast.json")
STAMP_JSON = os.path.join(PUBLIC_DIR, "last_update.json")
OUT_CSV_DAILY = os.path.join(PUBLIC_DIR, "llamadas_por_dia.csv")

# --- Parámetros de Pre-procesamiento y Erlang (deben coincidir con el entrenamiento) ---
MAD_K = 3.5
SUAVIZADO = "cap"
COL_FECHA = "fecha"
COL_HORA = "hora"
COL_LLAMADAS = "recibidos"
COL_TMO = "tmo (segundos)"
TIMEZONE = "America/Santiago"
FREQ = "H"
SLA_TARGET = 0.90; ASA_TARGET_S = 22; MAX_OCC = 0.85; SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0; BREAKS_MIN = [15, 15]; AUX_RATE = 0.15; ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True; MEAN_PATIENCE_S = 60.0; ABANDON_MAX = 0.06
AWT_MAX_S = 120.0; INTERCALL_GAP_S = 10.0

# --- FUNCIONES DE PRE-PROCESAMIENTO (COPIADAS DEL SCRIPT DE ENTRENAMIENTO) ---
def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int,float)): return float(val)
    s = str(val).strip()
    if s.isdigit(): return float(s)
    parts = s.split(":")
    try:
        if len(parts)==3: h,m,sec = map(float,parts); return h*3600+m*60+sec
        if len(parts)==2: m,sec = map(float,parts); return m*60+sec
        return float(s)
    except: return np.nan

def read_data(path):
    if path.lower().endswith(".csv"): return pd.read_csv(path)
    return pd.read_excel(path)

def ensure_ts(df):
    df["fecha_dt"] = pd.to_datetime(df[COL_FECHA], errors="coerce")
    df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str)+" "+df[COL_HORA].astype(str).str[:5], errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts")

def add_time_keys(df):
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour
    return df

def robust_baseline_by_dow_hour(df, target_col):
    grp = df.groupby(["dow","hour"])[target_col].agg(["median"]).rename(columns={"median":"med"})
    def mad(x):
        med = np.median(x); return np.median(np.abs(x-med))
    grp["mad"] = df.groupby(["dow","hour"])[target_col].apply(mad).values
    return grp

def detect_peaks(df, target_col, mad_k):
    base = robust_baseline_by_dow_hour(df, target_col)
    df = df.merge(base, left_on=["dow","hour"], right_index=True, how="left")
    df["upper_cap"] = df["med"] + mad_k * df["mad"].replace(0, df["mad"].median())
    df["is_peak"] = (df[target_col] > df["upper_cap"]).astype(int)
    return df

def smooth_series(df, target_col, method="cap"):
    if method=="cap":
        df[target_col+"_smooth"] = np.where(df["is_peak"]==1, df["upper_cap"], df[target_col])
    else:
        df[target_col+"_smooth"] = np.where(df["is_peak"]==1, df["med"], df[target_col])
    return df

# --- Funciones de Erlang (sin cambios) ---
def erlang_c(R, N):
    if N <= R: return 0.0
    inv_erlang_b = 1.0;
    for i in range(1, int(N) + 1): inv_erlang_b = 1.0 + (i / R) * inv_erlang_b
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
    erlangs = (llamadas * tmo_seg) / 3600.0; min_agents = int(np.ceil(erlangs)) + 1
    for agents in range(min_agents, min_agents + 100):
        sla, _, asa = erlang_a(erlangs, agents, asa_target_s, tmo_seg)
        if sla >= sla_target and asa <= asa_target_s: return agents
    return min_agents + 100

def get_prod_factor(shift_h, lunch_h, breaks_m):
    total_paid, non_prod = shift_h * 60, (lunch_h * 60) + sum(breaks_m)
    return ((total_paid - non_prod) / total_paid) if total_paid > 0 else 0

# --- LÓGICA DE PREDICCIÓN PARA TRANSFORMER ---
def predict_sequence(model, scaler, historical_data, seq_len, horizon):
    last_sequence = historical_data[-seq_len:].values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    input_data = last_sequence_scaled.reshape(1, seq_len, 1)
    pred_scaled = model.predict(input_data)
    pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return np.maximum(0, pred_unscaled).flatten().round().astype(int)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(PUBLIC_DIR, exist_ok=True)

    print(f"Descargando modelos desde: {OWNER}/{REPO}")
    assets = [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]
    for asset_name in assets:
        download_asset_from_latest(OWNER, REPO, asset_name, MODELS_DIR)
    print("✔ Modelos y scalers descargados.")

    print("Cargando modelos y scalers…")
    model_ll = keras.models.load_model(os.path.join(MODELS_DIR, ASSET_LLAMADAS), compile=False)
    scaler_ll = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_LLAMADAS))
    model_tmo = keras.models.load_model(os.path.join(MODELS_DIR, ASSET_TMO), compile=False)
    scaler_tmo = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_TMO))

    print(f"Leyendo y pre-procesando datos históricos desde '{HISTORICAL_DATA_PATH}'...")
    df_hist = read_data(HISTORICAL_DATA_PATH)
    df_hist = ensure_ts(df_hist)
    df_hist = add_time_keys(df_hist)
    df_hist[COL_TMO] = df_hist[COL_TMO].apply(parse_tmo_to_seconds)
    df_hist = detect_peaks(df_hist, COL_LLAMADAS, MAD_K)
    df_hist = smooth_series(df_hist, COL_LLAMADAS, SUAVIZADO)
    df_hist['recibidos_smooth'] = df_hist[COL_LLAMADAS + "_smooth"]
    df_hist['tmo_seg'] = df_hist[COL_TMO].ffill().bfill()
    
    if len(df_hist) < SEQ_LEN:
        raise ValueError(f"Historial insuficiente. Se necesitan {SEQ_LEN} registros, se encontraron {len(df_hist)}.")
    print(f"Se procesaron {len(df_hist)} registros históricos.")

    print("Prediciendo secuencias futuras...")
    pred_ll = predict_sequence(model_ll, scaler_ll, df_hist['recibidos_smooth'], SEQ_LEN, HORIZON)
    pred_tmo = predict_sequence(model_tmo, scaler_tmo, df_hist['tmo_seg'], SEQ_LEN, HORIZON)

    last_historical_date = df_hist['ts'].max()
    prediction_dates = pd.date_range(start=last_historical_date + pd.Timedelta(hours=1), periods=HORIZON, freq=FREQ)
    df_new_preds = pd.DataFrame({"ts": prediction_dates, "pred_llamadas": pred_ll, "pred_tmo_seg": pred_tmo})
    
    # Preparamos el historial para concatenar, usando los datos reales (suavizados)
    df_hist_out = df_hist[['ts']].copy()
    df_hist_out['pred_llamadas'] = df_hist['recibidos_smooth']
    df_hist_out['pred_tmo_seg'] = df_hist['tmo_seg']

    df_full = pd.concat([df_hist_out, df_new_preds], ignore_index=True).drop_duplicates(subset='ts', keep='last')
    
    out = df_full.copy()
    out['ts'] = out['ts'].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_dict = out.to_dict(orient="records")

    print(f"Guardando {len(out)} filas en CSV/JSON (carpeta public)...")
    out.to_csv(OUT_CSV_PRED, index=False, encoding="utf-8")
    with open(OUT_JSON_PRED, "w", encoding="utf-8") as f: json.dump(out_dict, f, ensure_ascii=False, indent=2)

    print("Generando totales diarios y dimensionamiento...")
    # ... (El resto de la lógica de Erlang y guardado de archivos es idéntica y no necesita cambios)
    tmp = out.copy(); tmp["ts"] = pd.to_datetime(tmp["ts"])
    daily = (tmp.assign(date=tmp["ts"].dt.date).groupby("date", as_index=False)["pred_llamadas"].sum().rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False, encoding="utf-8")

    erlang_rows = []; prod_factor = get_prod_factor(SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN)
    effective_shrinkage = 1 - (prod_factor * (1 - ABSENTEEISM_RATE))
    for row in out_dict:
        llamadas, tmo = row["pred_llamadas"], row["pred_tmo_seg"]
        agentes_prod = calculate_agents(llamadas, tmo, SLA_TARGET, ASA_TARGET_S)
        agentes_agendados = int(np.ceil(agentes_prod / (1 - effective_shrinkage))) if effective_shrinkage < 1 else agentes_prod
        erlangs = (llamadas * tmo) / 3600.0; occupancy = erlangs / agentes_prod if agentes_prod > 0 else 0
        sla, abn, asa = erlang_a(erlangs, agentes_prod, ASA_TARGET_S, tmo)
        erlang_rows.append({ "ts": row["ts"], "llamadas": llamadas, "tmo_seg": tmo, "erlangs": round(erlangs, 2), "agentes_productivos": agentes_prod, "agentes_agendados": agentes_agendados, "occupancy": round(occupancy, 4), "service_level": round(sla, 4), "abandon_rate": round(abn, 4), "avg_wait_s": round(asa, 2) })
        
    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f: json.dump(erlang_rows, f, ensure_ascii=False, indent=2)
    stamp = {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(STAMP_JSON, "w") as f: json.dump(stamp, f)

    print("Proceso de inferencia completado.")

if __name__ == "__main__":
    main()
