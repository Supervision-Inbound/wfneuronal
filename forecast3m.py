# =======================================================================
# forecast3m.py (Transformer temporal, usa últimos 90 días de /data/Hosting ia.xlsx)
# Salidas intactas en /public (predicciones.csv/json, llamadas_por_dia.csv, erlang_forecast.json, last_update.json)
# =======================================================================

import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from utils_release import download_asset_from_latest

# ---------- Parámetros ----------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"

# Assets (mismos nombres que la versión previa)
ASSET_LLAMADAS = "modelo_llamadas_nn.h5"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_TMO = "modelo_tmo_nn.h5"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

# Directorios
MODELS_DIR = "models"
DATA_DIR   = "data"

# Entradas de datos
HOSTING_FILE = os.path.join(DATA_DIR, "Hosting ia.xlsx")
FERIADOS_FILE = os.path.join(DATA_DIR, "Feriados_Chile_2023_2027.xlsx")

# --- SALIDAS (todas en /public) ---
OUT_CSV_DATAOUT   = "public/predicciones.csv"
OUT_JSON_PUBLIC   = "public/predicciones.json"
OUT_JSON_ERLANG   = "public/erlang_forecast.json"
STAMP_JSON        = "public/last_update.json"
OUT_CSV_DAILY     = "public/llamadas_por_dia.csv"

# --- Parámetros ---
TIMEZONE = "America/Santiago"
FREQ     = "H"
SEQ_LEN  = 24*90   # 90 días de historia
HORIZON  = 24*90   # 90 días futuros
TARGET_LLAMADAS = "recibidos"
TARGET_TMO      = "tmo (segundos)"

# --- Erlang configuraciones (se mantienen) ---
SLA_TARGET = 0.90; ASA_TARGET_S = 22
MAX_OCC = 0.85; SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0; BREAKS_MIN = [15, 15]; AUX_RATE = 0.15; ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True; MEAN_PATIENCE_S = 60.0; ABANDON_MAX = 0.06
AWT_MAX_S = 120.0; INTERCALL_GAP_S = 10.0

# ----------------- Utilidades -----------------
def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int,float)): return float(val)
    s = str(val).strip()
    if s.isdigit(): return float(s)
    parts = s.split(":")
    try:
        if len(parts)==3:
            h,m,sec = map(float,parts); return h*3600+m*60+sec
        if len(parts)==2:
            m,sec = map(float,parts); return m*60+sec
        return float(s)
    except: return np.nan

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
    aht_s = tmo_seg + INTERCALL_GAP_S
    R = (llamadas * aht_s) / 3600.0
    N = max(1, int(np.ceil(R / MAX_OCC)))
    if USE_ERLANG_A:
        sla, abn, asa = erlang_a(R, N, MEAN_PATIENCE_S, aht_s)
    else:
        p_wait = erlang_c(R, N)
        asa = (p_wait * aht_s) / max(N - R, 1e-6)
        sla = 1 - p_wait * np.exp(-AWT_MAX_S * (N - R) / aht_s)
        abn = 0.0
    n = N
    iterations = 0
    while ((sla < sla_target) or (asa > asa_target_s)) and iterations < 200:
        n += 1; iterations += 1
        if USE_ERLANG_A:
            sla, abn, asa = erlang_a(R, n, MEAN_PATIENCE_S, aht_s)
        else:
            p_wait = erlang_c(R, n)
            asa = (p_wait * aht_s) / max(n - R, 1e-6)
            sla = 1 - p_wait * np.exp(-AWT_MAX_S * (n - R) / aht_s)
    return n

def get_prod_factor(shift_hours, lunch_hours, breaks_min):
    work_h = shift_hours - lunch_hours - sum(breaks_min)/60.0
    return max(0.0, work_h/shift_hours)*(1-AUX_RATE)

# ----------------- Carga de data reciente (90 días) -----------------
def load_recent_history():
    if not os.path.exists(HOSTING_FILE):
        raise FileNotFoundError(f"No se encontró {HOSTING_FILE}.")
    df = pd.read_excel(HOSTING_FILE)
    # columnas estándar
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "hora" in df.columns:
        df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str)+" "+df["hora"].astype(str).str[:5], errors="coerce")
    else:
        df["ts"] = df["fecha_dt"]
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["tmo (segundos)"] = df["tmo (segundos)"].apply(parse_tmo_to_seconds)
    # Tomar últimos 90 días completos
    end_ts = df["ts"].max()
    start_ts = end_ts - pd.Timedelta(days=90)
    recent = df[df["ts"] > start_ts].copy()
    # Completar huecos horarios si los hubiese
    full_rng = pd.date_range(recent["ts"].min().floor("H"), end_ts.ceil("H"), freq="H")
    recent = recent.set_index("ts").reindex(full_rng).rename_axis("ts").reset_index()
    recent.rename(columns={"index":"ts"}, inplace=True)
    # Rellenos simples
    for c in [TARGET_LLAMADAS, TARGET_TMO]:
        if c in recent.columns:
            recent[c] = recent[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
    return recent

# ----------------- Inference -----------------
def main():
    os.makedirs("public", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1) Descargar modelos y scalers desde el último Release
    for asset_name in [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]:
        download_asset_from_latest(OWNER, REPO, asset_name, MODELS_DIR)
        if not os.path.exists(os.path.join(MODELS_DIR, asset_name)):
            raise FileNotFoundError(f"No se encontró {asset_name} en {MODELS_DIR}.")

    # 2) Cargar
    model_ll = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_LLAMADAS))
    model_tmo = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_TMO))
    scaler_ll = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_LLAMADAS))
    scaler_tmo = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_TMO))

    # 3) Cargar historia reciente (90 días)
    hist = load_recent_history()
    last_ts = hist["ts"].max()

    # 4) Construir secuencias de entrada (univariadas)
    series_ll = hist[TARGET_LLAMADAS].astype(float).values.reshape(-1,1)
    series_tmo = hist[TARGET_TMO].astype(float).values.reshape(-1,1)

    seq_ll = scaler_ll.transform(series_ll).flatten()
    seq_tmo = scaler_tmo.transform(series_tmo).flatten()

    if len(seq_ll) < SEQ_LEN or len(seq_tmo) < SEQ_LEN:
        raise ValueError("Historia insuficiente: se requieren 90 días completos de datos.")

    x_ll = seq_ll[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    x_tmo = seq_tmo[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)

    # 5) Predecir 90 días futuros (horizonte)
    yhat_ll_s = model_ll.predict(x_ll, verbose=0)[0]
    yhat_tmo_s = model_tmo.predict(x_tmo, verbose=0)[0]

    yhat_ll = scaler_ll.inverse_transform(yhat_ll_s.reshape(-1,1)).flatten()
    yhat_tmo = scaler_tmo.inverse_transform(yhat_tmo_s.reshape(-1,1)).flatten()

    # 6) Construir timeline futura por hora
    future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=HORIZON, freq="H", tz=None)
    out = pd.DataFrame({
        "ts": future_index.tz_localize(None).astype(str),
        "pred_llamadas": np.maximum(0, np.round(yhat_ll,0)).astype(int),
        "pred_tmo_seg": np.maximum(0, np.round(yhat_tmo,0)).astype(int),
    })

    # 7) Export CSV/JSON
    out.to_csv(OUT_CSV_DATAOUT, index=False, encoding="utf-8")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    # 7.1) Totales diarios
    tmp = out.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"])
    daily = (tmp.assign(date=tmp["ts"].dt.date).groupby("date", as_index=False)["pred_llamadas"].sum()
             .rename(columns={"pred_llamadas":"total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False, encoding="utf-8")

    # 8) Dimensionamiento Erlang
    erlang_rows = []
    prod_factor = get_prod_factor(SHIFT_HOURS, LUNCH_HOURS, BREAKS_MIN)
    derived_shrinkage   = 1 - prod_factor
    effective_shrinkage = 1 - (prod_factor * (1 - ABSENTEEISM_RATE))

    for row in out.to_dict(orient="records"):
        llamadas = row["pred_llamadas"]
        tmo = row["pred_tmo_seg"]
        agentes_prod = calculate_agents(llamadas, tmo, SLA_TARGET, ASA_TARGET_S)
        agentes_agendados = int(np.ceil(agentes_prod / (1 - effective_shrinkage))) if effective_shrinkage < 1 else agentes_prod
        erlang_rows.append({
            "ts": row["ts"],
            "pred_llamadas": llamadas,
            "pred_tmo_seg": tmo,
            "agents_productive": agentes_prod,
            "agents_scheduled": agentes_agendados,
            "params": {
                "SLA_TARGET": SLA_TARGET, "ASA_TARGET_S": ASA_TARGET_S,
                "MAX_OCC": MAX_OCC, "SHIFT_HOURS": SHIFT_HOURS, "LUNCH_HOURS": LUNCH_HOURS,
                "BREAKS_MIN": BREAKS_MIN, "AUX_RATE": AUX_RATE,
                "DERIVED_SHRINKAGE": round(derived_shrinkage,4),
                "PRODUCTIVITY_FACTOR": round(prod_factor,4),
                "USE_ERLANG_A": USE_ERLANG_A, "MEAN_PATIENCE_S": MEAN_PATIENCE_S,
                "ABANDON_MAX": ABANDON_MAX, "AWT_MAX_S": AWT_MAX_S, "INTERCALL_GAP_S": INTERCALL_GAP_S,
                "ABSENTEEISM_RATE": ABSENTEEISM_RATE, "EFFECTIVE_SHRINKAGE": round(effective_shrinkage,4)
            }
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # 9) Timestamp
    with open(STAMP_JSON, "w") as f:
        json.dump({"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}, f)

    print("✔ Proceso de inferencia completado (salidas en /public).")

if __name__ == "__main__":
    main()

