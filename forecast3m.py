# =======================================================================
# forecast3m.py (tolerante a nombres y a 1 solo scaler)
# - Usa últimos 90 días desde data/Hosting ia.xlsx
# - Predice 90 días futuros y publica /public/*
# - Carga modelos con fallback (Keras 3, tf.keras y compat.v1) con FIX
#   Requiere: TF 2.17 + Keras 3 (ver requirements.txt)
# =======================================================================

import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras  # Keras 3 nativa
import joblib
from utils_release import download_asset_from_latest

# ---------- Parámetros del repo ----------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"

# ---------- Directorios ----------
MODELS_DIR = "models"
DATA_DIR   = "data"

# ---------- Entradas de data ----------
HOSTING_FILE   = os.path.join(DATA_DIR, "Hosting ia.xlsx")
FERIADOS_FILE  = os.path.join(DATA_DIR, "Feriados_Chile_2023_2027.xlsx")  # (opcional)

# ---------- Salidas ----------
OUT_CSV_DATAOUT = "public/predicciones.csv"
OUT_JSON_PUBLIC = "public/predicciones.json"
OUT_JSON_ERLANG = "public/erlang_forecast.json"
STAMP_JSON      = "public/last_update.json"
OUT_CSV_DAILY   = "public/llamadas_por_dia.csv"

# ---------- Parámetros de ventana/horizonte ----------
SEQ_LEN  = 24*90
HORIZON  = 24*90
TARGET_LLAMADAS = "recibidos"
TARGET_TMO      = "tmo (segundos)"

# ---------- Erlang (igual que siempre) ----------
SLA_TARGET = 0.90; ASA_TARGET_S = 22
MAX_OCC = 0.85; SHIFT_HOURS = 10.0
LUNCH_HOURS = 1.0; BREAKS_MIN = [15, 15]; AUX_RATE = 0.15; ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True; MEAN_PATIENCE_S = 60.0; ABANDON_MAX = 0.06
AWT_MAX_S = 120.0; INTERCALL_GAP_S = 10.0

# ---------- Utilidades ----------
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
    n = N; iterations = 0
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

# ---------- Carga robusta de modelos (FIX) ----------
def load_any_keras_model(path):
    e1 = e2 = e3 = None  # evitar UnboundLocalError en el raise final
    # 1) Keras 3 nativo (safe_mode=False relaja validaciones del config legacy)
    try:
        m = keras.models.load_model(path, compile=False, safe_mode=False)
        print(f"[LOAD] Keras3 OK: {os.path.basename(path)}")
        return m
    except Exception as _e1:
        e1 = _e1
        print(f"[WARN] Keras3 load falló para {path}: {e1}")
    # 2) tf.keras
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"[LOAD] tf.keras OK: {os.path.basename(path)}")
        return m
    except Exception as _e2:
        e2 = _e2
        print(f"[WARN] tf.keras load falló para {path}: {e2}")
    # 3) compat v1
    try:
        m = tf.compat.v1.keras.models.load_model(path, compile=False)
        print(f"[LOAD] tf.compat.v1 OK: {os.path.basename(path)}")
        return m
    except Exception as _e3:
        e3 = _e3
        raise RuntimeError(
            "No pude cargar el modelo '{}'.\n"
            "- Keras3: {}\n- tf.keras: {}\n- compat.v1: {}\n\n"
            "Solución sin re-entrenar: convierte el .h5 a .keras y súbelo al release:\n"
            "  import tensorflow as tf\n"
            "  m=tf.keras.models.load_model('modelo_llamadas_tf.h5', compile=False); m.save('modelo_llamadas.keras')\n"
            "  m=tf.keras.models.load_model('modelo_tmo_tf.h5', compile=False);        m.save('modelo_tmo.keras')\n"
            .format(path, e1, e2, e3)
        )

# ---------- Descarga tolerante a nombres ----------
def fetch_first_asset(candidates, dest_dir):
    last_err = None
    for name in candidates:
        try:
            download_asset_from_latest(OWNER, REPO, name, dest_dir)
            local_path = os.path.join(dest_dir, name)
            if os.path.exists(local_path):
                print(f"[OK] Asset encontrado: {name}")
                return local_path
        except FileNotFoundError as e:
            last_err = e
    raise FileNotFoundError(f"No se encontró ninguno de estos assets en el último release: {candidates}. "
                            f"Último error: {last_err}")

# ---------- Carga de historia (90 días) ----------
def load_recent_history():
    if not os.path.exists(HOSTING_FILE):
        raise FileNotFoundError(f"No se encontró {HOSTING_FILE}.")
    df = pd.read_excel(HOSTING_FILE)
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "hora" in df.columns:
        df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str)+" "+df["hora"].astype(str).str[:5], errors="coerce")
    else:
        df["ts"] = df["fecha_dt"]
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["tmo (segundos)"] = df["tmo (segundos)"].apply(parse_tmo_to_seconds)
    end_ts = df["ts"].max()
    start_ts = end_ts - pd.Timedelta(days=90)
    recent = df[df["ts"] > start_ts].copy()
    full_rng = pd.date_range(recent["ts"].min().floor("H"), end_ts.ceil("H"), freq="H")
    recent = recent.set_index("ts").reindex(full_rng).rename_axis("ts").reset_index()
    for c in [TARGET_LLAMADAS, TARGET_TMO]:
        if c in recent.columns:
            recent[c] = recent[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
    return recent

# ---------- Main ----------
def main():
    os.makedirs("public", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Intentar múltiples nombres por compatibilidad (prioriza .keras, luego .h5)
    path_ll = fetch_first_asset(
        ["modelo_llamadas.keras", "modelo_llamadas_tf.h5", "modelo_llamadas_nn.h5", "modelo_llamadas.h5"], MODELS_DIR
    )
    path_tmo = fetch_first_asset(
        ["modelo_tmo.keras", "modelo_tmo_tf.h5", "modelo_tmo_nn.h5", "modelo_tmo.h5"], MODELS_DIR
    )

    # scalers: aceptar también 'scaler.pkl' único
    candidates_scaler_ll = ["scaler_llamadas.pkl", "scaler_llamadas_tf.pkl", "scaler.pkl"]
    candidates_scaler_tmo = ["scaler_tmo.pkl", "scaler_tmo_tf.pkl", "scaler.pkl"]

    try:
        path_sc_ll = fetch_first_asset(candidates_scaler_ll, MODELS_DIR)
    except FileNotFoundError:
        path_sc_ll = None
    try:
        path_sc_tmo = fetch_first_asset(candidates_scaler_tmo, MODELS_DIR)
    except FileNotFoundError:
        path_sc_tmo = None

    if path_sc_ll is None and path_sc_tmo is None:
        raise FileNotFoundError("No se encontró ningún scaler (busqué scaler_llamadas*, scaler_tmo*, scaler.pkl).")
    if path_sc_ll is None and path_sc_tmo is not None:
        print("[WARN] No hay scaler de llamadas; usaré el mismo scaler para TMO y llamadas.")
        path_sc_ll = path_sc_tmo
    if path_sc_tmo is None and path_sc_ll is not None:
        print("[WARN] No hay scaler de TMO; usaré el mismo scaler que el de llamadas.")
        path_sc_tmo = path_sc_ll

    # Carga modelos con fallback robusto
    model_ll  = load_any_keras_model(path_ll)
    model_tmo = load_any_keras_model(path_tmo)
    scaler_ll  = joblib.load(path_sc_ll)
    scaler_tmo = joblib.load(path_sc_tmo)

    hist = load_recent_history()
    last_ts = hist["ts"].max()

    series_ll  = hist[TARGET_LLAMADAS].astype(float).values.reshape(-1,1)
    series_tmo = hist[TARGET_TMO].astype(float).values.reshape(-1,1)

    seq_ll  = scaler_ll.transform(series_ll).flatten()
    seq_tmo = scaler_tmo.transform(series_tmo).flatten()

    if len(seq_ll) < SEQ_LEN or len(seq_tmo) < SEQ_LEN:
        raise ValueError("Historia insuficiente: se requieren 90 días completos de datos.")

    x_ll  = seq_ll[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    x_tmo = seq_tmo[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)

    yhat_ll_s  = model_ll.predict(x_ll,  verbose=0)[0]
    yhat_tmo_s = model_tmo.predict(x_tmo, verbose=0)[0]

    yhat_ll  = scaler_ll.inverse_transform(yhat_ll_s.reshape(-1,1)).flatten()
    yhat_tmo = scaler_tmo.inverse_transform(yhat_tmo_s.reshape(-1,1)).flatten()

    future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=HORIZON, freq="H", tz=None)
    out = pd.DataFrame({
        "ts": future_index.tz_localize(None).astype(str),
        "pred_llamadas": np.maximum(0, np.round(yhat_ll,0)).astype(int),
        "pred_tmo_seg": np.maximum(0, np.round(yhat_tmo,0)).astype(int),
    })

    # Exports
    out.to_csv(OUT_CSV_DATAOUT, index=False, encoding="utf-8")
    with open(OUT_JSON_PUBLIC, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    tmp = out.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"])
    daily = (tmp.assign(date=tmp["ts"].dt.date).groupby("date", as_index=False)["pred_llamadas"].sum()
             .rename(columns={"pred_llamadas":"total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False, encoding="utf-8")

    # Erlang
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
            "agents_scheduled": agentes_agendados
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    with open(STAMP_JSON, "w") as f:
        json.dump({"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}, f)

    print("✔ Proceso de inferencia completado (salidas en /public).")

if __name__ == "__main__":
    main()

