# =======================================================================
# forecast3m.py  (carga robusta de historical_data.csv con ;) 
# DESCARGA DE RELEASE, PREDICCIÓN ITERATIVA (120 DÍAS),
# RECALIBRACIÓN ESTACIONAL (DOW-HOUR), AJUSTE POR FERIADOS,
# SUAVIZADO ROBUSTO BASADO EN HISTÓRICO y DIMENSIONAMIENTO ERLANG-C
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
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"   # CSV con columna 'Fecha' (DD-MM-YYYY)

OUT_CSV_DATAOUT = "public/predicciones.csv"
OUT_JSON_PUBLIC = "public/predicciones.json"
OUT_CSV_DAILY = "public/llamadas_por_dia.csv"
STAMP_JSON = "public/last_update.json"
OUT_JSON_ERLANG = "public/erlang_forecast.json"

ASSET_LLAMADAS = "modelo_llamadas_nn.h5"
ASSET_SCALER_LLAMADAS = "scaler_llamadas.pkl"
ASSET_TMO = "modelo_tmo_nn.h5"
ASSET_SCALER_TMO = "scaler_tmo.pkl"

TIMEZONE = "America/Santiago"
FREQ = "H"
TARGET_LLAMADAS = "recibidos"
TARGET_TMO = "tmo_seg"

# Horizonte de predicción (DÍAS)
HORIZON_DAYS = 120  # <---- 120 días

# Suavizado (menos agresivo)
MAD_K = 5.0            # K base (lunes-viernes)
MAD_K_WEEKEND = 6.5    # K fin de semana

# -------------------- Parámetros Erlang / Dimensionamiento -------------
SLA_TARGET        = 0.90   # nivel de servicio objetivo
ASA_TARGET_S      = 22     # ASA objetivo (segundos)
INTERVAL_S        = 3600   # duración del intervalo (1 hora)
MAX_OCC           = 0.85   # ocupación máxima deseada
SHRINKAGE         = 0.30   # shrinkage (pausas, capacitaciones, etc.)
ABSENTEEISM_RATE  = 0.23   # ausentismo esperado

# =======================================================================
# Utilidades robustas para leer historical_data.csv con ';' y headers raros
# =======================================================================
def read_csv_semicolon_smart(path):
    """Lee CSV con separador ';' probando encodings y 'despega' encabezados colapsados."""
    encodings = ("utf-8", "latin1", "cp1252")
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            # Si vino todo en una sola columna con ';' en el nombre, rearmar encabezados
            if df.shape[1] == 1 and ";" in df.columns[0]:
                header = df.columns[0].split(";")
                tmp = df.iloc[:, 0].astype(str).str.split(";", expand=True)
                tmp.columns = [h.strip() for h in header]
                df = tmp
            # strip de columnas
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return pd.read_csv(path, sep=";")  # último intento

def normalize_hist_columns_and_index(df):
    """
    Normaliza nombres mínimos requeridos y construye índice horario tz-aware.
    Reglas:
      - Necesita 'fecha' y 'hora' o una columna datetime (datetime/datatime/ts/timestamp).
      - Asegura columna 'recibidos'.
      - Si existe 'tmo (segundos)' la convierte a numérico y crea 'tmo_seg'.
    Devuelve: df con índice tz-aware en America/Santiago y columnas limpias.
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    low = {c.lower(): c for c in d.columns}

    # Mapear 'recibidos' si vino con otro alias
    if "recibidos" not in low:
        for alt in ("llamadas", "total_llamadas", "calls"):
            if alt in low:
                d = d.rename(columns={low[alt]: "recibidos"})
                break
    else:
        d = d.rename(columns={low["recibidos"]: "recibidos"})

    # Construir ts
    fecha_col = low.get("fecha")
    hora_col  = low.get("hora")
    if fecha_col and hora_col:
        fecha_dt = pd.to_datetime(d[fecha_col], errors="coerce", dayfirst=True)
        hora_str = d[hora_col].astype(str).str.slice(0, 5)
        ts = pd.to_datetime(fecha_dt.dt.date.astype(str) + " " + hora_str, errors="coerce")
    else:
        # fallback: datetime ya armado
        dt_col = None
        for cand in ("datetime","datatime","ts","timestamp"):
            if cand in low:
                dt_col = low[cand]; break
        if dt_col is None:
            raise ValueError("historical_data.csv requiere 'fecha'+'hora' o una columna datetime ('datetime'/'datatime'/'ts').")
        ts = pd.to_datetime(d[dt_col], errors="coerce")

    d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    if d.index.tz is None:
        d.index = d.index.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        d = d[~d.index.isna()]

    # Validar recibidos
    if "recibidos" not in d.columns:
        raise ValueError(f"No se encontró 'recibidos'. Columnas: {list(d.columns)}")
    d["recibidos"] = pd.to_numeric(d["recibidos"], errors="coerce").fillna(0)

    # TMO a segundos si existe
    tmo_col = low.get("tmo (segundos)")
    if tmo_col and "tmo_seg" not in d.columns:
        d["tmo_seg"] = d[tmo_col].apply(parse_tmo_to_seconds)

    return d

# =======================================================================
# Utilidades de fecha/tiempo
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
    df_copy[f"{target_col}_ma24"]   = df_copy[target_col].shift(1).rolling(24, min_periods=1).mean()
    df_copy[f"{target_col}_ma168"]  = df_copy[target_col].shift(1).rolling(24*7, min_periods=1).mean()
    return df_copy

# =======================================================================
# Features para inferencia NN
# =======================================================================
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

# =======================================================================
# Recalibración estacional (dow-hour)
# =======================================================================
def compute_seasonal_weights(df_hist, col, weeks=8, clip_min=0.75, clip_max=1.30):
    """
    w(dow,h) = mediana_hist(dow,h) / mediana_hist_por_hora(h)
    Usa por defecto las últimas 'weeks' semanas (si hay suficientes datos).
    """
    d = df_hist.copy()
    if len(d) == 0:
        return { (dow,h): 1.0 for dow in range(7) for h in range(24) }

    end = d.index.max()
    start = end - pd.Timedelta(weeks=weeks)
    d = d.loc[d.index >= start]

    d2 = add_time_features(d[[col]])
    med_dow_hour = d2.groupby(["dow","hour"])[col].median()
    med_hour     = d2.groupby("hour")[col].median()

    weights = {}
    for dow in range(7):
        for h in range(24):
            num = med_dow_hour.get((dow,h), np.nan)
            den = med_hour.get(h, np.nan)
            w = 1.0
            if not np.isnan(num) and not np.isnan(den) and den != 0:
                w = float(num / den)
            weights[(dow,h)] = float(np.clip(w, clip_min, clip_max))
    return weights

def apply_seasonal_weights(df_future, weights):
    df = add_time_features(df_future.copy())
    idx = list(zip(df["dow"].values, df["hour"].values))
    w = np.array([weights.get(key, 1.0) for key in idx], dtype=float)
    df["pred_llamadas"] = (df["pred_llamadas"].astype(float) * w).round().astype(int)
    return df.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

# =======================================================================
# Suavizado robusto basado en histórico
# =======================================================================
def baseline_from_history(df_hist, col):
    d = add_time_features(df_hist[[col]].copy())
    g = d.groupby(["dow", "hour"])[col]
    base = g.median().rename("med").to_frame()
    mad = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    q95 = g.quantile(0.95).rename("q95")
    base = base.join([mad, q95])
    if base["mad"].isna().all():
        base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    base["q95"] = base["q95"].fillna(base["med"])
    return base

def apply_peak_smoothing_history(df_future, col, base, k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND):
    df = add_time_features(df_future.copy())
    keys = list(zip(df["dow"].values, df["hour"].values))
    b = base.reindex(keys)
    b = b.fillna(base.median(numeric_only=True))
    K = np.where(df["dow"].isin([5, 6]), k_weekend, k_weekday).astype(float)  # 5=sáb,6=dom
    upper_cap = b["med"].values + K * b["mad"].values
    mask = (df[col].astype(float).values > upper_cap) & (df[col].astype(float).values > b["q95"].values)
    df.loc[mask, col] = upper_cap[mask]
    return df.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

# =======================================================================
# Feriados: lectura y ajuste
# =======================================================================
def load_holidays(csv_path, tz=TIMEZONE):
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
    tz = getattr(index, "tz", None)
    idx_dates = index.tz_convert(TIMEZONE).date if tz is not None else index.date
    return pd.Series([d in holidays_set for d in idx_dates], index=index, dtype=bool, name="is_holiday")

def _safe_ratio(num, den, fallback=1.0):
    num = float(num) if num is not None and not np.isnan(num) else np.nan
    den = float(den) if num is not None and den is not None and not np.isnan(den) and den != 0 else np.nan
    if np.isnan(num) or np.isnan(den) or den == 0:
        return fallback
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

    factors_calls_by_hour = {h: _safe_ratio(med_hol_calls.get(h, np.nan), med_nor_calls.get(h, np.nan), fallback=global_calls_factor) for h in range(24)}
    factors_tmo_by_hour   = {h: _safe_ratio(med_hol_tmo.get(h, np.nan),   med_nor_tmo.get(h, np.nan),   fallback=global_tmo_factor)   for h in range(24)}

    factors_calls_by_hour = {h: float(np.clip(v, 0.10, 1.20)) for h, v in factors_calls_by_hour.items()}
    factors_tmo_by_hour   = {h: float(np.clip(v, 0.70, 1.50)) for h, v in factors_tmo_by_hour.items()}
    return factors_calls_by_hour, factors_tmo_by_hour, global_calls_factor, global_tmo_factor

def apply_holiday_adjustment(df_future, holidays_set, factors_calls_by_hour, factors_tmo_by_hour):
    df = add_time_features(df_future.copy())
    is_hol = mark_holidays_index(df.index, holidays_set).values
    hours = df["hour"].values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in hours])
    tmo_f  = np.array([factors_tmo_by_hour.get(int(h), 1.0) for h in hours])

    df.loc[is_hol, "pred_llamadas"] = (df.loc[is_hol, "pred_llamadas"].astype(float) * call_f[is_hol]).round().astype(int)
    df.loc[is_hol, "pred_tmo_seg"]  = (df.loc[is_hol, "pred_tmo_seg"].astype(float)  * tmo_f[is_hol]).round().astype(int)
    return df[["pred_llamadas", "pred_tmo_seg"]]

# =======================================================================
# Erlang C helpers
# =======================================================================
def erlang_c_prob_wait(agents, load_erlangs):
    if agents <= 0:
        return 1.0
    if load_erlangs <= 0:
        return 0.0
    rho = load_erlangs / agents
    if rho >= 1.0:
        return 1.0
    # Suma estable por recurrencia
    summation = 0.0
    term = 1.0
    for n in range(0, agents):
        if n > 0:
            term *= load_erlangs / n
        summation += term
    pn = term * (load_erlangs / agents) / (1 - rho)
    p_wait = pn / (summation + pn)
    return float(np.clip(p_wait, 0.0, 1.0))

def required_agents(arrivals, aht_s, asa_target_s=ASA_TARGET_S, sla_target=SLA_TARGET, interval_s=INTERVAL_S, max_occ=MAX_OCC):
    """ Devuelve (agents_productivos, carga_erlangs) cumpliendo SLA/ASA con cap de ocupación. """
    aht = max(float(aht_s), 1.0)
    lam = max(float(arrivals), 0.0)
    load = lam * aht / interval_s  # Erlangs
    base = max(int(np.ceil(load / max_occ)), 1)  # mínimo por ocupación
    agents = base
    for _ in range(2000):
        p_wait = erlang_c_prob_wait(agents, load)
        # ASA (Erlang C)
        asa = (p_wait * aht) / max(agents - load, 1e-9)
        # SLA aproximado para ASA target
        sla = 1.0 - p_wait * np.exp(-(agents - load) * (asa_target_s / aht))
        if sla >= sla_target and asa <= asa_target_s:
            break
        agents += 1
    return agents, load

# =======================================================================
# Predicción iterativa
# =======================================================================
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

# =======================================================================
# Main
# =======================================================================
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # Descargar modelos
    print("Descargando modelos desde el release de GitHub...")
    for asset in [ASSET_LLAMADAS, ASSET_SCALER_LLAMADAS, ASSET_TMO, ASSET_SCALER_TMO]:
        download_asset_from_latest(OWNER, REPO, asset, MODELS_DIR)

    # Cargar modelos y scalers
    print("Cargando modelos y scalers...")
    model_ll = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_LLAMADAS}")
    scaler_ll = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_LLAMADAS}")
    model_tmo = tf.keras.models.load_model(f"{MODELS_DIR}/{ASSET_TMO}")
    scaler_tmo = joblib.load(f"{MODELS_DIR}/{ASSET_SCALER_TMO}")

    # Cargar y procesar histórico (ROBUSTO con ;)
    print(f"Cargando datos históricos desde {DATA_FILE} ...")
    df_hist_raw = read_csv_semicolon_smart(DATA_FILE)
    print("Columnas leídas:", list(df_hist_raw.columns))
    df_hist = normalize_hist_columns_and_index(df_hist_raw)

    # Si existe tmo_seg, mantenlo; si no, intenta derivarlo desde 'tmo (segundos)' (ya lo hicimos arriba)
    cols_keep = [c for c in [TARGET_LLAMADAS, TARGET_TMO] if c in df_hist.columns]
    df_hist = df_hist[cols_keep].dropna(subset=[TARGET_LLAMADAS])

    # Cargar feriados
    print(f"Cargando feriados desde {HOLIDAYS_FILE}...")
    holidays_set = load_holidays(HOLIDAYS_FILE)

    # Horizonte de predicción (120 días exactos desde la última hora conocida)
    last_known_date = df_hist.index.max()
    start_pred = last_known_date + pd.Timedelta(hours=1)
    end_pred = (start_pred + pd.Timedelta(days=HORIZON_DAYS)) - pd.Timedelta(hours=1)
    future_ts = pd.date_range(start=start_pred, end=end_pred, freq=FREQ, tz=TIMEZONE)
    print(f"Se predecirán {len(future_ts)} horas desde {start_pred} hasta {end_pred} (≈ {HORIZON_DAYS} días).")

    # Predicción iterativa
    print("Realizando predicción iterativa de llamadas...")
    pred_ll = predecir_futuro_iterativo(df_hist, model_ll, scaler_ll, TARGET_LLAMADAS, future_ts)
    df_final = pd.DataFrame(index=future_ts)
    df_final["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    # Predicción iterativa de TMO (opcional si está en histórico y existen artefactos)
    if TARGET_TMO in df_hist.columns:
        print("Realizando predicción iterativa de TMO...")
        pred_tmo = predecir_futuro_iterativo(df_hist, model_tmo, scaler_tmo, TARGET_TMO, future_ts)
        df_final["pred_tmo_seg"] = np.maximum(0, np.round(pred_tmo)).astype(int)
    else:
        df_final["pred_tmo_seg"] = 0  # fallback inocuo para que el pipeline siga

    # Recalibración estacional (dow-hour)
    print("Aplicando recalibración estacional (dow-hour) basada en histórico reciente...")
    seasonal_w = compute_seasonal_weights(df_hist, TARGET_LLAMADAS, weeks=8, clip_min=0.75, clip_max=1.30)
    df_final = apply_seasonal_weights(df_final, seasonal_w)

    # Suavizado robusto (sólo recorta outliers)
    print("Aplicando suavizado robusto (baseline histórico)...")
    base_hist = baseline_from_history(df_hist, TARGET_LLAMADAS)
    df_tmp = df_final.copy()
    df_tmp["pred_llamadas"] = df_tmp["pred_llamadas"].astype(float)
    df_final_smoothed = apply_peak_smoothing_history(
        df_tmp, "pred_llamadas", base_hist,
        k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND
    )
    df_final_smoothed["pred_llamadas"] = df_final_smoothed["pred_llamadas"].round().astype(int)

    # Ajuste por feriados
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

    # Guardar predicciones
    print("Guardando archivos de salida (predicciones)...")
    out = df_final_adj.reset_index().rename(columns={"index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_CSV_DATAOUT, index=False)
    out.to_json(OUT_JSON_PUBLIC, orient="records", indent=2)

    # Agregado diario
    daily = (out.assign(date=pd.to_datetime(out["ts"]).dt.date)
               .groupby("date", as_index=False)["pred_llamadas"]
               .sum()
               .rename(columns={"pred_llamadas": "total_llamadas"}))
    daily.to_csv(OUT_CSV_DAILY, index=False)

    # =================== ERLANG: dimensionamiento por hora ===================
    print("Generando erlang_forecast.json...")
    df_er = df_final_adj.copy()
    df_er = df_er.rename(columns={"pred_llamadas": "calls", "pred_tmo_seg": "aht_s"})
    df_er["agents_prod"] = 0
    df_er["erlangs"] = 0.0

    for ts in df_er.index:
        agents, load = required_agents(
            arrivals=float(df_er.at[ts, "calls"]),
            aht_s=float(df_er.at[ts, "aht_s"]),
            asa_target_s=ASA_TARGET_S,
            sla_target=SLA_TARGET,
            interval_s=INTERVAL_S,
            max_occ=MAX_OCC,
        )
        df_er.at[ts, "agents_prod"] = int(agents)
        df_er.at[ts, "erlangs"] = float(load)

    # Programación de agentes (shrinkage + ausentismo)
    df_er["agents_sched"] = np.ceil(
        df_er["agents_prod"] / max(1e-9, (1 - SHRINKAGE)) / max(1e-9, (1 - ABSENTEEISM_RATE))
    ).astype(int)

    erjson = (df_er.reset_index()
                    .rename(columns={"index": "ts"})
                    .assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
                    .to_dict(orient="records"))
    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erjson, f, ensure_ascii=False, indent=2)

    # Timestamp
    json.dump(
        {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        open(STAMP_JSON, "w")
    )
    print("✔ Inferencia + Erlang completadas con éxito.")

if __name__ == "__main__":
    main()

