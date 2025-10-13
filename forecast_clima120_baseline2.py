# =======================================================================
# forecast_clima120_baseline2.py  (con feriados)
#
# Salidas:
#   • public/predicciones2.json        (por hora, SIN clima)
#   • public/llamadas_por_dia2.json    (por día, SIN clima, fecha YYYY-MM-DD)
#   • public/alertas_clima2.json       (alertas por comuna, CON clima)
#
# Requisitos:
#   - data/historical_data.csv         (separador ';')
#   - data/Comunas_Cordenadas.csv      (separador ';')
#   - data/Feriados_Chilev2.csv        (columna 'Fecha' DD-MM-YYYY)
#   - Release con artefactos:
#       > Llamadas:  modelo_llamadas_nn.h5, scaler_llamadas.pkl, training_columns_llamadas.json
#       > Alertas :  modelo_alertas_clima.h5, scaler_alertas_clima.pkl, training_columns_alertas_clima.json
# =======================================================================

import os, json, time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import requests_cache
from retry_requests import retry

# ------------------ Repositorio / assets ------------------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("public", exist_ok=True)

# --- Artefactos de LLAMADAS (idénticos al forecast original) ---
ASSET_LLAMADAS_MODEL = "modelo_llamadas_nn.h5"
ASSET_LLAMADAS_SCAL  = "scaler_llamadas.pkl"
ASSET_LLAMADAS_COLS  = "training_columns_llamadas.json"

# --- Artefactos de ALERTAS CLIMA ---
ASSET_ALERTAS_MODEL = "modelo_alertas_clima.h5"
ASSET_ALERTAS_SCAL  = "scaler_alertas_clima.pkl"
ASSET_ALERTAS_COLS  = "training_columns_alertas_clima.json"

# ------------------ Entradas y salidas ---------------------
HIST_CALLS   = "data/historical_data.csv"      # ';'
LOC_CSV      = "data/Comunas_Cordenadas.csv"   # ';'
HOLIDAYS_CSV = "data/Feriados_Chilev2.csv"     # columna 'Fecha' DD-MM-YYYY

OUT_HOURLY_JSON = "public/predicciones2.json"
OUT_DAILY_JSON  = "public/llamadas_por_dia2.json"
OUT_ALERTAS_JSON= "public/alertas_clima2.json"

# ------------------ Configuración general ------------------
TIMEZONE     = "America/Santiago"
FREQ         = "h"        # (usar 'h' evita FutureWarning)
HORIZON_DAYS = 120

# Ajustes estacionales y suavizado (idénticos al original)
MAD_K_WEEKDAY = 5.0
MAD_K_WEEKEND = 6.5

# ------------------ API clima (open-meteo) -----------------
CLIMA_API_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 14
HOURLY_VARS   = ["temperature_2m","precipitation","rain","wind_speed_10m","wind_gusts_10m"]
UNITS         = {"temperature_unit":"celsius","wind_speed_unit":"kmh","precipitation_unit":"mm"}
SLEEP_S       = 1.0
MAX_RETRIES   = 3
BACKOFF       = 1.8
COOL_429      = 60
ALERTA_THRESHOLD = 0.40  # >40% sobre baseline local

# =======================================================================
# Utilidades comunes
# =======================================================================

def download_asset_from_latest(owner, repo, asset_name, target_dir):
    """Descarga un asset del release más reciente si no existe localmente."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, asset_name)
    if os.path.exists(target_path):
        return target_path

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    if not token:
        print("ADVERTENCIA: GITHUB_TOKEN no encontrado; intentando descarga pública.")

    latest_release_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    r = requests.get(latest_release_url, headers=headers)
    r.raise_for_status()
    assets = r.json().get("assets", [])
    asset_url = None
    for a in assets:
        if a.get("name") == asset_name:
            asset_url = a.get("url"); break
    if not asset_url:
        raise FileNotFoundError(f"Asset '{asset_name}' no está en el último release de {owner}/{repo}.")

    headers["Accept"] = "application/octet-stream"
    with requests.get(asset_url, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"⬇️  Descargado: {target_path}")
    return target_path

def read_csv_semicolon(path):
    """Lee CSV con sep=';' probando encodings comunes; repara headers colapsados."""
    encodings = ("utf-8","utf-8-sig","latin1","cp1252")
    last = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            if df.shape[1] == 1 and ";" in df.columns[0]:
                # Header colapsado; intentar split
                colname = df.columns[0]
                tmp = df[colname].astype(str).str.split(";", expand=True)
                if tmp.shape[1] >= 2:
                    df = tmp
                else:
                    raise ValueError("Archivo no tiene el separador ';' correcto.")
            return df
        except Exception as e:
            last = e
            continue
    raise last or ValueError(f"No pude leer {path} con sep=';'")

def normalize_coords(df):
    """Normaliza columnas de coordenadas a (comuna, lat, lon)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in low: return low[c]
        return None
    c  = pick("comuna","municipio","localidad","ciudad","nombre","name")
    la = pick("lat","latitude","latitud","y")
    lo = pick("lon","lng","long","longitude","longitud","x")
    if not c or not la or not lo:
        raise ValueError(f"Encabezados inválidos en coordenadas: {list(df.columns)}")
    df = df.rename(columns={c:"comuna", la:"lat", lo:"lon"})
    for k in ["lat","lon"]:
        if df[k].dtype==object:
            df[k] = df[k].astype(str).str.replace(",",".",regex=False).str.strip()
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    df = df[(df.lat.between(-90,90)) & (df.lon.between(-180,180))]
    return df.drop_duplicates(subset=["comuna"]).reset_index(drop=True)

def ensure_datetime_calls(df):
    """Construye índice datetime tz-aware a partir de (datetime) o (fecha+hora)."""
    original_cols = df.columns.tolist()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    # 1) datetime directo
    for cand in ["datetime","datatime","ts","timestamp","fecha_hora","fechahora","fechayhora","fecha y hora"]:
        if cand in low:
            ts = pd.to_datetime(df[low[cand]], errors="coerce", dayfirst=True)
            if ts.isna().mean() > 0.5:
                ts = pd.to_datetime(df[low[cand]], errors="coerce", dayfirst=False)
            ts = ts.dropna()
            if ts.empty:
                break
            out = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
            out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
            return out.dropna(subset=["ts"]).set_index("ts")

    # 2) fecha + hora
    fecha_cols = [c for c in df.columns if "fecha" in c.lower() or c.lower()=="date"]
    hora_cols  = [c for c in df.columns if c.lower() in ("hora","hour","time","hr") or c.lower().startswith("hora_")]
    if not fecha_cols or not hora_cols:
        raise ValueError(f"historical_data.csv debe tener 'datetime' o 'fecha'+'hora'. Columnas: {original_cols}")

    fcol, hcol = fecha_cols[0], hora_cols[0]
    fecha_dt = pd.to_datetime(df[fcol], errors="coerce", dayfirst=True)
    if fecha_dt.isna().mean()>0.5:
        fecha_dt = pd.to_datetime(df[fcol], errors="coerce", dayfirst=False)
    hora_str  = df[hcol].astype(str).str.extract(r'(\d{1,2}:\d{1,2}(?::\d{1,2})?)', expand=False).fillna("00:00").str.slice(0,5)
    ts = pd.to_datetime(fecha_dt.dt.date.astype(str) + " " + hora_str, errors="coerce")
    out = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
    out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    out = out.dropna(subset=["ts"]).set_index("ts")
    print(f"✔ detectado en historical_data.csv -> fecha='{fcol}', hora='{hcol}'  (filas válidas: {len(out)})")
    return out

# =======================================================================
# Predicción de LLAMADAS (igual al flujo original)
# =======================================================================

def add_time_features_idx(idx: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=idx)
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["hour"] = d.index.hour
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7)
    return d

def rolling_features_df(df, target_col="recibidos"):
    d = df.copy()
    d[f"{target_col}_lag24"] = d[target_col].shift(24)
    d[f"{target_col}_ma24"]  = d[target_col].shift(1).rolling(24, min_periods=1).mean()
    d[f"{target_col}_ma168"] = d[target_col].shift(1).rolling(24*7, min_periods=1).mean()
    return d

def build_X_from_df(df, training_columns, target_col="recibidos"):
    dummies = pd.get_dummies(df[["dow","month"]], drop_first=False, dtype=int)
    base = ["sin_hour","cos_hour","sin_dow","cos_dow",
            f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"]
    have = [c for c in base if c in df.columns]
    X = pd.concat([df[have], dummies], axis=1)
    # asegurar mismas columnas que en entrenamiento
    for c in training_columns:
        if c not in X.columns:
            X[c] = 0
    return X[training_columns].fillna(0)

def seasonal_weights(df_hist, col="recibidos", weeks=8, clip=(0.75,1.30)):
    d = df_hist.copy()[[col]].dropna()
    if d.empty:
        return {(dow,h):1.0 for dow in range(7) for h in range(24)}
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    end = d.index.max(); start = end - pd.Timedelta(weeks=weeks)
    d = d.loc[d.index>=start]
    med_dow_h = d.groupby(["dow","hour"])[col].median()
    med_h = d.groupby("hour")[col].median()
    w={}
    for dow in range(7):
        for h in range(24):
            num = med_dow_h.get((dow,h), np.nan)
            den = med_h.get(h, np.nan)
            val = 1.0 if (np.isnan(num) or np.isnan(den) or den==0) else float(num/den)
            w[(dow,h)] = float(np.clip(val, clip[0], clip[1]))
    return w

def apply_seasonal_series(series, weights):
    d = pd.DataFrame(index=series.index); d["y"]=series.values
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    idx = list(zip(d["dow"].values, d["hour"].values))
    w = np.array([weights.get(key,1.0) for key in idx], dtype=float)
    return pd.Series(d["y"].values*w, index=d.index, name="y")

def baseline_from_history(df_hist, col="recibidos"):
    d = df_hist.copy()
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    g = d.groupby(["dow","hour"])[col]
    base = g.median().rename("med").to_frame()
    mad  = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    q95  = g.quantile(0.95).rename("q95")
    base = base.join([mad,q95])
    if base["mad"].isna().all(): base["mad"] = 0
    m = base["mad"].median()
    base["mad"] = base["mad"].replace(0, m if not np.isnan(m) else 1.0)
    base["q95"] = base["q95"].fillna(base["med"])
    return base

def smooth_with_mad(series, base, k_weekday=MAD_K_WEEKDAY, k_weekend=MAD_K_WEEKEND):
    d = pd.DataFrame(index=series.index); d["y"]=series.values
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    keys = list(zip(d["dow"].values, d["hour"].values))
    b = base.reindex(keys).fillna(base.median(numeric_only=True))
    K = np.where(d["dow"].isin([5,6]), k_weekend, k_weekday).astype(float)
    upper = b["med"].values + K*b["mad"].values
    y = d["y"].astype(float).values
    mask = (y > upper) & (y > b["q95"].values)
    y[mask] = upper[mask]
    return pd.Series(y, index=series.index, name="y")

def predict_iterative(df_hist, model, scaler, training_columns, target_col="recibidos", future_ts=None):
    d = df_hist[[target_col]].copy()
    for ts in future_ts:
        tmp = pd.concat([d, pd.DataFrame(index=[ts])])
        feats = add_time_features_idx(tmp.index)
        tmp = tmp.join(feats)
        tmp = rolling_features_df(tmp, target_col)
        X = build_X_from_df(tmp.tail(1), training_columns, target_col)
        Xs = scaler.transform(X)
        yhat = float(model.predict(Xs, verbose=0).flatten()[0])
        d.loc[ts, target_col] = yhat
    return d.loc[future_ts, target_col]

# ---------------------- Feriados (idéntico al original) -----------------

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
    if np.isnan(num) or np.isnan(den) or den == 0: return fallback
    return num / den

def compute_holiday_factors(df_hist, holidays_set, col_calls="recibidos"):
    d = df_hist.copy()
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    d["is_holiday"] = mark_holidays_index(d.index, holidays_set).values

    med_hol = d[d["is_holiday"]].groupby("hour")[col_calls].median()
    med_nor = d[~d["is_holiday"]].groupby("hour")[col_calls].median()

    g_hol = d[d["is_holiday"]][col_calls].median()
    g_nor = d[~d["is_holiday"]][col_calls].median()

    global_f = _safe_ratio(g_hol, g_nor, fallback=0.75)
    f_by_hour = {h: _safe_ratio(med_hol.get(h, np.nan), med_nor.get(h, np.nan), fallback=global_f) for h in range(24)}
    f_by_hour = {h: float(np.clip(v, 0.10, 1.20)) for h, v in f_by_hour.items()}
    return f_by_hour, global_f

def apply_holiday_adjustment_hourly(df_future_series, holidays_set, factors_calls_by_hour):
    df = pd.DataFrame({"y": df_future_series.copy()})
    df["hour"] = df.index.hour
    is_hol = mark_holidays_index(df.index, holidays_set).values
    call_f = np.array([factors_calls_by_hour.get(int(h), 1.0) for h in df["hour"].values])
    y = df["y"].astype(float).values
    y[is_hol] = np.round(y[is_hol] * call_f[is_hol]).astype(float)
    return pd.Series(y, index=df.index, name="y")

# =======================================================================
# Alertas climáticas (igual a versión anterior)
# =======================================================================

def http_client():
    cache = requests_cache.CachedSession(".openmeteo_cache_forecast", expire_after=3600)
    return retry(cache, retries=MAX_RETRIES, backoff_factor=BACKOFF)

def fetch_forecast(client, lat, lon):
    p = {"latitude":float(lat),"longitude":float(lon),
         "hourly":",".join(HOURLY_VARS),"forecast_days":int(FORECAST_DAYS),
         "timezone":TIMEZONE, **UNITS}
    r = client.get(CLIMA_API_URL, params=p)
    if r.status_code == 429:
        time.sleep(COOL_429); r = client.get(CLIMA_API_URL, params=p)
    r.raise_for_status()
    return r.json()

def clima_df_from_json(j, comuna):
    if "hourly" not in j or "time" not in j["hourly"]:
        return pd.DataFrame()
    t = pd.to_datetime(j["hourly"]["time"])
    df = pd.DataFrame({"ts": t})
    for v in HOURLY_VARS:
        df[v] = j["hourly"].get(v, np.nan)
    df["comuna"]=comuna
    return df

def add_time_feats_df(df):
    df["dow"]=df["ts"].dt.dayofweek
    df["hour"]=df["ts"].dt.hour
    df["month"]=df["ts"].dt.month
    df["sin_hour"]=np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"]=np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"]=np.sin(2*np.pi*df["dow"]/7)
    df["cos_dow"]=np.cos(2*np.pi*df["dow"]/7)
    return df

def rangos_consecutivos(df_alerts):
    if df_alerts.empty: return []
    d = df_alerts.copy()
    d["fecha"]=d["ts"].dt.date
    d["hora"]=d["ts"].dt.hour
    d = d.sort_values(["fecha","hora"])
    out=[]; curF=None; h0=None; h1=None; vals=[]
    def push():
        nonlocal out, curF, h0, h1, vals
        if curF is None: return
        n = (h1-h0+1)
        out.append({
            "fecha": curF.strftime("%Y-%m-%d"),
            "desde": f"{h0:02d}:00",
            "hasta": f"{h1:02d}:00",
            "n_horas": int(n),
            "incremento_promedio_pct": round(np.mean(vals)*100,1),
            "incremento_max_pct": round(np.max(vals)*100,1)
        })
    for _, r in d.iterrows():
        f = r["fecha"]; h=int(r["hora"]); pct=float(r["porcentaje_incremento"])
        if curF is None:
            curF=f; h0=h; h1=h; vals=[pct]; continue
        if f==curF and h==h1+1:
            h1=h; vals.append(pct)
        else:
            push(); curF=f; h0=h; h1=h; vals=[pct]
    push()
    return out

# =======================================================================
# MAIN
# =======================================================================

def main():
    # ---------- Descargar artefactos ----------
    # Llamadas
    download_asset_from_latest(OWNER, REPO, ASSET_LLAMADAS_MODEL, MODELS_DIR)
    download_asset_from_latest(OWNER, REPO, ASSET_LLAMADAS_SCAL,  MODELS_DIR)
    download_asset_from_latest(OWNER, REPO, ASSET_LLAMADAS_COLS,  MODELS_DIR)
    # Alertas
    download_asset_from_latest(OWNER, REPO, ASSET_ALERTAS_MODEL,  MODELS_DIR)
    download_asset_from_latest(OWNER, REPO, ASSET_ALERTAS_SCAL,   MODELS_DIR)
    download_asset_from_latest(OWNER, REPO, ASSET_ALERTAS_COLS,   MODELS_DIR)

    # ---------- Cargar artefactos ----------
    model_ll  = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_LLAMADAS_MODEL), compile=False)
    scaler_ll = joblib.load(os.path.join(MODELS_DIR, ASSET_LLAMADAS_SCAL))
    with open(os.path.join(MODELS_DIR, ASSET_LLAMADAS_COLS), "r", encoding="utf-8") as f:
        train_cols_ll = json.load(f)

    model_alert  = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_ALERTAS_MODEL), compile=False)
    scaler_alert = joblib.load(os.path.join(MODELS_DIR, ASSET_ALERTAS_SCAL))
    with open(os.path.join(MODELS_DIR, ASSET_ALERTAS_COLS), "r", encoding="utf-8") as f:
        train_cols_alert = json.load(f)

    # ===================== Pronóstico 120d SIN clima =====================
    print("📥 Cargando histórico de llamadas...")
    df_hist = read_csv_semicolon(HIST_CALLS)
    df_hist = df_hist.rename(columns=lambda c: c.strip())
    print("Columnas historical_data.csv:", df_hist.columns.tolist())
    df_hist = ensure_datetime_calls(df_hist)

    # columna 'recibidos'
    reci_col = next((c for c in df_hist.columns if c.lower()=="recibidos"), None)
    if not reci_col:
        raise ValueError("No encuentro columna 'recibidos' en historical_data.csv.")
    df_hist = df_hist.rename(columns={reci_col:"recibidos"})

    # Horizonte exacto
    last_known = df_hist.index.max()
    start = last_known + pd.Timedelta(hours=1)
    end   = start + pd.Timedelta(days=HORIZON_DAYS) - pd.Timedelta(hours=1)
    future_ts = pd.date_range(start=start, end=end, freq=FREQ, tz=TIMEZONE)

    print("🔮 Predicción iterativa con NN (idéntica al forecast original)...")
    y = predict_iterative(df_hist, model_ll, scaler_ll, train_cols_ll, "recibidos", future_ts)

    # Ajustes post-proceso (igual a original)
    print("⚖️  Ajuste estacional + suavizado MAD...")
    w = seasonal_weights(df_hist, col="recibidos", weeks=8, clip=(0.75,1.30))
    y_adj = apply_seasonal_series(y, w)
    base  = baseline_from_history(df_hist, col="recibidos")
    y_final = smooth_with_mad(y_adj, base, k_weekday=MAD_K_WEEKDAY, k_weekend=MAD_K_WEEKEND)

    # === Feriados ===
    print("📅 Cargando feriados y aplicando ajuste...")
    holidays_set = load_holidays(HOLIDAYS_CSV)
    if holidays_set:
        f_by_hour, g_calls = compute_holiday_factors(df_hist, holidays_set, col_calls="recibidos")
        y_final = apply_holiday_adjustment_hourly(y_final, holidays_set, f_by_hour)
    else:
        print("ADVERTENCIA: sin feriados; se omite ajuste.")

    y_final = y_final.round().clip(lower=0).astype(int)

    # Guardar por hora
    out_hourly = (pd.DataFrame({"ts": y_final.index.tz_convert(TIMEZONE), "pred_llamadas": y_final.values})
                    .assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")))
    out_hourly.to_json(OUT_HOURLY_JSON, orient="records", indent=2)

    # Guardar por día (fecha como 'YYYY-MM-DD')
    out_daily = (out_hourly
                 .assign(fecha=pd.to_datetime(out_hourly["ts"]).dt.strftime("%Y-%m-%d"))
                 .groupby("fecha", as_index=False)["pred_llamadas"]
                 .sum()
                 .rename(columns={"pred_llamadas":"total_llamadas"}))
    out_daily.to_json(OUT_DAILY_JSON, orient="records", indent=2)

    # ===================== Alertas CON clima =====================
    print("📍 Cargando coordenadas y consultando clima...")
    df_loc = normalize_coords(read_csv_semicolon(LOC_CSV))
    client = http_client()
    alert_items = []

    for _, row in df_loc.iterrows():
        comuna, lat, lon = row["comuna"], row["lat"], row["lon"]
        try:
            raw = fetch_forecast(client, lat, lon)
            d = clima_df_from_json(raw, comuna)
            if d.empty:
                alert_items.append({"comuna":comuna,"rango_alertas":[],"detalles":[]})
                continue

            d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
            d = d.dropna(subset=["ts"])
            d = add_time_feats_df(d)

            X = d[["temperature_2m","precipitation","rain","wind_speed_10m","wind_gusts_10m",
                   "sin_hour","cos_hour","sin_dow","cos_dow","dow","hour","month"]]
            X = pd.get_dummies(X, columns=["dow","hour","month"], drop_first=False)
            for c in train_cols_alert:
                if c not in X.columns: X[c]=0
            X = X[train_cols_alert].fillna(0)

            y_est = model_alert.predict(scaler_alert.transform(X), verbose=0).flatten()
            baseline = float(np.nanmean(y_est)) if np.isfinite(y_est).any() else 1e-6
            if baseline <= 0: baseline = 1e-6

            d["llamadas_estimadas"]    = y_est
            d["porcentaje_incremento"] = (y_est / baseline) - 1.0
            d["alerta"]                = d["porcentaje_incremento"] > ALERTA_THRESHOLD

            rangos = rangos_consecutivos(d[d["alerta"]==True][["ts","porcentaje_incremento"]])
            detalles = (d.assign(
                            porcentaje_incremento=lambda x:(x["porcentaje_incremento"]*100).round(1),
                            llamadas_estimadas=lambda x:x["llamadas_estimadas"].round(1),
                            alerta=lambda x:x["alerta"].astype(bool),
                            ts=lambda x:x["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        )[["ts","llamadas_estimadas","porcentaje_incremento","alerta"]]
                        .to_dict(orient="records"))
            alert_items.append({"comuna":comuna,"rango_alertas":rangos,"detalles":detalles})
            time.sleep(SLEEP_S)
        except Exception:
            # Si falla la comuna, igual deja entrada vacía (no corta el flujo)
            alert_items.append({"comuna":comuna,"rango_alertas":[],"detalles":[]})
            continue

    # Ordenar: con alertas primero
    alert_items.sort(key=lambda x: (len(x.get("rango_alertas",[]))==0, x["comuna"]))

    with open(OUT_ALERTAS_JSON,"w",encoding="utf-8") as f:
        json.dump(alert_items, f, ensure_ascii=False, indent=2)

    print("✅ Generado:", OUT_HOURLY_JSON)
    print("✅ Generado:", OUT_DAILY_JSON)
    print("✅ Generado:", OUT_ALERTAS_JSON)
    print("✔ Flujo completado.")

if __name__ == "__main__":
    main()

