# =======================================================================
# forecast_llamadas120_baseline2.py
# INFERENCIA SIN CLIMA usando modelo_unificado.h5 (2 inputs: core + weather)
# - Descarga assets core + weather desde Release
# - Carga con compile=False (evita error keras.metrics.*)
# - Predicción iterativa 120 días por hora
# - Recalibración estacional (dow-hour) + Suavizado robusto
# - Salidas: public/predicciones2.json y public/llamadas_por_dia2.json
# =======================================================================

import os, json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------- Fallback de descarga si utils_release no exporta la función ----------
try:
    from utils_release import download_asset_from_latest
except Exception:
    import requests
    def download_asset_from_latest(owner, repo, asset_name, target_dir):
        token = os.getenv("GITHUB_TOKEN")
        headers = {"Authorization": f"token {token}"} if token else {}
        if not token:
            print("ADVERTENCIA: GITHUB_TOKEN no encontrado. Las descargas pueden fallar.")
        latest_release_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        r = requests.get(latest_release_url, headers=headers)
        r.raise_for_status()
        assets = r.json().get("assets", [])
        url = None
        for a in assets:
            if a["name"] == asset_name:
                url = a["url"]; break
        if not url:
            raise FileNotFoundError(f"Asset '{asset_name}' no está en el último release de {owner}/{repo}.")
        headers["Accept"] = "application/octet-stream"
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, asset_name)
        with requests.get(url, headers=headers, stream=True) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(8192): f.write(chunk)
        print(f"⬇️  Descargado: {out_path}")
        return out_path

# -------------------- Parámetros generales ----------------------------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"
MODELS_DIR = "models"
DATA_FILE  = "data/historical_data.csv"

TIMEZONE = "America/Santiago"
FREQ = "H"                  # freq horaria
TARGET = "recibidos"
HORIZON_DAYS = 120

# Salidas (sufijo "2")
OUT_JSON_HOURLY = "public/predicciones2.json"
OUT_JSON_DAILY  = "public/llamadas_por_dia2.json"

# Suavizado (picos)
MAD_K         = 5.0
MAD_K_WEEKEND = 6.5

# -------------------- IO helpers ---------------------------
def read_csv_semicolon(path):
    encodings = ["utf-8", "latin1", "cp1252"]
    last = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            if df.shape[1] == 1 and ";" in df.columns[0]:
                parts = df.columns[0].split(";")
                df = pd.read_csv(path, sep=";", encoding=enc, header=None, names=parts)
            return df
        except Exception as e:
            last = e
    raise last or ValueError(f"No pude leer {path} con sep=';'")

def ensure_datetime_calls(df):
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    low = {c.lower(): c for c in d.columns}
    # 1) datetime / datatime directo
    for cand in ("datetime", "datatime"):
        if cand in low:
            ts = pd.to_datetime(d[low[cand]], errors="coerce", dayfirst=True)
            d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
            d["ts"] = d["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
            d = d.dropna(subset=["ts"]).set_index("ts")
            return d
    # 2) fecha + hora
    if "fecha" in low and "hora" in low:
        fecha_dt = pd.to_datetime(d[low["fecha"]], errors="coerce", dayfirst=True)
        hora_str = d[low["hora"]].astype(str).str.slice(0,5)
        ts = pd.to_datetime(fecha_dt.dt.date.astype(str) + " " + hora_str, errors="coerce")
        d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
        d["ts"] = d["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        d = d.dropna(subset=["ts"]).set_index("ts")
        return d
    # 3) fallback: cualquier columna que parezca timestamp
    for col in d.columns:
        try:
            ts = pd.to_datetime(d[col], errors="coerce", dayfirst=True)
            if ts.notna().sum() >= len(d)*0.7:
                d = d.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
                d["ts"] = d["ts"].dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
                d = d.dropna(subset=["ts"]).set_index("ts")
                return d
        except Exception:
            pass
    raise ValueError("historical_data.csv debe tener 'datetime'/'datatime' o 'fecha' y 'hora'.")

# -------------------- Feature engineering -------------------
def add_time_features(df_idxed):
    x = df_idxed.copy()
    x["dow"]   = x.index.dayofweek
    x["month"] = x.index.month
    x["hour"]  = x.index.hour
    x["sin_hour"] = np.sin(2*np.pi*x["hour"]/24)
    x["cos_hour"] = np.cos(2*np.pi*x["hour"]/24)
    x["sin_dow"]  = np.sin(2*np.pi*x["dow"]/7)
    x["cos_dow"]  = np.cos(2*np.pi*x["dow"]/7)
    return x

def rolling_features(df_idxed, col):
    x = df_idxed.copy()
    x[f"{col}_lag24"]  = x[col].shift(24)
    x[f"{col}_ma24"]   = x[col].shift(1).rolling(24, min_periods=1).mean()
    x[f"{col}_ma168"]  = x[col].shift(1).rolling(24*7, min_periods=1).mean()
    return x

def build_feature_matrix_core(df_idxed, training_columns, target_col):
    d = add_time_features(df_idxed)
    d = rolling_features(d, target_col)
    dummies = pd.get_dummies(d[["dow","month"]], drop_first=False, dtype=int)
    base = [
        "sin_hour","cos_hour","sin_dow","cos_dow",
        f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"
    ]
    exist = [c for c in base if c in d.columns]
    X = pd.concat([d[exist], dummies], axis=1)
    for c in set(training_columns) - set(X.columns):
        X[c] = 0
    X = X[training_columns].replace([np.inf,-np.inf], np.nan).fillna(0)
    return X

def build_feature_matrix_weather_zeros(training_columns_weather, index_like):
    # matriz de clima en ceros (sin aportar señal)
    Xw = pd.DataFrame(
        np.zeros((len(index_like), len(training_columns_weather)), dtype=float),
        index=index_like,
        columns=training_columns_weather
    )
    return Xw

# -------------------- Recalibración / Suavizado --------------
def compute_seasonal_weights(df_hist, col, weeks=8, clip_min=0.75, clip_max=1.30):
    if len(df_hist) == 0:
        return {(dow,h):1.0 for dow in range(7) for h in range(24)}
    end = df_hist.index.max()
    start = end - pd.Timedelta(weeks=weeks)
    recent = df_hist.loc[df_hist.index >= start, [col]]
    d2 = add_time_features(recent)
    med_dh = d2.groupby(["dow","hour"])[col].median()
    med_h  = d2.groupby("hour")[col].median()
    w = {}
    for dow in range(7):
        for h in range(24):
            num = med_dh.get((dow,h), np.nan)
            den = med_h.get(h, np.nan)
            val = 1.0
            if not np.isnan(num) and not np.isnan(den) and den != 0:
                val = float(num/den)
            w[(dow,h)] = float(np.clip(val, clip_min, clip_max))
    return w

def apply_seasonal_weights(df_future, weights, col="pred_llamadas"):
    d = add_time_features(df_future.copy())
    idx = list(zip(d["dow"].values, d["hour"].values))
    w = np.array([weights.get(k,1.0) for k in idx], dtype=float)
    d[col] = (d[col].astype(float) * w).round().astype(int)
    return d.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

def baseline_from_history(df_hist, col):
    dh = add_time_features(df_hist[[col]].copy())
    g = dh.groupby(["dow","hour"])[col]
    base = g.median().rename("med").to_frame()
    mad  = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    q95  = g.quantile(0.95).rename("q95")
    base = base.join([mad,q95])
    if base["mad"].isna().all(): base["mad"] = 0
    base["mad"] = base["mad"].replace(0, base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0)
    base["q95"] = base["q95"].fillna(base["med"])
    return base

def apply_peak_smoothing_history(df_future, col, base, k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND):
    d = add_time_features(df_future.copy())
    keys = list(zip(d["dow"].values, d["hour"].values))
    b = base.reindex(keys).fillna(base.median(numeric_only=True))
    K = np.where(d["dow"].isin([5,6]), k_weekend, k_weekday).astype(float)
    upper = b["med"].values + K*b["mad"].values
    mask = (d[col].astype(float).values > upper) & (d[col].astype(float).values > b["q95"].values)
    d.loc[mask, col] = upper[mask]
    return d.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

# -------------------- Predicción iterativa -------------------
def predict_iterative(df_hist_idxed, model, scaler_core, scaler_weather, cols_core, cols_weather, target_col, future_index):
    df_pred = df_hist_idxed.copy()
    for ts in future_index:
        tmp = pd.DataFrame(index=[ts])
        full = pd.concat([df_pred, tmp])
        Xc = build_feature_matrix_core(full, cols_core, target_col).tail(1)
        Xw = build_feature_matrix_weather_zeros(cols_weather, Xc.index)
        Xc_s = scaler_core.transform(Xc)
        Xw_s = scaler_weather.transform(Xw)
        # IMPORTANTE: el modelo espera dict con nombres de input entrenados
        yhat = model.predict({"X_core": Xc_s, "X_weather": Xw_s}, verbose=0).flatten()[0]
        df_pred.loc[ts, target_col] = yhat
    return df_pred.loc[future_index, target_col]

# =============================== MAIN ======================= 
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    print("Descargando assets del Release...")
    ASSET_MODEL   = "modelo_unificado.h5"
    ASSET_SC_CORE = "scaler_core.pkl"
    ASSET_COL_CORE= "training_columns_core.json"
    ASSET_SC_WTHR = "scaler_weather.pkl"
    ASSET_COL_WTHR= "training_columns_weather.json"

    for a in [ASSET_MODEL, ASSET_SC_CORE, ASSET_COL_CORE, ASSET_SC_WTHR, ASSET_COL_WTHR]:
        download_asset_from_latest(OWNER, REPO, a, MODELS_DIR)

    print("Cargando modelo y scalers...")
    model   = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_MODEL), compile=False)
    sc_core = joblib.load(os.path.join(MODELS_DIR, ASSET_SC_CORE))
    sc_wthr = joblib.load(os.path.join(MODELS_DIR, ASSET_SC_WTHR))
    with open(os.path.join(MODELS_DIR, ASSET_COL_CORE), "r", encoding="utf-8") as f:
        cols_core = json.load(f)
    with open(os.path.join(MODELS_DIR, ASSET_COL_WTHR), "r", encoding="utf-8") as f:
        cols_wthr = json.load(f)
    if isinstance(cols_core, dict) and "columns" in cols_core: cols_core = cols_core["columns"]
    if isinstance(cols_wthr, dict) and "columns" in cols_wthr: cols_wthr = cols_wthr["columns"]

    print(f"Cargando histórico: {DATA_FILE}")
    df_raw = read_csv_semicolon(DATA_FILE)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_hist = ensure_datetime_calls(df_raw)

    if TARGET not in df_hist.columns:
        lowmap = {c.lower(): c for c in df_hist.columns}
        if TARGET in lowmap:
            df_hist.rename(columns={lowmap[TARGET]: TARGET}, inplace=True)
        else:
            raise KeyError(f"No encuentro columna '{TARGET}' en {list(df_hist.columns)}")

    df_hist = df_hist[[TARGET]].dropna()
    if len(df_hist) < 24*14:
        print("ADVERTENCIA: histórico muy corto; resultados pueden ser inestables.")

    last = df_hist.index.max()
    start = last + pd.Timedelta(hours=1)
    end   = start + pd.Timedelta(days=HORIZON_DAYS) - pd.Timedelta(hours=1)
    # evitar warning de 'H' deprecado en algunas versiones
    future_ts = pd.date_range(start=start, end=end, freq="h", tz=TIMEZONE)
    print(f"Horizonte: {len(future_ts)} horas ({HORIZON_DAYS} días)")

    print("Prediciendo llamadas (SIN clima: rama clima = 0)...")
    pred = predict_iterative(df_hist, model, sc_core, sc_wthr, cols_core, cols_wthr, TARGET, future_ts)
    df_out = pd.DataFrame(index=future_ts, data={"pred_llamadas": np.maximum(0, np.round(pred)).astype(int)})

    print("Aplicando recalibración estacional y suavizado...")
    w = compute_seasonal_weights(df_hist, TARGET, weeks=8, clip_min=0.75, clip_max=1.30)
    df_out = apply_seasonal_weights(df_out, w, col="pred_llamadas")
    base = baseline_from_history(df_hist, TARGET)
    df_out = apply_peak_smoothing_history(df_out.astype(float), "pred_llamadas", base,
                                          k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND)
    df_out["pred_llamadas"] = df_out["pred_llamadas"].round().astype(int)

    print("Guardando JSON horario y diario...")
    hourly = (df_out.reset_index()
                    .rename(columns={"index":"ts"}))
    hourly["ts"] = hourly["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    hourly.to_json(OUT_JSON_HOURLY, orient="records", indent=2, force_ascii=False)

    daily = (df_out.copy()
                  .assign(date=lambda d: d.index.tz_convert(TIMEZONE).date)
                  .groupby("date", as_index=False)["pred_llamadas"].sum()
                  .rename(columns={"pred_llamadas":"total_llamadas"}))
    daily["fecha"] = daily["date"].astype(str)
    daily = daily[["fecha","total_llamadas"]]
    daily.to_json(OUT_JSON_DAILY, orient="records", indent=2, force_ascii=False)

    print("✔ Listo. Salidas:")
    print(f" - {OUT_JSON_HOURLY}")
    print(f" - {OUT_JSON_DAILY}")

if __name__ == "__main__":
    main()
