# =======================================================================
# forecast_unificado120.py
# INFERENCIA 120 días con MODELO UNIFICADO (base + clima)
# - Uplift de clima desactivado (X_weather = 0) para comparar con el modelo original
# - Recalibración estacional (dow-hour)
# - Ajuste por feriados
# - Suavizado robusto (cap por MAD)
# Salidas:
#   public/predicciones2.json  (horario)
#   public/llamadas_por_dia2.json (diario)
# =======================================================================

import os, io, json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ============== Config repo / assets / paths ==============
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"
MODELS_DIR = "models"

ASSET_MODEL = "modelo_unificado.h5"
ASSET_SCALER_CORE = "scaler_core.pkl"
ASSET_SCALER_WEAT = "scaler_weather.pkl"
ASSET_COLS_CORE   = "training_columns_core.json"
ASSET_COLS_WEAT   = "training_columns_weather.json"

DATA_FILE     = "data/historical_data.csv"       # CSV semicolon
HOLIDAYS_FILE = "data/Feriados_Chilev2.csv"      # CSV con columna 'Fecha' (DD-MM-YYYY)

OUT_JSON_HOURLY = "public/predicciones2.json"
OUT_JSON_DAILY  = "public/llamadas_por_dia2.json"

TIMEZONE = "America/Santiago"
FREQ     = "h"       # pandas recomienda 'h' (minúscula)
TARGET   = "recibidos"
HORIZON_DAYS = 120

# Suavizado robusto
MAD_K         = 5.0
MAD_K_WEEKEND = 6.5

# ============== Utils Release ==============
def download_asset_from_latest(owner, repo, asset_name, target_dir):
    import requests
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    latest_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    r = requests.get(latest_url, headers=headers); r.raise_for_status()
    assets = r.json().get("assets", [])
    asset = next((a for a in assets if a["name"] == asset_name), None)
    if not asset:
        raise FileNotFoundError(f"Asset '{asset_name}' no está en el último release de {owner}/{repo}.")
    headers["Accept"] = "application/octet-stream"
    os.makedirs(target_dir, exist_ok=True)
    tgt = os.path.join(target_dir, asset_name)
    with requests.get(asset["url"], headers=headers, stream=True) as resp:
        resp.raise_for_status()
        with open(tgt, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
    print(f"⬇️  Descargado: {tgt}")
    return tgt

# ============== Lectura robusta CSV (;) ==============
def read_csv_semicolon(path):
    # intenta varios encodings; si header viene colapsado "a;b;c", lo reexpande
    encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    last = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
            if df.shape[1] == 1 and ";" in df.columns[0]:
                # reintenta autodetect
                text = open(path, "r", encoding=enc, errors="ignore").read()
                df = pd.read_csv(io.StringIO(text), sep=";", low_memory=False)
            return df
        except Exception as e:
            last = e
    raise last or ValueError(f"No pude leer {path} con sep=';'")

# ============== Timestamps & features ==============
def ensure_datetime_calls(df, col_fecha="fecha", col_hora="hora"):
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    low = {c.lower(): c for c in d.columns}
    if col_fecha not in low or col_hora not in low:
        # intenta deducir
        f = next((c for c in d.columns if "fecha" in c.lower()), None)
        h = next((c for c in d.columns if c.lower()=="hora" or c.lower().startswith("hora")), None)
        if f is None or h is None:
            raise ValueError("historical_data.csv debe tener columnas 'fecha' y 'hora'.")
        col_fecha, col_hora = f, h
    else:
        col_fecha, col_hora = low[col_fecha], low[col_hora]

    fecha_dt = pd.to_datetime(d[col_fecha], errors="coerce", dayfirst=True)
    if fecha_dt.isna().mean() > 0.5:
        fecha_dt = pd.to_datetime(d[col_fecha], errors="coerce", dayfirst=False)

    hora_str = (d[col_hora].astype(str)
                .str.extract(r'(\d{1,2}:\d{1,2}(?::\d{1,2})?)', expand=False)
                .fillna("00:00").str.slice(0,5))
    ts = pd.to_datetime(fecha_dt.dt.date.astype(str) + " " + hora_str, errors="coerce")
    d["ts"] = ts
    d = d.dropna(subset=["ts"]).sort_values("ts")
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    d = d.dropna(subset=["ts"]).set_index("ts")
    return d

def add_time_features(df_idx):
    d = pd.DataFrame(index=df_idx.index)
    d["dow"]   = d.index.dayofweek
    d["month"] = d.index.month
    d["hour"]  = d.index.hour
    d["sin_hour"] = np.sin(2*np.pi*d["hour"]/24)
    d["cos_hour"] = np.cos(2*np.pi*d["hour"]/24)
    d["sin_dow"]  = np.sin(2*np.pi*d["dow"]/7)
    d["cos_dow"]  = np.cos(2*np.pi*d["dow"]/7)
    return d

def rolling_features(df, target_col):
    d = df.copy()
    d[f"{target_col}_lag24"]  = d[target_col].shift(24)
    d[f"{target_col}_ma24"]   = d[target_col].shift(1).rolling(24,     min_periods=1).mean()
    d[f"{target_col}_ma168"]  = d[target_col].shift(1).rolling(24*7,   min_periods=1).mean()
    return d

def build_X_core(df, training_cols_core, target_col):
    df_dum = pd.get_dummies(df[["dow","month"]], drop_first=False, dtype=int)
    base = ["sin_hour","cos_hour","sin_dow","cos_dow",
            f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"]
    exist = [c for c in base if c in df.columns]
    X = pd.concat([df[exist], df_dum], axis=1)
    for c in training_cols_core:
        if c not in X.columns:
            X[c] = 0
    return X[training_cols_core].fillna(0)

# ============== Recalibración & suavizado & feriados ==============
def compute_seasonal_weights(df_hist, col, weeks=8, clip_min=0.75, clip_max=1.30):
    d = df_hist[[col]].copy()
    if len(d) == 0:
        return {(dow,h):1.0 for dow in range(7) for h in range(24)}
    end = d.index.max()
    start = end - pd.Timedelta(weeks=weeks)
    d = d.loc[d.index >= start]
    d2 = add_time_features(d)
    med_dh = d2.groupby(["dow","hour"])[col].median()
    med_h  = d2.groupby("hour")[col].median()
    weights = {}
    for dow in range(7):
        for h in range(24):
            num = med_dh.get((dow,h), np.nan)
            den = med_h.get(h, np.nan)
            w = 1.0
            if not np.isnan(num) and not np.isnan(den) and den != 0:
                w = float(num/den)
            weights[(dow,h)] = float(np.clip(w, clip_min, clip_max))
    return weights

def apply_seasonal_weights(df_future, weights):
    df = add_time_features(df_future.copy())
    keys = list(zip(df["dow"].values, df["hour"].values))
    w = np.array([weights.get(k,1.0) for k in keys], dtype=float)
    df["pred_llamadas"] = (df["pred_llamadas"].astype(float)*w).round().astype(int)
    return df.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

def baseline_from_history(df_hist, col):
    d2 = add_time_features(df_hist[[col]].copy())
    g = d2.groupby(["dow","hour"])[col]
    base = g.median().rename("med").to_frame()
    mad  = g.apply(lambda x: np.median(np.abs(x - np.median(x)))).rename("mad")
    q95  = g.quantile(0.95).rename("q95")
    base = base.join([mad,q95])
    if base["mad"].isna().all(): base["mad"] = 0
    medmad = base["mad"].median() if not np.isnan(base["mad"].median()) else 1.0
    base["mad"] = base["mad"].replace(0, medmad)
    base["q95"] = base["q95"].fillna(base["med"])
    return base

def apply_peak_smoothing_history(df_future, col, base, k_weekday=MAD_K, k_weekend=MAD_K_WEEKEND):
    df = add_time_features(df_future.copy())
    keys = list(zip(df["dow"].values, df["hour"].values))
    b = base.reindex(keys).fillna(base.median(numeric_only=True))
    K = np.where(df["dow"].isin([5,6]), k_weekend, k_weekday).astype(float)
    up = b["med"].values + K*b["mad"].values
    mask = (df[col].astype(float).values > up) & (df[col].astype(float).values > b["q95"].values)
    df.loc[mask, col] = up[mask]
    return df.drop(columns=["dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow"], errors="ignore")

def load_holidays(csv_path):
    if not os.path.exists(csv_path):
        print(f"ADVERTENCIA: No hay feriados ({csv_path}).")
        return set()
    fer = pd.read_csv(csv_path)
    if "Fecha" not in fer.columns:
        print("ADVERTENCIA: CSV feriados sin columna 'Fecha'.")
        return set()
    fechas = pd.to_datetime(fer["Fecha"].astype(str), dayfirst=True, errors="coerce").dropna().dt.date
    return set(fechas)

def mark_holidays_index(index, holidays_set):
    tz = getattr(index, "tz", None)
    dates = index.tz_convert(TIMEZONE).date if tz is not None else index.date
    return pd.Series([d in holidays_set for d in dates], index=index, dtype=bool)

def apply_holiday_adjustment(df_future, holidays_set, col="pred_llamadas"):
    if not holidays_set:
        return df_future[[col]]
    d = add_time_features(df_future.copy())
    is_hol = mark_holidays_index(d.index, holidays_set).values
    # ratio por hora (mediana feriado / mediana normal)
    med_hol = d.loc[is_hol].groupby("hour")[col].median()
    med_nor = d.loc[~is_hol].groupby("hour")[col].median()
    ratios = {h: (float(med_hol.get(h, np.nan))/float(med_nor.get(h, np.nan))
                  if not np.isnan(med_hol.get(h, np.nan)) and not np.isnan(med_nor.get(h, np.nan)) and med_nor.get(h, np.nan)!=0
                  else 1.0)
              for h in range(24)}
    ratios = {h: float(np.clip(v, 0.10, 1.20)) for h,v in ratios.items()}
    hours = d["hour"].values
    f = np.array([ratios.get(int(h),1.0) for h in hours])
    d.loc[is_hol, col] = (d.loc[is_hol, col].astype(float) * f[is_hol]).round().astype(int)
    return d[[col]]

# ============== Predicción iterativa (uplift=0) ==============
def iterative_forecast_unified(df_hist, model, scaler_core, scaler_w, cols_core, cols_wea, target_col, future_ts):
    df_pred = df_hist.copy()
    out_vals = []
    # matriz meteorológica nula -> uplift = 0
    zeros_weather = pd.DataFrame(np.zeros((1, len(cols_wea))), columns=cols_wea)
    for ts in future_ts:
        temp = pd.DataFrame(index=[ts])
        df_tmp = pd.concat([df_pred, temp])
        feats = add_time_features(df_tmp)
        feats = pd.concat([df_tmp[[target_col]], feats], axis=1)
        feats = rolling_features(feats, target_col)
        Xc = build_X_core(feats.tail(1), cols_core, target_col)
        Xc_s = scaler_core.transform(Xc)
        Xw_s = scaler_w.transform(zeros_weather)
        y_hat, _upl = model.predict([Xc_s, Xw_s], verbose=0)
        val = float(y_hat.flatten()[0])
        df_pred.loc[ts, target_col] = val
        out_vals.append(val)
    return pd.Series(out_vals, index=future_ts, name="pred_llamadas")

# =============================== MAIN ===============================
def main():
    os.makedirs("public", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Descargando assets del Release...")
    for asset in [ASSET_MODEL, ASSET_SCALER_CORE, ASSET_SCALER_WEAT, ASSET_COLS_CORE, ASSET_COLS_WEAT]:
        download_asset_from_latest(OWNER, REPO, asset, MODELS_DIR)

    print("Cargando modelo y scalers...")
    model   = tf.keras.models.load_model(os.path.join(MODELS_DIR, ASSET_MODEL))
    sc_core = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_CORE))
    sc_wea  = joblib.load(os.path.join(MODELS_DIR, ASSET_SCALER_WEAT))
    cols_core = json.load(open(os.path.join(MODELS_DIR, ASSET_COLS_CORE), "r", encoding="utf-8"))
    cols_wea  = json.load(open(os.path.join(MODELS_DIR, ASSET_COLS_WEAT),  "r", encoding="utf-8"))

    print(f"Cargando histórico: {DATA_FILE} ...")
    df_hist_raw = read_csv_semicolon(DATA_FILE)
    df_hist_raw.columns = [c.strip() for c in df_hist_raw.columns]
    if TARGET not in [c.lower() for c in df_hist_raw.columns]:
        # buscar 'recibidos' con mayúsculas/espacios raros
        rec_col = next((c for c in df_hist_raw.columns if c.strip().lower()=="recibidos"), None)
        if rec_col is None:
            raise ValueError("No encuentro columna 'recibidos' en historical_data.csv")
        df_hist_raw = df_hist_raw.rename(columns={rec_col: TARGET})

    df_hist = ensure_datetime_calls(df_hist_raw, col_fecha="fecha", col_hora="hora")
    df_hist = df_hist[[TARGET]].astype(float)

    # Horizonte de predicción
    last = df_hist.index.max()
    start = last + pd.Timedelta(hours=1)
    end   = (start + pd.Timedelta(days=HORIZON_DAYS)) - pd.Timedelta(hours=1)
    future_ts = pd.date_range(start=start, end=end, freq=FREQ, tz=TIMEZONE)
    print(f"Horizonte: {len(future_ts)} horas ({HORIZON_DAYS} días) desde {start} hasta {end}")

    # Predicción iterativa (clima desactivado)
    print("Prediciendo (uplift clima = 0)...")
    pred_ll = iterative_forecast_unified(df_hist, model, sc_core, sc_wea, cols_core, cols_wea, TARGET, future_ts)
    df_fut = pd.DataFrame(index=future_ts)
    df_fut["pred_llamadas"] = np.maximum(0, np.round(pred_ll)).astype(int)

    # Recalibración estacional
    print("Aplicando recalibración estacional...")
    w = compute_seasonal_weights(df_hist, TARGET, weeks=8, clip_min=0.75, clip_max=1.30)
    df_fut = apply_seasonal_weights(df_fut, w)

    # Suavizado robusto (cap por MAD)
    print("Aplicando suavizado robusto...")
    base = baseline_from_history(df_hist, TARGET)
    df_tmp = df_fut.copy().astype(float)
    df_smooth = apply_peak_smoothing_history(df_tmp, "pred_llamadas", base, MAD_K, MAD_K_WEEKEND)
    df_smooth["pred_llamadas"] = df_smooth["pred_llamadas"].round().astype(int)

    # Ajuste por feriados
    print("Aplicando ajuste por feriados...")
    holidays = load_holidays(HOLIDAYS_FILE)
    df_adj = apply_holiday_adjustment(df_smooth, holidays, col="pred_llamadas")
    df_adj = df_adj.rename(columns={"pred_llamadas": "llamadas"})

    # Salida por hora
    out_h = (df_adj.rename(columns={"llamadas":"pred_llamadas"})
                    .reset_index()
                    .rename(columns={"index":"ts"}))
    out_h["ts"] = out_h["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_h.to_json(OUT_JSON_HOURLY, orient="records", indent=2, force_ascii=False)

    # Salida por día
    out_d = (df_adj.copy()
                    .assign(fecha=lambda d: d.index.tz_convert(TIMEZONE).date)
                    .groupby("fecha", as_index=False)["llamadas"].sum()
                    .rename(columns={"llamadas":"total_llamadas"}))
    out_d["fecha"] = pd.to_datetime(out_d["fecha"]).dt.strftime("%Y-%m-%d")
    out_d.to_json(OUT_JSON_DAILY, orient="records", indent=2, force_ascii=False)

    print("✔ Listo:")
    print(f" - {OUT_JSON_HOURLY}")
    print(f" - {OUT_JSON_DAILY}")

if __name__ == "__main__":
    main()
