# =======================================================================
# forecast_clima120_baseline2.py
# Una sola IA:
#   • Pronóstico nacional 120 días SIN clima (hora y día) -> *2.json
#   • Alertas por comuna CON clima (rangos por horas consecutivas) -> *2.json
#
# Requiere:
#   - data/historical_data.csv   (SEPARADO POR ; -> este script lo fuerza)
#   - data/Comunas_Cordenadas.csv (SEPARADO POR ; -> también lo forzamos)
#   - modelos de alertas en Release:
#       modelo_alertas_clima.h5, scaler_alertas_clima.pkl, training_columns_alertas_clima.json
# =======================================================================

import os, json, time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import requests_cache
from retry_requests import retry

# --------- Repo para descargar modelos si faltan ----------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"

MODELS_DIR = "models"
MODEL_NAME = "modelo_alertas_clima.h5"
SCALER_NAME = "scaler_alertas_clima.pkl"
COLS_NAME = "training_columns_alertas_clima.json"

# --------- Datos de entrada ---------
HIST_CALLS = "data/historical_data.csv"       # ; separador
LOC_CSV    = "data/Comunas_Cordenadas.csv"    # ; separador

# --------- Salidas (terminadas en 2) ---------
OUT_HOURLY_JSON = "public/predicciones2.json"          # por hora
OUT_DAILY_JSON  = "public/llamadas_por_dia2.json"      # por día
OUT_ALERTAS     = "public/alertas_clima2.json"         # alertas por comuna

# --------- Parámetros de horizonte / zona horaria ---------
HORIZON_DAYS = 120
FREQ = "h"  # minúscula para evitar FutureWarning
TIMEZONE = "America/Santiago"

# --------- API clima (para Alertas) ---------
CLIMA_API_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 14
HOURLY_VARS = ["temperature_2m","precipitation","rain","wind_speed_10m","wind_gusts_10m"]
UNITS = {"temperature_unit":"celsius","wind_speed_unit":"kmh","precipitation_unit":"mm"}
SLEEP_S = 1.0
MAX_RETRIES = 3
BACKOFF = 1.8
COOL_429 = 60

# --------- Umbral de alerta ---------
ALERTA_THRESHOLD = 0.40   # >40% sobre baseline del propio horizonte de la comuna

# =======================================================================
# Utilidades
# =======================================================================

def download_asset_from_latest(owner, repo, asset_name, target_dir):
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

def read_csv_historical_semicolon(path):
    """Lee el histórico FORZANDO separador ';' y probando encodings típicos."""
    encodings = ("utf-8","utf-8-sig","latin1","cp1252")
    last = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            if df.shape[1] == 1 and ";" in df.columns[0]:
                raise ValueError("Header sigue colapsado; encoding incorrecto.")
            return df
        except Exception as e:
            last = e
            continue
    raise last or ValueError(f"No pude leer {path} con sep=';'")

def read_csv_coords_semicolon(path):
    """Lee coordenadas FORZANDO ';' y repara si quedó en una sola columna."""
    encodings = ("utf-8","utf-8-sig","latin1","cp1252")
    last = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            if df.shape[1] == 1 and ";" in df.columns[0]:
                colname = df.columns[0]
                tmp = df[colname].astype(str).str.split(";", expand=True)
                if tmp.shape[1] >= 3:
                    def is_num(x):
                        try:
                            float(str(x).replace(",", "."))
                            return True
                        except:
                            return False
                    maybe_header = tmp.iloc[0].tolist()
                    if (not is_num(maybe_header[1])) or (not is_num(maybe_header[2])):
                        tmp.columns = [c.strip().lower() for c in maybe_header[:tmp.shape[1]]]
                        tmp = tmp.iloc[1:].reset_index(drop=True)
                    else:
                        tmp.columns = ["comuna","lat","lon"] + [f"extra_{i}" for i in range(tmp.shape[1]-3)]
                    df = tmp
                else:
                    raise ValueError("El archivo de coordenadas no tiene formato esperable.")
            return df
        except Exception as e:
            last = e
            continue
    raise last or ValueError(f"No pude leer {path} con sep=';'")

def normalize_loc(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    def pick(cols, cands):
        m = {c.lower().strip(): c for c in cols}
        for k in cands:
            if k in m: return m[k]
        return None

    c = pick(df.columns, ["comuna","municipio","localidad","ciudad","name","nombre"])
    la = pick(df.columns, ["lat","latitude","latitud","y"])
    lo = pick(df.columns, ["lon","lng","long","longitude","longitud","x"])
    if not c or not la or not lo:
        raise ValueError(f"Encabezados inválidos en coordenadas: {list(df.columns)}")
    df = df.rename(columns={c:"comuna", la:"lat", lo:"lon"})
    for k in ["lat","lon"]:
        if df[k].dtype==object:
            df[k]=df[k].astype(str).str.replace(",",".",regex=False).str.strip()
        df[k]=pd.to_numeric(df[k],errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    df = df[(df.lat.between(-90,90)) & (df.lon.between(-180,180))]
    return df.drop_duplicates(subset=["comuna"]).reset_index(drop=True)

def ensure_datetime_calls(df):
    original_cols = df.columns.tolist()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lowmap = {c.lower(): c for c in df.columns}
    cols_low = [c.lower() for c in df.columns]

    # 1) datetime directo
    dt_candidates = ["datetime","datatime","ts","timestamp","fecha_hora","fechahora","fechayhora","fecha y hora"]
    dt_col = next((lowmap[c] for c in dt_candidates if c in lowmap), None)
    if dt_col is not None:
        ts = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
        if ts.isna().mean() > 0.5:
            ts = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=False)
        ts = ts.dropna()
        if ts.empty:
            raise ValueError(f"No pude parsear la columna '{dt_col}' como datetime. Columnas originales: {original_cols}")
        out = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts")
        return out.set_index("ts")

    # 2) fecha + hora por separado
    fecha_candidates = [c for c in cols_low if ("fecha" in c) or (c=="date")]
    hora_candidates  = [c for c in cols_low if (c=="hora") or ("hora_" in c) or (c=="hour") or (c=="time") or (c=="hr")]
    if not fecha_candidates or not hora_candidates:
        raise ValueError(
            f"historical_data.csv debe tener 'datetime' o 'fecha'+'hora'. "
            f"Se encontraron columnas: {original_cols}"
        )
    fecha_col = lowmap[fecha_candidates[0]]
    hora_col  = lowmap[hora_candidates[0]]

    fecha_dt = pd.to_datetime(df[fecha_col], errors="coerce", dayfirst=True).dt.date
    if pd.isna(fecha_dt).mean() > 0.5:
        fecha_dt = pd.to_datetime(df[fecha_col], errors="coerce", dayfirst=False).dt.date

    hora_str = df[hora_col].astype(str).str.strip()
    hora_str = hora_str.str.extract(r'(\d{1,2}:\d{1,2}(?::\d{1,2})?)', expand=False)
    hora_str = hora_str.fillna("00:00").str.slice(0,5)

    ts = pd.to_datetime(pd.Series(fecha_dt).astype(str) + " " + hora_str, errors="coerce")
    out = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    if out.empty:
        raise ValueError(f"No pude construir 'ts' desde '{fecha_col}' + '{hora_col}'. Columnas originales: {original_cols}")
    print(f"✔ detectado en historical_data.csv -> fecha='{fecha_col}', hora='{hora_col}'  (filas válidas: {len(out)})")
    return out

def seasonal_weights(df_hist, col="recibidos", weeks=8, clip=(0.75,1.30)):
    d = df_hist.copy()
    d = d[[col]].dropna()
    if d.empty: return {(dow,h):1.0 for dow in range(7) for h in range(24)}
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    end = d.index.max()
    start = end - pd.Timedelta(weeks=weeks)
    d = d.loc[d.index>=start]
    med_dow_h = d.groupby(["dow","hour"])[col].median()
    med_h = d.groupby("hour")[col].median()
    w={}
    for dow in range(7):
        for h in range(24):
            num = med_dow_h.get((dow,h), np.nan)
            den = med_h.get(h, np.nan)
            val = 1.0
            if not np.isnan(num) and not np.isnan(den) and den!=0:
                val = float(num/den)
            w[(dow,h)] = float(np.clip(val, clip[0], clip[1]))
    return w

def apply_seasonal(series, weights):
    d = series.copy().to_frame("y")
    d["dow"]=d.index.dayofweek; d["hour"]=d.index.hour
    idx = list(zip(d["dow"].values, d["hour"].values))
    w = np.array([weights.get(key,1.0) for key in idx], dtype=float)
    return pd.Series((d["y"].values*w), index=d.index, name="y")

# --------- Clima para alertas ---------
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

def add_time_feats(df):
    df["dow"]=df["ts"].dt.dayofweek
    df["hour"]=df["ts"].dt.hour
    df["month"]=df["ts"].dt.month
    df["sin_hour"]=np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"]=np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"]=np.sin(2*np.pi*df["dow"]/7)
    df["cos_dow"]=np.cos(2*np.pi*df["dow"]/7)
    return df

def clima_df_from_json(j, comuna):
    if "hourly" not in j or "time" not in j["hourly"]:
        return pd.DataFrame()
    t = pd.to_datetime(j["hourly"]["time"])
    df = pd.DataFrame({"ts": t})
    for v in HOURLY_VARS:
        df[v] = j["hourly"].get(v, np.nan)
    df["comuna"]=comuna
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
    os.makedirs("public", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Asegurar modelos de alertas
    download_asset_from_latest(OWNER, REPO, MODEL_NAME, MODELS_DIR)
    download_asset_from_latest(OWNER, REPO, SCALER_NAME, MODELS_DIR)
    download_asset_from_latest(OWNER, REPO, COLS_NAME, MODELS_DIR)

    model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, MODEL_NAME), compile=False)
    scaler = joblib.load(os.path.join(MODELS_DIR, SCALER_NAME))
    with open(os.path.join(MODELS_DIR, COLS_NAME), "r") as f:
        train_cols = json.load(f)

    # ===================== Pronóstico 120d SIN clima (nacional) =====================
    df_hist = read_csv_historical_semicolon(HIST_CALLS)
    df_hist = df_hist.rename(columns=lambda c: c.strip())
    print("Columnas historical_data.csv:", df_hist.columns.tolist())
    df_hist = ensure_datetime_calls(df_hist)

    lowmap = {c.lower(): c for c in df_hist.columns}
    if "recibidos" not in lowmap:
        raise ValueError(f"En historical_data.csv no encuentro columna 'recibidos'. Columnas: {df_hist.columns.tolist()}")
    col_rec = lowmap["recibidos"]

    last_known = df_hist.index.max()
    start = last_known + pd.Timedelta(hours=1)
    end   = start + pd.Timedelta(days=HORIZON_DAYS) - pd.Timedelta(hours=1)
    future_ts = pd.date_range(start=start, end=end, freq=FREQ, tz=TIMEZONE)

    recent = df_hist.loc[df_hist.index >= (df_hist.index.max() - pd.Timedelta(weeks=2))]
    base_hour_med = recent.resample("h")[col_rec].median()
    base_hour_med = base_hour_med.reindex(future_ts, method=None).fillna(base_hour_med.median())
    weights = seasonal_weights(df_hist[[col_rec]].rename(columns={col_rec:"recibidos"}), col="recibidos", weeks=8, clip=(0.75,1.30))
    y120 = apply_seasonal(base_hour_med, weights).clip(lower=0).round().astype(int)

    # --- Guardar JSON por hora (ts string) ---
    out_hourly = (pd.DataFrame({"ts": y120.index.tz_convert(TIMEZONE), "pred_llamadas": y120.values})
                    .assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")))
    out_hourly.to_json(OUT_HOURLY_JSON, orient="records", indent=2)

    # --- Guardar JSON por día (fecha YYYY-MM-DD, no epoch) ---
    out_daily = (out_hourly
                 .assign(fecha=pd.to_datetime(out_hourly["ts"]).dt.strftime("%Y-%m-%d"))
                 .groupby("fecha", as_index=False)["pred_llamadas"]
                 .sum()
                 .rename(columns={"pred_llamadas":"total_llamadas"}))
    out_daily.to_json(OUT_DAILY_JSON, orient="records", indent=2)

    # ===================== Alertas CON clima =====================
    df_loc = normalize_loc(read_csv_coords_semicolon(LOC_CSV))
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
            d = add_time_feats(d)
            X = d[["temperature_2m","precipitation","rain","wind_speed_10m","wind_gusts_10m",
                   "sin_hour","cos_hour","sin_dow","cos_dow","dow","hour","month"]]
            X = pd.get_dummies(X, columns=["dow","hour","month"], drop_first=False)
            for c in train_cols:
                if c not in X.columns: X[c]=0
            X = X[train_cols].fillna(0)
            y = model.predict(scaler.transform(X), verbose=0).flatten()

            baseline = float(np.nanmean(y)) if np.isfinite(y).any() else 1e-6
            if baseline <= 0: baseline = 1e-6

            d["llamadas_estimadas"]   = y
            d["porcentaje_incremento"]= (y / baseline) - 1.0
            d["alerta"]               = d["porcentaje_incremento"] > ALERTA_THRESHOLD

            r = rangos_consecutivos(d[d["alerta"]==True][["ts","porcentaje_incremento"]])

            detalles = (d.assign(
                            porcentaje_incremento=lambda x:(x["porcentaje_incremento"]*100).round(1),
                            llamadas_estimadas=lambda x:x["llamadas_estimadas"].round(1),
                            alerta=lambda x:x["alerta"].astype(bool),
                            ts=lambda x:x["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        )[["ts","llamadas_estimadas","porcentaje_incremento","alerta"]]
                        .to_dict(orient="records"))

            alert_items.append({"comuna":comuna,"rango_alertas":r,"detalles":detalles})
            time.sleep(SLEEP_S)
        except Exception:
            alert_items.append({"comuna":comuna,"rango_alertas":[],"detalles":[]})
            continue

    # Ordenar: comunas con alertas primero
    alert_items.sort(key=lambda x: (len(x.get("rango_alertas",[]))==0, x["comuna"]))

    with open(OUT_ALERTAS,"w",encoding="utf-8") as f:
        json.dump(alert_items, f, ensure_ascii=False, indent=2)

    print(f"✅ {OUT_HOURLY_JSON}")
    print(f"✅ {OUT_DAILY_JSON}")
    print(f"✅ {OUT_ALERTAS}")
    print("Listo: pronóstico 120d SIN clima + alertas CON clima.")

if __name__ == "__main__":
    main()

