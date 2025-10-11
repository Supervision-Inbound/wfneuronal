# =======================================================================
# forecast3m.py — Inferencia con historial sembrado, lags/MAs y predicción recursiva
# Salidas: /public/predicciones.json, /public/erlang_forecast.json, /public/last_update.json
# Mantiene compatibilidad con utils_release.download_asset_from_latest (owner, repo, asset_name, target_dir)
# =======================================================================

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from utils_release import download_asset_from_latest

# ----------------------- Parámetros del repositorio ----------------------
OWNER = "Supervision-Inbound"
REPO  = "wfneuronal"

# ----------------------- Parámetros de horizonte -------------------------
def month_bounds(anchor=None):
    """Del 1 del mes anterior al fin del próximo mes (versión estable)"""
    anchor = pd.Timestamp.today().normalize() if anchor is None else pd.Timestamp(anchor)
    first_of_month = anchor.replace(day=1)
    first_prev = (first_of_month - pd.offsets.MonthBegin(1))
    last_next = (first_of_month + pd.offsets.MonthEnd(2))
    return first_prev, last_next

# ----------------------- Paths de salida ---------------------------------
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

OUT_JSON_PRED = PUBLIC_DIR / "predicciones.json"
OUT_JSON_ERLANG = PUBLIC_DIR / "erlang_forecast.json"
STAMP_JSON = PUBLIC_DIR / "last_update.json"

# ----------------------- Directorio de modelos ---------------------------
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ----------------------- Nombres de artefactos esperados -----------------
ARTIFACTS = {
    "calls_model": [
        "modelo_llamadas_nn.h5",  # nombre que guardas desde el entrenamiento NN
        "modelo_llamadas.h5",
        "model_llamadas.h5",
        "model.h5"
    ],
    "calls_scaler": [
        "scaler_llamadas.pkl",
        "scaler.pkl"
    ],
    "calls_features": [
        "training_columns_llamadas.json",
        "training_columns.json"
    ]
}

TMO_ARTIFACTS = {
    "tmo_model": ["modelo_tmo_nn.h5", "modelo_tmo.h5", "model_tmo.h5"],
    "tmo_scaler": ["scaler_tmo.pkl"],
    "tmo_features": ["training_columns_tmo.json"]
}

# ----------------------- Parámetros de Erlang ----------------------------
SLA_TARGET = 0.90
ASA_TARGET_S = 22.0
MAX_OCC = 0.85
SHRINKAGE = 0.30
ABSENTEEISM_RATE = 0.23
USE_ERLANG_A = True
MEAN_PATIENCE_S = 60.0
ABANDON_MAX = 0.06
AWT_MAX_S = 120.0

# ======================= Utilidades generales ============================
def try_download(names, target_dir):
    """Intenta descargar una lista de posibles nombres desde el último Release hacia target_dir."""
    for fname in names:
        try:
            print(f"[Descarga] Buscando asset en release: {fname}")
            path = download_asset_from_latest(OWNER, REPO, fname, str(target_dir))
            if path and os.path.exists(path):
                print(f"[Descarga] OK -> {path}")
                return path
        except Exception as e:
            print(f"[Descarga] No se pudo descargar {fname}: {e}")
    return None

def load_artifacts():
    """Carga modelo/scaler/columns de llamadas y opcionalmente TMO desde Release o desde /models."""
    # Llamadas
    model_calls_path = try_download(ARTIFACTS["calls_model"], MODELS_DIR) or str(MODELS_DIR / "modelo_llamadas_nn.h5")
    scaler_calls_path = try_download(ARTIFACTS["calls_scaler"], MODELS_DIR) or str(MODELS_DIR / "scaler_llamadas.pkl")
    feat_calls_path = try_download(ARTIFACTS["calls_features"], MODELS_DIR) or str(MODELS_DIR / "training_columns_llamadas.json")

    print(f"[Artefactos] model_calls_path={model_calls_path}")
    print(f"[Artefactos] scaler_calls_path={scaler_calls_path}")
    print(f"[Artefactos] feat_calls_path={feat_calls_path}")

    # Modelo de llamadas
    try:
        import tensorflow as tf
        model_calls = tf.keras.models.load_model(model_calls_path)
    except Exception as e:
        searched = ARTIFACTS["calls_model"] + [str(MODELS_DIR / "modelo_llamadas_nn.h5")]
        raise RuntimeError(
            f"No pude cargar el modelo de llamadas. Rutas probadas: {searched}. Error base: {e}"
        )

    # Scaler de llamadas
    scaler_calls = joblib.load(scaler_calls_path) if os.path.exists(scaler_calls_path) else None

    # Columnas de entrenamiento de llamadas
    feat_calls = None
    if feat_calls_path and os.path.exists(feat_calls_path):
        with open(feat_calls_path, "r", encoding="utf-8") as f:
            try:
                feat_calls = json.load(f)
            except Exception:
                # fallback por si viene como lista en texto
                txt = f.read()
                try:
                    feat_calls = json.loads(txt)
                except Exception:
                    feat_calls = None

    # TMO (si existe en Release)
    model_tmo = scaler_tmo = feat_tmo = None
    tmo_model_path = try_download(TMO_ARTIFACTS["tmo_model"], MODELS_DIR)
    tmo_scaler_path = try_download(TMO_ARTIFACTS["tmo_scaler"], MODELS_DIR)
    tmo_feat_path   = try_download(TMO_ARTIFACTS["tmo_features"], MODELS_DIR)

    if tmo_model_path and os.path.exists(tmo_model_path):
        import tensorflow as tf
        model_tmo = tf.keras.models.load_model(tmo_model_path)
        scaler_tmo = joblib.load(tmo_scaler_path) if tmo_scaler_path and os.path.exists(tmo_scaler_path) else None
        if tmo_feat_path and os.path.exists(tmo_feat_path):
            with open(tmo_feat_path, "r", encoding="utf-8") as f:
                feat_tmo = json.load(f)

    return (model_calls, scaler_calls, feat_calls, model_tmo, scaler_tmo, feat_tmo)

def try_load_history():
    """
    Intenta conseguir historial de llamadas para sembrar lags/MAs.
    Prioridad:
    1) Hosting ia.xlsx (hoja 0) con columnas [fecha, hora, recibidos] o [datetime, recibidos]
    2) public/predicciones.json (si ya existe)
    """
    # 1) Excel
    for path in ["Hosting ia.xlsx", "/mnt/data/Hosting ia.xlsx"]:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path, sheet_name=0)
                cols = [c.lower() for c in df.columns]
                df.columns = cols
                if "datetime" in cols:
                    df["ts"] = pd.to_datetime(df["datetime"])
                elif "datatime" in cols:
                    df["ts"] = pd.to_datetime(df["datatime"])
                elif "fecha" in cols and "hora" in cols:
                    df["ts"] = pd.to_datetime(df["fecha"].astype(str) + " " + df["hora"].astype(str), dayfirst=True, errors="coerce")
                else:
                    continue
                if "recibidos" not in cols:
                    continue
                hist = df[["ts","recibidos"]].dropna().sort_values("ts")
                hist = hist[hist["ts"] >= (hist["ts"].max() - pd.Timedelta(days=60))]
                return hist
            except Exception:
                pass
    # 2) predicciones.json existente
    if OUT_JSON_PRED.exists():
        try:
            data = json.loads(OUT_JSON_PRED.read_text(encoding="utf-8"))
            df = pd.DataFrame(data)
            if "ts" in df.columns and "recibidos" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"])
                hist = df[["ts","recibidos"]].dropna().sort_values("ts")
                return hist
        except Exception:
            pass
    return None

def calendar_df(start, end):
    idx = pd.date_range(start, end, freq="H", inclusive="left")
    df = pd.DataFrame({"ts": idx})
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    return df

def add_time_features(df):
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]  = np.cos(2 * np.pi * df["dow"] / 7)
    return df

def add_lags_ma(df, col):
    df[f"{col}_lag24"] = df[col].shift(24)
    df[f"{col}_lag48"] = df[col].shift(48)
    df[f"{col}_lag72"] = df[col].shift(72)
    df[f"{col}_ma24"]  = df[col].rolling(24, min_periods=1).mean()
    df[f"{col}_ma72"]  = df[col].rolling(72, min_periods=1).mean()
    df[f"{col}_ma168"] = df[col].rolling(24*7, min_periods=1).mean()
    return df

def ensure_columns(X, training_columns):
    if training_columns is None:
        return X
    for c in training_columns:
        if c not in X.columns:
            X[c] = 0.0
    X = X[training_columns]
    return X

def build_X(df, target_col, training_columns=None):
    dummies = pd.get_dummies(df[["dow","month"]], columns=["dow","month"], drop_first=False)
    feats = df[[
        "sin_hour","cos_hour","sin_dow","cos_dow",
        f"{target_col}_lag24", f"{target_col}_lag48", f"{target_col}_lag72",
        f"{target_col}_ma24",  f"{target_col}_ma72",  f"{target_col}_ma168"
    ]].copy()
    X = pd.concat([feats, dummies], axis=1).replace([np.inf,-np.inf], np.nan).fillna(0)
    X = ensure_columns(X, training_columns)
    return X

# ====================== Erlang (C/A simplificado) ========================
def erlang_c_prob_wait(a, n):
    rho = a / n
    if rho >= 1: 
        return 1.0
    summation = sum((a**k)/math.factorial(k) for k in range(n))
    pn = (a**n)/(math.factorial(n)*(1-rho)) / (summation + (a**n)/(math.factorial(n)*(1-rho)))
    return pn

def service_level_erlang_c(a, n, asa_target):
    rho = a / n
    if rho >= 1:
        return 0.0
    pw = erlang_c_prob_wait(a, n)
    return 1 - pw * math.exp(-(n - a) * (asa_target / 3600.0))

def required_agents(contacts_per_hour, aht_s, sla=SLA_TARGET, asa_s=ASA_TARGET_S, max_occ=MAX_OCC):
    if aht_s <= 0:
        aht_s = 1.0
    a = contacts_per_hour * (aht_s / 3600.0)
    n = max(1, int(math.ceil(a / max_occ)))
    for _ in range(200):
        sl = service_level_erlang_c(a, n, asa_s)
        if sl >= sla:
            break
        n += 1
    scheduled = math.ceil(n / (1 - SHRINKAGE))
    return n, scheduled

# ======================= Predicción recursiva ============================
def recursive_predict(model, scaler, training_columns, hist_df, future_calendar, target_col="recibidos"):
    hist_df = hist_df.copy().sort_values("ts").reset_index(drop=True)
    hist_df = add_time_features(hist_df.assign(hour=hist_df["ts"].dt.hour, 
                                               dow=hist_df["ts"].dt.dayofweek,
                                               month=hist_df["ts"].dt.month))
    hist_df = add_lags_ma(hist_df, target_col)

    out_rows = []
    concat = pd.concat([
        hist_df[["ts", target_col, "dow","month","hour","sin_hour","cos_hour","sin_dow","cos_dow",
                 f"{target_col}_lag24", f"{target_col}_lag48", f"{target_col}_lag72",
                 f"{target_col}_ma24",  f"{target_col}_ma72",  f"{target_col}_ma168"
                ]]
    ], axis=0, ignore_index=True)

    for ts in future_calendar["ts"]:
        row = {"ts": ts, "dow": ts.dayofweek, "month": ts.month, "hour": ts.hour}
        tmp = pd.DataFrame([row])
        tmp = add_time_features(tmp)
        concat = pd.concat([concat, tmp.assign(recibidos=np.nan)], ignore_index=True)

        concat = add_lags_ma(concat, target_col)
        X_row = build_X(concat.tail(1), target_col, training_columns)
        if scaler is not None:
            Xs = scaler.transform(X_row.values)
        else:
            Xs = X_row.values
        yhat = float(model.predict(Xs, verbose=0).reshape(-1)[0])
        concat.iloc[-1, concat.columns.get_loc(target_col)] = max(yhat, 0.0)
        out_rows.append({"ts": ts, target_col: max(yhat, 0.0)})

    return pd.DataFrame(out_rows)

# ============================== MAIN =====================================
def main():
    # 1) Cargar artefactos (desde Release -> /models)
    (model_calls, scaler_calls, feat_calls, model_tmo, scaler_tmo, feat_tmo) = load_artifacts()

    # 2) Horizonte
    start, end = month_bounds()
    cal_fut = calendar_df(start, end)

    # 3) Historial
    hist = try_load_history()
    if hist is None or len(hist) < 24*7:
        raise RuntimeError("No se encontró historial suficiente (mín. 7 días). Asegura 'Hosting ia.xlsx' o un JSON previo.")

    # 4) Predicción recursiva de llamadas
    pred_calls = recursive_predict(model_calls, scaler_calls, feat_calls, hist, cal_fut, target_col="recibidos")

    # 5) TMO (si hay modelo). Si no, usa patrón robusto.
    if model_tmo is not None:
        df_tmo = cal_fut.copy()
        df_tmo = add_time_features(df_tmo)
        dummies = pd.get_dummies(df_tmo[["dow","month"]], columns=["dow","month"], drop_first=False)
        feats = df_tmo[["sin_hour","cos_hour","sin_dow","cos_dow"]].copy()
        X_tmo = pd.concat([feats, dummies], axis=1).replace([np.inf,-np.inf], np.nan).fillna(0)
        if feat_tmo is not None:
            for c in feat_tmo:
                if c not in X_tmo.columns:
                    X_tmo[c] = 0.0
            X_tmo = X_tmo[feat_tmo]
        Xs = scaler_tmo.transform(X_tmo.values) if scaler_tmo is not None else X_tmo.values
        tmo_pred = model_tmo.predict(Xs, verbose=0).reshape(-1)
        tmo_pred = np.clip(tmo_pred, 60, 1200)
        tmo_series = pd.Series(tmo_pred, index=cal_fut["ts"])
    else:
        # Patrón simple si no hay modelo TMO: constante por hora (ajustable)
        hour_mean = hist.assign(hour=hist["ts"].dt.hour).groupby("hour")["recibidos"].mean()
        tmo_series = cal_fut["hour"].map(hour_mean).fillna(hour_mean.mean())
        tmo_series = pd.Series(np.clip(tmo_series.values*0 + 300.0, 240, 480), index=cal_fut["ts"])

    # 6) Construir JSON de predicciones
    pred = pred_calls.copy()
    pred["tmo_seg"] = tmo_series.values
    pred["ts"] = pred["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    with open(OUT_JSON_PRED, "w", encoding="utf-8") as f:
        json.dump(pred.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    # 7) Dimensionamiento (Erlang-C aproximado)
    erlang_rows = []
    for _, r in pred.iterrows():
        ts = r["ts"]
        calls = float(r["recibidos"])
        aht = float(r["tmo_seg"])
        productive, scheduled = required_agents(calls, aht, sla=SLA_TARGET, asa_s=ASA_TARGET_S, max_occ=MAX_OCC)
        erlang_rows.append({
            "ts": ts,
            "llamadas": round(calls),
            "tmo_seg": round(aht),
            "agentes_productivos": int(productive),
            "agentes_programados": int(scheduled)
        })

    with open(OUT_JSON_ERLANG, "w", encoding="utf-8") as f:
        json.dump(erlang_rows, f, ensure_ascii=False, indent=2)

    # 8) Timestamp
    stamp = {"generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(STAMP_JSON, "w", encoding="utf-8") as f:
        json.dump(stamp, f)

    print("✔ Proceso de inferencia completado. Archivos creados en /public")

if __name__ == "__main__":
    main()
