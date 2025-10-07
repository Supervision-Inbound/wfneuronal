# =======================================================================
# SCRIPT UNIFICADO Y FLEXIBLE PARA GOOGLE COLAB
# 1. Pide un archivo y trabaja con el nombre que sea.
# 2. Entrena los modelos de Red Neuronal.
# 3. Guarda los resultados para descargarlos.
# =======================================================================

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from google.colab import files # Importamos la librería de Colab

# --- PASO 1: Carga de archivo inteligente y flexible ---
print("Por favor, selecciona tu archivo de datos (CSV o XLSX) desde tu escritorio.")
# Pedimos al usuario que suba el archivo
uploaded = files.upload()

# Verificamos que se haya subido al menos un archivo
if not uploaded:
    raise ValueError("No se subió ningún archivo. Por favor, vuelve a ejecutar la celda.")

# Obtenemos el nombre del PRIMER archivo que se subió
# Esto hace que el script sea flexible a cualquier nombre de archivo
NOMBRE_ARCHIVO = list(uploaded.keys())[0]
print(f'\n¡Éxito! Se usará el archivo "{NOMBRE_ARCHIVO}" para el entrenamiento.')


# ---------- PARÁMETROS (El resto del script no cambia) ----------
HOJA_EXCEL = 0 # Usamos la primera hoja si es un Excel
COL_FECHA    = "fecha"
COL_HORA     = "hora"
COL_LLAMADAS = "recibidos"
COL_TMO      = "tmo (segundos)"
MAD_K = 6.0
MIN_CONSECUTIVOS = 1
SUAVIZADO = "cap"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
# ----------------------------------------------------------------


# --- PASO 2: Script de Entrenamiento ---

os.makedirs("/content/models", exist_ok=True)
os.makedirs("/content/data_out", exist_ok=True)

# Utilitarios
def parse_tmo_to_seconds(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)): return float(val)
    s = str(val).strip();
    if s.isdigit(): return float(s)
    parts = s.split(":")
    try:
        if len(parts) == 3: h, m, sec = map(float, parts); return h * 3600 + m * 60 + sec
        if len(parts) == 2: m, sec = map(float, parts); return m * 60 + sec
        return float(s)
    except: return np.nan

def read_data(path, hoja=None):
    if ".csv" in path.lower(): return pd.read_csv(path)
    elif ".xlsx" in path.lower() or ".xls" in path.lower(): return pd.read_excel(path, sheet_name=hoja)
    else: raise ValueError("Formato no soportado.")

def ensure_datetime(df, col_fecha, col_hora):
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce").dt.date
    df["hora_str"] = df[col_hora].astype(str).str.slice(0, 5)
    df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str) + " " + df["hora_str"], errors="coerce")
    return df

def add_time_features(df):
    df["dow"] = df["ts"].dt.dayofweek; df["month"] = df["ts"].dt.month; df["hour"] = df["ts"].dt.hour
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24); df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7); df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df

def rolling_features(df, target_col):
    df = df.sort_values("ts").copy()
    df[f"{target_col}_lag24"] = df[target_col].shift(24)
    df[f"{target_col}_ma24"] = df[target_col].rolling(24, min_periods=1).mean()
    df[f"{target_col}_ma168"] = df[target_col].rolling(24 * 7, min_periods=1).mean()
    return df

def robust_baseline_by_dow_hour(df, target_col):
    grp = df.groupby(["dow", "hour"])[target_col].agg(["median"]).rename(columns={"median": "med"})
    def mad(x): med = np.median(x); return np.median(np.abs(x - med))
    grp["mad"] = df.groupby(["dow", "hour"])[target_col].apply(mad).values
    return grp

def detect_peaks(df, target_col, mad_k=6.0, min_consec=1):
    base = robust_baseline_by_dow_hour(df, target_col)
    df = df.merge(base, left_on=["dow", "hour"], right_index=True, how="left")
    df["upper_cap"] = df["med"] + mad_k * df["mad"].replace(0, df["mad"].median())
    df["is_peak"] = (df[target_col] > df["upper_cap"]).astype(int)
    if min_consec > 1:
        df = df.sort_values("ts"); runs = (df["is_peak"].diff(1) != 0).cumsum()
        sizes = df.groupby(runs)["is_peak"].transform("sum")
        df["is_peak"] = np.where((df["is_peak"] == 1) & (sizes >= min_consec), 1, 0)
    return df

def smooth_series(df, target_col, method="cap"):
    if method == "cap": df[target_col + "_smooth"] = np.where(df["is_peak"] == 1, df["upper_cap"], df[target_col])
    else: df[target_col + "_smooth"] = np.where(df["is_peak"] == 1, df["med"], df[target_col])
    return df

def build_feature_matrix_nn(df, target_col):
    base_feats = ["sin_hour", "cos_hour", "sin_dow", "cos_dow", f"{target_col}_lag24", f"{target_col}_ma24", f"{target_col}_ma168"]
    cat_feats = ["dow", "month"]
    df_dummies = pd.get_dummies(df[cat_feats], columns=cat_feats, drop_first=False)
    X = pd.concat([df[base_feats], df_dummies], axis=1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)

def crear_modelo_nn(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(n_features,)), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    print("\n--- INICIANDO PROCESO DE ENTRENAMIENTO ---")
    print("Leyendo datos:", NOMBRE_ARCHIVO)
    df = read_data(NOMBRE_ARCHIVO, hoja=HOJA_EXCEL).copy()
    df = ensure_datetime(df, COL_FECHA, COL_HORA).dropna(subset=["ts"]).copy()
    df["tmo_seg"] = df[COL_TMO].apply(parse_tmo_to_seconds)
    df = add_time_features(df); df = rolling_features(df, COL_LLAMADAS)
    df_pk = detect_peaks(df.copy(), COL_LLAMADAS, mad_k=MAD_K, min_consec=MIN_CONSECUTIVOS)
    df_pk = smooth_series(df_pk, COL_LLAMADAS, method=SUAVIZADO)
    peaks_out = df_pk[["ts", COL_LLAMADAS, "med", "mad", "upper_cap", "is_peak"]].copy()
    peaks_out.to_csv("/content/data_out/peaks_llamadas.csv", index=False)
    print("Guardado: /content/data_out/peaks_llamadas.csv")
    df_train_ll = df_pk.copy(); df_train_ll[COL_LLAMADAS] = df_train_ll[COL_LLAMADAS + "_smooth"]
    
    print("\n--- Preparando Modelo de LLAMADAS ---")
    X_ll = build_feature_matrix_nn(df_train_ll, COL_LLAMADAS)
    y_ll = df_train_ll.loc[X_ll.index, COL_LLAMADAS].astype(float)
    Xtr_ll, Xte_ll, ytr_ll, yte_ll = train_test_split(X_ll, y_ll, test_size=TEST_SIZE, shuffle=False)
    scaler_llamadas = StandardScaler(); Xtr_ll_scaled = scaler_llamadas.fit_transform(Xtr_ll); Xte_ll_scaled = scaler_llamadas.transform(Xte_ll)
    print("Entrenando Modelo de Llamadas con Red Neuronal...")
    model_ll = crear_modelo_nn(Xtr_ll_scaled.shape[1])
    model_ll.fit(Xtr_ll_scaled, ytr_ll, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    pred_te_ll = model_ll.predict(Xte_ll_scaled).flatten()
    print(f"[LLAMADAS] MAE={mean_absolute_error(yte_ll, pred_te_ll):.2f}  R2={r2_score(yte_ll, pred_te_ll):.3f}")
    
    print("\n--- Preparando Modelo de TMO ---")
    df_tmo = df_pk.copy(); df_tmo = rolling_features(df_tmo, "tmo_seg")
    X_tmo = build_feature_matrix_nn(df_tmo, "tmo_seg"); y_tmo = df_tmo.loc[X_tmo.index, "tmo_seg"].astype(float)
    mask_ok = y_tmo.notna() & (y_tmo > 0); X_tmo = X_tmo.loc[mask_ok]; y_tmo = y_tmo.loc[mask_ok]
    if len(y_tmo) > 100:
        Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(X_tmo, y_tmo, test_size=TEST_SIZE, shuffle=False)
        scaler_tmo = StandardScaler(); Xtr_t_scaled = scaler_tmo.fit_transform(Xtr_t); Xte_t_scaled = scaler_tmo.transform(Xte_t)
        print(f"Entrenando Modelo de TMO... (muestras válidas: {len(y_tmo)})")
        model_tmo = crear_modelo_nn(Xtr_t_scaled.shape[1])
        model_tmo.fit(Xtr_t_scaled, ytr_t, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
        pred_te_t = model_tmo.predict(Xte_t_scaled).flatten()
        print(f"[TMO] MAE={mean_absolute_error(yte_t, pred_te_t):.2f}  R2={r2_score(yte_t, pred_te_t):.3f}")
    else: model_tmo, scaler_tmo = None, None; print(f"AVISO: TMO no entrenado (muestras válidas={len(y_tmo)}).")

    print("\n--- Guardando artefactos ---")
    model_ll.save("/content/models/modelo_llamadas_nn.h5")
    joblib.dump(scaler_llamadas, "/content/models/scaler_llamadas.pkl")
    print("Guardado: /content/models/modelo_llamadas_nn.h5 y scaler_llamadas.pkl")
    if model_tmo is not None:
        model_tmo.save("/content/models/modelo_tmo_nn.h5")
        joblib.dump(scaler_tmo, "/content/models/scaler_tmo.pkl")
        print("Guardado: /content/models/modelo_tmo_nn.h5 y scaler_tmo.pkl")
    
    print("\n¡PROCESO COMPLETADO!")

# Ejecutamos la función principal
main()
