# =======================================================================
# SCRIPT UNIFICADO Y FLEXIBLE PARA GOOGLE COLAB
# 1. Pide un archivo y trabaja con el nombre que sea.
# 2. Entrena los modelos de Red Neuronal (llamadas y, si aplica, TMO).
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

# Si corres en Colab:
try:
    from google.colab import files  # type: ignore
    _IN_COLAB = True
except Exception:
    _IN_COLAB = False

# ------------------------ PARÁMETROS GLOBALES ----------------------------
TARGET_CALLS = "recibidos"
TARGET_TMO   = "tmo_seg"           # si tu dataset trae TMO en segundos
MAD_K        = 3.5                 # solo para diagnóstico; no suavizamos en train
CAP_METHOD   = "cap"               # 'cap' o 'med' (no se aplica en train)
EPOCHS       = 100
BATCH_SIZE   = 256
VAL_SPLIT    = 0.1
SEED         = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------- UTILITARIOS ---------------------------------
def parse_tmo_to_seconds(val):
    """Convierte 'mm:ss' o 'hh:mm:ss' a segundos. Si ya es número, lo usa."""
    if pd.isna(val): 
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)): 
        return float(val)
    s = str(val).strip()
    if s.isdigit(): 
        return float(s)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = map(float, parts); 
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = map(float, parts); 
            return m * 60 + sec
        return float(s)
    except:
        return np.nan

def read_data(path, hoja=None):
    if ".csv" in path.lower(): 
        return pd.read_csv(path)
    elif ".xlsx" in path.lower() or ".xls" in path.lower(): 
        return pd.read_excel(path, sheet_name=hoja if hoja is not None else 0)
    else: 
        raise ValueError("Formato no soportado. Usa CSV o XLSX.")

def ensure_datetime(df, col_fecha, col_hora):
    """Construye 'ts' desde columnas fecha y hora."""
    df["fecha_dt"] = pd.to_datetime(df[col_fecha], errors="coerce").dt.date
    df["hora_str"] = df[col_hora].astype(str).str.slice(0, 5)
    df["ts"] = pd.to_datetime(df["fecha_dt"].astype(str) + " " + df["hora_str"], errors="coerce")
    return df

def ensure_ts(df):
    """Genera timestamp robusto buscando columnas típicas."""
    cols = {c.lower(): c for c in df.columns}
    # Normalizamos nombres
    df.columns = [c.lower() for c in df.columns]

    # Caso 1: datetime directo
    if "datetime" in df.columns:
        df["ts"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "datatime" in df.columns:
        df["ts"] = pd.to_datetime(df["datatime"], errors="coerce")
    # Caso 2: fecha + hora
    elif "fecha" in df.columns and "hora" in df.columns:
        df = ensure_datetime(df, "fecha", "hora")
    else:
        # heurística por si hay nombres raros
        date_cols = [c for c in df.columns if "fecha" in c]
        hour_cols = [c for c in df.columns if "hora" == c or c.startswith("hora_")]
        if date_cols and hour_cols:
            df = ensure_datetime(df, date_cols[0], hour_cols[0])
        else:
            raise ValueError("No pude construir 'ts'. Asegura columnas 'datetime' o 'fecha' + 'hora'.")

    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def add_time_parts(df):
    df["dow"]   = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["hour"]  = df["ts"].dt.hour
    # codificación circular
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"]  = np.sin(2 * np.pi * df["dow"]  / 7)
    df["cos_dow"]  = np.cos(2 * np.pi * df["dow"]  / 7)
    return df

def rolling_features(df, target_col):
    """Lags/MAs para el target. NUEVO: lag48, lag72, ma72."""
    df = df.sort_values("ts").copy()
    # Lags
    df[f"{target_col}_lag24"] = df[target_col].shift(24)
    df[f"{target_col}_lag48"] = df[target_col].shift(48)   # NUEVO
    df[f"{target_col}_lag72"] = df[target_col].shift(72)   # NUEVO
    # Medias móviles
    df[f"{target_col}_ma24"]  = df[target_col].rolling(24,     min_periods=1).mean()
    df[f"{target_col}_ma72"]  = df[target_col].rolling(72,     min_periods=1).mean()      # NUEVO
    df[f"{target_col}_ma168"] = df[target_col].rolling(24*7,   min_periods=1).mean()
    return df

def robust_baseline_by_dow_hour(df, target_col):
    """Baseline robusto por (dow, hour) para diagnóstico de picos."""
    grp = df.groupby(["dow", "hour"])[target_col].agg(["median"]).rename(columns={"median": "med"})
    def mad(x):
        med = np.median(x); 
        return np.median(np.abs(x - med))
    grp["mad"] = df.groupby(["dow", "hour"])[target_col].apply(mad).values
    return grp

def detect_peaks(df, target_col, mad_k=3.5, min_consec=1):
    """Marca picos por MAD, opcionalmente exige al menos 'min_consec' consecutivos."""
    base = robust_baseline_by_dow_hour(df, target_col)
    df = df.merge(base, left_on=["dow", "hour"], right_index=True, how="left")
    # evitar mad=0: reemplazar por la mediana global de mad
    df["mad"] = df["mad"].replace(0, df["mad"].median())
    df["upper_cap"] = df["med"] + mad_k * df["mad"]
    df["is_peak"] = (df[target_col] > df["upper_cap"]).astype(int)
    if min_consec > 1:
        df = df.sort_values("ts")
        runs = (df["is_peak"].diff(1) != 0).cumsum()
        sizes = df.groupby(runs)["is_peak"].transform("sum")
        df["is_peak"] = np.where((df["is_peak"] == 1) & (sizes >= min_consec), 1, 0)
    return df

def smooth_series(df, target_col, method="cap"):
    """No se usa en TRAIN (solo diagnóstico). Dejo para que tengas a mano."""
    if method == "cap":
        df[target_col + "_smooth"] = np.where(df["is_peak"] == 1, df["upper_cap"], df[target_col])
    else:
        df[target_col + "_smooth"] = np.where(df["is_peak"] == 1, df["med"], df[target_col])
    return df

def build_feature_matrix_nn(df, target_col, include_peak_flag=True):
    """
    Arma la matriz de features para la red densa.
    Usa senos/cosenos + lags/MA + dummies de dow/month.
    """
    base_feats = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        f"{target_col}_lag24", f"{target_col}_lag48", f"{target_col}_lag72",
        f"{target_col}_ma24",  f"{target_col}_ma72",  f"{target_col}_ma168"
    ]
    feats = list(base_feats)
    if include_peak_flag and "is_peak" in df.columns:
        feats.append("is_peak")  # compatibilidad si quieres testearlo

    # dummies calendario
    cat_feats = ["dow", "month"]
    dummies = pd.get_dummies(df[cat_feats], columns=cat_feats, drop_first=False)

    X = pd.concat([df[feats], dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def crear_modelo_nn(n_features):
    """Red densa mejorada: 256 → 128 → 64 + Dropout(0.2)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

# ------------------------ ENTRENAMIENTO END-TO-END -----------------------
def main():
    # --- PASO 1: Carga de archivo flexible (Colab) ---
    if _IN_COLAB:
        print("Por favor, selecciona tu archivo de datos (CSV o XLSX).")
        uploaded = files.upload()  # type: ignore
        if not uploaded:
            raise ValueError("No se subió ningún archivo. Vuelve a ejecutar.")
        input_file = list(uploaded.keys())[0]
    else:
        # Si no estás en Colab, apunta aquí el nombre de tu archivo:
        input_file = "Hosting ia.xlsx"
        if not os.path.exists(input_file) and os.path.exists("/mnt/data/Hosting ia.xlsx"):
            input_file = "/mnt/data/Hosting ia.xlsx"

    print(f"\n--- Archivo de entrada: {input_file} ---")
    df = read_data(input_file)

    # --- PASO 2: Normalización de columnas y timestamp ---
    df = ensure_ts(df)
    df = add_time_parts(df)

    # Asegurar columnas objetivo si existen
    if TARGET_CALLS not in df.columns:
        raise ValueError(f"No encuentro la columna '{TARGET_CALLS}' en el dataset.")
    if TARGET_TMO in df.columns:
        df[TARGET_TMO] = df[TARGET_TMO].apply(parse_tmo_to_seconds)

    # --- PASO 3: Diagnóstico de picos (no se usa en entrenamiento) ---
    df_diag = detect_peaks(df.copy(), TARGET_CALLS, mad_k=MAD_K, min_consec=1)
    peak_rate = df_diag["is_peak"].mean() if "is_peak" in df_diag.columns else 0.0
    print(f"Tasa de picos (MAD_K={MAD_K}): {peak_rate*100:.2f}%  [Solo diagnóstico]")

    # --- PASO 4: Features rolling para Llamadas ---
    df_ll = rolling_features(df.copy(), TARGET_CALLS)
    X_ll = build_feature_matrix_nn(df_ll, TARGET_CALLS, include_peak_flag=False)
    y_ll = df_ll.loc[X_ll.index, TARGET_CALLS].astype(float)

    # Split temporal (sin mezclar)
    Xtr_ll, Xte_ll, ytr_ll, yte_ll = train_test_split(
        X_ll, y_ll, test_size=0.2, shuffle=False
    )

    scaler_llamadas = StandardScaler()
    Xtr_ll_scaled = scaler_llamadas.fit_transform(pd.DataFrame(Xtr_ll, columns=Xtr_ll.columns))
    Xte_ll_scaled = scaler_llamadas.transform(pd.DataFrame(Xte_ll, columns=Xte_ll.columns))

    print("\n--- Entrenando Modelo de Llamadas (NN 256-128-64 + Dropout) ---")
    model_ll = crear_modelo_nn(Xtr_ll_scaled.shape[1])
    model_ll.fit(
        Xtr_ll_scaled, ytr_ll,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    pred_te_ll = model_ll.predict(Xte_ll_scaled, verbose=0).flatten()
    print(f"[LLAMADAS] MAE={mean_absolute_error(yte_ll, pred_te_ll):.2f}  R2={r2_score(yte_ll, pred_te_ll):.3f}")

    # --- PASO 5: Modelo de TMO (opcional si existe columna) -------------
    model_tmo = None
    scaler_tmo = None
    if TARGET_TMO in df.columns:
        print("\n--- Preparando Modelo de TMO ---")
        df_tmo = rolling_features(df.copy(), TARGET_TMO)
        X_tmo = build_feature_matrix_nn(df_tmo, TARGET_TMO, include_peak_flag=False)
        y_tmo = df_tmo.loc[X_tmo.index, TARGET_TMO].astype(float)
        mask_ok = y_tmo.notna() & (y_tmo > 0)
        X_tmo = X_tmo.loc[mask_ok]
        y_tmo = y_tmo.loc[mask_ok]

        if len(y_tmo) > 200:
            Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
                X_tmo, y_tmo, test_size=0.2, shuffle=False
            )
            scaler_tmo = StandardScaler()
            Xtr_t_scaled = scaler_tmo.fit_transform(pd.DataFrame(Xtr_t, columns=Xtr_t.columns))
            Xte_t_scaled = scaler_tmo.transform(pd.DataFrame(Xte_t, columns=Xte_t.columns))

            model_tmo = crear_modelo_nn(Xtr_t_scaled.shape[1])
            print("\n--- Entrenando Modelo de TMO ---")
            model_tmo.fit(
                Xtr_t_scaled, ytr_t,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_split=VAL_SPLIT, verbose=1,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
            )
            pred_te_t = model_tmo.predict(Xte_t_scaled, verbose=0).flatten()
            print(f"[TMO] MAE={mean_absolute_error(yte_t, pred_te_t):.2f}  R2={r2_score(yte_t, pred_te_t):.3f}")
        else:
            print("TMO: datos insuficientes para entrenar (se omite).")

    # --- PASO 6: Guardado de artefactos ---------------------------------
    os.makedirs("/content/models", exist_ok=True)
    # Guardar modelo de llamadas y scaler
    model_ll.save("/content/models/modelo_llamadas_nn.h5")
    joblib.dump(scaler_llamadas, "/content/models/scaler_llamadas.pkl")
    # Guardar columnas de entrenamiento (para inferencia segura)
    cols_ll = list(pd.DataFrame(X_ll, columns=X_ll.columns).columns)
    with open("/content/models/training_columns_llamadas.json", "w", encoding="utf-8") as f:
        import json; json.dump(cols_ll, f, ensure_ascii=False, indent=2)
    print("Guardado: modelo_llamadas_nn.h5, scaler_llamadas.pkl y training_columns_llamadas.json")

    # Guardar TMO si existe
    if model_tmo is not None and scaler_tmo is not None:
        model_tmo.save("/content/models/modelo_tmo_nn.h5")
        joblib.dump(scaler_tmo, "/content/models/scaler_tmo.pkl")
        cols_tmo = list(pd.DataFrame(X_tmo, columns=X_tmo.columns).columns)
        with open("/content/models/training_columns_tmo.json", "w", encoding="utf-8") as f:
            import json; json.dump(cols_tmo, f, ensure_ascii=False, indent=2)
        print("Guardado: modelo_tmo_nn.h5, scaler_tmo.pkl y training_columns_tmo.json")

    print("\n¡PROCESO COMPLETADO!")

# ------------------------------------------------------------------------
# Ejecutamos la función principal
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

