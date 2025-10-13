# --- helpers robustos para cargar coordenadas ---
def read_csv_smart(path):
    """Lee CSV probando encodings y separadores comunes."""
    encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    seps = (None, ",", ";", "\t", "|")  # None => autodetect con engine='python'
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    return pd.read_csv(path, encoding=enc, engine="python")
                else:
                    return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                last_err = e
                continue
    raise last_err or ValueError(f"No pude leer {path} con encodings/separadores estándar.")

def _pick_col(cols, candidates):
    """Devuelve el primer nombre de columna que exista en 'cols' entre 'candidates'."""
    cols_set = {c.lower().strip(): c for c in cols}
    for c in candidates:
        key = c.lower().strip()
        if key in cols_set:
            return cols_set[key]
    return None

def normalize_location_columns(df):
    """Normaliza columnas a: comuna, lat, lon (acepta alias y arregla decimales)."""
    original_cols = df.columns.tolist()
    # mapa de candidatos
    comuna_cands = ["comuna", "municipio", "localidad", "ciudad", "name", "nombre"]
    lat_cands    = ["lat", "latitude", "latitud", "y"]
    lon_cands    = ["lon", "lng", "long", "longitude", "longitud", "x"]

    comuna_col = _pick_col(original_cols, comuna_cands)
    lat_col    = _pick_col(original_cols, lat_cands)
    lon_col    = _pick_col(original_cols, lon_cands)

    if comuna_col is None:
        raise ValueError("No encuentro columna de 'comuna' (intenta: comuna/municipio/localidad/ciudad/name/nombre).")
    if lat_col is None or lon_col is None:
        raise ValueError("No encuentro columnas de lat/lon (intenta: lat/latitude/latitud/y y lon/lng/long/longitude/longitud/x).")

    df = df.rename(columns={comuna_col: "comuna", lat_col: "lat", lon_col: "lon"}).copy()

    # arreglar comas decimales si vienen como string con ','
    for c in ["lat", "lon"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # limpiar filas inválidas
    df = df.dropna(subset=["lat", "lon"])
    # valores razonables de lat/lon
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    # quitar duplicados por comuna
    df = df.drop_duplicates(subset=["comuna"]).reset_index(drop=True)
    return df

# ---------------- Main ----------------
def main():
    print("📦 Cargando modelo y artefactos...")
    model = tf.keras.models.load_model(f"models/{MODEL_NAME}", compile=False)
    scaler = joblib.load(f"models/{SCALER_NAME}")
    with open(f"models/{COLS_NAME}", "r") as f:
        training_cols = json.load(f)

    print("📍 Cargando coordenadas de comunas...")
    df_loc_raw = read_csv_smart(LOC_CSV)
    df_loc_raw.columns = [c.strip() for c in df_loc_raw.columns]  # conserva mayúsculas originales
    df_loc = normalize_location_columns(df_loc_raw)  # <- normalización clave

    client = build_http_client()
    all_preds = []
    # ... (resto de tu main tal cual)
