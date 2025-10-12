# =======================================================================
# CLIMA HORARIO por LAT/LON — FALLBACK ROBUSTO (SIN API KEYS)
# Fuentes: 1. Open-Meteo (Archive) → 2. Meteostat
# Desde 2022-01-01 hasta HOY | America/Santiago
#
# v4: Optimizado para rangos históricos largos mediante "chunking" anual.
# =======================================================================

# Instalar dependencias si no están presentes
!pip -q install openmeteo-requests retry-requests pandas requests-cache python-dateutil openpyxl chardet meteostat pytz

import io
import time
import chardet
import warnings
import pandas as pd
import datetime as dt
from dateutil import tz
from google.colab import files

# APIs de clima
import openmeteo_requests
from retry_requests import retry
import requests
import requests_cache
from meteostat import Hourly as MsHourly, Point as MsPoint

# Para escritura optimizada en Excel
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils.exceptions import IllegalCharacterError

warnings.filterwarnings("ignore")

# ---------- Parámetros ----------
OVERALL_START_DATE = "2022-01-01"
TZ_STR = "America/Santiago"
# <-- La fecha final sigue siendo dinámica (hoy)
OVERALL_END_DATE = dt.datetime.now(tz.gettz(TZ_STR)).date().isoformat()


# Variables HORARIAS normalizadas
HOURLY_VARS = [
    "temperature_2m", "precipitation", "rain",
    "wind_speed_10m", "wind_gusts_10m"
]
UNITS = {"temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm"}
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"

# Throttling / backoff
BASE_SLEEP_S = 1.8
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
COOL_DOWN_429 = 70

# Nombres de archivos de salida
OUT_CSV = "clima_por_comuna_horario.csv"
OUT_XLSX = "clima_por_comuna_horario.xlsx"
ERR_GEO_CSV = "errores_geograficos.csv"
FAILED_CSV = "fallidas.csv"
RETRIED_CSV = "reintentadas.csv"

# ---------- Lector robusto de CSV ----------
def read_locations(bytes_blob):
    enc_guess = chardet.detect(bytes_blob).get("encoding") or "utf-8"
    attempts = [
        (enc_guess, None), (enc_guess, ";"), ("utf-8-sig", None), ("utf-8-sig", ";"),
        ("latin-1", None), ("latin-1", ";"), ("cp1252", None), ("cp1252", ";"),
    ]
    last_err = None
    for enc, sep in attempts:
        try:
            df = pd.read_csv(io.BytesIO(bytes_blob), encoding=enc, sep=sep, engine="python")
            df.columns = [c.strip().lower() for c in df.columns]
            if "comunas" in df.columns and "comuna" not in df.columns:
                df = df.rename(columns={"comunas": "comuna"})
            req = {"comuna", "lat", "lon"}
            if not req.issubset(df.columns): continue
            for col in ("lat", "lon"):
                df[col] = (df[col].astype(str).str.replace(",", ".", regex=False).str.strip())
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["comuna", "lat", "lon"]).reset_index(drop=True)
            return df
        except Exception as e:
            last_err = e
    raise last_err or ValueError("No se pudo leer el CSV con ningún intento")

# ---------- FUENTE 1: OPEN-METEO (ARCHIVE API) ----------
# <-- CAMBIO: Acepta start y end para poder llamarla en bucle
def fetch_openmeteo_archive(client, lat, lon, start_date, end_date):
    params = {
        "latitude": float(lat), "longitude": float(lon),
        "start_date": start_date, "end_date": end_date,
        "timezone": TZ_STR, "hourly": HOURLY_VARS, **UNITS
    }
    return client.weather_api(OM_ARCHIVE_URL, params=params)[0]

def build_from_openmeteo(resp, comuna):
    hourly = resp.Hourly()
    dt_index = pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(TZ_STR)
    data = {"datetime": dt_index, **{var: hourly.Variables(idx).ValuesAsNumpy() for idx, var in enumerate(HOURLY_VARS)}}
    df = pd.DataFrame(data)
    df["fecha"] = df["datetime"].dt.date.astype(str)
    df["hora"] = df["datetime"].dt.strftime("%H:%M")
    df.insert(0, "comuna", comuna)
    return df[["comuna", "fecha", "hora"] + HOURLY_VARS]

# ---------- FUENTE 2: METEOSTAT (FALLBACK) ----------
# <-- CAMBIO: Acepta start y end para poder llamarla en bucle
def fetch_meteostat(lat, lon, start_date, end_date):
    start = dt.datetime.fromisoformat(start_date)
    end = dt.datetime.fromisoformat(end_date)
    loc = MsPoint(float(lat), float(lon))
    df = MsHourly(loc, start, end, timezone=TZ_STR).fetch()
    if df is None or df.empty: return None
    out = pd.DataFrame({
        "temperature_2m": df.get("temp"), "precipitation": df.get("prcp"),
        "rain": df.get("prcp"), "wind_speed_10m": df.get("wspd"),
        "wind_gusts_10m": df.get("wpgt")
    })
    out["datetime"] = out.index
    return out

def build_from_meteostat(df_ms, comuna):
    df = df_ms.copy()
    for c in HOURLY_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype('Float64')
    df["fecha"] = pd.to_datetime(df["datetime"]).dt.date.astype(str)
    df["hora"] = pd.to_datetime(df["datetime"]).dt.strftime("%H:%M")
    df.insert(0, "comuna", comuna)
    return df[["comuna", "fecha", "hora"] + HOURLY_VARS]

# ---------- Funciones de I/O y Utilidades ----------
def safe_sheet_name(name: str) -> str:
    s = str(name); s = s.translate(str.maketrans({c: ' ' for c in '[]*?/\\:'}))
    return s[:31]

def guardar_datos(df_chunk, comuna, workbook, is_first_write):
    df_chunk.to_csv(OUT_CSV, index=False, encoding="utf-8", header=is_first_write, mode="w" if is_first_write else "a")
    ws = workbook.create_sheet(title=safe_sheet_name(comuna))
    for r in dataframe_to_rows(df_chunk, index=False, header=True):
        sanitized_row = [None if cell is pd.NA else cell for cell in r]
        try:
            ws.append(sanitized_row)
        except IllegalCharacterError:
            ws.append([str(c).replace("\x00", "") if c is not None else "" for c in sanitized_row])
    return False

# ---------- Loop Principal ----------
def main():
    print("🔼 Sube tu CSV con 'comuna,lat,lon'…")
    uploaded = files.upload()
    if not uploaded: raise RuntimeError("No se subió ningún archivo.")

    df_locations = read_locations(list(uploaded.values())[0])

    bad_geo = df_locations[(df_locations["lat"].abs() > 90) | (df_locations["lon"].abs() > 180)]
    if not bad_geo.empty:
        bad_geo.to_csv(ERR_GEO_CSV, index=False, encoding="utf-8")
        print(f"⚠️ Coordenadas fuera de rango guardadas en {ERR_GEO_CSV} (se excluirán).")
        df_locations = df_locations[~df_locations.index.isin(bad_geo.index)].reset_index(drop=True)

    df_locations = df_locations.drop_duplicates(subset=["comuna", "lat", "lon"]).reset_index(drop=True)
    print(f"📍 Ubicaciones válidas a procesar: {len(df_locations)}")

    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600*24*7)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.6)
    om_client = openmeteo_requests.Client(session=retry_session)

    wb = Workbook(); wb.remove(wb.active)
    fallidas, reintentadas_ok = [], []
    header_ya_escrito = False

    # <-- CAMBIO: Generar rangos de fechas anuales para hacer las peticiones en trozos
    date_chunks = pd.date_range(start=OVERALL_START_DATE, end=OVERALL_END_DATE, freq='AS') # AS = Año-Start
    date_ranges = []
    for i, start_date in enumerate(date_chunks):
        end_date = date_chunks[i+1] - dt.timedelta(days=1) if i + 1 < len(date_chunks) else pd.to_datetime(OVERALL_END_DATE)
        date_ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))


    for i, row in df_locations.iterrows():
        comuna, lat, lon = row["comuna"], row["lat"], row["lon"]
        print(f"[{i+1}/{len(df_locations)}] Processing {comuna}...")

        all_data_for_location = [] # <-- Lista para guardar los dataframes de cada año
        fuente_exitosa_final = "N/A"

        # <-- CAMBIO: Bucle anidado para iterar sobre cada rango de fechas (cada año)
        for start_chunk, end_chunk in date_ranges:
            print(f"  ↳ Obteniendo datos para el período: {start_chunk} a {end_chunk}")
            df_chunk, fuente_exitosa, necesito_log_reintento = None, None, False
            delay = BASE_SLEEP_S

            # --- 1) Intentar Open-Meteo (Archive API) ---
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df_chunk = build_from_openmeteo(fetch_openmeteo_archive(om_client, lat, lon, start_chunk, end_chunk), comuna)
                    fuente_exitosa = "OM-Archive"
                    if attempt > 1: necesito_log_reintento = True
                    break
                except requests.exceptions.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 429:
                        print(f"    - OM-Archive Límite API. Esperando {COOL_DOWN_429}s…")
                        time.sleep(COOL_DOWN_429)
                    else:
                        print(f"    - OM-Archive HTTP {status}. Reintento {attempt}/{MAX_RETRIES} en {delay:.1f}s…")
                        time.sleep(delay); delay *= BACKOFF_FACTOR
                except Exception as e:
                    print(f"    - OM-Archive Error: {e}. Reintento {attempt}/{MAX_RETRIES} en {delay:.1f}s…")
                    time.sleep(delay); delay *= BACKOFF_FACTOR

            # --- 2) Si falló, intentar con Meteostat ---
            if df_chunk is None:
                print(f"    ↳ OM-Archive falló. Intentando fallback con Meteostat...")
                try:
                    df_ms = fetch_meteostat(lat, lon, start_chunk, end_chunk)
                    if df_ms is not None and not df_ms.empty:
                        df_chunk, fuente_exitosa = build_from_meteostat(df_ms, comuna), "Meteostat"
                except Exception as e:
                    print(f"    ↳ MS ERROR: {e}")

            if df_chunk is not None:
                all_data_for_location.append(df_chunk)
                fuente_exitosa_final = fuente_exitosa
                if necesito_log_reintento and comuna not in reintentadas_ok:
                    reintentadas_ok.append(comuna)

            time.sleep(BASE_SLEEP_S) # Pausa entre cada petición anual

        # --- 3) Procesar el resultado final para la comuna ---
        if all_data_for_location:
            # <-- CAMBIO: Concatenar todos los trozos anuales en un solo DataFrame
            df_final = pd.concat(all_data_for_location, ignore_index=True)
            header_ya_escrito = guardar_datos(df_final, comuna, wb, not header_ya_escrito)
            print(f"[{i+1}/{len(df_locations)}] ✅ OK ({fuente_exitosa_final}): {comuna} ({lat:.5f},{lon:.5f}) - {len(df_final)} filas obtenidas.")
        else:
            print(f"[{i+1}/{len(df_locations)}] ❌ FALLÓ: {comuna} con ambos proveedores para todos los períodos.")
            fallidas.append(comuna)

    # --- Guardar archivos finales y logs ---
    wb.save(OUT_XLSX)
    if fallidas: pd.DataFrame({"comuna": fallidas}).to_csv(FAILED_CSV, index=False, encoding="utf-8")
    if reintentadas_ok: pd.DataFrame({"comuna": reintentadas_ok}).to_csv(RETRIED_CSV, index=False, encoding="utf-8")

    print("\n✅ Proceso finalizado.")
    print(f" - CSV combinado: {OUT_CSV}")
    print(f" - Excel multi-hojas: {OUT_XLSX}")
    if fallidas: print(f" - Fallidas: {FAILED_CSV} ({len(fallidas)})")
    if reintentadas_ok: print(f" - Reintentadas y OK: {RETRIED_CSV} ({len(reintentadas_ok)})")

    files.download(OUT_CSV); files.download(OUT_XLSX)
    if fallidas: files.download(FAILED_CSV)
    if reintentadas_ok: files.download(RETRIED_CSV)
    if not bad_geo.empty: files.download(ERR_GEO_CSV)

if __name__ == "__main__":
    main()
