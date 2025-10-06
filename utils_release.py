import os
import requests

def download_asset_from_latest(owner, repo, asset_name, target_dir):
    """
    Descarga un 'asset' (archivo) desde el release más reciente de un repositorio de GitHub.

    Args:
        owner (str): El nombre del propietario del repositorio.
        repo (str): El nombre del repositorio.
        asset_name (str): El nombre exacto del archivo a descargar (ej: "modelo.pkl").
        target_dir (str): El directorio local donde se guardará el archivo.
    """
    # Obtener el token de GitHub desde las variables de entorno (así funciona en GitHub Actions)
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ADVERTENCIA: GITHUB_TOKEN no encontrado. Las descargas pueden fallar por límites de tasa.")
        headers = {}
    else:
        headers = {"Authorization": f"token {token}"}

    # URL para obtener la información del último release
    latest_release_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"

    print(f"Buscando el último release en {owner}/{repo}...")
    response = requests.get(latest_release_url, headers=headers)
    response.raise_for_status()  # Se detendrá si hay un error (ej: 404 Not Found)
    
    release_data = response.json()
    assets = release_data.get("assets", [])

    asset_url = None
    for asset in assets:
        if asset["name"] == asset_name:
            asset_url = asset["url"]
            break

    if not asset_url:
        raise FileNotFoundError(f"No se pudo encontrar el asset '{asset_name}' en el último release.")

    print(f"Descargando '{asset_name}'...")
    
    # Para descargar el asset, se necesita un header específico
    headers["Accept"] = "application/octet-stream"
    response = requests.get(asset_url, headers=headers, stream=True)
    response.raise_for_status()

    # Guardar el archivo en el directorio de destino
    target_path = os.path.join(target_dir, asset_name)
    os.makedirs(target_dir, exist_ok=True)
    
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"'{asset_name}' descargado exitosamente en '{target_path}'")

if __name__ == '__main__':
    # Ejemplo de cómo usar la función (no se ejecuta en producción)
    # Reemplaza con tus datos para probarlo localmente si es necesario
    # NOTA: Para probarlo en tu PC, necesitarías crear un token de GitHub y configurarlo
    # como una variable de entorno llamada GITHUB_TOKEN.
    
    # TEST_OWNER = "Supervision-Inbound"
    # TEST_REPO = "wf-Analytics-AI2.5"
    # TEST_ASSET = "modelo_llamadas_nn.h5" # Cambia esto por un asset que exista
    # TEST_DIR = "models_test"
    
    # print("--- Ejecutando prueba de descarga ---")
    # try:
    #     download_asset_from_latest(TEST_OWNER, TEST_REPO, TEST_ASSET, TEST_DIR)
    # except Exception as e:
    #     print(f"Error en la prueba: {e}")
    pass
