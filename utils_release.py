import os
import requests

def download_assets(owner, repo, asset_name=None, pattern=None, target_dir="models"):
    """
    Descarga uno o varios assets desde el último release de GitHub.

    - Si se entrega asset_name, descarga solo ese archivo exacto.
    - Si se entrega pattern, descarga todos los archivos que contengan ese patrón.
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    if not token:
        print("ADVERTENCIA: GITHUB_TOKEN no encontrado. Las descargas pueden fallar.")

    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    print(f"🔍 Buscando último release en {owner}/{repo}...")
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    release = r.json()
    assets = release.get("assets", [])

    # Filtro: uno exacto o por patrón
    if asset_name:
        selected = [a for a in assets if a["name"] == asset_name]
    elif pattern:
        selected = [a for a in assets if pattern in a["name"]]
    else:
        raise ValueError("Debes especificar 'asset_name' o 'pattern'.")

    if not selected:
        raise FileNotFoundError("No se encontraron assets que coincidan.")

    os.makedirs(target_dir, exist_ok=True)
    headers["Accept"] = "application/octet-stream"

    for asset in selected:
        name = asset["name"]
        print(f"⬇️ Descargando {name}...")
        url = asset["url"]
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        path = os.path.join(target_dir, name)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Guardado en {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--asset", help="Nombre exacto del archivo a descargar")
    parser.add_argument("--pattern", help="Substring para descargar múltiples archivos")
    parser.add_argument("--target", default="models")
    args = parser.parse_args()

    download_assets(args.owner, args.repo, asset_name=args.asset, pattern=args.pattern, target_dir=args.target)

