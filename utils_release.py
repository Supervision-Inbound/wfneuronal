import os
import time
import requests

def _get_latest_release_json(owner, repo, headers=None):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    r = requests.get(url, headers=headers or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def _download_stream(url, target_path, headers=None, chunk_size=8192, retries=3, backoff=2.0):
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers or {}, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)

def download_asset_from_latest(owner, repo, asset_name, target_dir):
    """
    Descarga un asset del último Release público de GitHub usando browser_download_url.
    Si el repo es privado, define GITHUB_TOKEN en el entorno y seguirá funcionando.
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    if not token:
        print("INFO: GITHUB_TOKEN no definido. Intentaré descarga pública.")

    print(f"Buscando el último release en {owner}/{repo}…")
    release_data = _get_latest_release_json(owner, repo, headers=headers)
    assets = release_data.get("assets", []) or []

    browser_url = None
    for a in assets:
        if a.get("name") == asset_name:
            browser_url = a.get("browser_download_url")
            break

    if not browser_url:
        names = [a.get("name") for a in assets]
        raise FileNotFoundError(
            f"No se encontró el asset '{asset_name}' en el último release de '{owner}/{repo}'. "
            f"Assets disponibles: {names}"
        )

    target_path = os.path.join(target_dir, asset_name)
    print(f"Descargando '{asset_name}'…")
    _download_stream(browser_url, target_path, headers=None)
    print(f"OK: '{asset_name}' descargado en '{target_path}'")
    return target_path

if __name__ == '__main__':
    pass
