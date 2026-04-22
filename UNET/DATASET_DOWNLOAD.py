import os
import requests
from pathlib import Path

# Zenodo personal access token am besten als Umgebungsvariable setzen:
# Linux/macOS:
# export ZENODO_TOKEN="dein_token"
#
# Windows PowerShell:
# $env:ZENODO_TOKEN="dein_token"

ACCESS_TOKEN = os.getenv("ZENODO_TOKEN")
RECORD_ID = "14223624"   # LUNA25 Imaging Data
OUTPUT_DIR = Path("luna25_downloads")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

headers = {}
if ACCESS_TOKEN:
    headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

# 1) Record-Metadaten holen
meta_url = f"https://zenodo.org/api/records/{RECORD_ID}"
resp = requests.get(meta_url, headers=headers, timeout=60)
resp.raise_for_status()

record = resp.json()
files = record.get("files", [])

print(f"Gefundene Dateien: {len(files)}")

# 2) Dateien herunterladen
for i, f in enumerate(files, start=1):
    filename = f["key"]
    url = f["links"]["self"]
    out_path = OUTPUT_DIR / filename

    if out_path.exists():
        print(f"[{i}/{len(files)}] Überspringe bereits vorhandene Datei: {filename}")
        continue

    print(f"[{i}/{len(files)}] Lade herunter: {filename}")

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as out_file:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)

    print(f"Fertig: {filename}")

print("Alle Downloads abgeschlossen.")