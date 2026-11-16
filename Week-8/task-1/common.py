"""Ortak yardimcilar: model indirme ve yol tanimlari."""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "outputs"


def download_model(url: str) -> str:
    """Modeli models/ klasorune indirir (varsa tekrar indirmez), yolunu dondurur."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / url.split("/")[-1]
    if not path.exists():
        print(f"Model indiriliyor: {url}")
        urllib.request.urlretrieve(url, path)
        print(f"Kaydedildi: {path}")
    return str(path)


def asset_path(name: str) -> str:
    """assets/ klasorundeki dosyanin yolunu dondurur; yoksa anlasilir hata verir."""
    path = ASSETS_DIR / name
    if not path.exists():
        raise SystemExit(f"Dosya bulunamadi: {path}\nLutfen assets/ klasorune '{name}' dosyasini koy.")
    return str(path)
