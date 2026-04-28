from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "dog" / "breed"

valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

bad_files = []

for file in DATASET_DIR.rglob("*"):
    if file.is_file():
        if file.suffix.lower() not in valid_ext:
            bad_files.append(file)
            continue

        try:
            with Image.open(file) as img:
                img.verify()
        except Exception:
            bad_files.append(file)

print(f"Bad files found: {len(bad_files)}")

for file in bad_files:
    print("Deleting:", file)
    file.unlink()

print("✅ Bad images removed")