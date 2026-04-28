from common import load_datasets, build_model, save_class_names
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "cow" / "breed"
MODEL_PATH = BASE_DIR / "models" / "cow_breed_model.keras"
CLASS_PATH = BASE_DIR / "models" / "cow_breed_classes.json"

train_obj, val_obj, class_names = load_datasets(DATASET_DIR)

print("Classes:", class_names)

save_class_names(class_names, CLASS_PATH)

model = build_model(len(class_names))

model.fit(
    train_obj.dataset,
    validation_data=val_obj.dataset,
    epochs=15
)

model.save(MODEL_PATH)

print("✅ Cow breed model trained")