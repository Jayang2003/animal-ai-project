import os
from common import load_datasets, build_model, save_class_names

DATA_DIR = r"D:\animal_ai_project\datasets\cow\breed"
MODEL_PATH = r"D:\animal_ai_project\models\cow_breed_model.keras"
CLASS_PATH = r"D:\animal_ai_project\models\cow_breed_classes.json"

train_ds, val_ds = load_datasets(DATA_DIR)

class_names = train_ds.class_names
print("Classes:", class_names)

save_class_names(class_names, CLASS_PATH)

model = build_model(len(class_names))

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

model.save(MODEL_PATH)

print("✅ Cow breed model trained")