import json
import random
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from pathlib import Path

# Get project root automatically
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "animal" / "type"
MODEL_PATH = BASE_DIR / "models" / "animal_type_model.keras"
CLASS_PATH = BASE_DIR / "models" / "animal_type_classes.json"

MODEL_PATH = MODELS_DIR / "animal_type_model.keras"
CLASS_PATH = MODELS_DIR / "animal_type_classes.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
VAL_SPLIT = 0.2
SEED = 42
CLASS_NAMES = ["cat", "cow", "dog", "horse", "buffalo"]



def collect_examples():
    examples = []

    for label_index, animal_name in enumerate(CLASS_NAMES):
        breed_root = DATASETS_DIR / animal_name / "breed"
        if not breed_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {breed_root}")

        for file_path in breed_root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                examples.append((str(file_path), label_index))

    if not examples:
        raise RuntimeError("No training images found for animal type model.")

    random.Random(SEED).shuffle(examples)
    return examples


def decode_and_resize(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(samples, training):
    paths = [item[0] for item in samples]
    labels = [item[1] for item in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmenter = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.2),
        ])

        def augment(image, label):
            return augmenter(image, training=True), label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(min(len(samples), 2048), seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    samples = collect_examples()
    split_index = int(len(samples) * (1 - VAL_SPLIT))
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    train_ds = build_dataset(train_samples, training=True)
    val_ds = build_dataset(val_samples, training=False)

    model = build_model(len(CLASS_NAMES))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Training animal detector with {len(train_samples)} train and {len(val_samples)} val images")
    print(f"Animal classes: {CLASS_NAMES}")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with open(CLASS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(CLASS_NAMES, file_obj, indent=2)

    print(f"Saved animal detector to {MODEL_PATH}")


if __name__ == "__main__":
    main()
