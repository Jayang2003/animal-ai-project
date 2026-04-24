import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight


class DatasetWithClassNames:
    """Wrapper to preserve class_names attribute through transformations"""
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def load_datasets(data_dir, img_size=(224, 224), batch_size=16):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Save class names before transformations
    class_names = train_ds.class_names

    # Add data augmentation to training dataset
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.3),  # Increased from 0.2 for small datasets
        layers.RandomContrast(0.3),    # Increased from 0.2 for small datasets
    ])

    def augment(images, labels):
        return data_augmentation(images, training=True), labels

    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Wrap datasets to preserve class_names
    train_ds = DatasetWithClassNames(train_ds, class_names)
    val_ds = DatasetWithClassNames(val_ds, class_names)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    """Build EfficientNetB0 model with proper architecture.
    
    Initial state: Base model FROZEN for Phase 1 training.
    To unfreeze for Phase 2: Call unfreeze_base_model()
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Phase 1: Freeze entire base for first training phase
    base_model.trainable = False

    # Build improved classifier head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def unfreeze_base_model(model, num_layers_to_unfreeze=50):
    """Unfreeze last N layers of base model for Phase 2 fine-tuning."""
    base_model = model.layers[0]  # EfficientNetB0 is first layer
    base_model.trainable = True
    
    # Freeze all but last N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Compile with lower learning rate for Phase 2
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def get_class_weights(train_ds, num_classes):
    """Compute class weights to handle imbalanced datasets."""
    # Collect all labels from training dataset
    labels = []
    for _, batch_labels in train_ds:
        labels.extend(batch_labels.numpy())
    
    labels = np.array(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return {i: weight for i, weight in enumerate(class_weights)}


def get_callbacks(patience_early_stop=15):
    """Return callbacks for better training."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]


def save_class_names(class_names, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)