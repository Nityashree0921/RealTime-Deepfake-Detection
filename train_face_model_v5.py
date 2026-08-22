import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = "face_dataset_v5"
MODEL_DIR = "models"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_face_model_v5.keras"
)

# =========================================================
# CHECK DATASET
# =========================================================

print("=" * 60)
print("TRAINING FACE MODEL V5")
print("=" * 60)

if not os.path.exists(DATASET_DIR):
    print("ERROR: Dataset not found:")
    print(DATASET_DIR)
    exit()

print("\nDataset:", DATASET_DIR)

# =========================================================
# LOAD DATASET
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass names:")
print(train_ds.class_names)

# =========================================================
# PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# =========================================================
# BASE MODEL
# =========================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze most of the network
base_model.trainable = False

print("Base model loaded.")

# =========================================================
# MODEL
# =========================================================

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 expects pixels in [-1, 1]
x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("\nModel created.")
model.summary()

# =========================================================
# CALLBACKS
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================================================
# TRAIN
# =========================================================

print("\n")
print("=" * 60)
print("STARTING V5 TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\n")
print("=" * 60)
print("LOADING BEST V5 MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# =========================================================
# FINAL TEST
# =========================================================

print("\n")
print("=" * 60)
print("FINAL V5 TEST")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        f"{name}: {value:.4f}"
    )

print("\n")
print("=" * 60)
print("V5 TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved:")
print(MODEL_PATH)