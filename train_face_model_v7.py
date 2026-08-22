import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================================================
# REPRODUCIBILITY & SETTINGS
# =========================================================

tf.keras.utils.set_random_seed(42)

DATASET_DIR = "face_dataset_v7"
MODEL_DIR = "models"
REPORTS_DIR = "reports"

MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_face_model_v7.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("TRAINING FACE DEEPFAKE MODEL V7 (TRANSFER LEARNING + FROZEN BN)")
print("=" * 70)

# =========================================================
# 1. LOAD DATASET
# =========================================================

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

print("\n1. Loading Datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Class Names: {train_ds.class_names} (0 = FAKE, 1 = REAL, Output = P(REAL))")

# Class weights
fake_count = len(os.listdir(os.path.join(train_dir, "fake")))
real_count = len(os.listdir(os.path.join(train_dir, "real")))
total_count = fake_count + real_count

weight_fake = total_count / (2.0 * fake_count)
weight_real = total_count / (2.0 * real_count)
class_weights = {0: weight_fake, 1: weight_real}
print(f"Train Counts: FAKE={fake_count}, REAL={real_count} | Class Weights: {class_weights}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# 2. MODEL ARCHITECTURE
# =========================================================

print("\n2. Building Model Architecture...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="face_augmentation_v7")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3), name="input_image")
x = data_augmentation(inputs)
x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling_to_neg1_pos1")(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.40, name="dropout_1")(x)
x = layers.Dense(128, activation="relu", name="dense_128")(x)
x = layers.Dropout(0.30, name="dropout_2")(x)
outputs = layers.Dense(1, activation="sigmoid", name="sigmoid_out")(x)

model = models.Model(inputs, outputs, name="Deepfake_MobileNetV2_V7")

# =========================================================
# 3. PHASE 1: TRAIN CLASSIFICATION HEAD
# =========================================================

print("\n" + "=" * 70)
print("PHASE 1: TRAINING CLASSIFICATION HEAD (BASE FROZEN)")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks_p1 = [
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

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_p1
)

# =========================================================
# 4. PHASE 2: CONTROLLED FINE-TUNING (KEEP BATCHNORM FROZEN)
# =========================================================

print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING TOP 25 MOBILENETV2 LAYERS (BATCHNORM FROZEN)")
print("=" * 70)

base_model.trainable = True
fine_tune_start = 125  # Last ~29 layers

for layer in base_model.layers[:fine_tune_start]:
    layer.trainable = False

# Unfreeze top layers except BatchNorm
unfrozen_count = 0
bn_frozen_count = 0

for layer in base_model.layers[fine_tune_start:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
        bn_frozen_count += 1
    else:
        layer.trainable = True
        unfrozen_count += 1

print(f"Total Base Layers: {len(base_model.layers)}")
print(f"Unfrozen Feature Layers: {unfrozen_count}")
print(f"Frozen BatchNormalization Layers in Fine-Tuning Block: {bn_frozen_count}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks_p2 = [
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
        min_lr=1e-7,
        verbose=1
    )
]

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_p2
)

# =========================================================
# 5. COMBINE & SAVE TRAINING CURVES
# =========================================================

print("\nGenerating and saving training curves to 'reports/'...")

def combine_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

acc = combine_histories(history_p1, history_p2, "accuracy")
val_acc = combine_histories(history_p1, history_p2, "val_accuracy")
loss = combine_histories(history_p1, history_p2, "loss")
val_loss = combine_histories(history_p1, history_p2, "val_loss")
auc = combine_histories(history_p1, history_p2, "auc")
val_auc = combine_histories(history_p1, history_p2, "val_auc")

epochs_range = range(1, len(acc) + 1)
p1_len = len(history_p1.history.get("accuracy", []))

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label="Training Accuracy", color="royalblue", lw=2)
plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="darkorange", lw=2)
if p1_len < len(epochs_range):
    plt.axvline(x=p1_len, color="gray", linestyle="--", label="Start Fine-Tuning")
plt.title("V7 Training and Validation Accuracy", fontsize=12, fontweight="bold")
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "v7_training_accuracy.png"), dpi=300)
plt.close()

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label="Training Loss", color="royalblue", lw=2)
plt.plot(epochs_range, val_loss, label="Validation Loss", color="darkorange", lw=2)
if p1_len < len(epochs_range):
    plt.axvline(x=p1_len, color="gray", linestyle="--", label="Start Fine-Tuning")
plt.title("V7 Training and Validation Loss", fontsize=12, fontweight="bold")
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Binary Crossentropy Loss", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "v7_training_loss.png"), dpi=300)
plt.close()

# Plot AUC
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, auc, label="Training AUC", color="royalblue", lw=2)
plt.plot(epochs_range, val_auc, label="Validation AUC", color="darkorange", lw=2)
if p1_len < len(epochs_range):
    plt.axvline(x=p1_len, color="gray", linestyle="--", label="Start Fine-Tuning")
plt.title("V7 Training and Validation ROC-AUC", fontsize=12, fontweight="bold")
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("ROC-AUC", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "v7_training_auc.png"), dpi=300)
plt.close()

print("Training plots saved:")
print("  - reports/v7_training_accuracy.png")
print("  - reports/v7_training_loss.png")
print("  - reports/v7_training_auc.png")

# =========================================================
# 6. LOAD & VERIFY BEST MODEL
# =========================================================

best_v7_model = tf.keras.models.load_model(MODEL_PATH)
print("\n" + "=" * 70)
print(f"BEST V7 MODEL SAVED SUCCESSFULLY: {MODEL_PATH}")
print("=" * 70)