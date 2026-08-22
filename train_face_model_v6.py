import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================================================
# REPRODUCIBILITY & SETTINGS
# =========================================================

tf.keras.utils.set_random_seed(42)

DATASET_DIR = "face_dataset_v6"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_face_model_v6.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("TRAINING FACE DEEPFAKE MODEL V6 (TWO-PHASE TRANSFER LEARNING)")
print("=" * 70)

# =========================================================
# DATASET LOADING
# =========================================================

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

for d in [train_dir, val_dir, test_dir]:
    if not os.path.exists(d):
        raise FileNotFoundError(f"Missing dataset split directory: {d}")

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

# Label Verification
print("\n" + "-" * 70)
print("CLASS MAPPING:")
print("-" * 70)
print(f"Class Names: {train_ds.class_names}")
print("Index 0 = FAKE")
print("Index 1 = REAL")
print("Model Output: P(REAL) via Sigmoid")

# Count samples for class weights
fake_train_count = len(os.listdir(os.path.join(train_dir, "fake")))
real_train_count = len(os.listdir(os.path.join(train_dir, "real")))
total_train = fake_train_count + real_train_count

weight_for_fake = total_train / (2.0 * fake_train_count)
weight_for_real = total_train / (2.0 * real_train_count)
class_weights = {0: weight_for_fake, 1: weight_for_real}

print(f"\nTrain Class Counts: FAKE={fake_train_count}, REAL={real_train_count}")
print(f"Calculated Class Weights: {class_weights}")

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================================================
# DATA AUGMENTATION (GENTLE FOR FACES)
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="face_augmentation")

# =========================================================
# MODEL ARCHITECTURE
# =========================================================

print("\n2. Building Model Architecture...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze for Phase 1

inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)
x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling")(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.40, name="dropout_1")(x)
x = layers.Dense(128, activation="relu", name="dense_128")(x)
x = layers.Dropout(0.30, name="dropout_2")(x)
outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

model = models.Model(inputs, outputs, name="MobileNetV2_Deepfake_V6")

# =========================================================
# PHASE 1: TRAIN CLASSIFICATION HEAD
# =========================================================

print("\n" + "=" * 70)
print("PHASE 1: TRAINING CLASSIFICATION HEAD (BASE MODEL FROZEN)")
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
# PHASE 2: FINE-TUNING TOP MOBILENETV2 LAYERS
# =========================================================

print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING TOP LAYERS OF MOBILENETV2")
print("=" * 70)

# Unfreeze base model and freeze lower layers
base_model.trainable = True
fine_tune_at = 120  # Unfreeze layers from index 120 onwards

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

print(f"Total layers in MobileNetV2: {len(base_model.layers)}")
print(f"Trainable layers in MobileNetV2: {len(base_model.layers) - fine_tune_at}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
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
# LOAD BEST MODEL & TEST SET EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("LOADING BEST V6 MODEL & INITIAL TEST EVALUATION")
print("=" * 70)

best_model = tf.keras.models.load_model(MODEL_PATH)

results = best_model.evaluate(test_ds, verbose=1)
for name, val in zip(best_model.metrics_names, results):
    print(f"  Test {name.upper()}: {val:.4f}")

print("\nBest V6 model saved successfully at:", MODEL_PATH)
print("=" * 70)