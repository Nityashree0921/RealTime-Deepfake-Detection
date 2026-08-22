import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "face_dataset_v7"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

MODEL_PATH = "models/deepfake_face_model_v7.keras"

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRAINING FACE DEEPFAKE MODEL V7")
print("=" * 70)

# ============================================================
# CHECK DATASET
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not os.path.exists(folder):
        print("\nERROR: Dataset folder not found:")
        print(folder)
        raise SystemExit

print("\nDataset found:")
print(DATASET_DIR)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading TRAIN dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("\nLoading VALIDATION dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading TEST dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# CLASS MAPPING
# ============================================================

print("\n" + "=" * 70)
print("CLASS MAPPING")
print("=" * 70)

print("Class names:", train_ds.class_names)
print("0 = FAKE")
print("1 = REAL")

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ============================================================
# BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

print("MobileNetV2 loaded.")
print("Pretrained layers frozen.")

# ============================================================
# MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="face_input"
)

x = data_augmentation(inputs)

# MobileNetV2 expects pixels in [-1, 1]

x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="deepfake_probability"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="DeepfakeFaceModelV7"
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

callbacks = [

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
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING V7 TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL V7 TEST")
print("=" * 70)

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nTEST RESULTS")

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        f"{name}: {value:.4f}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n" + "=" * 70)
print("V7 TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved:")
print(MODEL_PATH)