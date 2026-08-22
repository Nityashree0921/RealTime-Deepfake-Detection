import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 16

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_face_model_v2.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

images = []
labels = []

print("=" * 60)
print("LOADING CROPPED FACE DATASET")
print("=" * 60)


def load_images(folder, label):

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(f"{folder}: {len(files)} images")

    for i, filename in enumerate(files):

        path = os.path.join(folder, filename)

        image = tf.keras.utils.load_img(
            path,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        image = tf.keras.utils.img_to_array(image)

        images.append(image)
        labels.append(label)

        if (i + 1) % 100 == 0:
            print(
                f"Loaded {i + 1}/{len(files)}"
            )


# REAL = 0
# FAKE = 1

load_images(REAL_DIR, 0)
load_images(FAKE_DIR, 1)


# =========================================================
# CONVERT DATA
# =========================================================

X = np.array(
    images,
    dtype="float32"
)

y = np.array(
    labels,
    dtype="float32"
)


print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)

print(
    "REAL samples:",
    np.sum(y == 0)
)

print(
    "FAKE samples:",
    np.sum(y == 1)
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.08),

    layers.RandomZoom(0.15),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomContrast(0.15),

])


# =========================================================
# MOBILENETV2
# =========================================================

print("\n")
print("=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)


base_model = MobileNetV2(

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze initially

base_model.trainable = False


# =========================================================
# BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


x = data_augmentation(inputs)


x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# =========================================================
# COMPILE STAGE 1
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1
    )
]


# =========================================================
# STAGE 1
# =========================================================

print("\n")
print("=" * 60)
print("STAGE 1: TRANSFER LEARNING")
print("=" * 60)


history1 = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=INITIAL_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    shuffle=True
)


# =========================================================
# STAGE 2
# FINE-TUNING
# =========================================================

print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING")
print("=" * 60)


base_model.trainable = True


# Keep first 100 layers frozen

fine_tune_at = 100


for layer in base_model.layers[
    :fine_tune_at
]:

    layer.trainable = False


print(
    "Trainable MobileNetV2 layers:",
    sum(
        layer.trainable
        for layer in base_model.layers
    )
)


# =========================================================
# RECOMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# =========================================================
# FINE-TUNE
# =========================================================

history2 = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=FINE_TUNE_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    shuffle=True
)


# =========================================================
# FINAL EVALUATION
# =========================================================

print("\n")
print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)


results = model.evaluate(

    X_val,

    y_val,

    verbose=1
)


print("\nValidation Results:")

for name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{name}: {value:.4f}"
    )


# =========================================================
# SAVE
# =========================================================

model.save(
    MODEL_PATH
)


print("\n")
print("=" * 60)
print("FACE MODEL TRAINING COMPLETED")
print("=" * 60)

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)