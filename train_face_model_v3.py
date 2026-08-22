import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 16

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

TRAIN_DIR = "face_dataset/train"
VAL_DIR = "face_dataset/val"
TEST_DIR = "face_dataset/test"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_face_model_v3.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# LOAD DATASETS
# =========================================================

print("=" * 60)
print("LOADING VIDEO-LEVEL DATASET")
print("=" * 60)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Class names:", train_ds.class_names)

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

print()
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

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

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
        learning_rate=1e-4
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

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
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

print()
print("=" * 60)
print("STAGE 1: TRANSFER LEARNING")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# =========================================================
# STAGE 2
# FINE-TUNING
# =========================================================

print()
print("=" * 60)
print("STAGE 2: FINE-TUNING")
print("=" * 60)

base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
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
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# =========================================================
# LOAD BEST MODEL
# =========================================================

print()
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# =========================================================
# TEST EVALUATION
# =========================================================

print()
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)

print()
print("Test Results:")

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        f"{name}: {value:.4f}"
    )

# =========================================================
# COLLECT TEST PREDICTIONS
# =========================================================

print()
print("=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    ).flatten()

    y_true.extend(
        labels.numpy().flatten()
    )

    y_pred.extend(
        predictions
    )

y_true = tf.convert_to_tensor(
    y_true
).numpy()

y_pred = tf.convert_to_tensor(
    y_pred
).numpy()

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

print()
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

best_threshold = 0.5
best_f1 = 0.0

for threshold in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]:

    predicted_labels = (
        y_pred >= threshold
    ).astype(int)

    tp = ((predicted_labels == 1) &
          (y_true == 1)).sum()

    fp = ((predicted_labels == 1) &
          (y_true == 0)).sum()

    fn = ((predicted_labels == 0) &
          (y_true == 1)).sum()

    tn = ((predicted_labels == 0) &
          (y_true == 0)).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (tp + tn) /
        len(y_true)
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={accuracy:.3f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# =========================================================
# CONFUSION MATRIX
# =========================================================

final_predictions = (
    y_pred >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_true,
    final_predictions
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        final_predictions,
        target_names=[
            "FAKE",
            "REAL"
        ]
    )
)

# =========================================================
# ROC AUC
# =========================================================

auc = roc_auc_score(
    y_true,
    y_pred
)

print()
print("=" * 60)
print("ROC-AUC")
print("=" * 60)

print(
    f"ROC-AUC: {auc:.4f}"
)

# =========================================================
# BEST THRESHOLD
# =========================================================

print()
print("=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Best F1 score: {best_f1:.4f}"
)

# =========================================================
# SAVE THRESHOLD
# =========================================================

with open(
    "models/face_threshold.txt",
    "w"
) as f:

    f.write(
        str(best_threshold)
    )

print()
print("Threshold saved to:")
print(
    "models/face_threshold.txt"
)

print()
print("=" * 60)
print("V3 TRAINING COMPLETED")
print("=" * 60)

print("Model:")
print(MODEL_PATH)