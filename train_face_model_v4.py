import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_face_model_v4.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# GET VIDEO ID
# ============================================================

def get_video_id(filename):
    """
    Example:
    real_018_0005.jpg -> 018
    fake_024_0010.jpg -> 024
    """

    parts = filename.split("_")

    if len(parts) < 3:
        return None

    return parts[1]


# ============================================================
# LOAD FILES
# ============================================================

real_files = [
    f for f in os.listdir(REAL_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

fake_files = [
    f for f in os.listdir(FAKE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]


# ============================================================
# GROUP BY VIDEO
# ============================================================

real_videos = {}

for filename in real_files:

    video_id = get_video_id(filename)

    if video_id is not None:

        real_videos.setdefault(
            video_id,
            []
        ).append(filename)


fake_videos = {}

for filename in fake_files:

    video_id = get_video_id(filename)

    if video_id is not None:

        fake_videos.setdefault(
            video_id,
            []
        ).append(filename)


print("=" * 60)
print("VIDEO-LEVEL DATASET")
print("=" * 60)

print("REAL videos:", len(real_videos))
print("FAKE videos:", len(fake_videos))

print("REAL frames:", len(real_files))
print("FAKE frames:", len(fake_files))


# ============================================================
# VIDEO-LEVEL SPLIT
# ============================================================

real_ids = sorted(real_videos.keys())
fake_ids = sorted(fake_videos.keys())


real_train, real_temp = train_test_split(
    real_ids,
    test_size=0.30,
    random_state=42
)

real_val, real_test = train_test_split(
    real_temp,
    test_size=0.50,
    random_state=42
)


fake_train, fake_temp = train_test_split(
    fake_ids,
    test_size=0.30,
    random_state=42
)

fake_val, fake_test = train_test_split(
    fake_temp,
    test_size=0.50,
    random_state=42
)


print("\nVIDEO SPLIT")

print("REAL train:", real_train)
print("REAL val:", real_val)
print("REAL test:", real_test)

print("FAKE train:", fake_train)
print("FAKE val:", fake_val)
print("FAKE test:", fake_test)


# ============================================================
# BUILD DATASET FILE LIST
# ============================================================

def build_file_list(video_dict, video_ids):

    files = []

    for video_id in video_ids:

        for filename in video_dict.get(
            video_id,
            []
        ):

            files.append(filename)

    return files


real_train_files = build_file_list(
    real_videos,
    real_train
)

real_val_files = build_file_list(
    real_videos,
    real_val
)

real_test_files = build_file_list(
    real_videos,
    real_test
)


fake_train_files = build_file_list(
    fake_videos,
    fake_train
)

fake_val_files = build_file_list(
    fake_videos,
    fake_val
)

fake_test_files = build_file_list(
    fake_videos,
    fake_test
)


train_files = (
    [(REAL_DIR, f, 0) for f in real_train_files]
    +
    [(FAKE_DIR, f, 1) for f in fake_train_files]
)

val_files = (
    [(REAL_DIR, f, 0) for f in real_val_files]
    +
    [(FAKE_DIR, f, 1) for f in fake_val_files]
)

test_files = (
    [(REAL_DIR, f, 0) for f in real_test_files]
    +
    [(FAKE_DIR, f, 1) for f in fake_test_files]
)


print("\nFRAME SPLIT")

print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))


# ============================================================
# LOAD IMAGES
# ============================================================

def load_dataset(file_list):

    images = []
    labels = []

    for i, (folder, filename, label) in enumerate(file_list):

        path = os.path.join(
            folder,
            filename
        )

        image = tf.keras.utils.load_img(
            path,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        image = tf.keras.utils.img_to_array(
            image
        )

        images.append(image)
        labels.append(label)

        if (i + 1) % 50 == 0:

            print(
                f"Loaded {i + 1}/{len(file_list)}"
            )

    return (
        np.array(images, dtype="float32"),
        np.array(labels, dtype="float32")
    )


print("\nLoading training data...")

X_train, y_train = load_dataset(
    train_files
)

print("\nLoading validation data...")

X_val, y_val = load_dataset(
    val_files
)

print("\nLoading test data...")

X_test, y_test = load_dataset(
    test_files
)


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

])


# ============================================================
# MOBILENETV2
# ============================================================

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


base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


x = augmentation(inputs)


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


# ============================================================
# COMPILE
# ============================================================

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


# ============================================================
# CALLBACKS
# ============================================================

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


# ============================================================
# STAGE 1
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 1: TRANSFER LEARNING")
print("=" * 60)


model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=INITIAL_EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=callbacks
)


# ============================================================
# STAGE 2
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING")
print("=" * 60)


base_model.trainable = True


fine_tune_at = 100


for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False


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


model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=FINE_TUNE_EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("LOADING BEST V4 MODEL")
print("=" * 60)


model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST")
print("=" * 60)


predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()


# ============================================================
# THRESHOLD
# ============================================================

print("\n")
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)


best_threshold = 0.5
best_f1 = 0


for threshold in np.arange(
    0.20,
    0.71,
    0.05
):

    predicted = (
        predictions >= threshold
    ).astype(int)

    tp = np.sum(
        (predicted == 1) &
        (y_test == 1)
    )

    fp = np.sum(
        (predicted == 1) &
        (y_test == 0)
    )

    fn = np.sum(
        (predicted == 0) &
        (y_test == 1)
    )

    tn = np.sum(
        (predicted == 0) &
        (y_test == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    accuracy = (
        (tp + tn) /
        len(y_test)
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
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


# ============================================================
# FINAL METRICS
# ============================================================

final_predictions = (
    predictions >= best_threshold
).astype(int)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        final_predictions
    )
)


print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "REAL",
            "FAKE"
        ]
    )
)


print("\n")
print("=" * 60)
print("ROC-AUC")
print("=" * 60)

try:

    auc = roc_auc_score(
        y_test,
        predictions
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

except Exception as e:

    print(
        "Could not calculate ROC-AUC:",
        e
    )


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    MODEL_DIR,
    "face_threshold_v4.txt"
)

with open(
    threshold_path,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


print("\n")
print("=" * 60)
print("V4 TRAINING COMPLETED")
print("=" * 60)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Threshold:",
    best_threshold
)

print(
    "Threshold file:",
    threshold_path
)