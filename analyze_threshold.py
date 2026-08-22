import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


MODEL_PATH = "models/deepfake_face_model_v2.keras"

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

IMG_SIZE = 224


print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

images = []
labels = []


def load_images(folder, label):

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        img = tf.keras.utils.load_img(
            path,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        img = tf.keras.utils.img_to_array(img)

        images.append(img)
        labels.append(label)


load_images(REAL_DIR, 0)
load_images(FAKE_DIR, 1)


X = np.array(
    images,
    dtype="float32"
)
X = tf.keras.applications.mobilenet_v2.preprocess_input(X)
y_true = np.array(labels)


print("\nTotal images:", len(X))


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

print("\nRunning predictions...")

predictions = model.predict(
    X,
    batch_size=16,
    verbose=1
).reshape(-1)


# ---------------------------------------------------------
# THRESHOLD TEST
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("THRESHOLD RESULTS")
print("=" * 60)

print(
    "\nThreshold | Accuracy | Precision | Recall | F1"
)

print("-" * 60)


best_threshold = 0.5
best_f1 = 0


for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    y_pred = (
        predictions >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    print(
        f"{threshold:8.2f} | "
        f"{accuracy*100:8.2f}% | "
        f"{precision*100:9.2f}% | "
        f"{recall*100:6.2f}% | "
        f"{f1*100:6.2f}%"
    )


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold


print("\n")
print("=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"\nBest threshold: {best_threshold:.2f}"
)

print(
    f"Best F1 score: {best_f1*100:.2f}%"
)


# ---------------------------------------------------------
# BEST THRESHOLD CONFUSION
# ---------------------------------------------------------

y_best = (
    predictions >= best_threshold
).astype(int)


from sklearn.metrics import confusion_matrix


cm = confusion_matrix(
    y_true,
    y_best
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX AT BEST THRESHOLD")
print("=" * 60)

print("\n              Predicted")
print("              REAL  FAKE")

print(
    f"Actual REAL   {cm[0][0]:4d}  {cm[0][1]:4d}"
)

print(
    f"Actual FAKE   {cm[1][0]:4d}  {cm[1][1]:4d}"
)


print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)