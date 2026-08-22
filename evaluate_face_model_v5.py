import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "models/deepfake_face_model_v5.keras"
TEST_DIR = "face_dataset_v5/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("=" * 60)
print("EVALUATING FACE MODEL V5")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")
print("Input:", model.input_shape)
print("Output:", model.output_shape)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass names:", test_ds.class_names)

y_true = []
y_prob = []

print("\nRunning predictions...")

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0).flatten()
    y_true.extend(labels.numpy().astype(int))
    y_prob.extend(predictions)

y_true = np.array(y_true).flatten()
y_prob = np.array(y_prob).flatten()

fake_probs = y_prob[y_true == 0]
real_probs = y_prob[y_true == 1]

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print("\nFAKE images:")
print("Count :", len(fake_probs))
print("Min   :", fake_probs.min())
print("Max   :", fake_probs.max())
print("Mean  :", fake_probs.mean())
print("Median:", np.median(fake_probs))

print("\nREAL images:")
print("Count :", len(real_probs))
print("Min   :", real_probs.min())
print("Max   :", real_probs.max())
print("Mean  :", real_probs.mean())
print("Median:", np.median(real_probs))

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.20, 0.71, 0.05):

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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

print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print("Best threshold:", round(best_threshold, 2))
print("Best F1:", round(best_f1, 4))

y_pred = (y_prob >= best_threshold).astype(int)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_true, y_pred))

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_true,
    y_pred,
    target_names=["FAKE", "REAL"],
    zero_division=0
))

print("\n" + "=" * 60)
print("ROC-AUC")
print("=" * 60)

auc = roc_auc_score(y_true, y_prob)
print("ROC-AUC:", round(auc, 4))

with open("models/face_threshold_v5.txt", "w") as f:
    f.write(str(best_threshold))

print("\nThreshold saved:")
print("models/face_threshold_v5.txt")

print("\n" + "=" * 60)
print("V5 EVALUATION COMPLETED")
print("=" * 60)
