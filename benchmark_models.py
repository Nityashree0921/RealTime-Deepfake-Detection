import os
import numpy as np
import tensorflow as tf


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 16

REAL_DIR = "webcam_test/real"

MODELS = [
    "models/deepfake_face_model_v2.keras",
    "models/deepfake_face_model_v3.keras",
    "models/deepfake_face_model_v4.keras",
    "models/deepfake_face_model_v5.keras",
    "models/deepfake_face_model_v6.keras",
    "models/deepfake_face_model_v7.keras",
]


# =========================================================
# LOAD WEBCAM IMAGES
# =========================================================

print("=" * 70)
print("DEEPFAKE MODEL BENCHMARK")
print("=" * 70)

files = sorted([
    f for f in os.listdir(REAL_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"\nWebcam REAL images: {len(files)}")

images = []

for filename in files:

    path = os.path.join(REAL_DIR, filename)

    img = tf.keras.utils.load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img = tf.keras.utils.img_to_array(img)

    # IMPORTANT:
    # Do NOT divide by 255.
    # The models contain MobileNetV2 preprocess_input.

    images.append(img)


X = np.array(images, dtype=np.float32)

print("Input shape:", X.shape)
print("Input minimum:", X.min())
print("Input maximum:", X.max())


# =========================================================
# TEST EACH MODEL
# =========================================================

results = []


for model_path in MODELS:

    print("\n")
    print("=" * 70)
    print(f"TESTING: {os.path.basename(model_path)}")
    print("=" * 70)

    if not os.path.exists(model_path):

        print("MODEL NOT FOUND")

        continue


    try:

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("Model loaded.")

        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        predictions = model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=1
        ).flatten()


        # -------------------------------------------------
        # PREDICTION INTERPRETATION
        # -------------------------------------------------

        # Model convention:
        # 0 = REAL
        # 1 = FAKE

        # Use several thresholds for analysis.

        real_rates = {}

        for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

            predicted_real = np.sum(
                predictions < threshold
            )

            real_rate = (
                predicted_real / len(predictions)
            ) * 100

            real_rates[threshold] = real_rate


        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        avg_fake = np.mean(predictions) * 100
        min_fake = np.min(predictions) * 100
        max_fake = np.max(predictions) * 100
        median_fake = np.median(predictions) * 100


        print("\nRESULTS")

        print(
            f"Average FAKE probability : "
            f"{avg_fake:.2f}%"
        )

        print(
            f"Minimum FAKE probability : "
            f"{min_fake:.2f}%"
        )

        print(
            f"Maximum FAKE probability : "
            f"{max_fake:.2f}%"
        )

        print(
            f"Median FAKE probability  : "
            f"{median_fake:.2f}%"
        )


        print("\nREAL ACCEPTANCE RATE")

        for threshold, rate in real_rates.items():

            print(
                f"Threshold {threshold:.2f} : "
                f"{rate:.2f}% REAL"
            )


        results.append({
            "model": os.path.basename(model_path),
            "avg_fake": avg_fake,
            "min_fake": min_fake,
            "max_fake": max_fake,
            "median_fake": median_fake,
            "real_30": real_rates[0.30],
            "real_40": real_rates[0.40],
            "real_50": real_rates[0.50],
            "real_60": real_rates[0.60],
            "real_70": real_rates[0.70],
        })


        # Free memory

        del model

        tf.keras.backend.clear_session()


    except Exception as e:

        print("\nERROR:")
        print(type(e).__name__, e)


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    "\n"
    "MODEL                         "
    "AVG FAKE   "
    "MEDIAN     "
    "REAL@30    "
    "REAL@40    "
    "REAL@50    "
    "REAL@60    "
    "REAL@70"
)

print("-" * 110)


for r in results:

    print(
        f"{r['model']:<29}"
        f"{r['avg_fake']:>8.2f}%   "
        f"{r['median_fake']:>8.2f}%   "
        f"{r['real_30']:>8.2f}%   "
        f"{r['real_40']:>8.2f}%   "
        f"{r['real_50']:>8.2f}%   "
        f"{r['real_60']:>8.2f}%   "
        f"{r['real_70']:>8.2f}%"
    )


print("\n")
print("=" * 70)
print("BENCHMARK COMPLETED")
print("=" * 70)