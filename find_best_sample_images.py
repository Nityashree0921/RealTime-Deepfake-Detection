import os
import cv2
import numpy as np
import tensorflow as tf

MODELS = {
    "V6": "models/deepfake_face_model_v6.keras",
    "V7": "models/deepfake_face_model_v7.keras"
}
IMG_SIZE = 224

for name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        continue
    print(f"\nEvaluating images for model {name}...")
    model = tf.keras.models.load_model(model_path)
    
    # Check V6 or V7 test dirs
    test_dir = f"face_dataset_{name.lower()}/test"
    real_dir = os.path.join(test_dir, "real")
    fake_dir = os.path.join(test_dir, "fake")
    
    # REAL images
    best_real_img = None
    max_real_prob = -1.0
    
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(real_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(img_resized.astype("float32"), axis=0)
            prob = float(model.predict(img_array, verbose=0)[0][0])
            
            if prob > max_real_prob:
                max_real_prob = prob
                best_real_img = f
                
    # FAKE images
    best_fake_img = None
    min_fake_prob = 2.0
    
    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(fake_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(img_resized.astype("float32"), axis=0)
            prob = float(model.predict(img_array, verbose=0)[0][0])
            
            if prob < min_fake_prob:
                min_fake_prob = prob
                best_fake_img = f
                
    print(f"Model {name} Best REAL: {best_real_img} with P(REAL) = {max_real_prob*100:.2f}%")
    print(f"Model {name} Best FAKE (Lowest P(REAL)): {best_fake_img} with P(REAL) = {min_fake_prob*100:.2f}%")
