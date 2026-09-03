import os
import cv2
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()

print("\n" + "="*60)
print("TESTING CALIBRATED PIPELINE ON SAMPLE DEMO IMAGES")
print("="*60)

real_path = "sample_images/sample_real.jpg"
fake_path = "sample_images/sample_fake.jpg"

for img_path, true_lbl in [(real_path, "REAL"), (fake_path, "FAKE")]:
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
    img = cv2.imread(img_path)
    # The image is already cropped, but model.predict internally handles preprocessing resizing
    label, confidence, p_real = detector.predict(img, return_raw=True)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  True Label: {true_lbl}")
    print(f"  Raw P(REAL): {p_real*100:.2f}%")
    print(f"  Calibrated Predicted Label: {label}")
    print(f"  Calibrated Confidence: {confidence:.2f}%")
    print("-" * 40)
