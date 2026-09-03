"""
Test Grad-CAM and MC-Dropout on deepfake face model
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from face_detector import detect_faces

model_path = "models/deepfake_face_model_v6.keras"
model = tf.keras.models.load_model(model_path)

# Let's inspect MobileNetV2 base sub-layers to find the last conv layer
base_model = model.get_layer("mobilenetv2_1.00_224")
print("Base model name:", base_model.name)
conv_layers = [l.name for l in base_model.layers if "conv" in l.name.lower() or "relu" in l.name.lower()]
print("Last 5 conv/relu layers in base:", conv_layers[-5:])
target_conv_name = "Conv_1" if "Conv_1" in [l.name for l in base_model.layers] else conv_layers[-1]
print("Selected target conv layer for Grad-CAM:", target_conv_name)

# 1. GRAD-CAM FUNCTION
def compute_gradcam(model, face_array, target_conv_name="Conv_1", pred_index=None):
    """
    Computes genuine Grad-CAM heatmap for a face array (1, 224, 224, 3)
    """
    base_model = model.get_layer("mobilenetv2_1.00_224")
    target_layer = base_model.get_layer(target_conv_name)

    # Sub-model mapping base_model inputs to target conv layer output and base_model output
    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[target_layer.output, base_model.output]
    )

    # We need to trace gradients from top model loss to the conv activations
    # Layers after base_model:
    gap = model.get_layer("gap")
    d1 = model.get_layer("dropout_1")
    dense = model.get_layer("dense_128")
    d2 = model.get_layer("dropout_2")
    out = model.get_layer("output_sigmoid")

    # Pass through initial augmentation/rescaling if present
    rescaling = model.get_layer("rescaling")
    rescaled_input = rescaling(face_array)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = conv_model(rescaled_input)
        tape.watch(conv_outputs)
        
        # Pass base output through classifier head
        x = gap(base_output)
        x = d1(x, training=False)
        x = dense(x)
        x = d2(x, training=False)
        preds = out(x)
        
        # Model output is P(REAL). For FAKE explanation, we explain loss for FAKE: (1.0 - preds)
        p_real = preds[0][0]
        if p_real < 0.50:
            target_loss = 1.0 - p_real # Explain why FAKE
            explain_class = "FAKE"
        else:
            target_loss = p_real # Explain why REAL
            explain_class = "REAL"

    # Compute gradients of target_loss w.r.t. conv_outputs
    grads = tape.gradient(target_loss, conv_outputs)
    
    # Global average pooling of gradients: weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Apply ReLU to retain positive influences
    heatmap = np.maximum(heatmap.numpy(), 0)
    
    # Normalize heatmap between 0 and 1
    max_heat = np.max(heatmap)
    if max_heat > 0:
        heatmap = heatmap / max_heat
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap, explain_class, float(p_real)


# 2. MC-DROPOUT UNCERTAINTY FUNCTION
def compute_mc_dropout_uncertainty(model, face_array, num_passes=10):
    """
    Computes predictive mean, variance, and entropy via Monte Carlo Dropout
    """
    predictions = []
    for _ in range(num_passes):
        # training=True enables dropout_1 and dropout_2 during forward pass
        p = float(model(face_array, training=True)[0][0])
        predictions.append(p)

    predictions = np.array(predictions)
    mean_p = float(np.mean(predictions))
    var_p = float(np.var(predictions))
    std_p = float(np.std(predictions))
    
    # Predictive Entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
    eps = 1e-7
    p_safe = np.clip(mean_p, eps, 1.0 - eps)
    entropy = float(- (p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe)))
    
    # Categorical uncertainty level:
    # High variance (std >= 0.08) or high entropy near boundary
    if std_p >= 0.08 or (0.42 < mean_p < 0.58):
        uncertainty_level = "HIGH"
    elif std_p >= 0.04 or (0.35 < mean_p < 0.65):
        uncertainty_level = "MEDIUM"
    else:
        uncertainty_level = "LOW"

    return {
        "mean_p_real": mean_p,
        "variance": var_p,
        "std_dev": std_p,
        "entropy": entropy,
        "uncertainty_level": uncertainty_level,
        "passes": predictions.tolist()
    }


# Test on sample images
for sample_path in ["sample_images/sample_fake.jpg", "sample_images/sample_real.jpg"]:
    if not os.path.exists(sample_path):
        print(f"File not found: {sample_path}", flush=True)
        continue
    img = cv2.imread(sample_path)
    faces = detect_faces(img)
    if len(faces) == 0:
        print(f"No face detected in {sample_path}", flush=True)
        continue
    fx, fy, fw, fh = faces[0]
    p = int(0.12 * max(fw, fh))
    h, w = img.shape[:2]
    crop = img[max(0, fy-p):min(h, fy+fh+p), max(0, fx-p):min(w, fx+fw+p)]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_res = cv2.resize(crop_rgb, (224, 224))
    face_arr = np.expand_dims(crop_res.astype("float32"), axis=0)

    print("\n" + "=" * 60, flush=True)
    print(f"SAMPLE: {sample_path}", flush=True)
    print("=" * 60, flush=True)
    
    # Test Grad-CAM
    heatmap, explain_cls, p_real = compute_gradcam(model, face_arr, target_conv_name)
    print(f"Grad-CAM Explanation for {explain_cls} | P(REAL) = {p_real*100:.2f}%", flush=True)
    print(f"Heatmap shape: {heatmap.shape} | Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}", flush=True)

    # Generate colored heatmap overlay
    heatmap_resized = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop, 0.65, heatmap_colored, 0.35, 0)
    
    out_overlay_path = f"reports/xai_gradcam_{os.path.splitext(os.path.basename(sample_path))[0]}.jpg"
    os.makedirs("reports", exist_ok=True)
    cv2.imwrite(out_overlay_path, overlay)
    print(f"Saved Grad-CAM overlay to: {out_overlay_path}", flush=True)

    # Test MC-Dropout
    unc_res = compute_mc_dropout_uncertainty(model, face_arr, num_passes=10)
    print("MC-Dropout Results:", flush=True)
    print(f"  Mean P(REAL) : {unc_res['mean_p_real']*100:.2f}%", flush=True)
    print(f"  Std Dev      : {unc_res['std_dev']:.4f}", flush=True)
    print(f"  Entropy      : {unc_res['entropy']:.4f}", flush=True)
    print(f"  Uncertainty  : {unc_res['uncertainty_level']}", flush=True)
