"""
Inspect layer names and architecture of deepfake_face_model_v6.keras
"""

import os
import sys
import tensorflow as tf

model_path = "models/deepfake_face_model_v6.keras"
model = tf.keras.models.load_model(model_path)

print("=" * 60)
print("MODEL SUMMARY:")
print("=" * 60)
model.summary()

print("\n" + "=" * 60)
print("LAYER LIST:")
print("=" * 60)
for idx, layer in enumerate(model.layers):
    print(f"Layer {idx:3d}: {layer.name:30s} | Type: {type(layer).__name__:25s} | Output Shape: {layer.output_shape}")
    # If nested model (like MobileNetV2 base)
    if hasattr(layer, "layers"):
        print(f"  --> Nested Sub-layers ({len(layer.layers)} layers):")
        for s_idx, s_layer in enumerate(layer.layers[-10:]): # show last 10
            print(f"      Sub-Layer {s_idx}: {s_layer.name:30s} | Type: {type(s_layer).__name__:25s} | Shape: {s_layer.output_shape}")
