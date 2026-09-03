"""
Audio Deepfake Detection Quick Validation Script
Intelligent Real-Time Multimodal Deepfake Detection System
"""

import os
from evaluate_audio_model import evaluate_single_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_SAMPLE = os.path.join(BASE_DIR, "audio_dataset", "real", "real_human_speech_001.wav")
FAKE_SAMPLE = os.path.join(BASE_DIR, "audio_dataset", "fake", "fake_ai_generated_001.wav")

print("=" * 60)
print("     TESTING REAL AUDIO SAMPLE INFERENCE")
print("=" * 60)
if os.path.exists(REAL_SAMPLE):
    res_real = evaluate_single_file(REAL_SAMPLE)
else:
    print(f"Sample not found: {REAL_SAMPLE}")

print("\n" + "=" * 60)
print("     TESTING FAKE AUDIO SAMPLE INFERENCE")
print("=" * 60)
if os.path.exists(FAKE_SAMPLE):
    res_fake = evaluate_single_file(FAKE_SAMPLE)
else:
    print(f"Sample not found: {FAKE_SAMPLE}")

print("\n[SUCCESS] Audio Deepfake Model Validation Complete!")