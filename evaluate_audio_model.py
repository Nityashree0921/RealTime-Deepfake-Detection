"""
Audio Deepfake Model Evaluation and Benchmarking Suite
Intelligent Real-Time Multimodal Deepfake Detection System

Evaluates models/audio_deepfake_model.keras on audio datasets or single audio files.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, and Equal Error Rate (EER).
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import tensorflow as tf
from audio_preprocessor import AudioPreprocessor, LFCCExtractor, extract_lfcc_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(MODELS_DIR, "audio_deepfake_model.keras")
CONFIG_PATH = os.path.join(MODELS_DIR, "audio_calibration_config.json")


def compute_eer(y_true, y_score):
    """
    Computes the Equal Error Rate (EER) where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).
    Standard evaluation metric for ASVspoof audio spoofing benchmarks.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


def load_calibration_config():
    """
    Loads confidence calibration bounds and threshold settings.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "real_threshold_upper": 0.60,
        "fake_threshold_lower": 0.40,
        "calibration_mapping": {
            "real_min_conf": 85.0,
            "real_max_conf": 99.8,
            "fake_min_conf": 85.0,
            "fake_max_conf": 99.8
        }
    }


def calibrate_prediction(p_real, config=None):
    """
    Maps raw probability P(REAL) to calibrated REAL / FAKE / UNCERTAIN class and confidence percentage.
    """
    if config is None:
        config = load_calibration_config()

    t_upper = config.get("real_threshold_upper", 0.60)
    t_lower = config.get("fake_threshold_lower", 0.40)
    cal_map = config.get("calibration_mapping", {})
    real_min = cal_map.get("real_min_conf", 85.0)
    real_max = cal_map.get("real_max_conf", 99.8)
    fake_min = cal_map.get("fake_min_conf", 85.0)
    fake_max = cal_map.get("fake_max_conf", 99.8)

    if p_real >= t_upper:
        label = "REAL"
        norm = (p_real - t_upper) / (1.0 - t_upper) if t_upper < 1.0 else 1.0
        confidence = real_min + norm * (real_max - real_min)
    elif p_real <= t_lower:
        label = "FAKE"
        norm = (t_lower - p_real) / t_lower if t_lower > 0.0 else 1.0
        confidence = fake_min + norm * (fake_max - fake_min)
    else:
        label = "UNCERTAIN"
        dist_to_center = abs(p_real - 0.5) / 0.1
        confidence = 50.0 + dist_to_center * 15.0

    return label, min(99.9, max(50.0, confidence))


def evaluate_single_file(file_path, model_path=MODEL_PATH):
    """
    Performs inference on a single audio file and prints a comprehensive analysis report.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please train the model first.")

    print(f"Loading trained audio model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    preprocessor = AudioPreprocessor()
    extractor = LFCCExtractor()

    print(f"Preprocessing audio: {file_path}...")
    y, sr = preprocessor.process(file_path)
    duration = len(y) / float(sr)

    features = extractor.extract_features(y)
    features_batch = np.expand_dims(features, axis=0)  # Shape: (1, 200, 30, 3)

    p_real = float(model.predict(features_batch, verbose=0)[0][0])
    p_fake = 1.0 - p_real

    config = load_calibration_config()
    label, confidence = calibrate_prediction(p_real, config)

    print("\n" + "=" * 55)
    print("        AUDIO DEEPFAKE DETECTION ANALYSIS")
    print("=" * 55)
    print(f"File:        {os.path.basename(file_path)}")
    print(f"Path:        {file_path}")
    print(f"Duration:    {duration:.2f} seconds")
    print(f"Sample Rate: {sr} Hz (Mono)")
    print("-" * 55)
    print(f"Raw P(REAL): {p_real * 100:.2f}%")
    print(f"Raw P(FAKE): {p_fake * 100:.2f}%")
    print("-" * 55)
    print(f"VERDICT:     {label}")
    print(f"Confidence:  {confidence:.2f}%")
    if label == "FAKE":
        print("Status:      AI-GENERATED / MANIPULATED AUDIO DETECTED")
    elif label == "REAL":
        print("Status:      AUTHENTIC HUMAN VOICE DETECTED")
    else:
        print("Status:      UNCERTAIN / AMBIGUOUS ACOUSTIC PATTERNS")
    print("=" * 55 + "\n")

    return {
        "file": file_path,
        "duration": duration,
        "p_real": p_real,
        "p_fake": p_fake,
        "label": label,
        "confidence": confidence
    }


def evaluate_dataset(dataset_dir="audio_dataset", model_path=MODEL_PATH):
    """
    Evaluates the trained model on an entire dataset directory containing real/ and fake/ subfolders.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    from train_audio_model import load_dataset

    print(f"Loading trained model: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading and extracting features from dataset: {dataset_dir}...")
    X, y_true = load_dataset(dataset_dir)

    print("\nRunning Model Predictions...")
    y_pred_prob = model.predict(X, verbose=1).ravel()
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # Metrics computation
    cr = classification_report(y_true, y_pred_class, target_names=["FAKE", "REAL"])
    cm = confusion_matrix(y_true, y_pred_class)
    roc_auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 1.0
    eer = compute_eer(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0

    print("\n" + "=" * 60)
    print("           AUDIO MODEL EVALUATION BENCHMARK")
    print("=" * 60)
    print(cr)
    print(f"Confusion Matrix:\n{cm}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    print("=" * 60)

    # Save Plots
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Audio Confusion Matrix (Evaluation)", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["FAKE", "REAL"])
    plt.yticks(tick_marks, ["FAKE", "REAL"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2.0 else "black", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "audio_eval_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC Curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#20D67B", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#9AA8C7", lw=1.5, linestyle="--")
        plt.plot([eer], [1 - eer], marker="o", markersize=8, color="red", label=f"EER = {eer*100:.2f}%")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FAR)")
        plt.ylabel("True Positive Rate (1 - FRR)")
        plt.title("Audio Spoofing ROC Curve", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(REPORTS_DIR, "audio_eval_roc_curve.png")
        plt.savefig(roc_path, dpi=200)
        plt.close()

    print(f"\n[OK] Evaluation plots generated in {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Audio Deepfake Detector")
    parser.add_argument("--audio_file", type=str, default=None, help="Path to single audio file for testing")
    parser.add_argument("--dataset_dir", type=str, default="audio_dataset", help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to trained .keras model")

    args = parser.parse_args()

    if args.audio_file:
        evaluate_single_file(args.audio_file, model_path=args.model_path)
    else:
        evaluate_dataset(dataset_dir=args.dataset_dir, model_path=args.model_path)
