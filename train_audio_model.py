"""
Audio Deepfake Model Training Pipeline
Intelligent Real-Time Multimodal Deepfake Detection System

Architecture:
LFCC Features (200, 30, 3) -> 2D-CNN Blocks -> Feature Reshaping -> Bidirectional LSTM -> Dense & Dropout -> Sigmoid Output (Prob REAL)
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from audio_preprocessor import AudioPreprocessor, LFCCExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "audio_deepfake_model.keras")
METADATA_SAVE_PATH = os.path.join(MODELS_DIR, "audio_model_metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_audio_deepfake_model(input_shape=(200, 30, 3)):
    """
    Constructs a Deep Convolutional Recurrent Neural Network (CRNN with BiLSTM)
    specialized for LFCC spectral and temporal deepfake artifacts.
    """
    inputs = layers.Input(shape=input_shape, name="lfcc_input")

    # Block 1: Low-level Spectral Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Block 2: Intermediate Representation
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3: High-level Pattern Extraction
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Shape after 3 MaxPool2D(2,2):
    # Time: 200 -> 100 -> 50 -> 25
    # Freq: 30 -> 15 -> 7 -> 3
    # Channels: 128
    # Flatten spatial/frequency dimension per time-step for sequential LSTM
    target_time_steps = 25
    target_feature_dim = 3 * 128  # 384
    x = layers.Reshape((target_time_steps, target_feature_dim), name="reshape_for_recurrent")(x)

    # Bidirectional LSTM for Temporal Sequence Modeling
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3),
        name="bilstm_temporal"
    )(x)

    # Dense Classification Head
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.4, name="drop_dense")(x)

    x = layers.Dense(32, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3, name="drop_dense2")(x)

    # Sigmoid Output: Probability of Bonafide / Real (1.0 = Real, 0.0 = Fake)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_real")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LFCC_CNN_BiLSTM_AudioClassifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def load_dataset(dataset_dir):
    """
    Extracts LFCC features from real and fake audio directories.
    Label 1 = REAL (bonafide), Label 0 = FAKE (spoof/manipulated).
    """
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Dataset must contain 'real' and 'fake' subdirectories inside {dataset_dir}")

    preprocessor = AudioPreprocessor()
    extractor = LFCCExtractor()

    real_files = glob.glob(os.path.join(real_dir, "*.*"))
    fake_files = glob.glob(os.path.join(fake_dir, "*.*"))

    valid_exts = (".wav", ".mp3", ".flac", ".m4a")
    real_files = [f for f in real_files if f.lower().endswith(valid_exts)]
    fake_files = [f for f in fake_files if f.lower().endswith(valid_exts)]

    print(f"Found {len(real_files)} Real audio files and {len(fake_files)} Fake audio files.")

    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError("Dataset cannot be empty. Please ensure both 'real' and 'fake' directories have audio files.")

    X, y = [], []

    print("Extracting LFCC features for Real audio...")
    for idx, fpath in enumerate(real_files):
        try:
            audio, _ = preprocessor.process(fpath)
            feats = extractor.extract_features(audio)
            X.append(feats)
            y.append(1.0)  # Real
        except Exception as e:
            print(f"Warning: Skipped {os.path.basename(fpath)} due to error: {e}")

    print("Extracting LFCC features for Fake audio...")
    for idx, fpath in enumerate(fake_files):
        try:
            audio, _ = preprocessor.process(fpath)
            feats = extractor.extract_features(audio)
            X.append(feats)
            y.append(0.0)  # Fake
        except Exception as e:
            print(f"Warning: Skipped {os.path.basename(fpath)} due to error: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Total Extracted Features Shape: X={X.shape}, y={y.shape}")
    return X, y


def train_audio_classifier(dataset_dir="audio_dataset", epochs=30, batch_size=16):
    """
    Full training, validation, evaluation, and reporting pipeline.
    """
    X, y = load_dataset(dataset_dir)

    # 1. Stratified Train / Test / Validation Split (70% Train, 15% Val, 15% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # 2. Compute Class Weights for Imbalance Handling
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Computed Class Weights: {class_weight_dict}")

    # 3. Instantiate Model
    model = build_audio_deepfake_model(input_shape=X_train.shape[1:])
    model.summary()

    # 4. Training Callbacks
    training_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Fit Model
    print("Starting Model Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=training_callbacks,
        verbose=1
    )

    # 6. Save Model explicitly
    model.save(MODEL_SAVE_PATH)
    print(f"[OK] Trained Audio Deepfake Model saved to: {MODEL_SAVE_PATH}")

    # 7. Evaluate on Unseen Test Set
    print("\nEvaluating on Test Set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_metrics = dict(zip(model.metrics_names, test_results))
    print(f"Test Metrics: {test_metrics}")

    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # Classification Report
    cr = classification_report(y_test, y_pred_class, target_names=["FAKE", "REAL"], output_dict=True)
    print("\nClassification Report:\n", classification_report(y_test, y_pred_class, target_names=["FAKE", "REAL"]))

    cm = confusion_matrix(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 1.0

    # 8. Save Metrics Metadata
    metadata = {
        "model_name": "LFCC-CNN-BiLSTM Audio Deepfake Classifier",
        "input_shape": list(X_train.shape[1:]),
        "test_metrics": test_metrics,
        "classification_report": cr,
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test))
    }
    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Training metadata saved to: {METADATA_SAVE_PATH}")

    # 9. Plot and Save Training Graphs
    # Training History (Loss & Accuracy)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", color="#FF5577", lw=2)
    plt.plot(history.history["val_loss"], label="Val Loss", color="#00D9FF", lw=2)
    plt.title("Audio Model Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc", color="#20D67B", lw=2)
    plt.plot(history.history["val_accuracy"], label="Val Acc", color="#FF9D42", lw=2)
    plt.title("Audio Model Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    history_plot_path = os.path.join(REPORTS_DIR, "audio_training_history.png")
    plt.savefig(history_plot_path, dpi=200)
    plt.close()
    print(f"[OK] Training graphs saved to: {history_plot_path}")

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Audio Confusion Matrix", fontsize=12, fontweight="bold")
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
    cm_plot_path = os.path.join(REPORTS_DIR, "audio_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrix plot saved to: {cm_plot_path}")

    # ROC Curve Plot
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#20D67B", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#9AA8C7", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Audio ROC Curve", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_plot_path = os.path.join(REPORTS_DIR, "audio_roc_curve.png")
        plt.savefig(roc_plot_path, dpi=200)
        plt.close()
        print(f"[OK] ROC curve plot saved to: {roc_plot_path}")

    print("\n[SUCCESS] Audio Deepfake Model Training & Evaluation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LFCC-CNN-BiLSTM Audio Deepfake Detector")
    parser.add_argument("--dataset_dir", type=str, default="audio_dataset", help="Path to audio dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")

    args = parser.parse_args()
    train_audio_classifier(dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size)
