import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ----------------------------
# Load AI Model
# ----------------------------

MODEL_PATH = "models/audio_model"

print("Loading Audio Model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

print("Audio Model Loaded")


# ----------------------------
# Predict Function
# ----------------------------

def predict_audio(file_path):

    audio, sr = librosa.load(file_path, sr=16000)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    predicted = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][predicted].item() * 100

    label = model.config.id2label[predicted]

    if label.lower() == "bonafide":
        return "REAL", confidence
    else:
        return "FAKE", confidence


# ----------------------------
# Upload Button
# ----------------------------

def upload_audio():

    file_path = filedialog.askopenfilename(
        title="Select Audio",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.flac")
        ]
    )

    if not file_path:
        return

    try:

        label, confidence = predict_audio(file_path)

        from datetime import datetime
        import csv
        import os
        from database import save_detection
        from report_generator import generate_report

        now = datetime.now()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "detections.csv")

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Prediction", "Confidence"])

        # Save CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                "AUDIO-" + label,
                f"{confidence:.2f}"
            ])

        # Save SQLite
        save_detection(
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "AUDIO-" + label,
            confidence
        )

        # PDF report
        generate_report("AUDIO-" + label, confidence)

        # Email Alert
        if label == "FAKE":
            try:
                from email_alert import send_alert
                send_alert(file_path, confidence)
            except Exception as e:
                print("Email Error:", e)

        color = "green" if label == "REAL" else "red"

        result_label.config(
            text=f"{label}\nConfidence: {confidence:.2f}%",
            fg=color
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ----------------------------
# GUI
# ----------------------------

root = tk.Tk()

root.title("Audio Deepfake Detection")

root.geometry("500x350")

root.configure(bg="#1E1E2E")


title = tk.Label(

    root,

    text="🎤 Audio Deepfake Detection",

    font=("Arial",18,"bold"),

    bg="#1E1E2E",

    fg="white"
)

title.pack(pady=20)


btn = tk.Button(

    root,

    text="Upload Audio",

    command=upload_audio,

    font=("Arial",14),

    bg="#4CAF50",

    fg="white",

    width=18,

    height=2
)

btn.pack(pady=20)


result_label = tk.Label(

    root,

    text="",

    font=("Arial",18,"bold"),

    bg="#1E1E2E"
)

result_label.pack(pady=30)


root.mainloop()