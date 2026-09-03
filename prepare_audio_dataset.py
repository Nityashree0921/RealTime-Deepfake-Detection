"""
Audio Dataset Preparation and Bootstrapping Module
Intelligent Real-Time Multimodal Deepfake Detection System

Supports:
1. Bootstrapping synthetic speech and authentic speech sample datasets into audio_dataset/real and audio_dataset/fake.
2. Parsing and importing public audio spoofing datasets (e.g., ASVspoof 2019/2021, WaveFake, FakeAVCeleb).
"""

import os
import shutil
import argparse
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "audio_dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")


def generate_formant_speech(duration=3.0, sr=16000, f0=130, formants=[700, 1200, 2600], is_fake=False, seed=None):
    """
    Synthesizes acoustic speech signals with realistic pitch trajectories, formant resonances, and vocal tract filtering.
    If is_fake=True, injects vocoder phase artifacts, spectral buzz, and high-frequency aliasing characteristic of neural TTS/VC.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Pitch trajectory (Glottal source with natural intonation contour)
    if is_fake:
        # Synthetic speech often exhibits robotic/flat pitch or quantized steps
        pitch_contour = f0 + 2.0 * np.sin(2 * np.pi * 0.5 * t)
    else:
        # Natural human speech pitch modulation (micro-tremor + phrase intonation)
        pitch_contour = f0 + 15.0 * np.sin(2 * np.pi * 1.2 * t) + 4.0 * np.sin(2 * np.pi * 4.5 * t)

    # Glottal pulse train generation
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
    glottal_source = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.125 * np.sin(4 * phase)

    # 2. Formant Filtering (Vocal Tract Resonances)
    speech_signal = np.zeros_like(t)
    for formant_freq in formants:
        bandwidth = formant_freq / 10.0
        decay = np.exp(-np.pi * bandwidth / sr)
        freq_rad = 2 * np.pi * formant_freq / sr
        
        # Resonant impulse response
        filtered_formant = np.zeros_like(glottal_source)
        y1, y2 = 0.0, 0.0
        a1 = 2 * decay * np.cos(freq_rad)
        a2 = - (decay ** 2)
        gain = 1.0 - decay

        for i in range(len(glottal_source)):
            val = gain * glottal_source[i] + a1 * y1 + a2 * y2
            y2 = y1
            y1 = val
            filtered_formant[i] = val
            
        speech_signal += filtered_formant

    # 3. Artifact Injection for Fake/Manipulated Audio
    if is_fake:
        # Neural Vocoder Artifacts: high-frequency harmonic distortion & phase dispersion
        vocoder_carrier = np.sin(2 * np.pi * 3800 * t) * 0.15
        speech_signal += vocoder_carrier * np.abs(glottal_source)
        
        # Spectral buzz / quantization artifact
        speech_signal = np.round(speech_signal * 16.0) / 16.0
        
        # High-frequency linear spectral tilt
        hf_noise = np.random.normal(0, 0.02, len(speech_signal)) * (t / duration)
        speech_signal += hf_noise
    else:
        # Natural subtle room acoustics & breathiness
        room_noise = np.random.normal(0, 0.005, len(speech_signal))
        speech_signal += room_noise

    # 4. Amplitude Envelope (Attack-Decay-Sustain-Release)
    envelope = np.ones_like(t)
    attack_len = int(0.1 * sr)
    release_len = int(0.2 * sr)
    envelope[:attack_len] = np.linspace(0, 1, attack_len)
    envelope[-release_len:] = np.linspace(1, 0, release_len)
    speech_signal *= envelope

    # Normalize to [-0.95, 0.95]
    max_val = np.max(np.abs(speech_signal))
    if max_val > 1e-5:
        speech_signal = 0.90 * (speech_signal / max_val)

    return speech_signal.astype(np.float32)


def generate_bootstrap_samples(count_per_class=30):
    """
    Generates a balanced set of real and synthetic deepfake audio samples into audio_dataset/.
    """
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    print(f"Generating {count_per_class} REAL and {count_per_class} FAKE audio samples...")

    # Real Formants Pool (Human Voice ranges: Male, Female, Low/High Pitch)
    real_configs = [
        {"f0": 110, "formants": [650, 1100, 2500], "dur": 3.0},
        {"f0": 125, "formants": [720, 1240, 2600], "dur": 3.2},
        {"f0": 140, "formants": [800, 1350, 2700], "dur": 2.8},
        {"f0": 180, "formants": [850, 1400, 2800], "dur": 3.0},
        {"f0": 210, "formants": [900, 1500, 2900], "dur": 3.1},
        {"f0": 230, "formants": [950, 1600, 3100], "dur": 2.9},
    ]

    for i in range(count_per_class):
        cfg = real_configs[i % len(real_configs)]
        audio = generate_formant_speech(
            duration=cfg["dur"],
            sr=16000,
            f0=cfg["f0"] + np.random.uniform(-5, 5),
            formants=[f + np.random.uniform(-30, 30) for f in cfg["formants"]],
            is_fake=False,
            seed=1000 + i
        )
        out_path = os.path.join(REAL_DIR, f"real_human_speech_{i+1:03d}.wav")
        sf.write(out_path, audio, 16000)

    # Fake Formants Pool (TTS/VC Vocoder artifacts, pitch shifts, cloned voices)
    fake_configs = [
        {"f0": 105, "formants": [640, 1080, 2480], "dur": 3.0},
        {"f0": 130, "formants": [710, 1220, 2580], "dur": 3.1},
        {"f0": 150, "formants": [780, 1300, 2650], "dur": 2.8},
        {"f0": 190, "formants": [840, 1380, 2780], "dur": 3.2},
        {"f0": 220, "formants": [890, 1480, 2880], "dur": 3.0},
        {"f0": 240, "formants": [940, 1580, 3050], "dur": 2.9},
    ]

    for i in range(count_per_class):
        cfg = fake_configs[i % len(fake_configs)]
        audio = generate_formant_speech(
            duration=cfg["dur"],
            sr=16000,
            f0=cfg["f0"] + np.random.uniform(-4, 4),
            formants=[f + np.random.uniform(-25, 25) for f in cfg["formants"]],
            is_fake=True,
            seed=5000 + i
        )
        out_path = os.path.join(FAKE_DIR, f"fake_ai_generated_{i+1:03d}.wav")
        sf.write(out_path, audio, 16000)

    print(f"[OK] Generated {count_per_class} real audio files in: {REAL_DIR}")
    print(f"[OK] Generated {count_per_class} fake audio files in: {FAKE_DIR}")


def import_asvspoof_dataset(protocol_file, flac_dir, output_dir=DATASET_DIR, max_samples=None):
    """
    Parses ASVspoof 2019/2021 protocol files (e.g. ASVspoof2019.LA.cm.train.trn.txt)
    and copies/converts FLAC/WAV files into real/ and fake/ subdirectories.
    """
    if not os.path.exists(protocol_file):
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
    if not os.path.exists(flac_dir):
        raise FileNotFoundError(f"Audio directory not found: {flac_dir}")

    real_out = os.path.join(output_dir, "real")
    fake_out = os.path.join(output_dir, "fake")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    print(f"Reading ASVspoof protocol: {protocol_file}...")
    copied_real, copied_fake = 0, 0

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            file_id = parts[1]
            key = parts[4].lower()  # 'bonafide' or 'spoof'

            src_file = os.path.join(flac_dir, f"{file_id}.flac")
            if not os.path.exists(src_file):
                src_file = os.path.join(flac_dir, f"{file_id}.wav")

            if not os.path.exists(src_file):
                continue

            if key == "bonafide":
                dst_file = os.path.join(real_out, f"{file_id}.flac")
                shutil.copy2(src_file, dst_file)
                copied_real += 1
            elif key == "spoof":
                dst_file = os.path.join(fake_out, f"{file_id}.flac")
                shutil.copy2(src_file, dst_file)
                copied_fake += 1

            if max_samples and (copied_real + copied_fake) >= max_samples:
                break

    print(f"[OK] ASVspoof Import Complete: {copied_real} Real (bonafide), {copied_fake} Fake (spoof) audio files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Deepfake Dataset Preparation & Bootstrap")
    parser.add_argument("--generate_samples", action="store_true", help="Generate initial bootstrap samples")
    parser.add_argument("--count", type=int, default=40, help="Number of audio samples per class to generate")
    parser.add_argument("--asvspoof_protocol", type=str, default=None, help="Path to ASVspoof protocol TXT file")
    parser.add_argument("--asvspoof_audio_dir", type=str, default=None, help="Path to ASVspoof FLAC/WAV directory")

    args = parser.parse_args()

    if args.asvspoof_protocol and args.asvspoof_audio_dir:
        import_asvspoof_dataset(args.asvspoof_protocol, args.asvspoof_audio_dir)
    else:
        # Default action: Generate bootstrap dataset
        generate_bootstrap_samples(count_per_class=args.count)
