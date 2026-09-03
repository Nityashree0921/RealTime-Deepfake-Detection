"""
Audio Preprocessing and LFCC Feature Extraction Module
Intelligent Real-Time Multimodal Deepfake Detection System

This module handles:
1. Multi-format audio loading (WAV, MP3, FLAC, M4A).
2. Audio preprocessing (mono conversion, 16kHz resampling, amplitude normalization, silence trimming, padding/truncation).
3. LFCC (Linear Frequency Cepstral Coefficients) feature extraction with linear filterbanks and dynamic delta/delta-delta derivatives.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.fftpack import dct


class AudioPreprocessor:
    """
    Handles loading, cleaning, normalizing, and standardizing audio signals.
    """

    def __init__(self, target_sr=16000, target_duration=3.0):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)

    def load_audio(self, file_path):
        """
        Loads an audio file in any supported format (WAV, MP3, FLAC, M4A),
        converts to mono, and resamples to target_sr.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Load with librosa (supports multiple formats via audioread/soundfile backend)
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        except Exception as e:
            # Fallback to soundfile if librosa encounters a container error
            try:
                y, sr = sf.read(file_path)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != self.target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio from {file_path}: {e} (Fallback error: {e2})")

        return y, self.target_sr

    def normalize_audio(self, y):
        """
        Normalizes audio signal to [-1.0, 1.0] range using peak amplitude normalization.
        """
        max_val = np.max(np.abs(y))
        if max_val > 1e-6:
            return y / max_val
        return y

    def trim_silence(self, y, top_db=25):
        """
        Trims leading and trailing silence/low-energy frames.
        """
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(trimmed) > int(0.2 * self.target_sr):  # Keep if at least 200ms
            return trimmed
        return y

    def pad_or_truncate(self, y, target_samples=None):
        """
        Standardizes audio sample length by repeating/padding or truncating.
        """
        if target_samples is None:
            target_samples = self.target_samples

        current_samples = len(y)
        if current_samples < target_samples:
            # If shorter, tile/repeat the audio signal or zero-pad
            repeat_factor = int(np.ceil(target_samples / max(1, current_samples)))
            tiled = np.tile(y, repeat_factor)
            return tiled[:target_samples]
        elif current_samples > target_samples:
            # If longer, take the centered segment
            start = (current_samples - target_samples) // 2
            return y[start : start + target_samples]
        return y

    def segment_audio(self, y, segment_duration=3.0, hop_duration=1.5):
        """
        Divides long audio into overlapping segments for thorough multi-chunk evaluation.
        """
        seg_samples = int(self.target_sr * segment_duration)
        hop_samples = int(self.target_sr * hop_duration)

        if len(y) <= seg_samples:
            return [self.pad_or_truncate(y, seg_samples)]

        segments = []
        for start in range(0, len(y) - seg_samples + 1, hop_samples):
            segments.append(y[start : start + seg_samples])

        # Include remaining tail if not empty
        if len(y) > 0 and (len(y) - seg_samples) % hop_samples != 0:
            segments.append(self.pad_or_truncate(y[-seg_samples:], seg_samples))

        return segments

    def process(self, file_path):
        """
        Full end-to-end preprocessing pipeline for a single audio file.
        """
        y, sr = self.load_audio(file_path)
        y = self.trim_silence(y)
        y = self.normalize_audio(y)
        y = self.pad_or_truncate(y)
        return y, sr


class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients (LFCC) Feature Extractor.
    Constructs linearly spaced filterbanks and extracts static, delta, and delta-delta coefficients.
    """

    def __init__(
        self,
        sr=16000,
        n_fft=512,
        hop_length=256,
        win_length=512,
        n_filter=40,
        n_lfcc=30,
        f_min=0.0,
        f_max=8000.0,
        max_frames=200
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr / 2.0
        self.max_frames = max_frames
        
        # Precompute linear filterbank matrix
        self.filterbank = self._create_linear_filterbank()

    def _create_linear_filterbank(self):
        """
        Generates a linear triangular filterbank spanning [f_min, f_max].
        """
        num_fft_bins = self.n_fft // 2 + 1
        fft_freqs = np.linspace(0, self.sr / 2.0, num_fft_bins)
        
        # Linearly spaced filter center points
        filter_centers = np.linspace(self.f_min, self.f_max, self.n_filter + 2)
        filterbank = np.zeros((self.n_filter, num_fft_bins))

        for i in range(self.n_filter):
            f_left = filter_centers[i]
            f_center = filter_centers[i + 1]
            f_right = filter_centers[i + 2]

            # Up-slope
            mask_up = np.logical_and(fft_freqs >= f_left, fft_freqs <= f_center)
            if f_center > f_left:
                filterbank[i, mask_up] = (fft_freqs[mask_up] - f_left) / (f_center - f_left)

            # Down-slope
            mask_down = np.logical_and(fft_freqs >= f_center, fft_freqs <= f_right)
            if f_right > f_center:
                filterbank[i, mask_down] = (f_right - fft_freqs[mask_down]) / (f_right - f_center)

        return filterbank

    def extract_features(self, y):
        """
        Computes LFCC with Delta and Delta-Delta coefficients.
        Returns array of shape: (max_frames, n_lfcc, 3) where channels are [Static, Delta, Delta2].
        """
        # 1. Short-Time Fourier Transform (Power Spectrum)
        stft = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hamming"
        )
        power_spec = np.abs(stft) ** 2  # Shape: (n_fft // 2 + 1, n_frames)

        # 2. Linear Filterbank Matrix Multiplication
        filtered_energy = np.dot(self.filterbank, power_spec)  # Shape: (n_filter, n_frames)

        # 3. Logarithm of Filterbank Energies
        log_energy = np.log(filtered_energy + 1e-10)

        # 4. Discrete Cosine Transform (DCT Type-II)
        # Apply DCT across the filter dimension (axis=0)
        lfcc = dct(log_energy, type=2, axis=0, norm="ortho")[: self.n_lfcc, :] # Shape: (n_lfcc, n_frames)

        # 5. Compute Derivatives (Delta & Delta-Delta)
        delta1 = librosa.feature.delta(lfcc, order=1)
        delta2 = librosa.feature.delta(lfcc, order=2)

        # 6. Stack Static, Delta, and Delta-Delta: Shape -> (3, n_lfcc, n_frames)
        features = np.stack([lfcc, delta1, delta2], axis=0)

        # 7. Transpose to (n_frames, n_lfcc, 3)
        features = np.transpose(features, (2, 1, 0))

        # 8. Standardize Temporal Length to max_frames
        features = self._standardize_time_dimension(features)

        # 9. Feature Normalization (Mean-variance normalization across time)
        mean = np.mean(features, axis=(0, 1), keepdims=True)
        std = np.std(features, axis=(0, 1), keepdims=True) + 1e-7
        features = (features - mean) / std

        return features.astype(np.float32)

    def _standardize_time_dimension(self, features):
        """
        Pads or truncates the time frames dimension to self.max_frames.
        """
        n_frames = features.shape[0]
        if n_frames < self.max_frames:
            pad_width = self.max_frames - n_frames
            # Pad with edge repetition or zeros
            features = np.pad(features, ((0, pad_width), (0, 0), (0, 0)), mode="edge")
        elif n_frames > self.max_frames:
            features = features[: self.max_frames, :, :]
        return features


# Convenience global pipeline helper
_default_preprocessor = AudioPreprocessor()
_default_extractor = LFCCExtractor()


def extract_lfcc_pipeline(file_path, preprocessor=None, extractor=None):
    """
    End-to-end convenience utility to extract normalized LFCC features from any audio file.
    Returns: numpy array of shape (max_frames, n_lfcc, 3)
    """
    prep = preprocessor or _default_preprocessor
    ext = extractor or _default_extractor

    y, sr = prep.process(file_path)
    features = ext.extract_features(y)
    return features
