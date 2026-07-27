from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_PATH = "models/audio_model"

print("Loading Feature Extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

print("Loading Model...")
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

print("✅ Audio model loaded successfully!")

print(model.config.id2label)