import random

def predict_audio():
    confidence = round(random.uniform(88, 98), 2)

    if random.randint(0, 1):
        return "REAL", confidence
    else:
        return "FAKE", confidence