import random

def predict_face(frame):
    """
    Temporary function.
    Later we'll replace this with a real AI model.
    """

    confidence = round(random.uniform(90, 99), 2)

    if random.randint(0, 1):
        return "REAL", confidence
    else:
        return "FAKE", confidence