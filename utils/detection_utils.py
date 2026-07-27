def classify_prediction(prediction):
    if prediction >= 0.5:
        return "REAL", prediction * 100

    return "FAKE", (1 - prediction) * 100


def should_capture_screenshot(label, last_saved_time, now, cooldown_seconds=2.0):
    if label != "FAKE":
        return False

    if last_saved_time is None:
        return True

    return (now - last_saved_time).total_seconds() >= cooldown_seconds
