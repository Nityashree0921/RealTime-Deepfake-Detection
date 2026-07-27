from datetime import datetime, timedelta

from utils.detection_utils import classify_prediction, should_capture_screenshot


def test_fake_detection_triggers_capture_after_cooldown():
    now = datetime(2026, 7, 27, 19, 24, 30)
    last_time = now - timedelta(seconds=2)

    assert should_capture_screenshot("FAKE", last_time, now, cooldown_seconds=1.0) is True


def test_real_detection_never_triggers_capture():
    now = datetime(2026, 7, 27, 19, 24, 30)
    last_time = None

    assert should_capture_screenshot("REAL", last_time, now, cooldown_seconds=1.0) is False


def test_probability_above_half_is_real():
    label, confidence = classify_prediction(0.8)

    assert label == "REAL"
    assert confidence == 80.0


def test_probability_below_half_is_fake():
    label, confidence = classify_prediction(0.2)

    assert label == "FAKE"
    assert confidence == 80.0
