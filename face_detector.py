import cv2

# Load the face detection model
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces(frame):
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.60:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            fw = x2 - x1
            fh = y2 - y1

            if fw >= 35 and fh >= 35 and 0.45 <= (fw / max(1, fh)) <= 2.2:
                faces.append((x1, y1, fw, fh))

    # Sort largest face first
    faces.sort(key=lambda b: b[2] * b[3], reverse=True)
    return faces