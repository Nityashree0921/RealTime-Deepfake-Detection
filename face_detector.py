import cv2

# Load the face detection model
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces(frame):

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

        if confidence > 0.6:

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces