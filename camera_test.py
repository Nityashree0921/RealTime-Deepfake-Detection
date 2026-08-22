import cv2

print("Starting camera test...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ Camera could not be opened.")
    print("Trying another camera index...")

    camera.release()
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ No camera found.")
    print("Check Windows camera permissions and whether another app is using the camera.")
    exit()

print("✅ Camera opened successfully!")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("❌ Could not read frame from camera.")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()