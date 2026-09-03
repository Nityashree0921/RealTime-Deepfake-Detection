import cv2
import time
from camera import Camera

print("=" * 50)
print("DEEPGUARD WEBCAM DIAGNOSTIC & TEST")
print("=" * 50)

try:
    cam = Camera()
    print(f"✅ Camera successfully initialized on device index {cam.index} ({cam.backend})!")
    print("Opening live preview window... Press 'Q' or close window to exit.")

    WINDOW_NAME = "DeepGuard - Camera Diagnostic Test"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    frame_count = 0
    start_time = time.time()

    while True:
        frame = cam.get_frame()

        if frame is None:
            print("⚠️ Warning: Empty frame received, retrying...")
            time.sleep(0.03)
            continue

        frame_count += 1
        fps = frame_count / max(0.001, time.time() - start_time)

        # Draw overlay info
        cv2.putText(frame, f"Device: Index {cam.index} ({cam.backend})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Res: {frame.shape[1]}x{frame.shape[0]}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'Q' to Exit", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed successfully.")

except Exception as e:
    print(f"❌ Camera Error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Open Windows Settings -> Privacy & security -> Camera.")
    print("2. Ensure 'Camera access' and 'Let desktop apps access your camera' are turned ON.")
    print("3. Check if another app (Teams, Zoom, Google Meet, Chrome, OBS) is currently using your webcam.")