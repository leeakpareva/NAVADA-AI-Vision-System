import cv2
import time

def find_camera():
    """Find the first available camera, prioritizing DirectShow for stability"""
    print("[*] Searching for camera with DirectShow backend...")

    # Priority order - Use DirectShow first as it's proven stable
    configs_to_try = [
        (0, cv2.CAP_DSHOW, "DirectShow (stable backend)"),
        (1, cv2.CAP_DSHOW, "DirectShow index 1"),
        (2, cv2.CAP_DSHOW, "DirectShow index 2"),
    ]

    for idx, backend, desc in configs_to_try:
        try:
            print(f"[*] Checking index {idx} with {desc}...")
            cap = cv2.VideoCapture(idx, backend)

            if cap.isOpened():
                # Test if camera works
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[OK] Camera found at index {idx} using {desc}")
                    return cap
                cap.release()
        except Exception:
            pass

    # If still no camera found, try fallback without MSMF (it fails)
    print("[*] Trying fallback configurations...")
    fallback_configs = [
        (3, cv2.CAP_DSHOW, "DirectShow index 3"),
        (4, cv2.CAP_DSHOW, "DirectShow index 4"),
        (0, cv2.CAP_ANY, "Any backend index 0"),
        (1, cv2.CAP_ANY, "Any backend index 1"),
    ]

    for idx, backend, desc in fallback_configs:
        try:
            print(f"[*] Trying fallback: index {idx} with {desc}...")
            cap = cv2.VideoCapture(idx, backend)

            if cap.isOpened():
                # Brief pause to let camera initialize
                time.sleep(0.3)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[OK] Camera found at index {idx} using {desc}")
                    return cap
                cap.release()
        except Exception:
            pass

    return None

def get_camera_with_error_handling():
    """Get camera with proper error handling and user guidance"""
    cap = find_camera()
    if cap is None:
        print("\n[ERROR] No working camera found!")
        print("\nFor IVcam users:")
        print("  1. Ensure IVcam app is running on your phone")
        print("  2. Phone and PC are on the same WiFi network")
        print("  3. IVcam PC client should show 'Connected'")
        print("  4. Try restarting both IVcam apps")
        print("  5. Check if another app is using the camera")
        print("  6. Update camera drivers")
        print("\nPress any key to exit...")
        try:
            input()
        except EOFError:
            pass
        return None

    # Improve camera quality with error handling
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Warm up the camera carefully
        for i in range(5):
            try:
                ret, frame = cap.read()
                if not ret:
                    break
            except:
                break
        print("[*] Camera configured for better quality")
    except Exception as e:
        print(f"[Warning] Could not optimize camera settings: {e}")

    return cap