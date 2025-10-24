#!/usr/bin/env python3
"""
Simple camera test to verify camera functionality
"""

import cv2
import numpy as np
from camera_utils import get_camera_with_error_handling

def test_camera():
    """Test basic camera functionality"""
    print("Testing camera...")

    # Get camera
    cap = get_camera_with_error_handling()
    if cap is None:
        print("❌ Camera test failed - no camera available")
        return False

    print("✅ Camera initialized successfully")

    # Test frame capture
    for i in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Frame {i+1}: {frame.shape}")
        else:
            print(f"❌ Frame {i+1}: Failed to capture")
            cap.release()
            return False

    cap.release()
    print("✅ Camera test completed successfully")
    return True

if __name__ == "__main__":
    test_camera()