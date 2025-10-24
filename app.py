import chainlit as cl
import cv2
import numpy as np
import base64
import asyncio
import os
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import json
import pyttsx3
import speech_recognition as sr
from openai import OpenAI

# Load environment variables
load_dotenv()

# Try to import face recognition system
try:
    from face_recognition_system import SimpleFaceRecognizer
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Face recognition system not available")
    FACE_RECOGNITION_AVAILABLE = False

# Try to import object recognition system
try:
    from object_recognition_system import CustomObjectRecognizer
    OBJECT_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Custom object recognition system not available")
    OBJECT_RECOGNITION_AVAILABLE = False

# Try to import camera utils
try:
    from camera_utils import get_camera_with_error_handling
    CAMERA_UTILS_AVAILABLE = True
except ImportError:
    print("Camera utils not available, using basic camera detection")
    CAMERA_UTILS_AVAILABLE = False

# Configuration
USE_OPENAI = os.getenv('OPENAI_API_KEY') is not None
VOICE_PROMPT_ID = os.getenv('VOICE_PROMPT_ID', "pmpt_689ba50c5b5c81948e9cca71346b9dce0bc1568b20f11620")

# Global variables
yolo_model = None
face_recognizer = None
object_recognizer = None
live_feed_active = False
live_feed_thread = None
tts_engine = None
openai_client = None

def initialize_yolo():
    """Initialize YOLO model"""
    global yolo_model
    try:
        yolo_model = YOLO('yolov8s.pt')
        print("✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return False

def initialize_voice():
    """Initialize voice systems"""
    global tts_engine, openai_client

    # Initialize TTS engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.8)
        print("✅ TTS engine initialized")
    except Exception as e:
        print(f"⚠️ TTS engine failed to initialize: {e}")
        tts_engine = None

    # Initialize OpenAI client if API key available
    if USE_OPENAI:
        try:
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print("✅ OpenAI client initialized")
            return True
        except Exception as e:
            print(f"⚠️ OpenAI client failed to initialize: {e}")
            return False
    return False

def speak_text(text):
    """Convert text to speech"""
    global tts_engine
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

async def speak_with_openai(text):
    """Use OpenAI TTS for higher quality voice"""
    if not openai_client:
        return False

    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        # Save and play audio
        audio_path = "temp_speech.mp3"
        response.stream_to_file(audio_path)

        # You could implement audio playback here
        print(f"🎵 Audio saved to {audio_path}")
        return True
    except Exception as e:
        print(f"OpenAI TTS Error: {e}")
        return False

async def get_ai_response(message, detected_objects=None):
    """Get AI response with context about detected objects"""
    if not openai_client:
        return "I can see your message, but I need an OpenAI API key to provide intelligent responses."

    try:
        context = ""
        if detected_objects:
            context = f"I can see these objects in the image: {', '.join(detected_objects)}. "

        prompt = f"{context}User message: {message}"

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are NAVADA-AI, a friendly AI vision assistant. Be conversational and helpful. Keep responses under 100 words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"AI Response Error: {e}")
        return f"I detected your message but encountered an error: {str(e)}"

def get_camera():
    """Initialize camera with DirectShow backend for stability"""
    if CAMERA_UTILS_AVAILABLE:
        return get_camera_with_error_handling()

    # Based on diagnostic tests: Use DirectShow backend exclusively
    for index in [0, 1, 2]:  # Try camera indices 0-2
        try:
            print(f"[*] Testing camera index {index} with DirectShow...")
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

            if cap.isOpened():
                # Test if camera actually works
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"[OK] Camera {index}: Initial frame captured ({test_frame.shape})")

                    # Set optimal properties for DirectShow
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Verify camera works with new settings
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"[*] Camera {index} configured successfully")
                        print(f"[*] Final resolution: {test_frame.shape}")
                        return cap
                    else:
                        print(f"[ERROR] Camera {index}: Failed after property setup")
                else:
                    print(f"[ERROR] Camera {index}: No initial frame")

                cap.release()
            else:
                print(f"[ERROR] Camera {index}: Not opened")

        except Exception as e:
            print(f"[ERROR] Camera {index}: Exception - {e}")
            continue

    print("❌ No working camera found with DirectShow backend")
    return None

def capture_frame(cap):
    """Capture a frame from the camera with maximum robustness"""
    if not cap or not cap.isOpened():
        return None

    try:
        # Multiple attempts to read frame
        for attempt in range(3):
            ret, frame = cap.read()

            if ret and frame is not None and frame.size > 0:
                # Validate basic frame properties
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    continue

                h, w, c = frame.shape
                if h < 10 or w < 10:
                    continue

                # CRITICAL: Always create a completely new array to avoid stride issues
                # This fixes the OpenCV matrix stride error
                new_frame = np.zeros((h, w, c), dtype=np.uint8)
                new_frame[:] = frame[:]

                # Ensure the new frame is contiguous
                if not new_frame.flags['C_CONTIGUOUS']:
                    new_frame = np.ascontiguousarray(new_frame)

                # Final validation
                if new_frame.size > 0 and len(new_frame.shape) == 3:
                    return new_frame

            # Brief pause between attempts
            if attempt < 2:
                time.sleep(0.01)

        return None

    except Exception as e:
        print(f"Error capturing frame: {e}")
        # Try one more time with a fresh array approach
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                # Force copy to new memory location
                safe_frame = frame.copy()
                return np.ascontiguousarray(safe_frame)
        except:
            pass

        return None

def process_yolo_detection(frame):
    """Process frame with YOLO detection"""
    if yolo_model is None or frame is None:
        return frame, []

    try:
        # Validate input frame
        if frame.size == 0 or len(frame.shape) != 3:
            return frame, []

        # Ensure frame is contiguous and in correct format
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Create a working copy for YOLO processing
        work_frame = frame.copy()

        results = yolo_model(work_frame, conf=0.25, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = box.cls[0].item()

                        # Get class name
                        class_name = yolo_model.names[int(cls)]

                        # Size filtering - remove very small detections
                        box_area = (x2 - x1) * (y2 - y1)
                        frame_area = frame.shape[0] * frame.shape[1]
                        area_ratio = box_area / frame_area

                        if area_ratio > 0.01:  # Minimum 1% of frame area
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class': class_name,
                                'area_ratio': area_ratio
                            })

                            # Draw bounding box on original frame
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

                            # Draw label
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    except Exception as box_error:
                        print(f"Error processing detection box: {box_error}")
                        continue

        return frame, detections
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return frame, []

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Error converting frame to base64: {e}")
        return None

async def live_camera_feed():
    """True continuous live camera feed using CV2 window approach"""
    global live_feed_active

    cap = cl.user_session.get("camera")
    if not cap:
        await cl.Message(content="❌ No camera available for live feed").send()
        return

    live_feed_active = True
    frame_count = 0

    # Send instructions for true video experience
    await cl.Message(
        content="🔴 **LIVE CAMERA FEED WITH CV2 WINDOW**\n\n"
                "📺 **Opening dedicated video window for smooth streaming...**\n\n"
                "✨ **Features:**\n"
                "• True continuous video (not chat snapshots)\n"
                "• Real-time object detection overlay\n"
                "• Press 'q' in video window to stop\n"
                "• Press 's' to save current frame\n\n"
                "🎯 **YOLO will detect 80+ object types in real-time**"
    ).send()

    try:
        # Create CV2 window for true video streaming
        window_name = "NAVADA-AI Live Feed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 900)  # Large window

        # Set window position (optional)
        cv2.moveWindow(window_name, 100, 100)

        detection_history = []  # Keep track of recent detections
        fps_counter = 0
        fps_start_time = time.time()

        while live_feed_active:
            frame = capture_frame(cap)
            if frame is None:
                await cl.Message(content="❌ Camera disconnected. Stopping live feed.").send()
                break

            # Validate frame
            if frame.size == 0 or len(frame.shape) != 3:
                continue

            frame_count += 1
            fps_counter += 1

            try:
                # Resize frame for better viewing
                h, w = frame.shape[:2]
                # Target larger size for CV2 window
                scale_factor = min(1200/w, 900/h, 3.0)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)

                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                frame_copy = np.ascontiguousarray(frame_resized)

                # Run YOLO detection every 2nd frame for optimal performance
                if frame_count % 2 == 0:
                    processed_frame, detections = process_yolo_detection(frame_copy)
                    detection_history = detections  # Store latest detections
                else:
                    processed_frame = frame_copy.copy()
                    detections = detection_history  # Use previous detections

                # Calculate FPS
                if fps_counter % 30 == 0:  # Update FPS every 30 frames
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start_time)
                    fps_start_time = current_time
                    cl.user_session.set("current_fps", fps)

                current_fps = cl.user_session.get("current_fps", 0)

                # Enhanced overlay for CV2 window
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Create professional overlay
                overlay_height = 120
                overlay = processed_frame.copy()
                cv2.rectangle(overlay, (0, 0), (new_w, overlay_height), (0, 0, 0), -1)
                cv2.rectangle(overlay, (0, new_h-60), (new_w, new_h), (0, 0, 0), -1)
                processed_frame = cv2.addWeighted(processed_frame, 0.75, overlay, 0.25, 0)

                # Top overlay - Title and time
                cv2.putText(processed_frame, "NAVADA-AI LIVE DETECTION",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(processed_frame, f"Time: {timestamp}",
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # FPS and frame counter
                cv2.putText(processed_frame, f"FPS: {current_fps:.1f} | Frame: {frame_count}",
                           (new_w-300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Detection info
                if detections:
                    detected_objects = list(set([d['class'] for d in detections]))
                    detection_text = f"Detecting: {', '.join(detected_objects[:3])}"  # Show max 3
                    if len(detected_objects) > 3:
                        detection_text += f" +{len(detected_objects)-3} more"
                    cv2.putText(processed_frame, detection_text,
                               (20, new_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Objects: {len(detections)}",
                               (new_w-150, new_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_frame, "No objects detected",
                               (20, new_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Control instructions
                cv2.putText(processed_frame, "Press 'q' to quit | 's' to save frame",
                           (new_w-400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Display the frame in CV2 window (TRUE VIDEO STREAMING)
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    live_feed_active = False
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"navada_live_capture_{frame_count}_{timestamp.replace(':', '-')}.jpg"
                    cv2.imwrite(save_path, processed_frame)
                    await cl.Message(content=f"📸 Frame saved as: {save_path}").send()

                # Send periodic updates to chat (every 60 frames = ~2 seconds)
                if frame_count % 60 == 0:
                    if detections:
                        detected_classes = list(set([d['class'] for d in detections]))
                        await cl.Message(
                            content=f"📺 **Live Feed Active** (Frame {frame_count}) - "
                                   f"Detecting: {', '.join(detected_classes[:5])}"
                        ).send()
                    else:
                        await cl.Message(
                            content=f"📺 **Live Feed Active** (Frame {frame_count}) - Monitoring..."
                        ).send()

            except Exception as frame_error:
                print(f"Frame processing error: {frame_error}")
                continue

            # No sleep needed - CV2 handles timing with waitKey(1)

    except Exception as e:
        print(f"Error in live feed: {e}")
        await cl.Message(content=f"❌ Live feed error: {str(e)}").send()
    finally:
        live_feed_active = False
        cv2.destroyAllWindows()
        await cl.Message(
            content="🛑 **Live feed stopped.**\n\n"
                   "✅ CV2 video window closed\n"
                   "📱 Camera released and ready for next session"
        ).send()

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    await cl.Message(
        content="# **NAVADA-AI Vision System**\n\n"
                "Welcome to your AI-powered object detection platform."
    ).send()

    # Initialize YOLO model
    initialize_yolo()

    # Initialize voice systems
    voice_initialized = initialize_voice()

    # Initialize face recognition
    if FACE_RECOGNITION_AVAILABLE:
        global face_recognizer
        try:
            face_recognizer = SimpleFaceRecognizer()
        except Exception as e:
            print(f"Face recognition failed to load: {e}")

    # Initialize camera
    cap = get_camera()
    if cap:
        cl.user_session.set("camera", cap)
    else:
        print("Camera not found. Please check your camera connection or start iVCam.")

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    global live_feed_active

    # Get camera from session
    cap = cl.user_session.get("camera")
    user_input = message.content.lower().strip()

    # Live feed commands
    if any(cmd in user_input for cmd in ["live", "start live", "live feed", "start feed"]):
        if live_feed_active:
            await cl.Message(content="🔴 Live feed is already running! Use `stop` to end it.").send()
            return

        if not cap:
            await cl.Message(content="❌ No camera available. Please restart the app.").send()
            return

        await cl.Message(content="🔴 **Starting live camera feed with object detection...**\n\n"
                                "📱 Point your camera at objects\n"
                                "🎯 YOLO will detect 80+ object types\n"
                                "🛑 Type `stop` to end the live feed").send()

        # Start live feed in background task
        asyncio.create_task(live_camera_feed())
        return

    # Stop live feed
    if any(cmd in user_input for cmd in ["stop", "stop live", "end live", "stop feed"]):
        if live_feed_active:
            live_feed_active = False
            await cl.Message(content="🛑 **Live feed stopped.**").send()
        else:
            await cl.Message(content="ℹ️ No live feed is currently running.").send()
        return

    # Photo capture
    if any(cmd in user_input for cmd in ["capture", "photo", "take photo", "picture"]):
        if live_feed_active:
            await cl.Message(content="⚠️ Please stop the live feed first before taking photos.").send()
            return

        if cap:
            await cl.Message(content="📸 Capturing photo with object detection...").send()

            frame = capture_frame(cap)
            if frame is not None:
                # Process with YOLO
                processed_frame, detections = process_yolo_detection(frame.copy())

                # Add timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Convert frame to image
                img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

                # Save image
                temp_path = f"navada_photo_{timestamp}.jpg"
                img.save(temp_path)

                # Prepare detection summary
                if detections:
                    detected_objects = {}
                    for detection in detections:
                        obj_class = detection['class']
                        if obj_class in detected_objects:
                            detected_objects[obj_class] += 1
                        else:
                            detected_objects[obj_class] = 1

                    summary = "🎯 **Objects Detected:**\n"
                    for obj, count in detected_objects.items():
                        confidence_avg = sum(d['confidence'] for d in detections if d['class'] == obj) / count
                        summary += f"• **{obj}** ({count}x) - {confidence_avg:.1%} confidence\n"
                else:
                    summary = "🔍 No objects detected in this image."

                # Send image with analysis
                elements = [
                    cl.Image(name="captured_image", path=temp_path, display="inline")
                ]

                await cl.Message(
                    content=f"📸 **Photo Captured Successfully!**\n\n{summary}",
                    elements=elements
                ).send()

            else:
                await cl.Message(content="❌ Failed to capture image. Please check your camera.").send()
        else:
            await cl.Message(content="❌ No camera available. Please restart the app.").send()

    # Camera status
    elif "status" in user_input:
        if cap and cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            status_msg = f"✅ **Camera Status: ACTIVE**\n\n"
            status_msg += f"📊 **Camera Properties:**\n"
            status_msg += f"• Resolution: {width}x{height}\n"
            status_msg += f"• FPS: {fps}\n"
            status_msg += f"• YOLO Model: {'✅ Loaded' if yolo_model else '❌ Not loaded'}\n"
            status_msg += f"• Face Recognition: {'✅ Available' if FACE_RECOGNITION_AVAILABLE else '❌ Unavailable'}\n"
            status_msg += f"• Live Feed: {'🔴 Running' if live_feed_active else '⚪ Stopped'}\n"

            await cl.Message(content=status_msg).send()
        else:
            await cl.Message(content="❌ Camera is not available.").send()

    # Voice commands
    elif any(cmd in user_input for cmd in ["speak", "voice", "talk", "say"]):
        if tts_engine or openai_client:
            # Extract text to speak
            text_to_speak = user_input.replace("speak", "").replace("voice", "").replace("talk", "").replace("say", "").strip()
            if not text_to_speak:
                text_to_speak = "Hello! NAVADA-AI voice system is working."

            await cl.Message(content=f"🎵 **Speaking**: {text_to_speak}").send()

            # Try OpenAI TTS first, fallback to basic TTS
            if openai_client:
                success = await speak_with_openai(text_to_speak)
                if not success and tts_engine:
                    # Fallback to basic TTS in a thread to avoid blocking
                    threading.Thread(target=speak_text, args=(text_to_speak,), daemon=True).start()
            elif tts_engine:
                threading.Thread(target=speak_text, args=(text_to_speak,), daemon=True).start()
        else:
            await cl.Message(content="❌ Voice systems not available. Install required dependencies.").send()

    # AI Chat commands
    elif any(cmd in user_input for cmd in ["chat", "ask", "tell me", "what do you think"]):
        await cl.Message(content="🤖 **Thinking...**").send()

        # Get detected objects from last capture if available
        detected_objects = None  # You could store this from last detection

        ai_response = await get_ai_response(user_input, detected_objects)
        await cl.Message(content=f"🤖 **NAVADA-AI**: {ai_response}").send()

        # Speak the response if voice is available
        if tts_engine or openai_client:
            if openai_client:
                await speak_with_openai(ai_response)
            elif tts_engine:
                threading.Thread(target=speak_text, args=(ai_response,), daemon=True).start()

    # Help command
    elif "help" in user_input:
        help_msg = """# 📖 **NAVADA-AI Help Guide**

## 🎮 **Available Commands:**

### 📸 **Photo & Detection:**
• `capture`, `photo`, `picture` - Take a photo with object detection
• `live`, `start live` - Start live camera feed with real-time detection
• `stop`, `stop live` - Stop the live camera feed

### 🎵 **Voice & AI:**
• `speak [text]`, `say [text]` - Text-to-speech
• `chat [message]`, `ask [question]` - AI conversation
• `tell me about [topic]` - AI responses with voice

### 📊 **System Info:**
• `status` - Check camera and system status
• `help` - Show this help guide

## 🎯 **Features:**
• **Object Detection**: Detects 80+ objects using YOLOv8s
• **Live Feed**: Real-time object detection and tracking
• **Voice Interaction**: TTS and AI conversations
• **High Quality**: 640x480 resolution for stability
• **Camera Support**: Works with laptop cameras, iVCam, USB webcams

## 📱 **Camera Setup:**
• **Laptop Camera**: Works automatically
• **iPhone**: Install iVCam app and desktop software
• **USB Webcam**: Plug and play

## 🚀 **Quick Start:**
1. Type `capture` to take a test photo
2. Type `live` to start real-time detection
3. Type `speak hello` to test voice
4. Type `chat how are you` for AI conversation
5. Point camera at objects to see AI detection

**Ready to explore AI vision? Try `capture` or `live`!** 🚀"""

        await cl.Message(content=help_msg).send()

    # Default response
    else:
        await cl.Message(
            content=f"🤖 You said: **{message.content}**\n\n"
                    "💡 **Try these commands:**\n"
                    "• `capture` - Take a photo with AI detection\n"
                    "• `live` - Start live camera feed\n"
                    "• `speak hello` - Test voice system\n"
                    "• `chat hi` - AI conversation\n"
                    "• `help` - Full command guide\n\n"
                    "Ready to detect some objects? 🎯"
        ).send()

@cl.on_chat_end
def on_chat_end():
    """Clean up resources when chat ends"""
    global live_feed_active

    # Stop live feed
    live_feed_active = False

    # Release camera
    cap = cl.user_session.get("camera")
    if cap:
        cap.release()

    cv2.destroyAllWindows()
    print("✅ Resources cleaned up")

if __name__ == "__main__":
    print("🚀 Starting NAVADA-AI Chainlit App...")
    print("📱 Make sure iVCam is running if you want to use iPhone camera")
    print("🎯 Features: Live feed, Object detection, Photo capture")