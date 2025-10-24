#!/usr/bin/env python3
"""
NAVADA-AI Vision System - Streamlit Version
Ultra-simple interface with true continuous live streaming
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import json
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import openai
from typing import List, Dict

# Import camera utils for stable DirectShow backend
try:
    from camera_utils import get_camera_with_error_handling
    CAMERA_UTILS_AVAILABLE = True
except ImportError:
    CAMERA_UTILS_AVAILABLE = False

# Page configuration - minimal setup
st.set_page_config(
    page_title="NAVADA-AI Vision System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'live_feed_active' not in st.session_state:
    st.session_state.live_feed_active = False
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'current_detections' not in st.session_state:
    st.session_state.current_detections = []
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None
if 'refresh_sidebar' not in st.session_state:
    st.session_state.refresh_sidebar = 0
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

@st.cache_resource
def load_yolo_model():
    """Load YOLO model (cached)"""
    try:
        model = YOLO('yolov8s.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Database Management
@st.cache_resource
def init_database():
    """Initialize SQLite database for photo storage"""
    db_path = "navada_photos.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Create photos table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tags TEXT,
            location TEXT,
            activity TEXT,
            people TEXT,
            notes TEXT,
            detected_objects TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create chat history table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn

def save_photo_to_db(conn, photo_data):
    """Save photo metadata to database"""
    try:
        conn.execute("""
            INSERT OR REPLACE INTO photos
            (filename, file_path, timestamp, tags, location, activity, people, notes, detected_objects)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            photo_data['filename'],
            photo_data['file_path'],
            photo_data['timestamp'],
            json.dumps(photo_data['tags']),
            photo_data['location'],
            photo_data['activity'],
            photo_data['people'],
            photo_data['notes'],
            json.dumps(photo_data['detected_objects'])
        ))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def get_photos_from_db(conn, search_term=""):
    """Get photos from database with optional search"""
    try:
        if search_term:
            query = """
                SELECT * FROM photos
                WHERE tags LIKE ? OR location LIKE ? OR activity LIKE ?
                OR people LIKE ? OR notes LIKE ? OR detected_objects LIKE ?
                ORDER BY created_at DESC
            """
            search_param = f"%{search_term}%"
            cursor = conn.execute(query, (search_param, search_param, search_param,
                                        search_param, search_param, search_param))
        else:
            cursor = conn.execute("SELECT * FROM photos ORDER BY created_at DESC")

        photos = []
        for row in cursor.fetchall():
            photos.append({
                'id': row[0],
                'filename': row[1],
                'path': row[2],
                'timestamp': row[3],
                'tags': json.loads(row[4]) if row[4] else [],
                'location': row[5] or '',
                'activity': row[6] or '',
                'people': row[7] or '',
                'notes': row[8] or '',
                'detected_objects': json.loads(row[9]) if row[9] else [],
                'created_at': row[10]
            })
        return photos
    except Exception as e:
        st.error(f"Database query error: {e}")
        return []

def get_ai_response(message, photos_context=""):
    """Get AI response using simple mock for now (can be enhanced with real API)"""
    try:
        # For now, return a simple response - you can integrate with OpenAI later
        if "photo" in message.lower() or "image" in message.lower():
            return f"I can see you're asking about photos. {photos_context} How can I help you analyze your images?"
        elif "detect" in message.lower():
            return "I can help you understand the object detection results. What would you like to know?"
        elif "hello" in message.lower() or "hi" in message.lower():
            return "Hello! I'm your NAVADA-AI assistant. I can help you with photo analysis, object detection, and managing your image gallery. What would you like to know?"
        else:
            return f"I understand you're asking: '{message}'. I'm here to help with your photo analysis and object detection needs. Could you be more specific about what you'd like to know?"
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def save_chat_to_db(conn, message, response):
    """Save chat conversation to database"""
    try:
        conn.execute("""
            INSERT INTO chat_history (message, response)
            VALUES (?, ?)
        """, (message, response))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Chat save error: {e}")
        return False

def get_camera():
    """Initialize camera with DirectShow backend"""
    if CAMERA_UTILS_AVAILABLE:
        return get_camera_with_error_handling()

    # Fallback to DirectShow
    for index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Set properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
                cap.release()
        except Exception:
            continue
    return None

def capture_frame(cap):
    """Capture frame with robustness"""
    if not cap or not cap.isOpened():
        return None

    try:
        for attempt in range(3):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    h, w, c = frame.shape
                    if h > 10 and w > 10:
                        # Create contiguous array
                        new_frame = np.ascontiguousarray(frame)
                        return new_frame
    except Exception as e:
        st.error(f"Frame capture error: {e}")
    return None

def process_yolo_detection(frame, model):
    """Process YOLO detection on frame"""
    try:
        results = model(frame, conf=0.25, verbose=False)
        detections = []

        # Draw bounding boxes
        processed_frame = frame.copy()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                # Draw bounding box
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(processed_frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

        return processed_frame, detections
    except Exception as e:
        st.error(f"YOLO processing error: {e}")
        return frame, []

def save_photo_with_tags(frame, detections, metadata_input):
    """Save photo with enhanced metadata and tags"""
    try:
        # Create photos directory if it doesn't exist
        photos_dir = "photos"
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)

        # Generate timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_filename = f"navada_photo_{timestamp}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)

        # Save the photo
        cv2.imwrite(photo_path, frame)

        # Handle both old and new metadata formats
        if isinstance(metadata_input, dict):
            # New enhanced format
            metadata = {
                "filename": photo_filename,
                "timestamp": datetime.now().isoformat(),
                "tags": metadata_input.get("tags", []),
                "location": metadata_input.get("location", ""),
                "activity": metadata_input.get("activity", ""),
                "people": metadata_input.get("people", ""),
                "notes": metadata_input.get("notes", ""),
                "detected_objects": [
                    {
                        "class": detection["class"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"]
                    }
                    for detection in detections
                ]
            }
        else:
            # Legacy format (just tags list)
            metadata = {
                "filename": photo_filename,
                "timestamp": datetime.now().isoformat(),
                "tags": metadata_input if isinstance(metadata_input, list) else [],
                "location": "",
                "activity": "",
                "people": "",
                "notes": "",
                "detected_objects": [
                    {
                        "class": detection["class"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"]
                    }
                    for detection in detections
                ]
            }

        # Save metadata to JSON file (backup)
        metadata_filename = f"navada_photo_{timestamp}.json"
        metadata_path = os.path.join(photos_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save to database
        if 'db_conn' in st.session_state and st.session_state.db_conn:
            db_photo_data = {
                'filename': photo_filename,
                'file_path': photo_path,
                'timestamp': metadata['timestamp'],
                'tags': metadata['tags'],
                'location': metadata['location'],
                'activity': metadata['activity'],
                'people': metadata['people'],
                'notes': metadata['notes'],
                'detected_objects': metadata['detected_objects']
            }
            save_photo_to_db(st.session_state.db_conn, db_photo_data)

        return photo_path, len(detections)
    except Exception as e:
        st.error(f"Error saving photo: {e}")
        return None, 0

def load_saved_photos():
    """Load all saved photos and their metadata"""
    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        return []

    photos = []
    for filename in os.listdir(photos_dir):
        if filename.endswith('.jpg'):
            photo_path = os.path.join(photos_dir, filename)
            json_path = os.path.join(photos_dir, filename.replace('.jpg', '.json'))

            # Load metadata if exists
            metadata = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass

            photos.append({
                'filename': filename,
                'path': photo_path,
                'metadata': metadata,
                'timestamp': metadata.get('timestamp', ''),
                'tags': metadata.get('tags', []),
                'detected_objects': metadata.get('detected_objects', []),
                'location': metadata.get('location', ''),
                'activity': metadata.get('activity', ''),
                'people': metadata.get('people', ''),
                'notes': metadata.get('notes', '')
            })

    # Sort by timestamp (newest first)
    photos.sort(key=lambda x: x['timestamp'], reverse=True)
    return photos

def search_photos(photos, search_term):
    """Search photos by tags, detected objects, or filename"""
    if not search_term:
        return photos

    search_term = search_term.lower()
    filtered_photos = []

    for photo in photos:
        # Search in tags
        if any(search_term in tag.lower() for tag in photo['tags']):
            filtered_photos.append(photo)
            continue

        # Search in detected objects
        if any(search_term in obj['class'].lower() for obj in photo['detected_objects']):
            filtered_photos.append(photo)
            continue

        # Search in filename
        if search_term in photo['filename'].lower():
            filtered_photos.append(photo)
            continue

        # Search in location
        if search_term in photo.get('location', '').lower():
            filtered_photos.append(photo)
            continue

        # Search in activity
        if search_term in photo.get('activity', '').lower():
            filtered_photos.append(photo)
            continue

        # Search in people
        if search_term in photo.get('people', '').lower():
            filtered_photos.append(photo)
            continue

        # Search in notes
        if search_term in photo.get('notes', '').lower():
            filtered_photos.append(photo)

    return filtered_photos

def main():
    """Ultra-simple Streamlit app - just title and video"""

    # Custom CSS for black buttons with white text
    st.markdown("""
    <style>
    .stButton > button[kind="secondary"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
    }
    .stSidebar {
        background-color: #f8f9fa !important;
    }
    .stExpander {
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Minimal header - exactly like Chainlit
    st.title("NAVADA-AI Vision System")

    # Initialize database connection
    if st.session_state.db_conn is None:
        st.session_state.db_conn = init_database()

    # Enhanced Sidebar with Tabs
    with st.sidebar:
        # Create tabs for different functions
        tab1, tab2, tab3, tab4 = st.tabs(["📸 Gallery", "🤖 AI Chat", "📊 Stats", "⚙️ Settings"])

        # Tab 1: Enhanced Photo Gallery
        with tab1:
            st.subheader("📁 Photo Gallery")

            # Load photos from database with refresh trigger
            db_photos = get_photos_from_db(st.session_state.db_conn)

            if db_photos:
                st.success(f"**{len(db_photos)} photos in database**")

                # Search functionality
                search_term = st.text_input("🔍 Search photos",
                                          placeholder="Search by tags, objects, location, activity, people, notes...",
                                          key="gallery_search")

                # Filter photos based on search
                if search_term:
                    filtered_photos = get_photos_from_db(st.session_state.db_conn, search_term)
                else:
                    filtered_photos = db_photos

                if search_term and not filtered_photos:
                    st.warning("No photos found matching your search.")
                elif filtered_photos:
                    st.info(f"Showing {len(filtered_photos)} photos")

                    # Display photos in sidebar with better layout
                    for i, photo in enumerate(filtered_photos[:10]):  # Limit to 10 recent photos
                        with st.expander(f"📸 {photo['filename'][:15]}...", expanded=(i==0)):
                            # Display thumbnail
                            if os.path.exists(photo['path']):
                                st.image(photo['path'], width=180, caption=f"Photo {i+1}")

                            # Show enhanced metadata with better styling
                            if photo['timestamp']:
                                st.markdown(f"🕐 **Date:** {photo['timestamp'][:19].replace('T', ' ')}")

                            if photo['tags']:
                                tags_str = ", ".join([f"`{tag}`" for tag in photo['tags']])
                                st.markdown(f"🏷️ **Tags:** {tags_str}")

                            if photo.get('location'):
                                st.markdown(f"📍 **Location:** {photo['location']}")

                            if photo.get('activity'):
                                st.markdown(f"🎯 **Activity:** {photo['activity']}")

                            if photo.get('people'):
                                st.markdown(f"👥 **People:** {photo['people']}")

                            if photo.get('notes'):
                                st.markdown(f"📋 **Notes:** {photo['notes']}")

                            if photo['detected_objects']:
                                objects = [f"`{obj['class']}` ({obj['confidence']:.2f})"
                                         for obj in photo['detected_objects']]
                                st.markdown(f"🤖 **Objects:** {', '.join(objects)}")

                            # Download button
                            if os.path.exists(photo['path']):
                                with open(photo['path'], 'rb') as f:
                                    st.download_button(
                                        "⬇️ Download",
                                        f.read(),
                                        file_name=photo['filename'],
                                        mime="image/jpeg",
                                        key=f"download_{photo['id']}"
                                    )

                    if len(filtered_photos) > 10:
                        st.info(f"Showing first 10 of {len(filtered_photos)} photos. Use search to narrow results.")

            else:
                st.info("📸 No photos saved yet.\n\nStart live feed and click 'Capture' to save your first photo!")

        # Tab 2: AI Chat Assistant
        with tab2:
            st.subheader("🤖 AI Assistant")

            # Chat interface
            chat_container = st.container()

            with chat_container:
                # Display chat history
                for i, (msg, resp) in enumerate(st.session_state.chat_messages[-5:]):  # Last 5 messages
                    st.markdown(f"**You:** {msg}")
                    st.markdown(f"**AI:** {resp}")
                    st.markdown("---")

            # Chat input
            user_message = st.text_input("Ask me about your photos or detection...",
                                       placeholder="e.g., 'What objects were detected in my latest photo?'",
                                       key="chat_input")

            if st.button("💬 Send", key="send_chat"):
                if user_message:
                    # Get context from recent photos
                    recent_photos = get_photos_from_db(st.session_state.db_conn)[:3]
                    photo_context = ""
                    if recent_photos:
                        photo_context = f"Your recent photos contain: {', '.join([obj['class'] for photo in recent_photos for obj in photo['detected_objects']])}"

                    # Get AI response
                    ai_response = get_ai_response(user_message, photo_context)

                    # Add to chat history
                    st.session_state.chat_messages.append((user_message, ai_response))

                    # Save to database
                    save_chat_to_db(st.session_state.db_conn, user_message, ai_response)

                    # Force refresh
                    st.rerun()

            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

        # Tab 3: Statistics and Analytics
        with tab3:
            st.subheader("📊 Photo Statistics")

            all_photos = get_photos_from_db(st.session_state.db_conn)

            if all_photos:
                # Basic stats
                st.metric("Total Photos", len(all_photos))

                # Most common objects
                all_objects = []
                for photo in all_photos:
                    for obj in photo['detected_objects']:
                        all_objects.append(obj['class'])

                if all_objects:
                    from collections import Counter
                    object_counts = Counter(all_objects)
                    st.write("**Most Detected Objects:**")
                    for obj, count in object_counts.most_common(5):
                        st.write(f"• {obj}: {count} times")

                # Recent activity
                st.write("**Recent Activity:**")
                for photo in all_photos[:3]:
                    st.write(f"• {photo['filename'][:20]}... - {len(photo['detected_objects'])} objects")

            else:
                st.info("📈 No statistics available yet. Take some photos first!")

        # Tab 4: Settings and Info
        with tab4:
            st.subheader("⚙️ Settings")

            # Storage info
            st.write("**📂 Storage Locations:**")
            st.code(f"Photos: {os.path.abspath('photos')}")
            st.code(f"Database: {os.path.abspath('navada_photos.db')}")

            # Refresh button
            if st.button("🔄 Refresh Gallery", key="refresh_gallery"):
                st.session_state.refresh_sidebar += 1
                st.success("Gallery refreshed!")
                st.rerun()

            # Database info
            if st.session_state.db_conn:
                cursor = st.session_state.db_conn.execute("SELECT COUNT(*) FROM photos")
                photo_count = cursor.fetchone()[0]
                cursor = st.session_state.db_conn.execute("SELECT COUNT(*) FROM chat_history")
                chat_count = cursor.fetchone()[0]

                st.write("**📊 Database Stats:**")
                st.write(f"• Photos: {photo_count}")
                st.write(f"• Chat messages: {chat_count}")

            # Clear data options
            st.write("**🗑️ Data Management:**")
            if st.button("Clear Chat History", key="clear_all_chat"):
                st.session_state.db_conn.execute("DELETE FROM chat_history")
                st.session_state.db_conn.commit()
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")

            st.markdown("---")
            st.markdown("**NAVADA-AI Vision System**")
            st.markdown("*Designed by Lee Akpareva MBA, MA*")

    # Auto-initialize everything on startup
    if st.session_state.camera is None:
        st.session_state.camera = get_camera()

    if st.session_state.yolo_model is None:
        st.session_state.yolo_model = load_yolo_model()

    # Simple controls - LIVE/STOP/CAPTURE buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

    with col1:
        if st.button("LIVE", type="secondary", width="stretch"):
            st.session_state.live_feed_active = True

    with col2:
        if st.button("STOP", type="secondary", width="stretch"):
            st.session_state.live_feed_active = False

    with col3:
        capture_photo = st.button("CAPTURE", type="secondary", width="stretch",
                                disabled=not st.session_state.live_feed_active)

    # Photo capture modal/section
    if capture_photo:
        if st.session_state.current_frame is not None:
            st.write("---")
            st.subheader("📸 Save Photo with Tags")

            # Show detected objects
            if st.session_state.current_detections:
                detected_list = [f"• {d['class']} ({d['confidence']:.2f})" for d in st.session_state.current_detections]
                st.write("**Detected Objects:**")
                st.write("\n".join(detected_list))

            # Enhanced tagging section
            st.write("**🏷️ Add Tags & Details:**")

            # Main tags
            tag_input = st.text_input("📝 Tags (comma-separated):",
                                     placeholder="e.g. family, kitchen, morning, meeting")

            # Additional details
            col_detail1, col_detail2 = st.columns(2)
            with col_detail1:
                location = st.text_input("📍 Location:", placeholder="e.g. Home, Office, Park")
                activity = st.text_input("🎯 Activity:", placeholder="e.g. Cooking, Meeting, Exercise")

            with col_detail2:
                people = st.text_input("👥 People:", placeholder="e.g. John, Sarah")
                notes = st.text_input("📋 Notes:", placeholder="Additional details")

            # Save buttons
            col_save1, col_save2 = st.columns(2)
            with col_save1:
                if st.button("💾 Save Photo", type="primary"):
                    # Combine all details into enhanced metadata
                    tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
                    enhanced_metadata = {
                        "tags": tags,
                        "location": location.strip() if location else "",
                        "activity": activity.strip() if activity else "",
                        "people": people.strip() if people else "",
                        "notes": notes.strip() if notes else ""
                    }
                    photo_path, obj_count = save_photo_with_tags(
                        st.session_state.current_frame,
                        st.session_state.current_detections,
                        enhanced_metadata
                    )
                    if photo_path:
                        st.success(f"✅ Photo saved! ({obj_count} objects detected)")
                        st.info(f"📁 Saved to: {photo_path}")

                        # Trigger real-time sidebar update
                        st.session_state.refresh_sidebar += 1

                        # Show immediate feedback and refresh
                        time.sleep(0.5)  # Brief pause to show success message
                        st.rerun()
                    else:
                        st.error("❌ Failed to save photo")

            with col_save2:
                if st.button("❌ Cancel"):
                    st.rerun()

            st.write("---")
        else:
            st.warning("⚠️ No frame available yet. Wait a moment for the camera to capture a frame.")

    # Main video area - full width
    video_placeholder = st.empty()

    # Simple status area
    status_placeholder = st.empty()

    # Live feed processing
    if st.session_state.live_feed_active and st.session_state.camera and st.session_state.yolo_model:
        frame_count = 0

        # Continuous loop for live feed
        while st.session_state.live_feed_active:
            frame = capture_frame(st.session_state.camera)

            if frame is None:
                status_placeholder.error("❌ Camera disconnected")
                break

            frame_count += 1

            # Process every 2nd frame for performance
            if frame_count % 2 == 0:
                # Process with YOLO
                processed_frame, detections = process_yolo_detection(
                    frame, st.session_state.yolo_model
                )

                # Store current frame and detections for photo capture
                st.session_state.current_frame = processed_frame.copy()
                st.session_state.current_detections = detections

                # Simple timestamp overlay
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(processed_frame, f"LIVE - {timestamp}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display frame (TRUE CONTINUOUS STREAMING)
                video_placeholder.image(processed_frame, channels="BGR", width="stretch")

                # Simple status update
                if detections:
                    detected_objects = [d['class'] for d in detections]
                    status_placeholder.success(f"🎯 Detecting: {', '.join(set(detected_objects))} | 📸 Click Capture to save")
                else:
                    status_placeholder.info("👁️ Monitoring... | 📸 Click Capture to save current view")

            # Fast refresh rate
            time.sleep(0.03)

    else:
        # Show static camera preview when not in live mode
        if st.session_state.camera:
            # Single frame preview when not recording
            preview_frame = capture_frame(st.session_state.camera)
            if preview_frame is not None:
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(preview_frame, f"READY - {timestamp}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                video_placeholder.image(preview_frame, channels="BGR", width="stretch")
                status_placeholder.info("📹 Camera ready - Click 'Start Live Feed' to begin detection")
            else:
                video_placeholder.error("❌ Camera not available")
                status_placeholder.error("Please check camera connection")
        else:
            video_placeholder.error("❌ No camera detected")
            status_placeholder.error("Please ensure camera is connected and not in use by another app")

if __name__ == "__main__":
    main()