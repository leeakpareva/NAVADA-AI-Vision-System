# NAVADA-AI Vision System

**Advanced Object Detection & Recognition Platform**
*Designed by Lee Akpareva MBA, MA*

## 🚀 Quick Start

**Run the app:**
```bash
streamlit run streamlit_app.py
```

### 📸 **Live Detection**
- Click **▶️ Start Live Feed** - Start real-time camera feed with detection
- Click **⏹️ Stop** - Stop the live camera feed

## 🎯 **Core Features**

### **Object Detection**
- **80+ Objects**: People, vehicles, electronics, food, household items
- **YOLOv8s Model**: High-accuracy real-time detection
- **Confidence Scoring**: Shows detection certainty levels
- **Smart Filtering**: Removes false positives automatically

### **Live Camera Feed**
- **Real-time Detection**: Live object identification
- **Multiple Camera Support**: Laptop, iPhone (iVCam), USB webcams
- **Performance Optimized**: Fast refresh rate for smooth streaming
- **Visual Overlays**: Timestamps and detection status

## 📱 **Camera Setup**

### **Laptop Camera**
- Works automatically - no setup required
- Built-in cameras detected first

### **iPhone via iVCam**
1. Install **iVCam** app on iPhone
2. Install **iVCam** desktop software on computer
3. Connect both to same WiFi network
4. Start iVCam → NAVADA-AI auto-detects it

### **USB Webcam**
- Plug and play support
- Automatically detected and configured

## 🔧 **System Requirements**

- **Camera**: USB webcam, built-in camera, or iPhone with iVCam
- **Python 3.8+** with required dependencies
- **OpenCV** for computer vision processing
- **YOLOv8s** model (downloads automatically)

## 🎯 **Detected Object Classes**

NAVADA-AI can detect 80+ object types including:

**People & Animals**: person, cat, dog, horse, cow, elephant, bird
**Vehicles**: car, motorcycle, bicycle, airplane, bus, truck, boat
**Electronics**: cell phone, laptop, tv, mouse, remote, keyboard
**Food**: banana, apple, sandwich, pizza, cake, orange, broccoli
**Household**: chair, couch, bed, microwave, oven, refrigerator
**Sports**: sports ball, tennis racket, skateboard, surfboard
**And many more...**

## 💡 **Tips for Best Results**

1. **Good Lighting**: Ensure adequate lighting for clear detection
2. **Object Size**: Objects should be clearly visible and not too small
3. **Camera Position**: Point camera directly at objects for best accuracy
4. **Stable Connection**: For iVCam, ensure strong WiFi connection

## 🆘 **Troubleshooting**

**Camera Not Detected:**
- Check camera connections
- For iVCam: Ensure both devices on same WiFi
- Try restarting the application

**Poor Detection:**
- Improve lighting conditions
- Move objects closer to camera
- Ensure objects are clearly visible

**Live Feed Issues:**
- Click Stop to end current feed before starting new one
- Check camera isn't being used by another application

## 📦 **Installation**

```bash
# Install dependencies
pip install streamlit opencv-python ultralytics numpy

# Run the application
streamlit run streamlit_app.py
```

## 🏗️ **Project Structure**

```
ml_terminal_project/
├── streamlit_app.py          # Main Streamlit application
├── camera_utils.py           # Camera initialization utilities
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

Ready to explore AI vision? Click **Start Live Feed** to begin! 🚀