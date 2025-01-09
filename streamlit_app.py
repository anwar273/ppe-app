import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import av

# Load YOLO pretrained model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)  # Load the YOLO model
    return model

# Class names except machinery and vehicle
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"
]

# Count classes except 'vehicle' and 'machinery'
def count_classes(boxes):
    counts = {name: 0 for name in CLASS_NAMES}
    for box in boxes:
        class_idx = int(box.cls)
        if class_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_idx]
            counts[class_name] += 1
    return counts

# VideoProcessor class for streamlit-webrtc
class VideoProcessor:
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.25

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection
        results = self.model(img, conf=self.confidence_threshold)
        filtered_boxes = []
        if hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                if int(box.cls) not in [8, 9]:
                    filtered_boxes.append(box)
            results[0].boxes = filtered_boxes

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamlit app layout
st.title("Object Detection App")
st.markdown("Upload an image, video, or use the webcam for real-time object detection using YOLO.")

# Sidebar logo and configuration
st.sidebar.image("SECURE LOGO.png", width=120, caption="SECURE Vision")
st.sidebar.title("Model Configuration")
model_path = st.sidebar.text_input("Enter the model path:", "best1.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Load the model
model = None
if model_path:
    with st.spinner("Loading model..."):
        try:
            model = load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Image upload
st.header("Image Detection")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(np.array(image), conf=confidence_threshold)
    img_res = results[0].plot()
    st.image(img_res, caption="Detected Image", use_container_width=True)

    detected_classes = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = CLASS_NAMES[class_id]
        confidence = box.conf.item()
        detected_classes.append({"Detected Classes": class_name, "Confidence": f"{confidence:.2f}"})

    detected_classes_df = pd.DataFrame(detected_classes)
    fig = px.bar(
        detected_classes_df,
        x="Detected Classes",
        y=detected_classes_df["Confidence"].astype(float),
        title="Detected Classes with Confidence Scores",
        labels={"Confidence": "Confidence Scores"},
        color="Confidence",
    )
    st.plotly_chart(fig)

# Video upload
st.header("Video Detection")
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
if video_file and model:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name
    with st.spinner("Processing video..."):
        processed_video = model.predict(video_path, conf=confidence_threshold, save=True)
        st.video(processed_video[0].save_path)

# Real-time webcam detection
st.header("Real-Time Webcam Detection")
if model:
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: VideoProcessor(),
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.warning("Model is not loaded. Please load the model to start detection.")
