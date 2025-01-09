import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import av

# Constants
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"
]
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Utility Functions
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def count_classes(boxes):
    counts = {name: 0 for name in CLASS_NAMES}
    for box in boxes:
        class_idx = int(box.cls)
        if class_idx < len(CLASS_NAMES):
            counts[CLASS_NAMES[class_idx]] += 1
    return counts

def process_image(image, model, conf):
    results = model.predict(image, conf=conf, device='cpu')
    results[0].boxes = [box for box in results[0].boxes if int(box.cls) not in [8, 9]]
    return results

# Video Processing Class
class VideoProcessor:
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.25

    def recv(self, frame):
        if not self.model:
            return frame

        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=self.confidence_threshold)
        results[0].boxes = [box for box in results[0].boxes if int(box.cls) not in [8, 9]]
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamlit UI
st.title("Object Detection App")
st.markdown("Upload an image, video, or use the webcam for real-time object detection using YOLO.")

# Sidebar Configuration
st.sidebar.image("SECURE LOGO.png", width=120, caption="SECURE Vision")
st.sidebar.title("Model Configuration")
model_path = st.sidebar.text_input("Enter the model path:", "best1.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Load Model
model = load_model(model_path) if model_path else None
if model:
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.error("Model not loaded.")

# Image Detection
st.header("Image Detection")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = process_image(np.array(image), model, confidence_threshold)
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Detected Image", use_container_width=True)

    detected_classes = [
        {"Detected Classes": CLASS_NAMES[int(box.cls)], "Confidence": f"{box.conf.item():.2f}"}
        for box in results[0].boxes
    ]
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

# Video Detection
st.header("Video Detection")
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
if video_file and model:
    with st.spinner("Processing video..."):
        # Save the uploaded video to a temporary path
        video_path = f"temp_{video_file.name}"
        with open(video_path, "wb") as temp_file:
            temp_file.write(video_file.read())
        
        # Process the video with the model
        processed_video = model.predict(video_path, conf=confidence_threshold, save=True)
        
        # Display the processed video
        st.video(processed_video[0].save_path)

        # Optional: Offer a download button for the processed video
        st.download_button(
            label="Download Processed Video",
            data=open(processed_video[0].save_path, "rb").read(),
            file_name="processed_video.mp4",
            mime="video/mp4",
        )


# Webcam Detection
st.header("Real-Time Webcam Detection")
if model:
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: VideoProcessor(),
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.warning("Model is not loaded. Please load the model to start detection.")
