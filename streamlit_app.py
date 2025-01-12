import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import tempfile
import pygame
import av

# Load YOLO pretrained model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

CLASS_NAMES = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"]
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Count classes except 'vehicle' and 'machinery'
def count_classes(boxes):
    counts = {name: 0 for name in CLASS_NAMES}
    for box in boxes:
        class_idx = int(box.cls)
        if class_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_idx]
            counts[class_name] += 1
    return counts

# Process image
def process_image(image, model, conf):
    results = model.predict(image, conf=conf)
    return results

# Real-time webcam detection
class VideoProcessor:
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=self.confidence_threshold)
        results[0].boxes = [box for box in results[0].boxes if int(box.cls) not in [8, 9]]
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("Object Detection App")
st.sidebar.image("SECURE LOGO.png", width=120, caption="SECURE Vision")
st.sidebar.title("Model Configuration")
model_path = st.sidebar.text_input("Enter the model path:", "best1.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

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
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if model:
        results = process_image(np.array(image), model, confidence_threshold)
        img_res = results[0].plot()
        st.image(img_res, caption="Detected Image", use_container_width=True)

        detected_classes = [{"Detected Classes": CLASS_NAMES[int(box.cls)], "Confidence": f"{box.conf.item():.2f}"} for box in results[0].boxes]
        detected_classes_df = pd.DataFrame(detected_classes)
        fig = px.bar(detected_classes_df, x="Detected Classes", y=detected_classes_df["Confidence"].astype(float))
        st.plotly_chart(fig)

# Webcam Detection
st.header("Real-Time Webcam Detection")
if model:
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: VideoProcessor(model, confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
    )
