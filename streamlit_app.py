import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import numpy as np

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)  # Load the YOLO model
    return model

# YOLO Video Transformer for WebRTC
class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model, conf_threshold, class_names):
        self.model = model
        self.conf_threshold = conf_threshold
        self.class_names = class_names

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform YOLO inference
        results = self.model(img, conf=self.conf_threshold, device="cpu")
        boxes = results[0].boxes  # Extract bounding boxes
        
        # Annotate the image with YOLO results
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            # Draw bounding boxes and labels
            label = f"{self.class_names[class_id]}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Sidebar Configuration
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

# WebRTC Section
st.header("Real-Time Detection with WebRTC")
if model:
    webrtc_streamer(
        key="yolo-realtime",
        video_transformer_factory=lambda: YOLOTransformer(
            model=model,
            conf_threshold=confidence_threshold,
            class_names=[
                "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", 
                "Person", "Safety Cone", "Safety Vest"
            ],
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,  # Enable async processing for better performance
    )
else:
    st.warning("Please load the model to enable real-time detection.")
