import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from PIL import Image
import av
import cv2
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import threading

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)  # Load YOLO model
    return model

# Thread lock for YOLO inference
yolo_lock = threading.Lock()

# Class names
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest"
]

# YOLO Video Processor for RTCWebcam
class YOLOProcessor(VideoProcessorBase):
    def __init__(self, model, conf_threshold, class_names):
        self.model = model
        self.conf_threshold = conf_threshold
        self.class_names = class_names

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Lock inference to avoid thread conflicts
        with yolo_lock:
            results = self.model(img, conf=self.conf_threshold, device="cpu")
            boxes = results[0].boxes

            # Annotate image with bounding boxes and labels
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = f"{self.class_names[class_id]}: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Process Image
def process_image(image, model, conf):
    results = model.predict(image, conf=conf, device="cpu")
    return results

# Process Video
def process_video(video_path, model, conf):
    cap = cv2.VideoCapture(video_path)
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, device="cpu")
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return temp_video.name

# Streamlit app UI
st.title("YOLO Object Detection App")
st.sidebar.title("Model Configuration")

# Sidebar inputs
model_path = st.sidebar.text_input("Enter the model path:", "best1.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Load model
model = None
if model_path:
    with st.spinner("Loading model..."):
        try:
            model = load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Image Detection Section
st.header("Image Detection")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image and model:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    results = process_image(np.array(image), model, confidence_threshold)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Processed Image", use_container_width=True)

    detected_classes = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = CLASS_NAMES[class_id]
        confidence = box.conf.item()
        detected_classes.append({"Class": class_name, "Confidence": confidence})
    
    if detected_classes:
        detected_df = pd.DataFrame(detected_classes)
        st.write("Detected Classes:")
        st.table(detected_df)

# Video Detection Section
st.header("Video Detection")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_video and model:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name
    
    with st.spinner("Processing video..."):
        processed_video_path = process_video(video_path, model, confidence_threshold)
        st.video(processed_video_path)
        st.download_button(
            label="Download Processed Video",
            data=open(processed_video_path, "rb").read(),
            file_name="processed_video.mp4"
        )

# WebRTC Section
st.header("Real-Time Object Detection with WebRTC")
if model:
    webrtc_streamer(
        key="yolo-webrtc",
        video_processor_factory=lambda: YOLOProcessor(
            model=model,
            conf_threshold=confidence_threshold,
            class_names=CLASS_NAMES
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.warning("Please load a model to enable real-time detection.")
