import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2
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

# YOLO Video Transformer
class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model, conf_threshold, class_names):
        self.model = model
        self.conf_threshold = conf_threshold
        self.class_names = class_names

    def transform(self, frame):
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

# Streamlit app UI
st.title("YOLO Real-Time Object Detection")
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

# WebRTC section
st.header("Real-Time Object Detection with WebRTC")
if model:
    webrtc_streamer(
        key="yolo-webrtc",
        video_transformer_factory=lambda: YOLOTransformer(
            model=model,
            conf_threshold=confidence_threshold,
            class_names=CLASS_NAMES
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
else:
    st.warning("Please load a model to enable real-time detection.")
