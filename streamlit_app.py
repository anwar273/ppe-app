import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import av
from dataclasses import dataclass
from typing import Dict, List, Optional
import tempfile
import pygame
import cv2
from pathlib import Path

@dataclass
class DetectionConfig:
    """Configuration settings for object detection"""
    CLASS_NAMES: List[str] = (
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", 
        "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"
    )
    EXCLUDED_CLASSES: List[int] = (8, 9)  # machinery and vehicle
    RTC_CONFIGURATION: Dict = None
    
    def __post_init__(self):
        self.RTC_CONFIGURATION = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

class DetectionModel:
    """Handles model loading and inference"""
    def __init__(self):
        self._model: Optional[YOLO] = None
        
    @st.cache_resource
    def load_model(self, model_path: str) -> YOLO:
        """Load and cache the YOLO model"""
        self._model = YOLO(model_path)
        return self._model
    
    def predict(self, image: np.ndarray, conf: float) -> List:
        """Run prediction and filter excluded classes"""
        if not self._model:
            raise ValueError("Model not loaded")
            
        results = self._model.predict(image, conf=conf, device='cpu')
        return self._filter_results(results)
    
    @staticmethod
    def _filter_results(results: List) -> List:
        """Filter out excluded classes from results"""
        if hasattr(results[0], 'boxes'):
            filtered_boxes = [
                box for box in results[0].boxes 
                if int(box.cls) not in DetectionConfig.EXCLUDED_CLASSES
            ]
            results[0].boxes = filtered_boxes
        return results

class AlertSystem:
    """Handles PPE detection alerts"""
    def __init__(self, sound_path: str):
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound(sound_path)
        
    def process(self, frame: np.ndarray, detected_classes: List[int]):
        """Process frame and trigger alerts if needed"""
        if any(cls in [2, 3, 4] for cls in detected_classes):
            self._trigger_alert(frame, "Alert: PPE not detected!", (0, 0, 255))
        else:
            self._trigger_all_clear(frame)
    
    def _trigger_alert(self, frame: np.ndarray, text: str, color: tuple):
        """Display alert on frame and play sound"""
        if not pygame.mixer.get_busy():
            pygame.mixer.Sound.play(self.alert_sound)
        self._draw_text(frame, text, color)
    
    def _trigger_all_clear(self, frame: np.ndarray):
        """Display all clear message and stop alert"""
        pygame.mixer.stop()
        self._draw_text(frame, "All clear: PPE detected!", (0, 255, 0))
    
    @staticmethod
    def _draw_text(frame: np.ndarray, text: str, color: tuple):
        """Draw text on frame"""
        position = ((frame.shape[1]//2) - 20, 30)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

class StreamlitInterface:
    """Manages Streamlit UI and interactions"""
    def __init__(self):
        self.config = DetectionConfig()
        self.model = DetectionModel()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup main UI components"""
        st.title("Object Detection App")
        st.markdown("Upload an image, video, or use the webcam for real-time object detection using YOLO.")
        
        self._setup_sidebar()
        self._setup_image_detection()
        self._setup_video_detection()
        self._setup_webcam_detection()
    
    def _setup_sidebar(self):
        """Setup sidebar configuration"""
        st.sidebar.image("SECURE LOGO.png", width=120, caption="SECURE Vision")
        st.sidebar.title("Model Configuration")
        
        model_path = st.sidebar.text_input("Enter the model path:", "best1.pt")
        if model_path:
            self._load_model(model_path)
        
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.25
        )
    
    def _load_model(self, model_path: str):
        """Load the detection model"""
        with st.spinner("Loading model..."):
            try:
                self.model.load_model(model_path)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    def _setup_image_detection(self):
        """Setup image detection section"""
        st.header("Image Detection")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file and self.model._model:
            self._process_image(uploaded_file)
    
    def _process_image(self, uploaded_file):
        """Process uploaded image"""
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        results = self.model.predict(np.array(image), self.confidence_threshold)
        self._display_results(results)
    
    def _display_results(self, results):
        """Display detection results"""
        img_res = results[0].plot()
        st.image(img_res, caption="Detected Image", use_container_width=True)
        
        detected_classes = self._get_detection_data(results)
        if detected_classes:
            self._plot_detection_results(detected_classes)
    
    def _get_detection_data(self, results) -> List[Dict]:
        """Extract detection data from results"""
        detected_classes = []
        for box in results[0].boxes:
            detected_classes.append({
                "Detected Classes": self.config.CLASS_NAMES[int(box.cls)],
                "Confidence": f"{box.conf.item():.2f}",
            })
        return detected_classes
    
    def _plot_detection_results(self, detected_classes: List[Dict]):
        """Create and display detection results plot"""
        df = pd.DataFrame(detected_classes)
        fig = px.bar(
            df,
            x="Detected Classes",
            y=df["Confidence"].astype(float),
            title="Detected Classes with Confidence Scores",
            labels={"Confidence": "Confidence Scores"},
            color="Confidence",
        )
        st.plotly_chart(fig)

    def _setup_video_detection(self):
        """Setup video detection section"""
        st.header("Video Detection")
        video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
        
        if video_file and self.model._model:
            self._process_video(video_file)
    
    def _process_video(self, video_file):
        """Process uploaded video"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            with st.spinner("Processing video..."):
                processed_video = self._run_video_detection(temp_file.name)
                self._display_video_results(processed_video)
    
    def _run_video_detection(self, video_path: str) -> str:
        """Run detection on video file"""
        output_path = self._setup_video_writer(video_path)
        self._process_video_frames(video_path, output_path)
        return output_path
    
    def _setup_video_writer(self, video_path: str) -> str:
        """Setup video writer for processing"""
        cap = cv2.VideoCapture(video_path)
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, frame_size)
        cap.release()
        return temp_video.name
    
    def _process_video_frames(self, video_path: str, output_path: str):
        """Process individual video frames"""
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model.predict(frame, self.confidence_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
        cap.release()
        out.release()
    
    def _display_video_results(self, video_path: str):
        """Display processed video and download button"""
        st.video(video_path)
        st.download_button(
            "Download Processed Video",
            Path(video_path).read_bytes(),
            "processed_video.mp4"
        )
    
    def _setup_webcam_detection(self):
        """Setup webcam detection section"""
        st.header("Real-Time Webcam Detection")
        if self.model._model:
            self._start_webcam_detection()
        else:
            st.warning("Model is not loaded. Please load the model to start detection.")
    
    def _start_webcam_detection(self):
        """Start webcam detection"""
        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=self.config.RTC_CONFIGURATION,
            video_processor_factory=self._create_video_processor,
            media_stream_constraints={"video": True, "audio": False},
        )
    
    def _create_video_processor(self):
        """Create video processor for webcam detection"""
        processor = VideoProcessor()
        processor.model = self.model._model
        processor.confidence_threshold = self.confidence_threshold
        return processor

class VideoProcessor:
    """Processes video frames for webcam detection"""
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.25
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process received video frame"""
        if not self.model:
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=self.confidence_threshold)
        results[0].boxes = [
            box for box in results[0].boxes 
            if int(box.cls) not in DetectionConfig.EXCLUDED_CLASSES
        ]
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

if __name__ == "__main__":
    app = StreamlitInterface()
