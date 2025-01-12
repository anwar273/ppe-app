from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# RTC Configuration for WebRTC
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Define the video processor
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

# Webcam Detection Section
st.header("Real-Time Webcam Detection")
if model:
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: VideoProcessor(model, confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        st.success("Webcam is active. Object detection running!")
else:
    st.warning("Model is not loaded. Please load the model to enable webcam detection.")
