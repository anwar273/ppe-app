import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import pygame
import pandas as pd
import plotly.express as px

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

# Process image
def process_image(image, model, conf):
    results = model.predict(image, conf=conf, device='cpu')
    filtered_boxes = []
    if hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            if int(box.cls) not in [8, 9]:
                filtered_boxes.append(box)
        results[0].boxes = filtered_boxes
    return results

# Process video and save results
def process_video(video_path, model, conf):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (frame_width, frame_height))

    class_counts = {name: 0 for name in CLASS_NAMES}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, device='cpu')
        filtered_boxes = []
        if hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                if int(box.cls) not in [8, 9]:
                    filtered_boxes.append(box)
            results[0].boxes = filtered_boxes

            frame_counts = count_classes(filtered_boxes)
            for key in frame_counts:
                class_counts[key] += frame_counts[key]

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    out.release()
    st.success("Video processing complete!")
    st.write("Class Counts:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")

    return temp_video.name

# Alert System when detecting "NO-Hardhat", "NO-Mask", "NO-Safety Vest"
def alert_system(frame, detected_classes):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
        
    alert_sound = pygame.mixer.Sound("emergency-siren-alert-single-epic-stock-media-1-00-01.mp3")
    
    if any(cls in [2, 3, 4] for cls in detected_classes):  # Adjust class IDs based on your model
        if not pygame.mixer.get_busy():  # Play sound only if not already playing
            pygame.mixer.Sound.play(alert_sound)
        cv2.putText(frame, "Alert: PPE not detected!", ((frame.shape[1]//2) - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        pygame.mixer.stop()  # Stop all sounds
        cv2.putText(frame, "All clear: PPE detected!", ((frame.shape[1]//2) - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
# Real-time webcam detection
def process_webcam(model, conf):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Webcam")
    class_counts = {name: 0 for name in CLASS_NAMES}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        results = model(frame, conf=conf, device='cpu')
        filtered_boxes = []
        if hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                if int(box.cls) not in [8, 9]:
                    filtered_boxes.append(box)
    
            results[0].boxes = filtered_boxes
            
            # Extract class IDs from filtered boxes
            detected_classes = [int(box.cls) for box in filtered_boxes]
            alert_system(frame, detected_classes)
            
            frame_counts = count_classes(filtered_boxes)
            for key in frame_counts:
                class_counts[key] = frame_counts[key]

        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    pygame.mixer.quit()  # Ensure pygame resources are released
    st.success("Webcam stopped.")

# Streamlit app
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
detected_classes = []  # Initialize as a list
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if model:
        results = process_image(np.array(image), model, confidence_threshold)
        img_res = results[0].plot()
        st.image(img_res, caption="Detected Image", use_container_width=True)
        
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = CLASS_NAMES[class_id]
            confidence = box.conf.item()  # Convert tensor to float using .item()
            
            # Append data as a dictionary to the list
            detected_classes.append({
                "Detected Classes": class_name,
                "Confidence": f"{confidence:.2f}",
            })
        
        # Convert the list to a DataFrame after the loop
        detected_classes_df = pd.DataFrame(detected_classes)
        
        # Display the table
        #st.table(detected_classes_df)

        # Create a Plotly bar chart
        fig = px.bar(
            detected_classes_df,
            x="Detected Classes",
            y=detected_classes_df["Confidence"].astype(float),
            title="Detected Classes with Confidence Scores",
            labels={"Confidence": "Confidence Scores"},
            color="Confidence",  # Optional: color based on confidence
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

# Video upload
st.header("Video Detection")
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
if video_file and model:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name
    with st.spinner("Processing video..."):
        processed_video = process_video(video_path, model, confidence_threshold)
        st.video(processed_video)
        st.download_button("Download Processed Video", processed_video, "processed_video.mp4")

# Webcam real-time detection
st.header("Real-Time Webcam Detection")
if st.button("Start Webcam"):
    if model:
        st.write("Click 'Stop Webcam' to end the webcam feed.")
        process_webcam(model, confidence_threshold)
    else:
        st.error("Model not loaded yet.")
