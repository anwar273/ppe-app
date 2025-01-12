import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# Load YOLO pretrained model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)  # Load the YOLO model
    return model

# Class names except machinery and vehicle
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"
]

# Count classes, including "Non-Person"
def count_classes(filtered_boxes, non_person_boxes):
    counts = {name: 0 for name in CLASS_NAMES + ["Non-Person"]}

    # Count valid classes
    for box in filtered_boxes:
        class_idx = int(box.cls)
        if class_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_idx]
            counts[class_name] += 1

    # Count "Non-Person" objects
    counts["Non-Person"] += len(non_person_boxes)

    return counts

# Process image and classify "Non-Person"
def process_image(image, model, conf):
    results = model.predict(image, conf=conf, device='cpu')
    filtered_boxes = []
    non_person_boxes = []

    if hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            class_id = int(box.cls)
            bbox = box.xyxy[0]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Exclude vehicle (8) and machinery (9)
            if class_id in [8, 9]:
                continue

            # Identify small or non-"Person" objects
            if class_id == 5 and (width < 50 or height < 50):  # Small "Person" -> Non-Person
                non_person_boxes.append(box)
            elif class_id == 5:
                filtered_boxes.append(box)  # Valid "Person"
            else:
                non_person_boxes.append(box)  # Other objects -> Non-Person

        results[0].filtered_boxes = filtered_boxes
        results[0].non_person_boxes = non_person_boxes

    return results

# Process video and classify "Non-Person"
def process_video(video_path, model, conf):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (frame_width, frame_height))

    class_counts = {name: 0 for name in CLASS_NAMES + ["Non-Person"]}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = process_image(frame, model, conf)
        annotated_frame = results[0].plot()

        # Update counts
        frame_counts = count_classes(results[0].filtered_boxes, results[0].non_person_boxes)
        for key, value in frame_counts.items():
            class_counts[key] += value

        out.write(annotated_frame)
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    out.release()
    st.success("Video processing complete!")
    st.write("Class Counts:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")

    return temp_video.name

# Real-time webcam detection
def process_webcam(model, conf):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Webcam")
    class_counts = {name: 0 for name in CLASS_NAMES + ["Non-Person"]}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        results = process_image(frame, model, conf)
        annotated_frame = results[0].plot()

        # Update counts
        frame_counts = count_classes(results[0].filtered_boxes, results[0].non_person_boxes)
        for key, value in frame_counts.items():
            class_counts[key] = value  # Update dynamically

        # Display frame and counts
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        st.sidebar.write("Class Counts:")
        for class_name, count in class_counts.items():
            st.sidebar.write(f"{class_name}: {count}")

    cap.release()
    st.success("Webcam stopped.")

# Streamlit app
st.title("Object Detection App")
st.sidebar.image("C:/Users/Dell G3/OneDrive - ECOLE SUPÉRIEURE DES INDUSTRIES DU TEXTILE ET DE L'HABILLEMENT/4 ANNEE/PROJET 5/SECURE LOGO.png", width=120, caption="SECURE Vision")
st.sidebar.title("Model Configuration")
model_path = st.sidebar.text_input("Enter the model path:", "C:/Users/Dell G3/Desktop/best1.pt")
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

# Image Detection
st.header("Image Detection")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    results = process_image(np.array(image), model, confidence_threshold)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Image", use_container_width=True)

    if hasattr(results[0], 'filtered_boxes') and hasattr(results[0], 'non_person_boxes'):
        class_counts = count_classes(results[0].filtered_boxes, results[0].non_person_boxes)
        st.write("Class Counts:")
        for class_name, count in class_counts.items():
            st.write(f"{class_name}: {count}")

# Video Detection
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

# Webcam Detection
st.header("Real-Time Webcam Detection")
if st.button("Start Webcam"):
    if model:
        process_webcam(model, confidence_threshold)
    else:
        st.error("Model not loaded yet.")
