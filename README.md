
# Object Detection App with YOLO and Streamlit

This project is an object detection web application built with Streamlit and YOLO, designed to process images, videos, and real-time webcam feeds for object detection and alerting on specific safety violations.

## Features

- **Image Detection:** Upload an image file (JPEG/PNG), and the app will detect objects and display them with confidence scores.
- **Video Detection:** Upload a video file (MP4/AVI) for processing. Download the annotated video after detection.
- **Real-Time Webcam Detection:** Detect objects in real-time using a webcam feed.
- **Alert System:** Notifies users with an alert sound and visual cues when safety violations are detected (e.g., missing hardhat or mask).
- **Interactive Dashboard:** Displays detected classes and confidence scores in a table and bar chart using Plotly.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLO model weights (e.g., `best1.pt`) and place them in the project directory.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`.

## Configuration

- **Model Path:** Set the path to your YOLO model in the sidebar.
- **Confidence Threshold:** Adjust the detection confidence using the slider in the sidebar.

## File Requirements

- **Images:** Supported formats are JPG, JPEG, and PNG.
- **Videos:** Supported formats are MP4 and AVI.

## Dependencies

- Streamlit
- Ultralytics YOLO
- PIL
- OpenCV
- PyGame
- Pandas
- Plotly
- NumPy

## Folder Structure

```
project-directory/
├── streamlit_app.py         # Main Streamlit app script
├── requirements.txt         # Python dependencies
├── best1.pt                 # YOLO model weights (not included, download separately)
├── SECURE LOGO.png          # App logo
├── emergency-siren-alert-single-epic-stock-media-1-00-01.mp3  # Alert sound
```

## Known Issues

- Ensure your webcam is properly connected for real-time detection.
- Adjust the confidence threshold for optimal detection results.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [YOLO by Ultralytics](https://github.com/ultralytics/yolov5) for the pretrained model.
- [Streamlit](https://streamlit.io) for the user-friendly app interface.
- Open-source libraries that made this project possible.

---

**Developed with ❤ by Secure Vision**
