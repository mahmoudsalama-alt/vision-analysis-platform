# Vision Analysis Platform

A comprehensive computer vision analysis platform built with Streamlit and YOLOv8 for multiple use cases.

## Features

- **Safety Equipment Detection** - Detect PPE and safety equipment in images/videos
- **Hazard Detection** - Identify potential hazards and fire
- **Medical Imaging Analysis** - Bone fracture detection
- **Road Damage Detection** - Identify potholes and road damage
- **Product Counter** - Count products crossing a designated line
- **Object Recognition** - Detect common objects (person, laptop, phone, etc.)
- **Vehicle Registration Recognition** - License plate detection and OCR
- **Crowd Monitoring** - Monitor people count in designated areas with alerts

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd cv
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud Vision API:
   - Create a service account in Google Cloud Console
   - Download the JSON credentials file
   - Update the path in the Python files

4. Add model weights:
   - Place your YOLO model weights in the `weights/` folder
   - Update paths in the code if needed

## Usage

Run the Streamlit app:
```bash
streamlit run test8.py
```

## Model Files Required

- `weights/best.pt` - Safety equipment detection
- `weights/firenew.pt` - Hazard detection
- `weights/bone.pt` - Medical imaging
- `weights/pathholes.pt` - Road damage detection
- `weights/detector_de_chocolate.pt` - Product counter
- `license_plate_detector.pt` - License plate detection
- `yolov8n.pt` - Object recognition (downloaded automatically)

## Configuration

Update these paths in the code:
- `GOOGLE_CREDENTIALS_PATH` - Path to Google Cloud Vision API credentials
- `LICENSE_PLATE_IMAGES_PATH` - Path to license plate images folder

## License

All rights reserved.
