# import streamlit as st
# from pathlib import Path
# import PIL
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pytube import YouTube
# from collections import defaultdict
# import requests
# from PIL import Image
# import os
# from io import BytesIO
# import json
# from google.cloud import vision
# from google.oauth2 import service_account
# import time

# # Path to your Google Cloud Vision API service account JSON file
# GOOGLE_CREDENTIALS_PATH = "byteeit-testing-project-cloud-vision-api.json"  # REPLACE THIS WITH YOUR ACTUAL PATH

# # HARDCODED PATH for license plate images - replace with your actual path
# LICENSE_PLATE_IMAGES_PATH = "images" # REPLACE THIS WITH YOUR ACTUAL PATH

# # Extended model dictionary with renamed use cases
# MODELS = {
#     "Safety Equipment Detection": "weights/best.pt",
#     "Hazard Detection": "weights/firenew.pt",
#     "Medical Imaging Analysis": "weights/bone.pt",
#     "Road Damage Detection": "weights/pathholes.pt",
#     "Product Counter": "weights/detector_de_chocolate.pt",
#     "Object Recognition": "yolov8n.pt",
#     "Vehicle Registration Recognition": "license_plate_detector.pt"
# }

# # Classes for Object Recognition detection
# COMMON_CLASSES = {
#     0: 'person', 26: 'handbag', 27: 'tie', 28: 'suitcase',
#     39: 'bottle', 41: 'cup', 42: 'fork', 63: 'laptop',
#     64: 'mouse', 67: 'cell phone'
# }

# class LicensePlateProcessor:
#     def __init__(self, model, vision_client):
#         self.model = model
#         self.vision_client = vision_client
#         self.all_plates_info = []

#     def process_image(self, image_path):
#         try:
#             image = Image.open(image_path).convert('RGB')
#             image_np = np.array(image)
            
#             results = self.model(image_np)
#             plates_info = []
            
#             for result in results[0].boxes.data:
#                 x1, y1, x2, y2, conf, cls = result
#                 if conf > 0.5:
#                     plate_region = image_np[int(y1):int(y2), int(x1):int(x2)]
                    
#                     if len(plate_region.shape) == 2:
#                         plate_region = cv2.cvtColor(plate_region, cv2.COLOR_GRAY2RGB)
#                     elif plate_region.shape[2] == 4:
#                         plate_region = cv2.cvtColor(plate_region, cv2.COLOR_RGBA2RGB)
                    
#                     plate_text = self.extract_plate_text(plate_region)
                    
#                     if plate_text:
#                         plates_info.append({
#                             'region': plate_region,
#                             'text': plate_text,
#                             'image_path': image_path,
#                             'confidence': float(conf)
#                         })
            
#             self.all_plates_info.extend(plates_info)
#             return plates_info
#         except Exception as e:
#             st.error(f"Error processing image {image_path}: {str(e)}")
#             return []

#     def extract_plate_text(self, plate_region):
#         try:
#             resized_plate = self.resize_plate_region(plate_region)
#             success, encoded_image = cv2.imencode('.png', resized_plate)
#             image_bytes = encoded_image.tobytes()
            
#             image = vision.Image(content=image_bytes)
#             response = self.vision_client.text_detection(image=image)
#             texts = response.text_annotations
            
#             if texts:
#                 # The first text annotation contains the entire detected text
#                 return texts[0].description
#             return ""
#         except Exception as e:
#             st.error(f"Error in OCR processing: {str(e)}")
#             return ""

#     def resize_plate_region(self, plate_region):
#         height, width = plate_region.shape[:2]
#         min_dim = min(height, width)
#         if min_dim < 50:
#             scale = 50.0 / min_dim
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             plate_region = cv2.resize(plate_region, (new_width, new_height), 
#                                     interpolation=cv2.INTER_CUBIC)
#         return plate_region

#     def search_plates(self, search_query):
#         return [plate for plate in self.all_plates_info 
#                 if search_query.lower() in plate['text'].lower()]

# def process_image(image, model, confidence, selected_model):
#     if selected_model == "Object Recognition":
#         results = model.predict(image, conf=confidence)[0]
#         # Filter only desired classes
#         filtered_boxes = []
#         for box in results.boxes:
#             cls = int(box.cls[0].item())
#             if cls in COMMON_CLASSES:
#                 filtered_boxes.append(box)
#         results.boxes = filtered_boxes
#         return results.plot()
#     else:
#         results = model.predict(image, conf=confidence)[0]
#         return results.plot()

# def process_license_plate(frame, model, vision_client):
#     results = model(frame)[0]
#     annotated_frame = frame.copy()
    
#     for box in results.boxes.data:
#         x1, y1, x2, y2 = map(int, box[:4])
#         plate_region = frame[y1:y2, x1:x2]
        
#         # Extract text using Google Vision OCR
#         plate_text = extract_plate_text(plate_region, vision_client)
        
#         # Draw bounding box and text
#         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         if plate_text:
#             cv2.putText(annotated_frame, plate_text, (x1, y1-10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     return annotated_frame

# class ProductCounter:
#     def __init__(self):
#         self.counted_products = set()
#         self.total_count = 0
#         self.line_position = 0.5  # Default to middle of the frame
#         self.margin = 5  # Reduced margin for more precise counting

#     def process_frame(self, frame, tracking_results, line_y_ratio=None):
#         if line_y_ratio is not None:
#             self.line_position = line_y_ratio
            
#         height, width = frame.shape[:2]
#         line_y = int(height * self.line_position)
        
#         # Draw the counting line across the entire frame
#         cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        
#         if tracking_results.boxes is None or len(tracking_results.boxes) == 0:
#             # Draw count even when no boxes are detected
#             cv2.putText(frame, f"Total = {self.total_count}", 
#                        (10, 37), cv2.FONT_HERSHEY_DUPLEX, 
#                        1.3, (0, 0, 255), 2)
#             return frame, self.total_count

#         # Draw white background for the counter text
#         (text_width, text_height), _ = cv2.getTextSize(f"Total = {self.total_count}", 
#                                                       cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
#         cv2.rectangle(frame, (0, 0), 
#                      (text_width + 20, text_height + 20), 
#                      (255, 255, 255), -1)

#         for box in tracking_results.boxes:
#             if not hasattr(box, 'id'):
#                 continue
                
#             product_id = int(box.id.item())
#             box_coords = box.xyxy[0].cpu().numpy()
            
#             # Calculate center point
#             center_x = int((box_coords[0] + box_coords[2]) / 2)
#             center_y = int((box_coords[1] + box_coords[3]) / 2)
            
#             # Draw bounding box
#             cv2.rectangle(frame, 
#                          (int(box_coords[0]), int(box_coords[1])), 
#                          (int(box_coords[2]), int(box_coords[3])), 
#                          (255, 0, 0), 2)
            
#             # Draw center point
#             cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
#             # Check if product crosses the line
#             if line_y - self.margin < center_y < line_y + self.margin:
#                 if product_id not in self.counted_products:
#                     self.counted_products.add(product_id)
#                     self.total_count += 1

#         # Draw the total count with consistent style
#         cv2.putText(frame, f"Total = {self.total_count}", 
#                    (10, 37), cv2.FONT_HERSHEY_DUPLEX, 
#                    1.3, (0, 0, 255), 2)
        
#         return frame, self.total_count

# def resize_plate_region(plate_region):
#     """Resize plate region to meet OCR requirements"""
#     height, width = plate_region.shape[:2]
#     min_dim = min(height, width)
#     if min_dim < 50:
#         scale = 50.0 / min_dim
#         new_width = int(width * scale)
#         new_height = int(height * scale)
#         plate_region = cv2.resize(plate_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#     return plate_region

# def extract_plate_text(plate_region, vision_client):
#     """Extract text from license plate region using Google Vision OCR"""
#     try:
#         # Resize plate region
#         resized_plate = resize_plate_region(plate_region)
        
#         # Convert numpy array to bytes
#         success, encoded_image = cv2.imencode('.png', resized_plate)
#         image_bytes = encoded_image.tobytes()
        
#         # Call Google Vision OCR
#         image = vision.Image(content=image_bytes)
#         response = vision_client.text_detection(image=image)
#         texts = response.text_annotations
        
#         if texts:
#             # The first text annotation contains the entire detected text
#             return texts[0].description
#         return ""
#     except Exception as e:
#         st.error(f"Error in OCR processing: {str(e)}")
#         return ""

# def process_video(source, model, confidence, st_frame, selected_model, counter=None, 
#                  line_position=None, vision_client=None):
#     """
#     Process a video source (file, RTSP, or YouTube) and display detected objects in real-time.
#     """
#     try:
#         cap = cv2.VideoCapture(source)
#         if not cap.isOpened():
#             st.error("Error opening video source")
#             return

#         # Create a unique key for the stop button
#         button_key = f"stop_button_{selected_model}_{int(time.time())}"
#         stop_button_container = st.empty()
#         stop_pressed = False

#         while cap.isOpened() and not stop_pressed:
#             success, frame = cap.read()
#             if not success:
#                 break

#             frame = cv2.resize(frame, (1280, 720))
            
#             # Process the frame based on the selected model
#             if selected_model == "Vehicle Registration Recognition" and vision_client:
#                 processed_frame = process_license_plate(frame, model, vision_client)
#             elif selected_model == "Product Counter" and counter:
#                 results = model.track(frame, conf=confidence, persist=True)[0]
#                 processed_frame, _ = counter.process_frame(frame, results, line_position)
#             else:
#                 if selected_model == "Object Recognition":
#                     results = model.predict(frame, conf=confidence, classes=[0, 26, 27, 28, 39, 41, 42, 63, 64, 67])[0]
#                 else:
#                     results = model.predict(frame, conf=confidence)[0]
#                 processed_frame = results.plot()
            
#             # Display the processed frame
#             st_frame.image(processed_frame, channels="BGR", use_container_width=True)
            
#             # Update stop button with the unique key
#             stop_pressed = stop_button_container.button("Stop", key=button_key)

#     except Exception as e:
#         st.error(f"Error processing video: {str(e)}")
#     finally:
#         if 'cap' in locals() and cap.isOpened():
#             cap.release()

# def _display_detected_frames(conf, model, st_frame, image, selected_model, counter=None, line_position=None):
#     """Display detected frames with tracking"""
#     image = cv2.resize(image, (720, int(720*(9/16))))
    
#     if selected_model == "Product Counter" and counter:
#         # Use tracking for product counting
#         results = model.track(image, conf=conf, persist=True)[0]
#         # Pass the line position to the counter
#         processed_frame, count = counter.process_frame(image, results, line_position)
#         st_frame.image(processed_frame, channels="BGR", use_container_width=True)
#     else:
#         if selected_model == "Object Recognition":
#             results = model.predict(image, conf=conf, classes=[0,26,27,28,39,41,42,63,64,67])[0]
#         else:
#             results = model.predict(image, conf=conf)[0]
#         image = results.plot()
#         st_frame.image(image, channels="BGR", use_container_width=True)

# def play_stored_video(conf, model, selected_model):
#     """
#     Plays a stored video file. Detects objects in real-time using the YOLO model.
#     """
#     source_vid = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
#     if source_vid is not None:
#         # Save uploaded video to temporary file
#         temp_file = f"temp_video.{source_vid.name.split('.')[-1]}"
#         with open(temp_file, 'wb') as f:
#             f.write(source_vid.read())

#         try:
#             vid_cap = cv2.VideoCapture(temp_file)
#             st_frame = st.empty()
            
#             # Initialize counter if it's the product counter model
#             counter = None
#             line_position = None
#             if selected_model == "Product Counter":
#                 counter = ProductCounter()
#                 line_position = st.sidebar.slider("Counting Line Position", 0.1, 0.9, 0.5)
            
#             # Add stop button
#             stop_button = st.button("Stop", key="stop_video")
            
#             while vid_cap.isOpened() and not stop_button:
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(
#                         conf,
#                         model,
#                         st_frame,
#                         image,
#                         selected_model,
#                         counter,
#                         line_position
#                     )
#                 else:
#                     vid_cap.release()
#                     break
                
#                 if stop_button:
#                     vid_cap.release()
#                     break
                    
#         except Exception as e:
#             st.error(f"Error processing video: {str(e)}")
#         finally:
#             if 'vid_cap' in locals() and vid_cap.isOpened():
#                 vid_cap.release()
#             # Clean up temporary file
#             if os.path.exists(temp_file):
#                 os.remove(temp_file)

# def process_license_plate_images(license_processor, folder_path):
#     """Process all license plate images in the specified folder"""
#     if not os.path.exists(folder_path):
#         st.error(f"Folder not found: {folder_path}")
#         return False
#     elif not os.path.isdir(folder_path):
#         st.error(f"The path is not a directory: {folder_path}")
#         return False
    
#     # Process all images in the folder
#     image_files = [f for f in os.listdir(folder_path) 
#                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     if not image_files:
#         st.warning("No image files found in the specified folder.")
#         return False
    
#     with st.spinner("Processing license plate images..."):
#         progress_bar = st.progress(0)
#         for idx, image_file in enumerate(image_files):
#             image_path = os.path.join(folder_path, image_file)
#             license_processor.process_image(image_path)
#             progress = (idx + 1) / len(image_files)
#             progress_bar.progress(progress)
        
#     return True

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="Vision Analysis Platform", page_icon="🤖", layout="wide")
    
#     st.title("Vision Analysis Platform")
    
#     st.sidebar.title("Model Configuration")
#     selected_model = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
    
#     confidence = float(st.sidebar.slider("Confidence", 25, 100, 40)) / 100
    
#     is_product_counter = selected_model == "Product Counter"
#     is_license_plate = selected_model == "Vehicle Registration Recognition"
    
#     # Initialize Vision client directly with the hardcoded path
#     vision_client = None
#     if is_license_plate:
#         try:
#             credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)
#             vision_client = vision.ImageAnnotatorClient(credentials=credentials)
#             st.success("Google Vision API client initialized successfully")
#         except Exception as e:
#             st.error(f"Error initializing Google Vision client: {str(e)}")
#             st.info(f"Check if the file path is correct: {GOOGLE_CREDENTIALS_PATH}")
    
#     model = None
#     try:
#         model = YOLO(MODELS[selected_model])
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         st.info("Check if the model path is correct and the model file exists")
#         return
    
#     if selected_model == "Vehicle Registration Recognition" and vision_client:
#         # Initialize license plate processor
#         license_processor = LicensePlateProcessor(model, vision_client)
        
#         # Process images from the hardcoded path
#         if process_license_plate_images(license_processor, LICENSE_PLATE_IMAGES_PATH):
#             # Search functionality
#             st.subheader("Search for Vehicle Registration")
#             search_query = st.text_input("Enter registration number to search:")
            
#             if search_query:
#                 matching_plates = license_processor.search_plates(search_query)
#                 if matching_plates:
#                     st.success(f"Found {len(matching_plates)} matching registration(s)")
#                     for idx, plate in enumerate(matching_plates):
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.image(plate['image_path'], 
#                                    caption=f"Original Image {idx+1}", 
#                                    use_container_width=True)
#                         with col2:
#                             st.image(plate['region'], 
#                                    caption=f"Detected Registration {idx+1}: {plate['text']}", 
#                                    use_container_width=True)
#                 else:
#                     st.warning("No matching registrations found.")
    
#     # Initialize counter if needed
#     counter = ProductCounter() if is_product_counter else None
    
#     # Set up input source options - don't show any input options for license plate model
#     if selected_model == "Vehicle Registration Recognition":
#         # Skip the rest of the UI since we're using hardcoded path for this model
#         return
#     elif selected_model == "Product Counter":
#         source_options = ["Upload Video", "RTSP", "YouTube"]
#         # Add a unique key to the slider and store it in session state
#         line_position = st.sidebar.slider(
#             "Counting Line Position", 
#             0.1, 
#             0.9, 
#             0.5, 
#             key="main_line_position_slider"
#         )
#         st.session_state['line_position'] = line_position
#     else:
#         source_options = ["Image", "Upload Video", "RTSP", "YouTube"]
#         line_position = None
    
#     source_type = st.sidebar.radio("Select Input", source_options)
#     st_frame = st.empty()
    
#     # Handle Image input
#     if source_type == "Image" and not is_product_counter:
#         image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="image_uploader")
#         if image_file:
#             try:
#                 image = PIL.Image.open(image_file)
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.image(image, caption="Original Image", use_container_width=True)
                
#                 with col2:
#                     if st.button("Detect", key="detect_button"):
#                         if is_license_plate and vision_client:
#                             frame = np.array(image)
#                             result = process_license_plate(frame, model, vision_client)
#                             st.image(result, caption="Detected Vehicle Registrations", use_container_width=True)
#                         else:
#                             result = process_image(image, model, confidence, selected_model)
#                             st.image(result, caption="Detected Objects", use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error processing image: {str(e)}")
    
#     # Handle Video input
#     elif source_type == "Upload Video":
#         play_stored_video(confidence, model, selected_model)
    
#     # Handle YouTube input
#     elif source_type == "YouTube":
#         url = st.sidebar.text_input("YouTube URL", key="youtube_url")
#         if url and st.sidebar.button("Process", key="process_youtube_button"):
#             with st.spinner("Processing YouTube video..."):
#                 try:
#                     yt = YouTube(url)
#                     stream = yt.streams.filter(file_extension="mp4", res="720p").first()
#                     process_video(stream.url, model, confidence, st_frame,
#                                 selected_model, counter, line_position, vision_client)
#                 except Exception as e:
#                     st.error(f"Error processing YouTube video: {str(e)}")
    
#     # Handle RTSP input
#     elif source_type == "RTSP":
#         rtsp_url = st.sidebar.text_input("RTSP URL", key="rtsp_url")
#         if rtsp_url and st.sidebar.button("Start Stream", key="start_rtsp_button"):
#             process_video(rtsp_url, model, confidence, st_frame,
#                          selected_model, counter, line_position, vision_client)

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         import traceback
#         st.error(traceback.format_exc())
import streamlit as st
from pathlib import Path
import PIL
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import requests
from PIL import Image
import os
from io import BytesIO
import json
from google.cloud import vision
from google.oauth2 import service_account
import time

# Path to your Google Cloud Vision API service account JSON file
GOOGLE_CREDENTIALS_PATH = "byteeit-testing-project-cloud-vision-api.json"  # REPLACE THIS WITH YOUR ACTUAL PATH

# HARDCODED PATH for license plate images - replace with your actual path
LICENSE_PLATE_IMAGES_PATH = "images" # REPLACE THIS WITH YOUR ACTUAL PATH

# Extended model dictionary with renamed use cases
MODELS = {
    "Safety Equipment Detection": "weights/best.pt",
    "Hazard Detection": "weights/firenew.pt",
    "Medical Imaging Analysis": "weights/bone.pt",
    "Road Damage Detection": "weights/pathholes.pt",
    "Product Counter": "weights/detector_de_chocolate.pt",
    "Object Recognition": "yolov8n.pt",
    "Vehicle Registration Recognition": "license_plate_detector.pt"
}

# Classes for Object Recognition detection
COMMON_CLASSES = {
    0: 'person', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    39: 'bottle', 41: 'cup', 42: 'fork', 63: 'laptop',
    64: 'mouse', 67: 'cell phone'
}

class LicensePlateProcessor:
    def __init__(self, model, vision_client):
        self.model = model
        self.vision_client = vision_client
        self.all_plates_info = []

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            results = self.model(image_np)
            plates_info = []
            
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result
                if conf > 0.5:
                    plate_region = image_np[int(y1):int(y2), int(x1):int(x2)]
                    
                    if len(plate_region.shape) == 2:
                        plate_region = cv2.cvtColor(plate_region, cv2.COLOR_GRAY2RGB)
                    elif plate_region.shape[2] == 4:
                        plate_region = cv2.cvtColor(plate_region, cv2.COLOR_RGBA2RGB)
                    
                    plate_text = self.extract_plate_text(plate_region)
                    
                    if plate_text:
                        plates_info.append({
                            'region': plate_region,
                            'text': plate_text,
                            'image_path': image_path,
                            'confidence': float(conf)
                        })
            
            self.all_plates_info.extend(plates_info)
            return plates_info
        except Exception as e:
            st.error(f"Error processing image {image_path}: {str(e)}")
            return []

    def extract_plate_text(self, plate_region):
        try:
            resized_plate = self.resize_plate_region(plate_region)
            success, encoded_image = cv2.imencode('.png', resized_plate)
            image_bytes = encoded_image.tobytes()
            
            image = vision.Image(content=image_bytes)
            response = self.vision_client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # The first text annotation contains the entire detected text
                return texts[0].description
            return ""
        except Exception as e:
            st.error(f"Error in OCR processing: {str(e)}")
            return ""

    def resize_plate_region(self, plate_region):
        height, width = plate_region.shape[:2]
        min_dim = min(height, width)
        if min_dim < 50:
            scale = 50.0 / min_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            plate_region = cv2.resize(plate_region, (new_width, new_height), 
                                    interpolation=cv2.INTER_CUBIC)
        return plate_region

    def search_plates(self, search_query):
        return [plate for plate in self.all_plates_info 
                if search_query.lower() in plate['text'].lower()]

def process_image(image, model, confidence, selected_model):
    if selected_model == "Object Recognition":
        results = model.predict(image, conf=confidence)[0]
        # Filter only desired classes
        filtered_boxes = []
        for box in results.boxes:
            cls = int(box.cls[0].item())
            if cls in COMMON_CLASSES:
                filtered_boxes.append(box)
        results.boxes = filtered_boxes
        return results.plot()
    else:
        results = model.predict(image, conf=confidence)[0]
        return results.plot()

def process_license_plate(frame, model, vision_client):
    results = model(frame)[0]
    annotated_frame = frame.copy()
    
    for box in results.boxes.data:
        x1, y1, x2, y2 = map(int, box[:4])
        plate_region = frame[y1:y2, x1:x2]
        
        # Extract text using Google Vision OCR
        plate_text = extract_plate_text(plate_region, vision_client)
        
        # Draw bounding box and text
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if plate_text:
            cv2.putText(annotated_frame, plate_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return annotated_frame

class ProductCounter:
    def __init__(self):
        self.counted_products = set()
        self.total_count = 0
        self.line_position = 0.5  # Default to middle of the frame
        self.margin = 5  # Reduced margin for more precise counting

    def process_frame(self, frame, tracking_results, line_y_ratio=None):
        if line_y_ratio is not None:
            self.line_position = line_y_ratio
            
        height, width = frame.shape[:2]
        line_y = int(height * self.line_position)
        
        # Draw the counting line across the entire frame
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        
        if tracking_results.boxes is None or len(tracking_results.boxes) == 0:
            # Draw count even when no boxes are detected
            cv2.putText(frame, f"Total = {self.total_count}", 
                       (10, 37), cv2.FONT_HERSHEY_DUPLEX, 
                       1.3, (0, 0, 255), 2)
            return frame, self.total_count

        # Draw white background for the counter text
        (text_width, text_height), _ = cv2.getTextSize(f"Total = {self.total_count}", 
                                                      cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
        cv2.rectangle(frame, (0, 0), 
                     (text_width + 20, text_height + 20), 
                     (255, 255, 255), -1)

        for box in tracking_results.boxes:
            if not hasattr(box, 'id'):
                continue
                
            product_id = int(box.id.item())
            box_coords = box.xyxy[0].cpu().numpy()
            
            # Calculate center point
            center_x = int((box_coords[0] + box_coords[2]) / 2)
            center_y = int((box_coords[1] + box_coords[3]) / 2)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (int(box_coords[0]), int(box_coords[1])), 
                         (int(box_coords[2]), int(box_coords[3])), 
                         (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Check if product crosses the line
            if line_y - self.margin < center_y < line_y + self.margin:
                if product_id not in self.counted_products:
                    self.counted_products.add(product_id)
                    self.total_count += 1

        # Draw the total count with consistent style
        cv2.putText(frame, f"Total = {self.total_count}", 
                   (10, 37), cv2.FONT_HERSHEY_DUPLEX, 
                   1.3, (0, 0, 255), 2)
        
        return frame, self.total_count

def resize_plate_region(plate_region):
    """Resize plate region to meet OCR requirements"""
    height, width = plate_region.shape[:2]
    min_dim = min(height, width)
    if min_dim < 50:
        scale = 50.0 / min_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        plate_region = cv2.resize(plate_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return plate_region

def extract_plate_text(plate_region, vision_client):
    """Extract text from license plate region using Google Vision OCR"""
    try:
        # Resize plate region
        resized_plate = resize_plate_region(plate_region)
        
        # Convert numpy array to bytes
        success, encoded_image = cv2.imencode('.png', resized_plate)
        image_bytes = encoded_image.tobytes()
        
        # Call Google Vision OCR
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        
        if texts:
            # The first text annotation contains the entire detected text
            return texts[0].description
        return ""
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return ""

def process_video(source, model, confidence, st_frame, selected_model, counter=None, 
                 line_position=None, vision_client=None):
    """
    Process a video source (file or RTSP) and display detected objects in real-time.
    """
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error("Error opening video source")
            return

        # Create a unique key for the stop button
        button_key = f"stop_button_{selected_model}_{int(time.time())}"
        stop_button_container = st.empty()
        stop_pressed = False

        while cap.isOpened() and not stop_pressed:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (1280, 720))
            
            # Process the frame based on the selected model
            if selected_model == "Vehicle Registration Recognition" and vision_client:
                processed_frame = process_license_plate(frame, model, vision_client)
            elif selected_model == "Product Counter" and counter:
                results = model.track(frame, conf=confidence, persist=True)[0]
                processed_frame, _ = counter.process_frame(frame, results, line_position)
            else:
                if selected_model == "Object Recognition":
                    results = model.predict(frame, conf=confidence, classes=[0, 26, 27, 28, 39, 41, 42, 63, 64, 67])[0]
                else:
                    results = model.predict(frame, conf=confidence)[0]
                processed_frame = results.plot()
            
            # Display the processed frame
            st_frame.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Update stop button with the unique key
            stop_pressed = stop_button_container.button("Stop", key=button_key)

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def _display_detected_frames(conf, model, st_frame, image, selected_model, counter=None, line_position=None):
    """Display detected frames with tracking"""
    image = cv2.resize(image, (720, int(720*(9/16))))
    
    if selected_model == "Product Counter" and counter:
        # Use tracking for product counting
        results = model.track(image, conf=conf, persist=True)[0]
        # Pass the line position to the counter
        processed_frame, count = counter.process_frame(image, results, line_position)
        st_frame.image(processed_frame, channels="BGR", use_container_width=True)
    else:
        if selected_model == "Object Recognition":
            results = model.predict(image, conf=conf, classes=[0,26,27,28,39,41,42,63,64,67])[0]
        else:
            results = model.predict(image, conf=conf)[0]
        image = results.plot()
        st_frame.image(image, channels="BGR", use_container_width=True)

def play_stored_video(conf, model, selected_model):
    """
    Plays a stored video file. Detects objects in real-time using the YOLO model.
    """
    source_vid = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if source_vid is not None:
        # Save uploaded video to temporary file
        temp_file = f"temp_video.{source_vid.name.split('.')[-1]}"
        with open(temp_file, 'wb') as f:
            f.write(source_vid.read())

        try:
            vid_cap = cv2.VideoCapture(temp_file)
            st_frame = st.empty()
            
            # Initialize counter if it's the product counter model
            counter = None
            line_position = None
            if selected_model == "Product Counter":
                counter = ProductCounter()
                line_position = st.sidebar.slider("Counting Line Position", 0.1, 0.9, 0.5)
            
            # Add stop button
            stop_button = st.button("Stop", key="stop_video")
            
            while vid_cap.isOpened() and not stop_button:
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        selected_model,
                        counter,
                        line_position
                    )
                else:
                    vid_cap.release()
                    break
                
                if stop_button:
                    vid_cap.release()
                    break
                    
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        finally:
            if 'vid_cap' in locals() and vid_cap.isOpened():
                vid_cap.release()
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

def process_license_plate_images(license_processor, folder_path):
    """Process all license plate images in the specified folder"""
    if not os.path.exists(folder_path):
        st.error(f"Folder not found: {folder_path}")
        return False
    elif not os.path.isdir(folder_path):
        st.error(f"The path is not a directory: {folder_path}")
        return False
    
    # Process all images in the folder
    image_files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning("No image files found in the specified folder.")
        return False
    
    with st.spinner("Processing license plate images..."):
        progress_bar = st.progress(0)
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            license_processor.process_image(image_path)
            progress = (idx + 1) / len(image_files)
            progress_bar.progress(progress)
        
    return True

def main():
    # Set page configuration
    st.set_page_config(page_title="Vision Analysis Platform", page_icon="🤖", layout="wide")
    
    st.title("Vision Analysis Platform")
    
    st.sidebar.title("Model Configuration")
    selected_model = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
    
    confidence = float(st.sidebar.slider("Confidence", 25, 100, 40)) / 100
    
    is_product_counter = selected_model == "Product Counter"
    is_license_plate = selected_model == "Vehicle Registration Recognition"
    
    # Initialize Vision client directly with the hardcoded path
    vision_client = None
    if is_license_plate:
        try:
            credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            st.success("Google Vision API client initialized successfully")
        except Exception as e:
            st.error(f"Error initializing Google Vision client: {str(e)}")
            st.info(f"Check if the file path is correct: {GOOGLE_CREDENTIALS_PATH}")
    
    model = None
    try:
        model = YOLO(MODELS[selected_model])
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Check if the model path is correct and the model file exists")
        return
    
    if selected_model == "Vehicle Registration Recognition" and vision_client:
        # Initialize license plate processor
        license_processor = LicensePlateProcessor(model, vision_client)
        
        # Process images from the hardcoded path
        if process_license_plate_images(license_processor, LICENSE_PLATE_IMAGES_PATH):
            # Search functionality
            st.subheader("Search for Vehicle Registration")
            search_query = st.text_input("Enter registration number to search:")
            
            if search_query:
                matching_plates = license_processor.search_plates(search_query)
                if matching_plates:
                    st.success(f"Found {len(matching_plates)} matching registration(s)")
                    for idx, plate in enumerate(matching_plates):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(plate['image_path'], 
                                   caption=f"Original Image {idx+1}", 
                                   use_container_width=True)
                        with col2:
                            st.image(plate['region'], 
                                   caption=f"Detected Registration {idx+1}: {plate['text']}", 
                                   use_container_width=True)
                else:
                    st.warning("No matching registrations found.")
    
    # Initialize counter if needed
    counter = ProductCounter() if is_product_counter else None
    
    # Set up input source options - don't show any input options for license plate model
    if selected_model == "Vehicle Registration Recognition":
        # Skip the rest of the UI since we're using hardcoded path for this model
        return
    elif selected_model == "Product Counter":
        source_options = ["Upload Video", "RTSP"]
        # Add a unique key to the slider and store it in session state
        line_position = st.sidebar.slider(
            "Counting Line Position", 
            0.1, 
            0.9, 
            0.5, 
            key="main_line_position_slider"
        )
        st.session_state['line_position'] = line_position
    else:
        source_options = ["Image", "Upload Video", "RTSP"]
        line_position = None
    
    source_type = st.sidebar.radio("Select Input", source_options)
    st_frame = st.empty()
    
    # Handle Image input
    if source_type == "Image" and not is_product_counter:
        image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="image_uploader")
        if image_file:
            try:
                image = PIL.Image.open(image_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                
                with col2:
                    if st.button("Detect", key="detect_button"):
                        if is_license_plate and vision_client:
                            frame = np.array(image)
                            result = process_license_plate(frame, model, vision_client)
                            st.image(result, caption="Detected Vehicle Registrations", use_container_width=True)
                        else:
                            result = process_image(image, model, confidence, selected_model)
                            st.image(result, caption="Detected Objects", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Handle Video input
    elif source_type == "Upload Video":
        play_stored_video(confidence, model, selected_model)
    
    # Handle RTSP input
    elif source_type == "RTSP":
        rtsp_url = st.sidebar.text_input("RTSP URL", key="rtsp_url")
        if rtsp_url and st.sidebar.button("Start Stream", key="start_rtsp_button"):
            process_video(rtsp_url, model, confidence, st_frame,
                         selected_model, counter, line_position, vision_client)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())