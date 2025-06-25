# import platform
# import pathlib

# # Fix path compatibility issue between Windows and Linux
# if platform.system() != 'Windows':
#     pathlib.WindowsPath = pathlib.PosixPath

# import cv2
# import numpy as np
# import os
# import pytesseract
# import argparse
# import time
# import yolov5
# from collections import Counter

# def enhance_image_for_ocr(image):
#     """Apply preprocessing to improve OCR accuracy"""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply CLAHE for better contrast
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
    
#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY_INV, 11, 2)
    
#     # Apply morphological operations
#     kernel = np.ones((2, 2), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     return opening

# def extract_speed_limit(sign_image, debug=False):
#     """Extract speed limit number from sign image using OCR"""
#     # Skip if image is too small
#     if sign_image.shape[0] < 20 or sign_image.shape[1] < 20:
#         return "", "None"
    
#     # Resize image for better OCR performance
#     height, width = sign_image.shape[:2]
#     new_height = 200  # Increased size for better OCR
#     new_width = int(width * (new_height / height))
#     resized = cv2.resize(sign_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
#     # Define valid speed limits (common values in most regions)
#     valid_speed_limits = ['20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '100', '110', '120']
    
#     # Try different preprocessing methods
#     methods = [
#         ("Enhanced", enhance_image_for_ocr(resized)),
#         ("Grayscale", cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)),
#         ("Binary", cv2.threshold(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 
#                                  127, 255, cv2.THRESH_BINARY)[1]),
#         ("Otsu", cv2.threshold(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 
#                                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
#     ]
    
#     best_result = ""
#     confidence = "None"
    
#     for method_name, processed_image in methods:
#         # Configure Tesseract for digit recognition with multiple PSM modes
#         for psm in [7, 8, 6]:  # Try different page segmentation modes
#             config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789'
            
#             # Extract text
#             speed_limit = pytesseract.image_to_string(processed_image, config=config)
#             speed_limit = speed_limit.strip()
            
#             # Validate the result
#             if speed_limit and speed_limit.isdigit() and 1 <= len(speed_limit) <= 3:
#                 # Only accept values from our predefined list
#                 if speed_limit in valid_speed_limits:
#                     best_result = speed_limit
#                     confidence = "High"
#                     break
#                 elif not best_result and len(speed_limit) == 2:  # More likely to be valid if 2 digits
#                     best_result = speed_limit
#                     confidence = "Low"
        
#         # Debug mode info without GUI display
#         if debug and method_name:
#             print(f"Method: {method_name}, Result: {speed_limit}")
        
#         if confidence == "High":
#             break
    
#     # Debug mode info without GUI display
#     if debug and best_result:
#         print(f"Found speed limit: {best_result} with confidence: {confidence}")
    
#     return best_result, confidence

# def process_video(model_path, source, output_path=None, conf_threshold=0.4, debug=False):
#     """Process video for speed limit sign detection and recognition"""
#     # Load YOLOv5 model
#     model = yolov5.load(model_path)
#     model.conf = conf_threshold  # Set confidence threshold
#     print(f"Model loaded successfully from {model_path}")
    
#     # Open video source
#     if source.isdigit():
#         cap = cv2.VideoCapture(int(source))
#     else:
#         cap = cv2.VideoCapture(source)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {source}")
#         return
    
#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Create output video writer if specified
#     out = None
#     if output_path:
#         os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#         print(f"Writing output to {output_path}")
    
#     # Initialize variables
#     frame_count = 0
#     start_time = time.time()
#     recent_detections = []  # For tracking recent speed limits
#     detection_count = 0
    
#     # For tracking consistent detections
#     detection_history = []
#     current_speed_limit = None
    
#     print(f"Processing video from {source}...")
    
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
        
#         frame_count += 1
        
#         # Print progress every 100 frames
#         if frame_count % 100 == 0:
#             elapsed_time = time.time() - start_time
#             fps_current = frame_count / elapsed_time
#             print(f"Processed {frame_count} frames. Current FPS: {fps_current:.2f}")
        
#         # Run YOLOv5 inference
#         results = model(frame)
        
#         # Process each detection
#         for det in results.xyxy[0]:  # Process detections for first image
#             x1, y1, x2, y2, conf, cls = det.tolist()
            
#             # Convert coordinates to integers
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
#             # Extract the sign region with margin
#             margin = 10
#             y_min = max(0, y1 - margin)
#             y_max = min(frame.shape[0], y2 + margin)
#             x_min = max(0, x1 - margin)
#             x_max = min(frame.shape[1], x2 + margin)
            
#             sign_image = frame[y_min:y_max, x_min:x_max]
            
#             if sign_image.size > 0:
#                 # Extract the speed limit number
#                 speed_limit, confidence = extract_speed_limit(sign_image, debug)
                
#                 # Only process if we got a number
#                 if speed_limit:
#                     # Add to detection history for consistency checking
#                     detection_history.append(speed_limit)
#                     # Keep only the last 10 detections
#                     if len(detection_history) > 10:
#                         detection_history.pop(0)
                    
#                     # Count occurrences of each speed limit in recent history
#                     counter = Counter(detection_history)
#                     most_common = counter.most_common(1)
                    
#                     # Only accept a speed limit if it appears multiple times
#                     if most_common and most_common[0][1] >= 3:  # At least 3 occurrences
#                         current_speed_limit = most_common[0][0]
#                         detection_count += 1
                        
#                         # Print detection info
#                         print(f"Frame {frame_count}: Detected speed limit {current_speed_limit} km/h (confidence: {confidence})")
                        
#                         # Display result on frame (for output video)
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
#                         # Add confidence indicator
#                         confidence_color = (0, 255, 0) if confidence == "High" else (0, 165, 255)
#                         cv2.putText(frame, f"Speed: {current_speed_limit} km/h", 
#                                     (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
                        
#                         # Display detection confidence
#                         cv2.putText(frame, f"Conf: {conf:.2f}", 
#                                     (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # Display current speed limit on frame if available
#         if current_speed_limit:
#             cv2.putText(frame, f"Current Speed Limit: {current_speed_limit} km/h", 
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Add frame counter and FPS
#         elapsed_time = time.time() - start_time
#         fps_text = f"FPS: {frame_count / elapsed_time:.2f}"
#         cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Write the frame to output video if specified
#         if out:
#             out.write(frame)
    
#     # Release resources
#     cap.release()
#     if out:
#         out.release()
    
#     # Print summary
#     elapsed_time = time.time() - start_time
#     print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
#     print(f"Average FPS: {frame_count / elapsed_time:.2f}")
#     print(f"Total speed limit detections: {detection_count}")
#     if current_speed_limit:
#         print(f"Final detected speed limit: {current_speed_limit} km/h")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Speed Limit Detection and Recognition")
#     parser.add_argument("--model", type=str, default="/home/manideep/CV/P3Data/speed_limit/mani/runs/train/road_signs_yolov52/weights/best.pt", 
#                         help="Path to the trained YOLOv5 model")
#     parser.add_argument("--source", type=str, default="/home/manideep/CV/P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4", 
#                         help="Source for detection (0 for webcam, or video file path)")
#     parser.add_argument("--output", type=str, default="/home/manideep/CV/P3Data/speed_limit/output.avi", 
#                         help="Path to save output video")
#     parser.add_argument("--conf", type=float, default=0.2, 
#                         help="Confidence threshold for detection")
#     parser.add_argument("--debug", action="store_true", 
#                         help="Enable debug mode to print additional information")
    
#     args = parser.parse_args()
    
#     # Process the video
#     process_video(args.model, args.source, args.output, args.conf, args.debug)
import platform
import pathlib
import cv2
import numpy as np
import os
import pytesseract
import argparse
import time
import yolov5
import json
from collections import Counter
from transformers import pipeline
from PIL import Image
import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)


# Fix path compatibility issue between Windows and Linux
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
    

def enhance_image_for_ocr(image):
    """Apply preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening

def extract_speed_limit(sign_image, debug=False):
    """Extract speed limit number from sign image using OCR"""
    # Skip if image is too small
    if sign_image.shape[0] < 20 or sign_image.shape[1] < 20:
        return "", "None"
    
    # Resize image for better OCR performance
    height, width = sign_image.shape[:2]
    new_height = 200  # Increased size for better OCR
    new_width = int(width * (new_height / height))
    resized = cv2.resize(sign_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Define valid speed limits (common values in most regions)
    valid_speed_limits = ['20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '100', '110', '120']
    
    # Try different preprocessing methods
    methods = [
        ("Enhanced", enhance_image_for_ocr(resized)),
        ("Grayscale", cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)),
        ("Binary", cv2.threshold(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 
                                 127, 255, cv2.THRESH_BINARY)[1]),
        ("Otsu", cv2.threshold(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 
                               0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    ]
    
    best_result = ""
    confidence = "None"
    
    for method_name, processed_image in methods:
        # Configure Tesseract for digit recognition with multiple PSM modes
        for psm in [7, 8, 6]:  # Try different page segmentation modes
            config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789'
            
            # Extract text
            speed_limit = pytesseract.image_to_string(processed_image, config=config)
            speed_limit = speed_limit.strip()
            
            # Validate the result
            if speed_limit and speed_limit.isdigit() and 1 <= len(speed_limit) <= 3:
                # Only accept values from our predefined list
                if speed_limit in valid_speed_limits:
                    best_result = speed_limit
                    confidence = "High"
                    break
                elif not best_result and len(speed_limit) == 2:  # More likely to be valid if 2 digits
                    best_result = speed_limit
                    confidence = "Low"
        
        # Debug mode info without GUI display
        if debug and method_name:
            print(f"Method: {method_name}, Result: {speed_limit}")
        
        if confidence == "High":
            break
    
    # Debug mode info without GUI display
    if debug and best_result:
        print(f"Found speed limit: {best_result} with confidence: {confidence}")
    
    return best_result, confidence

def estimate_depth(frame, depth_estimator):
    """
    Estimate depth map using MiDaS
    Args:
        frame: Input video frame (OpenCV/NumPy format)
        depth_estimator: The depth estimation model
    Returns:
        depth_map: Estimated depth map
    """
    if depth_estimator is None:
        return None
    
    # Ensure frame is RGB (MiDaS expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Get depth map
    result = depth_estimator(pil_image)
    depth_map = result["depth"]
    
    # Convert to numpy array if it's not already
    if not isinstance(depth_map, np.ndarray):
        depth_map = np.array(depth_map)
    
    return depth_map

def convert_to_metric(depth_map):
    """
    Convert MiDaS depth output to metric depth using the inverse linear relationship
    Args:
        depth_map: Relative depth map from MiDaS
    Returns:
        metric_depth_map: Depth map in millimeters
    """
    # Constants for depth conversion (calibrated for your camera)
    A = 0.0018
    B = 0.0045
    
    # Apply the inverse linear transformation: true_depth = 1.0 / (A + (B * midas_output))
    metric_depth_map = 1.0 / (A + (B * depth_map))
    
    return metric_depth_map

def create_3d_bounding_box(bbox_2d, depth_value, camera_intrinsics):
    """
    Create a 3D bounding box from a 2D bounding box and depth
    Args:
        bbox_2d: 2D bounding box [x1, y1, x2, y2]
        depth_value: Metric depth value in millimeters
        camera_intrinsics: Camera parameters for 3D calculations
    Returns:
        bbox_3d: 3D bounding box information
    """
    x1, y1, x2, y2 = bbox_2d
    
    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Get camera intrinsics
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    # Convert pixel coordinates to world coordinates using pinhole camera model
    world_x = (center_x - cx) * depth_value / fx
    world_y = (center_y - cy) * depth_value / fy
    world_z = depth_value
    
    # Calculate width and height in world coordinates
    width_pixel = x2 - x1
    height_pixel = y2 - y1
    
    # Convert pixel dimensions to world dimensions
    width_world = width_pixel * depth_value / fx
    height_world = height_pixel * depth_value / fy
    
    # Estimate depth dimension based on typical speed limit sign proportions
    width = width_world  # Use calculated width
    height = height_world  # Use calculated height
    depth = width_world * 0.1  # Typical depth of a speed limit sign (thin)
    
    # Create 3D bounding box
    bbox_3d = {
        "center": [float(world_x), float(world_y), float(world_z)],
        "dimensions": {
            "width": float(width),
            "height": float(height),
            "depth": float(depth)
        },
        "orientation": 0.0,  # Assuming speed limit signs are perpendicular to the road
        "corners": calculate_3d_corners(world_x, world_y, world_z, width, height, depth)
    }
    
    return bbox_3d

def calculate_3d_corners(center_x, center_y, center_z, width, height, depth):
    """
    Calculate the 8 corners of a 3D bounding box
    Args:
        center_x, center_y, center_z: Center coordinates
        width, height, depth: Dimensions of the box
    Returns:
        corners: List of 8 corner coordinates
    """
    # Half dimensions
    hw = width / 2
    hh = height / 2
    hd = depth / 2
    
    # Define the 8 corners relative to center
    corners = [
        [center_x - hw, center_y - hh, center_z - hd],  # front bottom left
        [center_x + hw, center_y - hh, center_z - hd],  # front bottom right
        [center_x + hw, center_y + hh, center_z - hd],  # front top right
        [center_x - hw, center_y + hh, center_z - hd],  # front top left
        [center_x - hw, center_y - hh, center_z + hd],  # back bottom left
        [center_x + hw, center_y - hh, center_z + hd],  # back bottom right
        [center_x + hw, center_y + hh, center_z + hd],  # back top right
        [center_x - hw, center_y + hh, center_z + hd]   # back top left
    ]
    
    return corners

def process_video(model_path, source, output_path=None, conf_threshold=0.4, debug=False):
    """Process video for speed limit sign detection and recognition with 3D information"""
    # Load YOLOv5 model
    model = yolov5.load(model_path)
    model.conf = conf_threshold  # Set confidence threshold
    print(f"Model loaded successfully from {model_path}")
    
    # Initialize depth estimation model
    print("Loading depth estimation model...")
    try:
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        print("Depth model loaded successfully")
    except Exception as e:
        print(f"Error loading depth model: {e}")
        depth_estimator = None
    
    # Define camera intrinsics (adjust these values for your camera)
    camera_intrinsics = {
        'fx': 1594.7,
        'fy': 1607.7,
        'cx': 655.2961,
        'cy': 414.3627,
        'image_size': [960, 1280]
    }
    
    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if specified
    out = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing output to {output_path}")
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    recent_detections = []  # For tracking recent speed limits
    detection_count = 0
    
    # For tracking consistent detections
    detection_history = []
    current_speed_limit = None
    
    # For 3D data storage
    output_dir = os.path.join(os.path.dirname(output_path), "3d_data") if output_path else "3d_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing video from {source}...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_current = frame_count / elapsed_time
            print(f"Processed {frame_count} frames. Current FPS: {fps_current:.2f}")
        
        # Estimate depth map if depth estimator is available
        depth_map = None
        metric_depth_map = None
        if depth_estimator is not None:
            depth_map = estimate_depth(frame, depth_estimator)
            if depth_map is not None:
                metric_depth_map = convert_to_metric(depth_map)
        
        # Run YOLOv5 inference
        results = model(frame)
        
        # List to store detections for this frame
        frame_detections = []
        
        # Process each detection
        for det in results.xyxy[0]:  # Process detections for first image
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract the sign region with margin
            margin = 10
            y_min = max(0, y1 - margin)
            y_max = min(frame.shape[0], y2 + margin)
            x_min = max(0, x1 - margin)
            x_max = min(frame.shape[1], x2 + margin)
            
            sign_image = frame[y_min:y_max, x_min:x_max]
            
            if sign_image.size > 0:
                # Extract the speed limit number
                speed_limit, confidence = extract_speed_limit(sign_image, debug)
                
                # Only process if we got a number
                if speed_limit:
                    # Add to detection history for consistency checking
                    detection_history.append(speed_limit)
                    # Keep only the last 10 detections
                    if len(detection_history) > 10:
                        detection_history.pop(0)
                    
                    # Count occurrences of each speed limit in recent history
                    counter = Counter(detection_history)
                    most_common = counter.most_common(1)
                    
                    # Only accept a speed limit if it appears multiple times
                    if most_common and most_common[0][1] >= 3:  # At least 3 occurrences
                        current_speed_limit = most_common[0][0]
                        detection_count += 1
                        
                        # Calculate center of detection
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Get 3D information if depth data is available
                        bbox_3d = None
                        metric_depth = None
                        if depth_map is not None and metric_depth_map is not None:
                            if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                                # Get raw MiDaS depth value
                                raw_depth = float(depth_map[center_y, center_x])
                                # Get metric depth value
                                metric_depth = float(metric_depth_map[center_y, center_x])
                                # Create 3D bounding box
                                bbox_3d = create_3d_bounding_box(
                                    [x1, y1, x2, y2],
                                    metric_depth,
                                    camera_intrinsics
                                )
                        
                        # Create detection data
                        detection = {
                            'class': 'Speed Limit',
                            'value': speed_limit,
                            'confidence': float(conf),
                            'bounding_box': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2),
                                'center': {
                                    'x': float(center_x),
                                    'y': float(center_y)
                                }
                            }
                        }
                        
                        # Add depth information if available
                        if metric_depth is not None:
                            detection['metric_depth'] = metric_depth
                        if bbox_3d is not None:
                            detection['bbox_3d'] = bbox_3d
                        
                        frame_detections.append(detection)
                        
                        # Print detection info
                        print(f"Frame {frame_count}: Detected speed limit {current_speed_limit} km/h (confidence: {confidence})")
                        
                        # Display result on frame (for output video)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence indicator
                        confidence_color = (0, 255, 0) if confidence == "High" else (0, 165, 255)
                        cv2.putText(frame, f"Speed: {current_speed_limit} km/h", 
                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
                        
                        # Display detection confidence
                        cv2.putText(frame, f"Conf: {conf:.2f}", 
                                    (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Display depth information if available
                        if metric_depth is not None:
                            depth_text = f"D: {metric_depth:.2f}mm"
                            cv2.putText(frame, depth_text, (x1, y2+40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Display 3D coordinates if available
                            if bbox_3d is not None:
                                world_coords = bbox_3d["center"]
                                coords_text = f"X:{world_coords[0]:.1f},Y:{world_coords[1]:.1f},Z:{world_coords[2]:.1f}"
                                cv2.putText(frame, coords_text, (x1, y2+60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save 3D data for this frame
        if frame_detections:
            scene_data = {
                "frame_number": frame_count,
                "frame_dimensions": {
                    "width": width,
                    "height": height
                },
                "camera_intrinsics": camera_intrinsics,
                "objects": frame_detections,
                "has_depth_data": depth_estimator is not None
            }
            
            # Save JSON data
            json_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.json")
            with open(json_filename, 'w') as f:
                json.dump(scene_data, f, indent=2)
        
        # Display current speed limit on frame if available
        if current_speed_limit:
            cv2.putText(frame, f"Current Speed Limit: {current_speed_limit} km/h", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add frame counter and FPS
        elapsed_time = time.time() - start_time
        fps_text = f"FPS: {frame_count / elapsed_time:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write the frame to output video if specified
        if out:
            out.write(frame)
    
    # Release resources
    cap.release()
    if out:
        out.release()
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")
    print(f"Total speed limit detections: {detection_count}")
    if current_speed_limit:
        print(f"Final detected speed limit: {current_speed_limit} km/h")
    print(f"3D data saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speed Limit Detection and Recognition with 3D")
    parser.add_argument("--model", type=str, default=r"C:\Users\pavan\Documents\CV_environment\.venv\obj_detection\Models\speed_limit\road_signs_yolov52\weights\best.pt", 
                        help="Path to the trained YOLOv5 model")
    parser.add_argument("--source", type=str, default=r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene2\Undist\2023-03-03_10-31-11-front_undistort.mp4", 
                        help="Source for detection (0 for webcam, or video file path)")
    parser.add_argument("--output", type=str, default=r"C:\Users\pavan\Documents\CV_P3\output\3D_Speed_Limit\Scene_2", 
                        help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.4, 
                        help="Confidence threshold for detection")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode to print additional information")
    
    args = parser.parse_args()
    
    # Process the video
    process_video(args.model, args.source, args.output, args.conf, args.debug)
