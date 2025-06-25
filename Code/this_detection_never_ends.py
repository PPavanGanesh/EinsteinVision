import cv2
import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import json
import torch

from collections import deque
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from ultralytics import YOLO
from PIL import Image

RAFT_DIR = os.path.abspath(r"C:\Users\pavan\Documents\CV_P3\RAFT\RAFT")  # Update with your actual RAFT path
sys.path.insert(0, RAFT_DIR)
try:
    from raft import RAFT
    from utils.utils import InputPadder
    RAFT_AVAILABLE = True
except ImportError:
    print("RAFT not available. Vehicle motion detection will be disabled.")
    RAFT_AVAILABLE = False

class ComprehensiveSceneAnalyzer:
    def __init__(self,
                yolo_model_path='yolov8x.pt',
                segformer_model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                depth_model_name="Intel/dpt-hybrid-midas",
                # vehicle_type_classifier = VehicleTypeClassifier(model_path="vehicle_classifier_mobilenet.h5"),
                vehicle_orientation_model_path=r"C:\Users\pavan\Documents\CV_P3\best_orientation.pt",
                taillight_model_path=r"C:\Users\pavan\Documents\CV_P3\Manideep_breaklight\6-tail_light\runs\detect\TAIL_LIGHT_detection_model3\weights\best.pt",
                raft_model_path=os.path.join(RAFT_DIR,'models', 'raft-things.pth')):

        # Camera intrinsics from the provided image
        self.camera_intrinsics = {
            'fx': 1594.7,
            'fy': 1607.7,
            'cx': 655.2961,
            'cy': 414.3627,
            'image_size': [960, 1280]
        }
        # Initialize YOLO detector
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize SegFormer for lane detection
        print("Loading SegFormer model for lane detection...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(segformer_model_name)
            self.segformer_model.to(self.device)
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(segformer_model_name)
            print(f"SegFormer model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading SegFormer model: {e}")
            self.segformer_model = None
            self.feature_extractor = None
        
        # Initialize depth estimation model
        print("Loading depth estimation model...")
        try:
            from transformers import pipeline
            device_id = 0 if torch.cuda.is_available() else -1
            self.depth_estimator = pipeline("depth-estimation", model=depth_model_name, device=device_id)
            print("Depth model loaded successfully")
        except Exception as e:
            print(f"Error loading depth model: {e}")
            self.depth_estimator = None
            
        print("Loading tail light detection model...")
        try:
            self.taillight_model = YOLO(taillight_model_path)
            print(f"Tail light detection model loaded successfully")
        except Exception as e:
            print(f"Error loading tail light detection model: {e}")
            self.taillight_model = None
        
        # Initialize collision detector
        print("Initializing collision detector...")
        self.collision_detector = CollisionDetector(self.camera_intrinsics, collision_threshold_mm=20)
        
        # Initialize lane detector components
        self.vanishing_points_history = deque(maxlen=15)
        self.left_lane_history = deque(maxlen=10)
        self.right_lane_history = deque(maxlen=10)
        
        # Initialize RAFT for optical flow
        if RAFT_AVAILABLE:
            print("Loading RAFT model for vehicle motion detection...")
            try:
                import argparse
                raft_args = argparse.Namespace(
                    small=False,
                    mixed_precision=True,
                    alternate_corr=False,
                    path=raft_model_path
                )
                self.raft_model = RAFT(raft_args)
                self.raft_model = torch.nn.DataParallel(self.raft_model)
                self.raft_model.load_state_dict(torch.load(raft_args.path))
                self.raft_model = self.raft_model.module.eval().to(self.device)
                print("RAFT model loaded successfully")
                
                # Initialize vehicle tracker
                self.vehicle_tracker = VehicleTracker()
                self.prev_frame = None
                self.ego_speed = 0.0
            except Exception as e:
                print(f"Error loading RAFT model: {e}")
                self.raft_model = None
        else:
            self.raft_model = None
        
        # Enhanced class mapping with vehicle subtypes
        self.class_map = {
            'car': 'Sedan',  # Default car type, will be refined
            'truck': 'Truck',
            #'bus': 'Bus',
            'bicycle': 'Bicycle',
            'motorcycle': 'Motorcycle',
            'person': 'Pedestrian',
            'traffic light': 'Traffic Light',
            'stop sign': 'Stop Sign',
            'speed limit sign': 'Speed Limit Sign',
            'yield sign': 'Yield Sign',
            'do not enter sign': 'Do Not Enter Sign',
            'warning sign': 'Warning Sign',
            'dustbin': 'Dustbin',
            # 'suitcase': 'Dustbin',
            # 'backpack': 'Dustbin',
            'traffic pole': 'Traffic Pole',
            'traffic cone': 'Traffic Cone',
            'traffic cylinder': 'Traffic Cylinder',
            'arrow': 'Road Arrow'
        }

    def detect_traffic_light_color(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        cropped_img = frame[y1:y2, x1:x2]
        
        if cropped_img.size == 0:
            return "Unknown"

        # Normalize brightness and convert to HSV
        lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab = cv2.merge((l_eq, a, b))
        normalized_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)

        # Updated color ranges (to handle orangish red, dim green, yellow fade)
        color_ranges = {
            "Red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),    # True red to reddish-orange
                (np.array([11, 100, 100]), np.array([18, 255, 255])),   # Orangish red range
                (np.array([19, 100, 100]), np.array([25, 255, 255])),   # Orange to deep amber
                (np.array([170, 100, 100]), np.array([180, 255, 255]))  # Wrap-around red (at high hue end)
            ],
            "Yellow": [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
            "Green": [(np.array([35, 50, 50]), np.array([100, 255, 255]))]
        }
        

        color_scores = {}
        for color, ranges in color_ranges.items():
            score = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                score += np.sum(mask)
            color_scores[color] = score

        dominant_color = max(color_scores, key=color_scores.get)
        if color_scores[dominant_color] < (cropped_img.shape[0] * cropped_img.shape[1] * 10):
            return "Unknown"

        # ---- Arrow detection with hierarchy-aware contour analysis ----
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 5:
                (x, y, w, h) = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if 0.4 < aspect < 2.5:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        if cx < cropped_img.shape[1] * 0.33:
                            return f"{dominant_color} Left Arrow"
                        elif cx > cropped_img.shape[1] * 0.66:
                            return f"{dominant_color} Right Arrow"
                        else:
                            return f"{dominant_color} Straight Arrow"
        return dominant_color

    
    def detect_vehicle_type(self, frame, box):
        """
        Determine specific vehicle type (sedan, SUV, hatchback, pickup)
        based on geometric properties and visual features.
        """
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extract vehicle image
        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.size == 0:
            return "Sedan"  # Default if cropping fails
        
        # Calculate aspect ratio (width/height)
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        
        # Calculate relative size (area of bounding box)
        relative_size = width * height / (frame.shape[0] * frame.shape[1])
        
        # Calculate height ratio (height of vehicle / image height)
        height_ratio = height / frame.shape[0]
        
        # Calculate contour features
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate contour density (ratio of contour area to bounding box area)
        contour_area = 0
        if contours:
            contour_area = max(cv2.contourArea(cnt) for cnt in contours)
        contour_density = contour_area / (width * height) if width * height > 0 else 0
        
        # Get color information (check for typical SUV/truck colors)
        hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        # Check for dark colors (black, dark gray, dark blue)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size
        
        # Enhanced classification logic with more features
        if aspect_ratio > 1.8:  # Long vehicles
            if height_ratio > 0.45 and contour_density > 0.4:  # Taller with complex shape
                return "Pickup Truck"
            elif dark_ratio > 0.4:  # Darker color, likely a luxury sedan
                return "Sedan"
            else:
                return "Sedan"
        elif aspect_ratio > 1.4:  # Medium length
            if height_ratio > 0.5 and contour_density > 0.35:  # Taller with moderate complexity
                if dark_ratio > 0.35:  # Darker colors common in SUVs
                    return "SUV"
                else:
                    return "SUV"
            elif relative_size > 0.15:  # Larger vehicles likely SUVs
                return "SUV"
            else:
                return "Sedan"
        else:  # Shorter vehicles
            if height_ratio < 0.3:  # Very short vehicles
                return "SUV"
            elif contour_density > 0.40:  # More complex shape typical of hatchbacks
                return "Hatchback"
            else:
                return "Hatchback"

    # def detect_vehicle_facing(self, frame, bbox, depth_value=None):
    #     """
    #     Detect the basic facing direction of a vehicle (front, back, side).
    #     Uses simple visual cues without relying on complex models.
        
    #     Args:
    #         frame: Input video frame
    #         bbox: Bounding box coordinates [x1, y1, x2, y2]
            
    #     Returns:
    #         str: "front", "back", or "side"
    #     """
    #     x1, y1, x2, y2 = map(int, bbox)
    #     vehicle_img = frame[y1:y2, x1:x2]
        
    #     if vehicle_img.size == 0:
    #         return "back"  # Default if cropping fails
        
    #     height, width = vehicle_img.shape[:2]
        
    #     # Convert to HSV for better color detection
    #     hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
    #     # Check for taillights (red color in lower half)
    #     lower_half = hsv[height//2:, :]
    #     red_mask1 = cv2.inRange(lower_half, np.array([0, 70, 100]), np.array([25, 255, 255]))
    #     red_mask2 = cv2.inRange(lower_half, np.array([160, 70, 100]), np.array([180, 255, 255]))
    #     yellow_mask = cv2.inRange(lower_half, np.array([15, 100, 200]), np.array([40, 255, 255]))
    #     grey_mask = cv2.inRange(lower_half, np.array([0, 0, 50]), np.array([180, 50, 200]))
    #     red_mask = cv2.bitwise_or(red_mask1, red_mask2, yellow_mask)
    #     red_pixels = np.sum(red_mask) / (lower_half.shape[0] * lower_half.shape[1])
        
    #     # Check for headlights (white/yellow color in lower half)
    #     white_mask = cv2.inRange(lower_half, np.array([0, 0, 200]), np.array([180, 30, 255]))
    #     # yellow_mask = cv2.inRange(lower_half, np.array([15, 100, 200]), np.array([40, 255, 255]))
    #     # grey_mask = cv2.inRange(lower_half, np.array([0, 0, 50]), np.array([180, 50, 200]))

    #     front_mask = cv2.bitwise_or(white_mask, grey_mask)
    #     front_pixels = np.sum(front_mask) / (lower_half.shape[0] * lower_half.shape[1])
        
    #     # Calculate aspect ratio (width/height)
    #     aspect_ratio = width / height
        
    #     # Check symmetry
    #     gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(gray, 50, 150)
    #     left_half = edges[:, :width//2]
    #     right_half = edges[:, width//2:]
    #     left_edges = np.count_nonzero(left_half)
    #     right_edges = np.count_nonzero(right_half)
    #     symmetry_ratio = abs(left_edges - right_edges) / (left_edges + right_edges + 1e-5)
        
    #     # Decision logic
    #     if symmetry_ratio < 0.2:  # Highly symmetric
    #         # if red_pixels > front_pixels and red_pixels > 0.015:
    #         if red_pixels > front_pixels:
    #             return "backkk"
    #         elif front_pixels > red_pixels and front_pixels > 0.095:
    #             return "fronttt"
    #         else:
    #             # Default to front if can't determine
    #             return "backk"
    #     else:  # Less symmetric, likely side view
    #         return "side"
    
    def detect_vehicle_facing(self, frame, bbox, depth_value=None):
        """
        Detect the basic facing direction of a vehicle (front, back, side).
        Uses visual cues and position-based heuristics for US traffic rules.
        
        Args:
            frame: Input video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            depth_value: Optional depth value in mm
            
        Returns:
            str: "front", "back", or "side"
        """
        x1, y1, x2, y2 = map(int, bbox)
        vehicle_img = frame[y1:y2, x1:x2]
        
        if vehicle_img.size == 0 or vehicle_img.shape[0] < 10 or vehicle_img.shape[1] < 10:
            return "back"  # Default if cropping fails
        
        height, width = vehicle_img.shape[:2]
        frame_width = frame.shape[1]
        
        # Calculate position in frame (for US traffic rule heuristic)
        center_x = (x1 + x2) / 2
        position_ratio = center_x / frame_width  # 0 = far left, 1 = far right
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
        # Check for taillights (red color in lower portion)
        lower_portion = hsv[int(height*0.6):, :]
        red_mask1 = cv2.inRange(lower_portion, np.array([0, 70, 100]), np.array([25, 255, 255]))
        red_mask2 = cv2.inRange(lower_portion, np.array([160, 70, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = np.sum(red_mask) / (lower_portion.shape[0] * lower_portion.shape[1] + 1e-5)
        
        # Check for headlights (white/yellow color in lower portion)
        white_mask = cv2.inRange(lower_portion, np.array([0, 0, 200]), np.array([180, 30, 255]))
        yellow_mask = cv2.inRange(lower_portion, np.array([15, 100, 200]), np.array([40, 255, 255]))
        front_mask = cv2.bitwise_or(white_mask, yellow_mask)
        front_pixels = np.sum(front_mask) / (lower_portion.shape[0] * lower_portion.shape[1] + 1e-5)
        
        # Check symmetry
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        left_half = edges[:, :width//2]
        right_half = edges[:, width//2:]
        left_edges = np.count_nonzero(left_half)
        right_edges = np.count_nonzero(right_half)
        symmetry_ratio = abs(left_edges - right_edges) / (left_edges + right_edges + 1e-5)
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = width / height
        is_wide = aspect_ratio > 1.8  # Wide vehicles are more likely to be side views
        
        # Strong indicators for back view
        back_score = red_pixels * 3.0
        
        # Strong indicators for front view
        front_score = front_pixels * 2.5
        
        # Side view indicators
        side_score = 0
        if is_wide:
            side_score += 2.0
        side_score += symmetry_ratio * 2.0  # Higher asymmetry suggests side view
        
        # Apply US traffic rule heuristic for distant vehicles
        if depth_value is not None and depth_value > 30000:  # 30 meters
            if position_ratio > 0.6:  # Right side of frame
                back_score += 12.0 #oost back score
            elif position_ratio < 0.4:  # Left side of frame
                front_score += 7.0 # Boost front score
        
        # Strong evidence of taillights
        if red_pixels > 0.03 and red_pixels > front_pixels:
            return "back"
        
        # Strong evidence of headlights
        if front_pixels > 0.07 and front_pixels > red_pixels * 5.0:
            return "front"
        
        # Use scores for decision
        max_score = max(front_score, back_score, side_score)
        
        if max_score == side_score and side_score > 1.0:
            return "side"
        elif max_score == back_score:
            return "back"
        else:
            return "back"

    
    #######################################################################################################  BELOW IS THE OLD CODE FOR VEHICLE FACING DETECTION
    # def detect_vehicle_facing(self, frame, bbox, depth_value=None):
    #     """
    #     Detect the basic facing direction of a vehicle (front, back, side).
    #     Uses visual cues and depth information for more accurate detection.
        
    #     Args:
    #         frame: Input video frame
    #         bbox: Bounding box coordinates [x1, y1, x2, y2]
    #         depth_value: Estimated depth at the center of the object (optional)
            
    #     Returns:
    #         str: "front", "back", or "side"
    #     """
    #     x1, y1, x2, y2 = map(int, bbox)
    #     vehicle_img = frame[y1:y2, x1:x2]
        
    #     if vehicle_img.size == 0:
    #         return "front"  # Default if cropping fails
        
    #     height, width = vehicle_img.shape[:2]
        
    #     # Calculate aspect ratio (width/height)
    #     aspect_ratio = width / height
        
    #     # 1. Check symmetry (more robust method)
    #     gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 50, 150)
        
    #     # Divide image into left, middle, and right sections
    #     left_half = edges[:, :width//2]
    #     right_half = edges[:, width//2:]
    #     left_edges = np.count_nonzero(left_half)
    #     right_edges = np.count_nonzero(right_half)
        
    #     # Calculate symmetry score (0 = perfect symmetry, 1 = no symmetry)
    #     total_edges = left_edges + right_edges
    #     if total_edges > 0:
    #         symmetry_score = abs(left_edges - right_edges) / total_edges
    #     else:
    #         symmetry_score = 0
        
    #     # 2. Check for headlights/taillights (improved color detection)
    #     hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
    #     # Lower half of the vehicle (where lights are typically located)
    #     lower_region = hsv[int(height*0.6):, :]
        
    #     # Check for taillights (red color)
    #     red_mask1 = cv2.inRange(lower_region, np.array([0, 70, 100]), np.array([25, 255, 255]))
    #     red_mask2 = cv2.inRange(lower_region, np.array([160, 70, 100]), np.array([180, 255, 255]))
    #     red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    #     red_pixels = np.sum(red_mask) / (lower_region.shape[0] * lower_region.shape[1] + 1e-5)
        
    #     # Check for headlights (white/yellow color)
    #     white_mask = cv2.inRange(lower_region, np.array([0, 0, 200]), np.array([180, 30, 255]))
    #     yellow_mask = cv2.inRange(lower_region, np.array([15, 100, 200]), np.array([40, 255, 255]))
    #     front_mask = cv2.bitwise_or(white_mask, yellow_mask)
    #     front_pixels = np.sum(front_mask) / (lower_region.shape[0] * lower_region.shape[1] + 1e-5)
        
    #     # 3. Check for license plate (typically visible in front/back views)
    #     plate_region = gray[int(height*0.5):, int(width*0.25):int(width*0.75)]
    #     _, plate_thresh = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     plate_contours, _ = cv2.findContours(plate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     plate_detected = False
    #     for cnt in plate_contours:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         aspect = w / float(h) if h > 0 else 0
    #         if 2 < aspect < 5 and cv2.contourArea(cnt) > 100:
    #             plate_detected = True
    #             break
        
    #     # 4. Use depth information if available
    #     depth_factor = 1.0
    #     if depth_value is not None:
    #         # If depth is available, use it to adjust our confidence
    #         # Side views typically have larger width relative to depth
    #         # Front/back views have smaller width relative to depth
    #         width_depth_ratio = width / (depth_value / 1000)  # Convert depth to meters
    #         if width_depth_ratio > 0.3:  # Threshold determined empirically
    #             depth_factor = 1.5  # Increase confidence for side view
    #         else:
    #             depth_factor = 0.8  # Decrease confidence for side view
        
    #     # 5. Decision logic with weighted factors
    #     # Calculate scores for each orientation
    #     front_score = front_pixels * 2.0 + (1 - symmetry_score) * 0.5 + (plate_detected * 0.5)
    #     back_score = red_pixels * 2.0 + (1 - symmetry_score) * 0.5 + (plate_detected * 0.5)
    #     side_score = (symmetry_score * 1.5 + (aspect_ratio > 1.8) * 1.0) * depth_factor
        
    #     # Apply thresholds with hysteresis to prevent flickering
    #     if side_score > max(front_score, back_score) and side_score > 0.8:
    #         return "side"
    #     elif front_score > back_score and front_score > 0.6:
    #         return "front"
    #     elif back_score > front_score and back_score > 0.6:
    #         return "back"
    #     else:
    #         # Default based on simple heuristics
    #         if symmetry_score > 0.25:  # Not very symmetric
    #             return "side"
    #         elif red_pixels > front_pixels:
    #             return "back"
    #         else:
    #             return "front"
    
    # def detect_vehicle_facing(self, frame, bbox, depth_value=None):
    #     """
    #     Detect the basic facing direction of a vehicle (front, back, side).
    #     Uses visual cues and depth information for more accurate detection.
        
    #     Args:
    #         frame: Input video frame
    #         bbox: Bounding box coordinates [x1, y1, x2, y2]
    #         depth_value: Estimated depth at the center of the object (optional)
            
    #     Returns:
    #         str: "front", "back", or "side"
    #     """
    #     x1, y1, x2, y2 = map(int, bbox)
    #     vehicle_img = frame[y1:y2, x1:x2]
        
    #     if vehicle_img.size == 0:
    #         return "back"  # Default if cropping fails
        
    #     height, width = vehicle_img.shape[:2]
        
    #     # Calculate aspect ratio (width/height)
    #     aspect_ratio = width / height
        
    #     # 1. Check symmetry (more robust method)
    #     gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 50, 150)
        
    #     # Divide image into left, middle, and right sections
    #     left_half = edges[:, :width//2]
    #     right_half = edges[:, width//2:]
    #     left_edges = np.count_nonzero(left_half)
    #     right_edges = np.count_nonzero(right_half)
        
    #     # Calculate symmetry score (0 = perfect symmetry, 1 = no symmetry)
    #     total_edges = left_edges + right_edges
    #     if total_edges > 0:
    #         symmetry_score = abs(left_edges - right_edges) / total_edges
    #     else:
    #         symmetry_score = 0
        
    #     # 2. Check for headlights/taillights (improved color detection)
    #     hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
    #     # Lower half of the vehicle (where lights are typically located)
    #     lower_region = hsv[int(height*0.6):, :]
        
    #     # Check for taillights (red color)
    #     red_mask1 = cv2.inRange(lower_region, np.array([0, 70, 100]), np.array([25, 255, 255]))
    #     red_mask2 = cv2.inRange(lower_region, np.array([160, 70, 100]), np.array([180, 255, 255]))
    #     red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    #     red_pixels = np.sum(red_mask) / (lower_region.shape[0] * lower_region.shape[1] + 1e-5)
        
    #     # Check for headlights (white/yellow color)
    #     white_mask = cv2.inRange(lower_region, np.array([0, 0, 200]), np.array([180, 30, 255]))
    #     yellow_mask = cv2.inRange(lower_region, np.array([15, 100, 200]), np.array([40, 255, 255]))
    #     front_mask = cv2.bitwise_or(white_mask, yellow_mask)
    #     front_pixels = np.sum(front_mask) / (lower_region.shape[0] * lower_region.shape[1] + 1e-5)
        
    #     # 3. Check for license plate (typically visible in front/back views)
    #     plate_region = gray[int(height*0.5):, int(width*0.25):int(width*0.75)]
    #     _, plate_thresh = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     plate_contours, _ = cv2.findContours(plate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     plate_detected = False
    #     for cnt in plate_contours:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         ar = w / float(h) if h > 0 else 0
    #         if 2 < ar < 5 and cv2.contourArea(cnt) > 100:
    #             plate_detected = True
    #             break
        
    #     # 4. Use depth information if available
    #     depth_factor = 1.0
    #     if depth_value is not None:
    #         # If depth is available, use it to adjust our confidence
    #         # Side views typically have larger width relative to depth
    #         # Front/back views have smaller width relative to depth
    #         width_depth_ratio = width / (depth_value / 1000)  # Convert depth to meters
    #         if width_depth_ratio > 0.3:  # Threshold determined empirically
    #             depth_factor = 1.5  # Increase confidence for side view
    #         else:
    #             depth_factor = 0.8  # Decrease confidence for side view
        
    #     # 5. Decision logic with weighted factors
    #     # Calculate scores for each orientation
    #     front_score = front_pixels * 2.0 + (1 - symmetry_score) * 1.0 + (plate_detected * 1.0)
    #     back_score = red_pixels * 2.0 + (1 - symmetry_score) * 0.8 + (plate_detected * 0.5)
    #     side_score = (symmetry_score * 1.5 + (aspect_ratio > 1.8) * 1.0) * depth_factor
        
    #     # Apply thresholds with hysteresis to prevent flickering
    #     if side_score > max(front_score, back_score) and side_score > 0.8:
    #         return "side"
    #     elif front_score > back_score and front_score > 0.6:
    #         return "front"
    #     elif back_score > front_score and back_score > 0.6:
    #         return "back"
    #     else:
    #         # Default based on simple heuristics
    #         if symmetry_score > 0.25:  # Not very symmetric
    #             return "side"
    #         elif red_pixels > front_pixels:
    #             return "back"
    #         else:
    #             return "front"
    
    
    # def detect_vehicle_facing(self, frame, bbox, depth_value=None):
    #     """
    #     Enhanced vehicle orientation detector focusing on reliable front/back detection
        
    #     Args:
    #         frame: Input video frame
    #         bbox: Bounding box coordinates [x1, y1, x2, y2]
    #         depth_value: Estimated depth at the center of the object (optional)
            
    #     Returns:
    #         str: "front", "back", or "side"
    #     """
    #     x1, y1, x2, y2 = map(int, bbox)
    #     vehicle_img = frame[y1:y2, x1:x2]
        
    #     if vehicle_img.size == 0 or vehicle_img.shape[0] < 10 or vehicle_img.shape[1] < 10:
    #         return "front"  # Default if cropping fails
        
    #     height, width = vehicle_img.shape[:2]
        
    #     # 1. STRONG FRONT/BACK INDICATORS
    #     # Convert to HSV for better color detection
    #     hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
    #     # Check for taillights (red color in lower portion)
    #     lower_portion = hsv[int(height*0.5):, :]
    #     red_mask1 = cv2.inRange(lower_portion, np.array([0, 100, 100]), np.array([10, 255, 255]))
    #     red_mask2 = cv2.inRange(lower_portion, np.array([160, 100, 100]), np.array([180, 255, 255]))
    #     red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    #     red_pixels = np.sum(red_mask) / (lower_portion.shape[0] * lower_portion.shape[1] + 1e-5)
        
    #     # Check for headlights (white/yellow color in lower portion)
    #     white_mask = cv2.inRange(lower_portion, np.array([0, 0, 180]), np.array([180, 30, 255]))
    #     yellow_mask = cv2.inRange(lower_portion, np.array([15, 80, 180]), np.array([35, 255, 255]))
    #     front_mask = cv2.bitwise_or(white_mask, yellow_mask)
    #     front_pixels = np.sum(front_mask) / (lower_portion.shape[0] * lower_portion.shape[1] + 1e-5)
        
    #     # 2. LICENSE PLATE DETECTION (strong indicator for front/back)
    #     # Look for rectangular shapes with license plate aspect ratio
    #     gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 50, 150)
        
    #     # Find contours
    #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     plate_detected = False
    #     plate_area_ratio = 0
        
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if area < 100:  # Filter small contours
    #             continue
                
    #         # Approximate the contour
    #         peri = cv2.arcLength(cnt, True)
    #         approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
    #         # Check if it's rectangular (4 corners)
    #         if len(approx) == 4:
    #             (x, y, w, h) = cv2.boundingRect(approx)
    #             aspect_ratio = w / float(h)
                
    #             # License plates typically have aspect ratio between 2:1 and 5:1
    #             if 2.0 < aspect_ratio < 5.0:
    #                 # Check if it's in the lower half of the vehicle
    #                 if y > height * 0.4:
    #                     plate_detected = True
    #                     plate_area_ratio = (w * h) / (width * height)
    #                     break
        
    #     # 3. SYMMETRY ANALYSIS (front and back views are more symmetric)
    #     left_half = edges[:, :width//2]
    #     right_half = edges[:, width//2:]
    #     left_edges = np.count_nonzero(left_half)
    #     right_edges = np.count_nonzero(right_half)
        
    #     # Calculate symmetry score (0 = perfect symmetry, 1 = no symmetry)
    #     total_edges = left_edges + right_edges
    #     if total_edges > 0:
    #         symmetry_score = abs(left_edges - right_edges) / total_edges
    #     else:
    #         symmetry_score = 0
        
    #     # 4. ASPECT RATIO (side views tend to be wider)
    #     aspect_ratio = width / height
    #     is_wide = aspect_ratio > 1.8
        
    #     # 5. DECISION LOGIC WITH WEIGHTED SCORING
    #     # Strong indicators for back view
    #     back_score = red_pixels * 3.0
    #     if red_pixels > 0.1:  # Significant red presence
    #         back_score += 2.0
            
    #     # Strong indicators for front view
    #     front_score = front_pixels * 3.0
    #     if front_pixels > 0.1:  # Significant headlight presence
    #         front_score += 2.0
        
    #     # License plate is a good indicator for front/back
    #     if plate_detected:
    #         # Add to both front and back scores
    #         front_score += 1.0
    #         back_score += 1.0
        
    #     # Symmetry favors front/back views
    #     if symmetry_score < 0.2:  # Very symmetric
    #         front_score += 1.0
    #         back_score += 1.0
        
    #     # Wide aspect ratio favors side view
    #     side_score = 0
    #     if is_wide:
    #         side_score += 2.0
        
    #     # Add symmetry score (higher means less symmetric, favoring side view)
    #     side_score += symmetry_score * 3.0
        
    #     # Final decision with thresholds
    #     # If we have strong evidence of taillights, it's likely the back
    #     if red_pixels > 0.08 and red_pixels > front_pixels * 1.5:
    #         return "back"
        
    #     # If we have strong evidence of headlights, it's likely the front
    #     if front_pixels > 0.08 and front_pixels > red_pixels * 1.5:
    #         return "front"
        
    #     # Otherwise use the scores
    #     max_score = max(front_score, back_score, side_score)
        
    #     if max_score == front_score and front_score > 0.5:
    #         return "front"
    #     elif max_score == back_score and back_score > 0.5:
    #         return "back"
    #     elif max_score == side_score and side_score > 1.0:
    #         return "side"
    #     else:
    #         # Default fallback based on simple heuristics
    #         if red_pixels > front_pixels:
    #             return "back"
    #         else:
    #             return "front"  # Default to front as a safer choice
    
    # def detect_vehicle_facing(self, frame, bbox, depth_value=None):
    #     """
    #     Enhanced vehicle orientation detector focusing on reliable front/back detection
    #     """
    #     x1, y1, x2, y2 = map(int, bbox)
    #     vehicle_img = frame[y1:y2, x1:x2]
        
    #     if vehicle_img.size == 0 or vehicle_img.shape[0] < 10 or vehicle_img.shape[1] < 10:
    #         return "front"  # Default if cropping fails
        
    #     height, width = vehicle_img.shape[:2]
        
    #     # 1. STRONG FRONT/BACK INDICATORS - Improved color ranges
    #     hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
    #     # Check for taillights (red color in lower portion) - More sensitive detection
    #     lower_portion = hsv[int(height*0.5):, :]
    #     red_mask1 = cv2.inRange(lower_portion, np.array([0, 70, 100]), np.array([15, 255, 255]))
    #     red_mask2 = cv2.inRange(lower_portion, np.array([160, 70, 100]), np.array([180, 255, 255]))
    #     red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    #     red_pixels = np.sum(red_mask) / (lower_portion.shape[0] * lower_portion.shape[1] + 1e-5)
        
    #     # Check for headlights (white/yellow color in lower portion) - More sensitive detection
    #     white_mask = cv2.inRange(lower_portion, np.array([0, 0, 180]), np.array([180, 40, 255]))
    #     yellow_mask = cv2.inRange(lower_portion, np.array([15, 60, 180]), np.array([35, 255, 255]))
    #     front_mask = cv2.bitwise_or(white_mask, yellow_mask)
    #     front_pixels = np.sum(front_mask) / (lower_portion.shape[0] * lower_portion.shape[1] + 1e-5)
        
    #     # 2. SYMMETRY ANALYSIS (front and back views are more symmetric)
    #     gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 50, 150)
        
    #     left_half = edges[:, :width//2]
    #     right_half = edges[:, width//2:]
    #     left_edges = np.count_nonzero(left_half)
    #     right_edges = np.count_nonzero(right_half)
        
    #     # Calculate symmetry score (0 = perfect symmetry, 1 = no symmetry)
    #     total_edges = left_edges + right_edges
    #     if total_edges > 0:
    #         symmetry_score = abs(left_edges - right_edges) / total_edges
    #     else:
    #         symmetry_score = 0
        
    #     # 3. DECISION LOGIC WITH WEIGHTED SCORING - Adjusted thresholds
    #     # If we have strong evidence of taillights, it's likely the back
    #     if red_pixels > 0.05 and red_pixels > front_pixels * 1.2:
    #         return "back"
        
    #     # If we have strong evidence of headlights, it's likely the front
    #     if front_pixels > 0.05 and front_pixels > red_pixels * 1.2:
    #         return "front"
        
    #     # Default fallback based on simple heuristics with lower threshold for back detection
    #     if red_pixels > 0.03:
    #         return "back"
    #     else:
    #         return "front"  # Default to front as a safer choice


    # def detect_vehicle_facing_with_position(self, frame, bbox, depth_value=None):
    #     x1, y1, x2, y2 = map(int, bbox)
    #     width = frame.shape[1]
    #     cx = (x1 + x2) / 2

    #     facing = self.detect_vehicle_facing(frame, bbox, depth_value)

    #     # Only apply correction if visual cues are ambiguous or distance is large
    #     if depth_value is not None and depth_value > 40:  # meters, adjust as needed
    #         if cx > width * 0.6:
    #             return "back"
    #         elif cx < width * 0.4:
    #             return "front"
    #     return facing
    
    # def detect_vehicle_facing_with_position(self, frame, bbox, depth_value=None):
    #     x1, y1, x2, y2 = map(int, bbox)
    #     width = frame.shape[1]
    #     cx = (x1 + x2) / 2
        
    #     # Get base facing from visual cues
    #     facing = self.detect_vehicle_facing(frame, bbox, depth_value)
        
    #     # Apply US traffic rule correction
    #     # Convert depth from mm to meters for readability
    #     depth_meters = depth_value / 1000.0 if depth_value is not None else None
        
    #     # Only apply position-based correction if:
    #     # 1. We have depth information
    #     # 2. The vehicle is beyond a certain distance (less reliable visual cues)
    #     # 3. The visual detection confidence isn't very high
    #     if depth_meters is not None and depth_meters > 15:  # Vehicles further than 15m
    #         # Right side of road - vehicles likely facing away (back)
    #         if cx > width * 0.55:
    #             # Strong correction for right side
    #             return "back"
    #         # Left side of road - vehicles likely facing toward camera (front)
    #         elif cx < width * 0.45:
    #             # Strong correction for left side
    #             return "front"
        
    #     return facing


    # def is_side_view_special_case(self, frame, bbox, depth_value):
    #     """
    #     Check for special cases that indicate a side view using depth and visual cues
        
    #     Args:
    #         frame: Input video frame
    #         bbox: Bounding box coordinates [x1, y1, x2, y2]
    #         depth_value: Estimated depth at the center of the object
            
    #     Returns:
    #         bool: True if this is likely a side view
    #     """
    #     x1, y1, x2, y2 = map(int, bbox)
    #     width = x2 - x1
    #     height = y2 - y1
        
    #     # 1. Check aspect ratio - side views typically have larger width/height ratio
    #     aspect_ratio = width / height
    #     if aspect_ratio > 2.0:  # Very wide vehicles are likely side views
    #         return True
        
    #     # 2. Check width to depth ratio - side views appear wider relative to their depth
    #     width_depth_ratio = width / (depth_value / 1000)  # Convert depth to meters
    #     if width_depth_ratio > 0.35:  # Empirically determined threshold
    #         return True
        
    #     # 3. Check for wheel visibility - wheels are often visible in side views
    #     vehicle_img = frame[y1:y2, x1:x2]
    #     if vehicle_img.size == 0:
    #         return False
            
    #     # Look for circular shapes (wheels) in lower portion of the vehicle
    #     lower_third = vehicle_img[int(height*0.6):, :]
    #     gray = cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
    #     # Use Hough circles to detect wheels
    #     circles = cv2.HoughCircles(
    #         blurred, 
    #         cv2.HOUGH_GRADIENT, 
    #         dp=1, 
    #         minDist=20, 
    #         param1=50, 
    #         param2=30, 
    #         minRadius=int(height*0.05), 
    #         maxRadius=int(height*0.2)
    #     )
        
    #     if circles is not None and len(circles[0]) >= 2:
    #         # Multiple circular shapes detected - likely wheels
    #         return True
        
    #     return False

    def detect_vehicle_orientation(self, frame, box):
        """
        Detect the orientation of a vehicle (front, back, left side, right side).
        """
        import cv2
        import numpy as np

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.size == 0:
            return "Unknown"

        height, width = vehicle_img.shape[:2]

        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        left_edges = np.count_nonzero(edges[:, :width//3])
        middle_edges = np.count_nonzero(edges[:, width//3:2*width//3])
        right_edges = np.count_nonzero(edges[:, 2*width//3:])
        left_right_ratio = left_edges / (right_edges + 1e-5)

        hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)

        # Taillights (back)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 100]), np.array([25, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 70, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Headlights (front)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([40, 255, 255]))
        front_mask = cv2.bitwise_or(white_mask, yellow_mask)

        red_score = np.sum(red_mask) / (width * height)
        front_score = np.sum(front_mask) / (width * height)

        # Plate detection (lower half)
        lower_half = gray[height//2:, :]
        _, plate_thresh = cv2.threshold(lower_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(plate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plate_detected = False
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / float(h)
            if 2 < ar < 5 and cv2.contourArea(cnt) > 100:
                plate_detected = True
                break

        # Windshield analysis
        upper_middle = gray[height//4:height//2, width//3:2*width//3]
        _, windshield_thresh = cv2.threshold(upper_middle, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        windshield_brightness = np.mean(windshield_thresh)

        if 0.75 < left_right_ratio < 1.25:
            # Symmetrical: Front or Back
            if red_score > 0.08 and red_score > front_score:
                return "BACK"
            elif front_score > 0.08 or (plate_detected and windshield_brightness > 100):   ##changes form130
                return "FRONT"
            else:
                return "FRONT"
        else:
            # #Asymmetrical: Side view
            # hog = cv2.HOGDescriptor()
            # resized = cv2.resize(gray, (64, 64))
            # hog_feats = hog.compute(resized).reshape(-1)

            # left_half = resized[:, :32]
            # right_half = resized[:, 32:]

            # left_grad = np.mean(cv2.Sobel(left_half, cv2.CV_64F, 1, 0))
            # right_grad = np.mean(cv2.Sobel(right_half, cv2.CV_64F, 1, 0))

            # if left_grad > right_grad * 1.2:
            #     return "Right Side"
            # elif right_grad > left_grad * 1.2:
            #     return "Left Side"
            # else:
            #     return "Side"
            
            hog = cv2.HOGDescriptor(_winSize=(64, 64),
                                    _blockSize=(16, 16),
                                    _blockStride=(8, 8),
                                    _cellSize=(8, 8),
                                    _nbins=9)
            resized = cv2.resize(gray, (64, 64))
            hog_feats = hog.compute(resized).reshape(-1)

            left_half = resized[:, :32]
            right_half = resized[:, 32:]
            
            left_grad = np.mean(cv2.Sobel(left_half, cv2.CV_64F, 1, 0))
            right_grad = np.mean(cv2.Sobel(right_half, cv2.CV_64F, 1, 0))

            if left_grad > right_grad * 1.2:
                return "Left Side"
            elif right_grad > left_grad * 1.2:
                return "Right Side"
            else:
                return "Side"

    def detect_tail_lights(self, frame, vehicle_bbox):
        """
        Detect tail lights using the specialized model
        
        Args:
            frame: Input video frame
            vehicle_bbox: Bounding box of the vehicle [x1, y1, x2, y2]
            
        Returns:
            list: List of detected tail lights with position and side information
        """
        if self.taillight_model is None:
            return []
        
        # Extract vehicle region
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        vehicle_img = frame[y1:y2, x1:x2]
        
        if vehicle_img.size == 0:
            return []
        
        # Get vehicle dimensions
        vehicle_width = x2 - x1
        
        # Run tail light detection on the vehicle crop
        results = self.taillight_model.predict(source=vehicle_img, conf=0.5, save=False)
        
        tail_lights = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Get class and confidence
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Only process tail lights (class 0 in the model)
                if cls_id == 0:
                    # Get box coordinates within the vehicle crop
                    tx1, ty1, tx2, ty2 = box.xyxy[0].astype(int)
                    
                    # Calculate center point
                    center_x = (tx1 + tx2) // 2
                    
                    # Determine which side the light is on
                    # If center_x is in left half of vehicle, it's the RIGHT taillight (from car's perspective)
                    # If center_x is in right half of vehicle, it's the LEFT taillight (from car's perspective)
                    if center_x < vehicle_width / 2:
                        side = "right"
                    else:
                        side = "left"
                    
                    # Convert coordinates to original frame coordinates
                    frame_x1 = x1 + tx1
                    frame_y1 = y1 + ty1
                    frame_x2 = x1 + tx2
                    frame_y2 = y1 + ty2
                    
                    tail_lights.append({
                        "bbox": (frame_x1, frame_y1, frame_x2, frame_y2),
                        "side": side,
                        "confidence": confidence
                    })
        
        return tail_lights
    
    def analyze_tail_light_state(self, frame, light_bbox):
        """
        Analyze the state of a detected light (regular taillight vs. brake light)
        
        Args:
            frame: Input video frame
            light_bbox: Bounding box of the light [x1, y1, x2, y2]
            
        Returns:
            dict: Dictionary with light state information
        """
        x1, y1, x2, y2 = light_bbox
        light_img = frame[y1:y2, x1:x2]
        
        if light_img.size == 0:
            return {"is_on": False, "is_brake": False}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for tail lights
        # Regular tail lights (dimmer red)
        lower_red1 = np.array([0, 70, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 150])
        upper_red2 = np.array([180, 255, 255])
        
        # Brake lights (brighter, more intense red)
        lower_brake1 = np.array([0, 100, 200])
        upper_brake1 = np.array([10, 255, 255])
        lower_brake2 = np.array([170, 100, 200])
        upper_brake2 = np.array([180, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        brake_mask1 = cv2.inRange(hsv, lower_brake1, upper_brake1)
        brake_mask2 = cv2.inRange(hsv, lower_brake2, upper_brake2)
        brake_mask = cv2.bitwise_or(brake_mask1, brake_mask2)
        
        # Calculate percentages
        total_pixels = light_img.size // 3  # 3 channels
        red_ratio = cv2.countNonZero(red_mask) / total_pixels
        brake_ratio = cv2.countNonZero(brake_mask) / total_pixels
        
        # Determine light state
        is_on = red_ratio > 0.15
        is_brake = brake_ratio > 0.15
        
        return {
            "is_on": is_on,
            "is_brake": is_brake,
            "red_ratio": red_ratio,
            "brake_ratio": brake_ratio
        }
    
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO with enhanced vehicle classification."""
        results = self.yolo_model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get confidence and class
                conf = box.conf[0]
                cls_id = int(box.cls[0].item())
                cls = result.names[cls_id]
                
                # Skip low confidence detections
                if conf < 0.65:
                    continue
                    
                # Extract bounding box coordinates first
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                    
                # Process based on class type
                if cls == 'car':
                    # Process as vehicle
                    vehicle_type = self.detect_vehicle_type(frame, box)
                    
                    # Use the simplified vehicle facing detection
                    # facing = self.detect_vehicle_facing_with_position(frame, [int(x1), int(y1), int(x2), int(y2)])
                    facing = self.detect_vehicle_facing(frame, [int(x1), int(y1), int(x2), int(y2)])
                    
                    # Use the simplified light detection function
                    light_states = self.detect_vehicle_lights_simple(frame, box, vehicle_type, facing)
                    
                    # Add light information to the class name
                    light_info = []
                    if light_states["headlights"]:
                        light_info.append("Headlights On")
                    if light_states["taillights"]:
                        light_info.append("Taillights On")
                    if light_states["left_indicator"]:
                        light_info.append("Left Indicator")
                    if light_states["right_indicator"]:
                        light_info.append("Right Indicator")
                    
                    light_str = ", ".join(light_info)
                    if light_str:
                        mapped_cls = f"{vehicle_type} ({facing}, {light_str})"
                    else:
                        mapped_cls = f"{vehicle_type} ({facing})"
                elif cls == 'traffic light':
                    # Process as traffic light
                    light_color = self.detect_traffic_light_color(frame, box.xyxy[0])
                    mapped_cls = f"Traffic Light ({light_color})"
                else:
                    # Map to standard class
                    mapped_cls = self.class_map.get(cls, cls)
                
                # Create detection data
                detection_data = {
                    'class': mapped_cls,
                    'original_class': cls,
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
                
                # Add light states only for vehicles
                if cls == 'car':
                    detection_data['lights'] = light_states
                    
                    # Add facing information explicitly for easier access in JSON
                    detection_data['facing'] = facing
                
                detections.append(detection_data)
        
        return detections

    def detect_vehicle_lights_simple(self, frame, box, vehicle_type, orientation):
        """
        Simple detection of vehicle lights with basic on/off states
        
        Args:
            frame: Input video frame
            box: Bounding box of the vehicle [x1, y1, x2, y2]
            vehicle_type: Type of vehicle (sedan, SUV, etc.)
            orientation: Orientation of vehicle (front, back, side)
            
        Returns:
            dict: Dictionary containing light states
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Extract vehicle image
        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.size == 0:
            return {"headlights": False, "taillights": False, "left_indicator": False, "right_indicator": False}
        
        # Get dimensions
        height, width = vehicle_img.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
        
        # Results dictionary
        light_states = {
            "headlights": False,
            "taillights": False,
            "brake_lights": False,
            "left_indicator": False,
            "right_indicator": False
        }
        
        # Define regions of interest based on orientation
        if orientation.lower() == "front":
            # Check for headlights (typically in lower corners)
            left_headlight_roi = hsv[int(height*0.6):height, 0:int(width*0.3)]
            right_headlight_roi = hsv[int(height*0.6):height, int(width*0.7):width]
            
            # White/yellow color for headlights
            white_mask = cv2.inRange(left_headlight_roi, np.array([0, 0, 200]), np.array([180, 30, 255]))
            yellow_mask = cv2.inRange(left_headlight_roi, np.array([15, 100, 200]), np.array([40, 255, 255]))
            left_headlight_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            white_mask = cv2.inRange(right_headlight_roi, np.array([0, 0, 200]), np.array([180, 30, 255]))
            yellow_mask = cv2.inRange(right_headlight_roi, np.array([15, 100, 200]), np.array([40, 255, 255]))
            right_headlight_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Calculate percentage of bright pixels
            left_headlight_score = np.sum(left_headlight_mask) / (left_headlight_roi.shape[0] * left_headlight_roi.shape[1])
            right_headlight_score = np.sum(right_headlight_mask) / (right_headlight_roi.shape[0] * right_headlight_roi.shape[1])
            
            # Threshold for headlight detection
            if left_headlight_score > 0.15 or right_headlight_score > 0.15:
                light_states["headlights"] = True
                
        elif orientation.lower() == "back":
            # Check for taillights (typically in lower corners)
            left_taillight_roi = hsv[int(height*0.6):height, 0:int(width*0.3)]
            right_taillight_roi = hsv[int(height*0.6):height, int(width*0.7):width]
            
            # Red color for taillights
            red_mask1 = cv2.inRange(left_taillight_roi, np.array([0, 70, 150]), np.array([25, 255, 255]))
            red_mask2 = cv2.inRange(left_taillight_roi, np.array([160, 70, 150]), np.array([180, 255, 255]))
            left_taillight_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            red_mask1 = cv2.inRange(right_taillight_roi, np.array([0, 70, 150]), np.array([25, 255, 255]))
            red_mask2 = cv2.inRange(right_taillight_roi, np.array([160, 70, 150]), np.array([180, 255, 255]))
            right_taillight_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Calculate percentage of red pixels
            left_taillight_score = np.sum(left_taillight_mask) / (left_taillight_roi.shape[0] * left_taillight_roi.shape[1])
            right_taillight_score = np.sum(right_taillight_mask) / (right_taillight_roi.shape[0] * right_taillight_roi.shape[1])
            
            # Threshold for taillight detection
            if left_taillight_score > 0.15 or right_taillight_score > 0.15:
                light_states["taillights"] = True
        
        # Check for indicators (typically yellow/amber) regardless of orientation
        # Left indicator
        left_indicator_roi = hsv[int(height*0.4):int(height*0.7), 0:int(width*0.2)]
        amber_mask = cv2.inRange(left_indicator_roi, np.array([15, 150, 150]), np.array([35, 255, 255]))
        left_indicator_score = np.sum(amber_mask) / (left_indicator_roi.shape[0] * left_indicator_roi.shape[1])
        
        # Right indicator
        right_indicator_roi = hsv[int(height*0.4):int(height*0.7), int(width*0.8):width]
        amber_mask = cv2.inRange(right_indicator_roi, np.array([15, 150, 150]), np.array([35, 255, 255]))
        right_indicator_score = np.sum(amber_mask) / (right_indicator_roi.shape[0] * right_indicator_roi.shape[1])
        
        # Thresholds for indicators
        if left_indicator_score > 0.1:
            light_states["left_indicator"] = True
        
        if right_indicator_score > 0.1:
            light_states["right_indicator"] = True
        
        return light_states

    def estimate_depth(self, frame):
        """Estimate depth map for the frame."""
        if self.depth_estimator is None:
            return None
            
        try:
            # Convert frame to RGB and create PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run depth estimation
            result = self.depth_estimator(pil_image)
            
            # Get the depth map
            depth_map = np.array(result["depth"])
            
            return depth_map
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None
    
    def convert_to_metric(self, depth_map, a=None, b=None, known_depths=None, known_pixels=None):
        """
        Convert MiDaS depth map to metric depth.

        Args:
            depth_map: MiDaS depth map output
            a, b: Scale and shift parameters (if known)
            known_depths: List of known metric depths for calibration points
            known_pixels: List of (x,y) coordinates for calibration points
            
        Returns:
            Metric depth map in millimeters
        """
        # If we don't have a and b, but have calibration points, calculate them
        if (a is None or b is None) and known_depths is not None and known_pixels is not None:
            # Need at least 2 known points
            if len(known_depths) >= 2 and len(known_depths) == len(known_pixels):
                # Extract MiDaS values at known pixels
                midas_values = [depth_map[y, x] for x, y in known_pixels]
                
                # Set up linear system to solve for a and b
                # For each point: 1/d = (1/a) * midas + (b/a)
                inv_depths = [1.0 / d for d in known_depths]
                A = np.vstack([midas_values, np.ones(len(midas_values))]).T
                x = np.linalg.lstsq(A, inv_depths, rcond=None)[0]
                
                # Extract a and b
                b_over_a = x[1]
                one_over_a = x[0]
                a = 1.0 / one_over_a
                b = b_over_a * a
                
                print(f"Calculated parameters: a={a}, b={b}")
            else:
                raise ValueError("Need at least 2 calibration points")

        # If we still don't have a and b, use default approximate values
        if a is None:
            a = 3000.0  # Default scale factor for approximate mm conversion
        if b is None:
            b = 0.1  # Default shift
            
        # Apply the conversion formula
        metric_depth = a * (1.0 / (depth_map + b))

        return metric_depth
    
    def detect_lanes(self, frame):
        """Detect lanes in the frame."""
        height, width = frame.shape[:2]
        
        # Define the bottom half region
        bottom_half_y = int(height * 0.4)  # Start at 40% from the top
        bottom_half = frame[bottom_half_y:, :]
        
        # Use SegFormer for lane segmentation if available
        lane_mask = None
        if self.segformer_model is not None and self.feature_extractor is not None:
            try:
                lane_mask = self.segment_lanes_with_segformer(bottom_half)
            except Exception as e:
                print(f"Error in SegFormer lane detection: {e}")
        
        # Fall back to traditional methods if SegFormer fails
        if lane_mask is None:
            # Convert to HSV and apply thresholding
            hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
            
            # For white lanes
            lower_white = np.array([0, 0, 200], dtype=np.uint8)
            upper_white = np.array([180, 30, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # For yellow lanes
            lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
            upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Combine masks
            lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        
        # Find lane lines using Hough transform
        edges = cv2.Canny(lane_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 == 0:  # Avoid division by zero
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter lines by slope
                if abs(slope) < 0.1:  # Ignore horizontal lines
                    continue
                
                # Separate left and right lanes by slope and position
                if slope < 0 and x1 < width // 2:  # Left lane
                    left_lines.append(line[0])
                elif slope > 0 and x1 > width // 2:  # Right lane
                    right_lines.append(line[0])
        
        # Determine lane types
        left_lane_type = "Solid" if len(left_lines) > 5 else "Dashed"
        right_lane_type = "Solid" if len(right_lines) > 5 else "Dashed"
        
        # Prepare lane data
        lane_data = {
            "left_lane": {
                "detected": len(left_lines) > 0,
                "type": left_lane_type,
                "points": left_lines
            },
            "right_lane": {
                "detected": len(right_lines) > 0,
                "type": right_lane_type,
                "points": right_lines
            },
            "bottom_half_y": bottom_half_y
        }
        
        return lane_data

    def segment_lanes_with_segformer(self, frame):
        """Use SegFormer to segment lane markings in the image."""
        try:
            # Preprocess the image
            inputs = self.feature_extractor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)
                logits = outputs.logits
            
            # Resize logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits, 
                size=frame.shape[:2], 
                mode="bilinear", 
                align_corners=False
            )
            
            # Get predicted segmentation map
            seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            # Extract lane classes (road marking class is typically class 6 or 7 in Cityscapes)
            lane_mask = np.zeros_like(seg_map, dtype=np.uint8)
            
            # Road marking classes (adjust based on your model's class mapping)
            road_marking_classes = [6, 7]  # Example classes for lane markings
            
            for cls in road_marking_classes:
                lane_mask = np.logical_or(lane_mask, seg_map == cls)
            
            # Convert to uint8 binary mask
            lane_mask = (lane_mask * 255).astype(np.uint8)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
            
            return lane_mask
            
        except Exception as e:
            print(f"Error in SegFormer lane segmentation: {e}")
            return None

    def create_3d_bounding_box(self, bbox, depth_value, class_name):
        """
        Create a 3D bounding box from a 2D bounding box and depth value.
        Uses camera intrinsics to project 2D points to 3D.
        """
        # Extract orientation from class name if available
        orientation = "unknown"
        if "front" in class_name.lower():
            orientation = "front"
        elif "back" in class_name.lower():
            orientation = "back"
        elif "side" in class_name.lower():
            orientation = "side"
        
        # if orientation == "unknown" and hasattr(self, 'current_frame'):
        #     orientation = self.detect_vehicle_facing_with_position(
        #         self.current_frame, 
        #         [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
        #         depth_value
        #     )
            
        if orientation == "unknown" and hasattr(self, 'current_frame'):
            orientation = self.detect_vehicle_facing(
                self.current_frame, 
                [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                depth_value
            )
    
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        center_x, center_y = bbox['center']['x'], bbox['center']['y']
        
        # Camera intrinsics
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # Project 2D center to 3D
        X_world = (center_x - cx) * depth_value / fx
        Y_world = (center_y - cy) * depth_value / fy
        Z_world = depth_value
        
        # Create 3D bounding box by projecting 2D corners
        corners_3d = []
        for corner in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            x_pixel, y_pixel = corner
            
            # Apply the projection formula
            X_corner = (x_pixel - cx) * depth_value / fx
            Y_corner = (y_pixel - cy) * depth_value / fy
            Z_corner = depth_value
            
            corners_3d.append([float(X_corner), float(Y_corner), float(Z_corner)])
        
        # Estimate object dimensions
        width_3d = abs(corners_3d[1][0] - corners_3d[0][0])
        height_3d = abs(corners_3d[2][1] - corners_3d[1][1])
        
        # Estimate depth based on orientation
        if orientation == "side":
            # Side view typically shows the length of the vehicle
            depth_3d = width_3d * 2.5  # Typical car length/width ratio
        else:  # front or back
            # Front/back view shows the width of the vehicle
            depth_3d = width_3d * 2.0  # Estimated length
        
        # Generate full 3D bounding box (8 corners)
        bbox_3d = []
        for front_corner in corners_3d:
            # Front face corner
            bbox_3d.append(front_corner)
            
            # Back face corner (add depth along Z axis)
            back_corner = [
                front_corner[0],
                front_corner[1],
                front_corner[2] - depth_3d
            ]
            
            bbox_3d.append(back_corner)
        
        # Calculate yaw angle based on orientation
        yaw_angle = 0.0  # Default (front)
        if orientation == "back":
            yaw_angle = 3.14159  # π radians (180 degrees)
        elif orientation == "side":
            yaw_angle = 1.57079  # π/2 radians (90 degrees)
        
        return {
            'corners': bbox_3d,
            'center': [float(X_world), float(Y_world), float(Z_world)],
            'dimensions': {
                'width': float(width_3d),
                'height': float(height_3d),
                'depth': float(depth_3d)
            },
            'facing': orientation,
            'yaw_angle': float(yaw_angle)
        }
     
    def compute_ego_motion_flow(self, flow, frame_shape, sample_step=20):
        """Estimate ego motion using homography-based flow"""
        h, w = frame_shape[:2]
        
        # Sample grid points with proper reshaping
        y, x = np.mgrid[0:h:sample_step, 0:w:sample_step].reshape(2, -1).astype(int)
        samples = flow[y, x]
        src_pts = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
        dst_pts = src_pts + samples[:, :2]

        # Filter out outliers using RANSAC
        if len(src_pts) > 4:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return np.zeros_like(flow)
            
            # Generate ego flow from homography
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            pts = np.vstack([xx.ravel(), yy.ravel()]).T.astype(np.float32)
            warped = cv2.perspectiveTransform(pts.reshape(-1,1,2), H).reshape(-1,2)
            ego_flow = warped - pts
            return ego_flow.reshape(h, w, 2)
        
        return np.zeros_like(flow)

    def process_frame(self, frame, frame_number):
        """Process a single frame and return comprehensive scene data with 3D information and motion detection."""
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Store current frame for vehicle orientation detection
        self.current_frame = frame
        
        # 1. Detect objects with enhanced vehicle classification
        detections = self.detect_objects(frame)
        
        # 2. Estimate depth
        depth_map = self.estimate_depth(frame)
        
        # 3. Detect vehicle motion if RAFT is available
        if self.raft_model is not None:
            vehicle_detections = [d for d in detections if 'car' in d['original_class'] or 
                                'truck' in d['original_class'] or 
                                'bus' in d['original_class']]
            
            if self.prev_frame is not None:
                # Convert frames to RGB for RAFT
                current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prev_frame_rgb = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2RGB)
                
                # Compute optical flow
                with torch.no_grad():
                    frame_tensor = torch.from_numpy(current_frame_rgb).permute(2,0,1).float().to(self.device)
                    prev_frame_tensor = torch.from_numpy(prev_frame_rgb).permute(2,0,1).float().to(self.device)
                    
                    padder = InputPadder(prev_frame_tensor.shape)
                    image1, image2 = padder.pad(prev_frame_tensor[None], frame_tensor[None])
                    _, flow = self.raft_model(image1, image2, iters=20, test_mode=True)
                    flow = padder.unpad(flow[0]).cpu().numpy().transpose(1, 2, 0)
                
                # Estimate ego-motion flow
                ego_flow = self.compute_ego_motion_flow(flow, current_frame_rgb.shape)
                residual_flow = flow - ego_flow
                
                # Update ego speed (average flow magnitude)
                self.ego_speed = np.mean(np.linalg.norm(ego_flow, axis=2))
                
                # Calculate flow magnitudes for each vehicle
                vehicle_boxes = []
                flow_mags = []
                
                for detection in vehicle_detections:
                    bbox = detection['bounding_box']
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    vehicle_boxes.append([x1, y1, x2, y2])
                    
                    # Calculate flow magnitude for this vehicle
                    roi_flow = residual_flow[y1:y2, x1:x2]
                    if roi_flow.size > 0:
                        mag = np.median(np.linalg.norm(roi_flow, axis=2))
                    else:
                        mag = 0
                    flow_mags.append(mag)
                
                # Update tracker
                if vehicle_boxes and flow_mags:
                    tracks = self.vehicle_tracker.update(vehicle_boxes, flow_mags, self.ego_speed)
                    
                    # Update detections with motion status
                    for detection in vehicle_detections:
                        bbox = detection['bounding_box']
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        
                        # Find matching track
                        for tid in tracks:
                            track_box = tracks[tid]['box']
                            if (abs(x1 - track_box[0]) < 20 and 
                                abs(y1 - track_box[1]) < 20 and 
                                abs(x2 - track_box[2]) < 20 and 
                                abs(y2 - track_box[3]) < 20):
                                
                                # Add motion status to detection
                                detection['motion_status'] = tracks[tid]['status']
                                break
            
            # Store current frame for next iteration
            self.prev_frame = frame.copy()
        
        # 4. Add depth and 3D information to detections if available
        if depth_map is not None:
            # Convert to metric depth
            metric_depth_map = self.convert_to_metric(depth_map)
            
            for detection in detections:
                bbox = detection['bounding_box']
                center_x, center_y = int(bbox['center']['x']), int(bbox['center']['y'])
                
                # Get depth at the center of the bounding box
                if 0 <= center_y < height and 0 <= center_x < width:
                    # Get raw MiDaS depth value
                    raw_depth = float(depth_map[center_y, center_x])
                    # Get metric depth value
                    metric_depth = float(metric_depth_map[center_y, center_x])
                    
                    # Store both
                    detection['relative_depth'] = raw_depth
                    detection['metric_depth'] = metric_depth # In millimeters
                    
                    # Check for collision
                    detection['collision'] = self.collision_detector.check_collision(detection)
                    
                    # Create 3D bounding box using metric depth
                    bbox_3d = self.create_3d_bounding_box(
                        bbox,
                        metric_depth, # Use metric depth
                        detection['class']
                    )
                    detection['bbox_3d'] = bbox_3d
        
        # 5. Detect lanes
        lane_data = self.detect_lanes(frame)
        
        # 6. Combine all data
        scene_data = {
            "frame_number": frame_number,
            "frame_dimensions": {
                "width": width,
                "height": height
            },
            "camera_intrinsics": self.camera_intrinsics,
            "objects": detections,
            "lanes": lane_data,
            "has_depth_data": depth_map is not None,
            "ego_speed": self.ego_speed if hasattr(self, 'ego_speed') else 0.0
        }
        
        return scene_data
    
    
    


    def process_video(self, video_path, output_dir, start_frame=0, end_frame=None, frame_skip=0):
        """Process a video file and save JSON data for each frame."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust end_frame if not specified
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        print(f"Video properties:")
        print(f" - Resolution: {width}x{height}")
        print(f" - Frame Rate: {fps:.2f} fps")
        print(f" - Total Frames: {total_frames}")
        print(f" - Processing frames: {start_frame} to {end_frame}")
        print(f" - Frame Skip: {frame_skip} (processing 1 out of every {frame_skip+1} frames)")

        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = start_frame
        processed_count = 0

        # Process each frame
        while cap.isOpened() and frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if frame_skip > 0 and (frame_count - start_frame) % (frame_skip + 1) != 0:
                frame_count += 1
                continue
            
            # Process the frame
            scene_data = self.process_frame(frame, frame_count)
            
            # Save JSON data
            json_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.json")
            with open(json_filename, 'w') as f:
                json.dump(scene_data, f, indent=2, cls=NumpyEncoder)  # Use custom encoder for numpy types
            
            processed_count += 1
            frame_count += 1
            
            # Print progress every frame
            if processed_count % 1 == 0:
                print(f"Processed {processed_count} frames | Current frame: {frame_count}/{end_frame}")

        # Release resources
        cap.release()

        print(f"Processing complete. Saved data for {processed_count} frames to {output_dir}")

    def draw_bounding_boxes_and_lanes(self, frame, detections, lanes):
        """
        Draw bounding boxes and lanes on the frame with collision warnings and motion status.
        """
        # Draw bounding boxes
        for obj in detections:
            bbox = obj['bounding_box']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Check if collision status is available
            is_collision = obj.get('collision', False)
            
            # Determine color based on class and collision status
            if is_collision:
                color = (0, 0, 255)  # Red for collision
            elif 'motion_status' in obj and obj['motion_status'] == 'parked':
                color = (128, 128, 128)  # Gray for parked vehicles
            elif "sedan" in obj['class'].lower():
                color = (0, 255, 0)  # Green for sedans
            elif "suv" in obj['class'].lower():
                color = (0, 200, 0)  # Dark green for SUVs
            elif "hatchback" in obj['class'].lower():
                color = (0, 255, 100)  # Light green for hatchbacks
            elif "pickup" in obj['class'].lower():
                color = (0, 150, 0)  # Olive for pickup trucks
            elif "truck" in obj['class'].lower():
                color = (128, 128, 128)  # Gray for trucks
            elif "pedestrian" in obj['class'].lower():
                color = (255, 0, 0)  # Blue for pedestrians
            elif "traffic light" in obj['class'].lower():
                if "red" in obj['class'].lower():
                    color = (0, 0, 255)  # Red
                elif "yellow" in obj['class'].lower():
                    color = (0, 255, 255)  # Yellow
                elif "green" in obj['class'].lower():
                    color = (0, 255, 0)  # Green
                else:
                    color = (255, 255, 255)  # White
            else:
                color = (255, 255, 0)  # Yellow for other objects
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add class label and collision warning
            label = obj['class']
            if is_collision:
                label = "COLLISION WARNING: " + label
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # If depth is available, show it
            if 'metric_depth' in obj:
                depth_text = f"D: {obj['metric_depth']:.2f}mm"
                if is_collision:
                    depth_text += " - TOO CLOSE!"
                cv2.putText(frame, depth_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show motion status if available
            if 'motion_status' in obj:
                status_color = (0, 255, 0) if obj['motion_status'] == 'moving' else (128, 128, 128)
                cv2.putText(frame, obj['motion_status'].upper(), (x1, y2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Draw motion arrow for moving vehicles
                if obj['motion_status'] == 'moving':
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    arrow_len = 30
                    cv2.arrowedLine(frame, (cx, cy), (cx, cy - arrow_len), (0, 0, 255), 3)
            
            # Show facing direction if available
            if 'facing' in obj:
                facing_text = f"Facing: {obj['facing'].upper()}"
                cv2.putText(frame, facing_text, (x1, y2 + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw vehicle lights if available
            if 'lights' in obj:
                lights = obj['lights']
                light_y = y2 + 80  # Start lower to account for motion status and facing text
                
                if lights.get("headlights", False):
                    cv2.putText(frame, "Headlights", (x1, light_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    light_y += 20
                    
                if lights.get("brake_lights", False):  # Display brake lights differently
                    cv2.putText(frame, "BRAKE LIGHTS", (x1, light_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    light_y += 20
                elif lights.get("taillights", False):  # Only show taillights if brake lights aren't on
                    cv2.putText(frame, "Taillights", (x1, light_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    light_y += 20
                    
                if lights.get("left_indicator", False):
                    cv2.putText(frame, "Left Indicator", (x1, light_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    light_y += 20
                    
                if lights.get("right_indicator", False):
                    cv2.putText(frame, "Right Indicator", (x1, light_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw 3D bounding box projection if available
            if 'bbox_3d' in obj and 'corners' in obj['bbox_3d']:
                corners_3d = obj['bbox_3d']['corners']
                
                # Define connections between corners for drawing 3D box
                # Front face
                cv2.line(frame, (int(corners_3d[0][0]), int(corners_3d[0][1])), 
                        (int(corners_3d[1][0]), int(corners_3d[1][1])), (0, 255, 0), 1)
                cv2.line(frame, (int(corners_3d[1][0]), int(corners_3d[1][1])), 
                        (int(corners_3d[2][0]), int(corners_3d[2][1])), (0, 255, 0), 1)
                cv2.line(frame, (int(corners_3d[2][0]), int(corners_3d[2][1])), 
                        (int(corners_3d[3][0]), int(corners_3d[3][1])), (0, 255, 0), 1)
                cv2.line(frame, (int(corners_3d[3][0]), int(corners_3d[3][1])), 
                        (int(corners_3d[0][0]), int(corners_3d[0][1])), (0, 255, 0), 1)
                
                # Back face
                cv2.line(frame, (int(corners_3d[4][0]), int(corners_3d[4][1])), 
                        (int(corners_3d[5][0]), int(corners_3d[5][1])), (0, 0, 255), 1)
                cv2.line(frame, (int(corners_3d[5][0]), int(corners_3d[5][1])), 
                        (int(corners_3d[6][0]), int(corners_3d[6][1])), (0, 0, 255), 1)
                cv2.line(frame, (int(corners_3d[6][0]), int(corners_3d[6][1])), 
                        (int(corners_3d[7][0]), int(corners_3d[7][1])), (0, 0, 255), 1)
                cv2.line(frame, (int(corners_3d[7][0]), int(corners_3d[7][1])), 
                        (int(corners_3d[4][0]), int(corners_3d[4][1])), (0, 0, 255), 1)
                
                # Connections between front and back faces
                cv2.line(frame, (int(corners_3d[0][0]), int(corners_3d[0][1])), 
                        (int(corners_3d[4][0]), int(corners_3d[4][1])), (255, 0, 0), 1)
                cv2.line(frame, (int(corners_3d[1][0]), int(corners_3d[1][1])), 
                        (int(corners_3d[5][0]), int(corners_3d[5][1])), (255, 0, 0), 1)
                cv2.line(frame, (int(corners_3d[2][0]), int(corners_3d[2][1])), 
                        (int(corners_3d[6][0]), int(corners_3d[6][1])), (255, 0, 0), 1)
                cv2.line(frame, (int(corners_3d[3][0]), int(corners_3d[3][1])), 
                        (int(corners_3d[7][0]), int(corners_3d[7][1])), (255, 0, 0), 1)
        
        # Draw lanes
        if lanes['left_lane']['detected']:
            for point in lanes['left_lane']['points']:
                x1, y1, x2, y2 = point
                cv2.line(frame, 
                        (int(x1), int(y1) + lanes['bottom_half_y']), 
                        (int(x2), int(y2) + lanes['bottom_half_y']), 
                        (255, 0, 0), 2)
            
            # Add lane type annotation
            if len(lanes['left_lane']['points']) > 0:
                point = lanes['left_lane']['points'][0]
                x, y = int(point[0]), int(point[1]) + lanes['bottom_half_y']
                cv2.putText(frame, f"Left Lane: {lanes['left_lane']['type']}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if lanes['right_lane']['detected']:
            for point in lanes['right_lane']['points']:
                x1, y1, x2, y2 = point
                cv2.line(frame, 
                        (int(x1), int(y1) + lanes['bottom_half_y']), 
                        (int(x2), int(y2) + lanes['bottom_half_y']), 
                        (0, 0, 255), 2)
            
            # Add lane type annotation
            if len(lanes['right_lane']['points']) > 0:
                point = lanes['right_lane']['points'][0]
                x, y = int(point[0]), int(point[1]) + lanes['bottom_half_y']
                cv2.putText(frame, f"Right Lane: {lanes['right_lane']['type']}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add ego speed indicator
        if hasattr(self, 'ego_speed'):
            cv2.putText(frame, f"Ego Speed: {self.ego_speed:.1f} px/frame", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw frame information
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Frame Size: {width}x{height}", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection count
        cv2.putText(frame, f"Objects: {len(detections)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Count vehicles by motion status
        moving_count = sum(1 for obj in detections if obj.get('motion_status') == 'moving')
        parked_count = sum(1 for obj in detections if obj.get('motion_status') == 'parked')
        
        # Display vehicle counts
        if moving_count > 0 or parked_count > 0:
            cv2.putText(frame, f"Moving: {moving_count}, Parked: {parked_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def save_visualization(self, json_dir, video_path, output_path):
        """
        Create a visualization video with bounding boxes and lane detections.
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            json_path = os.path.join(json_dir, f"frame_{frame_number:06d}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                # Call the method to draw bounding boxes and lanes
                frame = self.draw_bounding_boxes_and_lanes(frame, data['objects'], data['lanes'])

            # Add frame number to the top-right corner
            text = f"Frame: {frame_number}"
            cv2.putText(frame, text, (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            frame_number += 1

        cap.release()
        out.release()
        print(f"Visualization saved to {output_path}")

    
class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class VehicleOrientationYOLO:
    def __init__(self, model_path=r"C:\Users\pavan\Documents\CV_P3\best_orientation.pt", conf_threshold=0.5):
        """
        Initialize the vehicle orientation detector using a specialized YOLO model
        """
        try:
            # Check if model file exists
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded vehicle orientation YOLO model from {model_path}")
            else:
                print(f"Warning: Model file not found at {model_path}, using pretrained model")
                # If no custom model is provided, use a pretrained model
                self.model = YOLO("yolov8m.pt")
                print("Using pretrained YOLOv8m model for orientation detection")
            
            self.conf_threshold = conf_threshold
            self.class_names = ['vehicle_front', 'vehicle_rear', 'vehicle_side']
            self.colors = {
                'vehicle_front': (0, 255, 0),    # Green
                'vehicle_rear': (0, 0, 255),     # Red
                'vehicle_side': (255, 0, 0)      # Blue
            }
            
            # Check if CUDA is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Vehicle orientation detector using device: {self.device}")
        except Exception as e:
            print(f"Error initializing YOLO orientation model: {e}")
            self.model = None
    
    def detect_orientation(self, frame, bbox):
        """
        Detect vehicle orientation using the specialized YOLO model
        
        Args:
            frame: The full image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            orientation: One of "front", "back", "side"
            confidence: Confidence score for the orientation
            yaw_angle: Estimated yaw angle in radians (approximate)
            details: Additional details about the detection
        """
        if self.model is None:
            return "unknown", 0.5, 0.0, {}
            
        try:
            # Extract vehicle crop
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size == 0:
                return "unknown", 0.5, 0.0, {}
            
            # Run inference on the crop
            results = self.model(vehicle_crop, conf=self.conf_threshold)
            
            # Default values
            orientation = "unknown"
            confidence = 0.5
            yaw_angle = 0.0
            details = {}
            
            # Process detections
            if results and len(results) > 0:
                result = results[0]  # Get first result
                boxes = result.boxes
                
                if len(boxes) > 0:
                    # Get the detection with highest confidence
                    best_idx = 0
                    best_conf = 0
                    
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            best_idx = i
                    
                    # Get class of best detection
                    class_id = int(boxes[best_idx].cls[0])
                    confidence = float(boxes[best_idx].conf[0])
                    
                    # Map class_id to orientation
                    if class_id < len(self.class_names):
                        yolo_orientation = self.class_names[class_id]
                        
                        # Convert to your orientation format
                        if yolo_orientation == "vehicle_front":
                            orientation = "front"
                            yaw_angle = 0.0
                        elif yolo_orientation == "vehicle_rear":
                            orientation = "back"
                            yaw_angle = 3.14159  # π radians (180 degrees)
                        elif yolo_orientation == "vehicle_side":
                            orientation = "side"
                            yaw_angle = 1.57079  # π/2 radians (90 degrees)
                        
                        # Add detailed information
                        details = {
                            'original_class': yolo_orientation,
                            'class_id': class_id,
                            'bbox_in_crop': boxes[best_idx].xyxy[0].tolist() if hasattr(boxes[best_idx], 'xyxy') else None
                        }
            
            return orientation, confidence, yaw_angle, details
        except Exception as e:
            print(f"Error in YOLO orientation detection: {e}")
            return "unknown", 0.5, 0.0, {'error': str(e)}
    
    def create_3d_bbox(self, frame, bbox, depth_value, camera_matrix):
        """
        Create a 3D bounding box using YOLO orientation detection
        
        Args:
            frame: The full image
            bbox: 2D bounding box [x1, y1, x2, y2]
            depth_value: Estimated depth at the center of the object
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            bbox_3d: Dictionary containing 3D bounding box parameters
        """
        try:
            # Get orientation and 3D parameters
            orientation, confidence, yaw_angle, _ = self.detect_orientation(frame, bbox)
            
            # Extract 2D box parameters
            x1, y1, x2, y2 = map(int, bbox)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Camera intrinsics
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # Project 2D center to 3D
            X_world = (center_x - cx) * depth_value / fx
            Y_world = (center_y - cy) * depth_value / fy
            Z_world = depth_value
            
            # Estimate dimensions based on vehicle type and orientation
            if orientation == "side":
                # Side view typically shows the length of the vehicle
                length = width * 2.5  # Typical car length/width ratio
                width_3d = width * 0.8  # Estimated width
                height_3d = height * 1.0  # Height from image
            else:  # front or back
                # Front/back view shows the width of the vehicle
                length = width * 2.0  # Estimated length
                width_3d = width * 1.0  # Width from image
                height_3d = height * 1.0  # Height from image
            
            # Create 3D bounding box with orientation
            corners_3d = []
            
            # Create rotation matrix from yaw angle
            cos_yaw = np.cos(yaw_angle)
            sin_yaw = np.sin(yaw_angle)
            R = np.array([
                [cos_yaw, 0, sin_yaw],
                [0, 1, 0],
                [-sin_yaw, 0, cos_yaw]
            ])
            
            # Define 3D box corners in vehicle coordinates (centered at origin)
            l, w, h = length/2, width_3d/2, height_3d/2
            x_corners = np.array([l, l, -l, -l, l, l, -l, -l])
            y_corners = np.array([h, -h, -h, h, h, -h, -h, h])
            z_corners = np.array([w, w, w, w, -w, -w, -w, -w])
            corners = np.vstack((x_corners, y_corners, z_corners))
            
            # Apply rotation
            corners = R @ corners
            
            # Translate to world coordinates
            corners[0, :] += X_world
            corners[1, :] += Y_world
            corners[2, :] += Z_world
            
            # Convert to list of points
            for i in range(8):
                corners_3d.append([float(corners[0, i]), float(corners[1, i]), float(corners[2, i])])
            
            return {
                'corners': corners_3d,
                'center': [float(X_world), float(Y_world), float(Z_world)],
                'dimensions': {
                    'length': float(length),
                    'width': float(width_3d),
                    'height': float(height_3d)
                },
                'orientation': orientation,
                'yaw_angle': float(yaw_angle),
                'confidence': float(confidence)
            }
        except Exception as e:
            print(f"Error creating 3D bbox: {e}")
            return self._create_basic_3d_bbox(bbox, depth_value, camera_matrix)
    
    def _create_basic_3d_bbox(self, bbox, depth_value, camera_matrix):
        """Fallback method to create basic 3D bbox without orientation"""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Camera intrinsics
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Project to 3D
        X_world = (center_x - cx) * depth_value / fx
        Y_world = (center_y - cy) * depth_value / fy
        Z_world = depth_value
        
        # Estimate dimensions
        width = x2 - x1
        height = y2 - y1
        length = width * 2.0
        width_3d = width * 1.0
        height_3d = height * 1.0
        
        # Create basic 3D box
        corners_3d = []
        
        # Front face (facing camera)
        corners_3d.append([X_world - width_3d/2, Y_world - height_3d/2, Z_world])
        corners_3d.append([X_world + width_3d/2, Y_world - height_3d/2, Z_world])
        corners_3d.append([X_world + width_3d/2, Y_world + height_3d/2, Z_world])
        corners_3d.append([X_world - width_3d/2, Y_world + height_3d/2, Z_world])
        
        # Back face
        corners_3d.append([X_world - width_3d/2, Y_world - height_3d/2, Z_world - length])
        corners_3d.append([X_world + width_3d/2, Y_world - height_3d/2, Z_world - length])
        corners_3d.append([X_world + width_3d/2, Y_world + height_3d/2, Z_world - length])
        corners_3d.append([X_world - width_3d/2, Y_world + height_3d/2, Z_world - length])
        
        return {
            'corners': corners_3d,
            'center': [float(X_world), float(Y_world), float(Z_world)],
            'dimensions': {
                'length': float(length),
                'width': float(width_3d),
                'height': float(height_3d)
            },
            'orientation': 'unknown',
            'yaw_angle': 0.0,
            'confidence': 0.7
        }


class CollisionDetector:
    def __init__(self, camera_intrinsics, collision_threshold_mm=16):
        """
        Initialize collision detector with configurable threshold
        
        Args:
            camera_intrinsics: Camera parameters for 3D calculations
            collision_threshold_mm: Distance threshold in mm for collision warnings
        """
        self.camera_intrinsics = camera_intrinsics
        self.collision_threshold_mm = collision_threshold_mm
        
    def check_collision(self, detection):
        """
        Check if an object is within the collision threshold
        
        Args:
            detection: Detection object with metric_depth value
            
        Returns:
            bool: True if object is within collision threshold
        """
        if 'metric_depth' not in detection:
            return False
            
        # Check if object is within collision threshold
        return detection['metric_depth'] <= self.collision_threshold_mm


class VehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_history = 10  # Increased history length
        self.min_detection_frames = 3  # Minimum frames before making a decision
        self.motion_confidence = {}  # Track confidence in motion classification
        
    def update(self, boxes, flow_magnitudes, ego_speed):
        new_tracks = {}
        
        # Adjust base threshold based on scene complexity
        base_threshold = 0.5 if ego_speed < 5.0 else 1.0
        
        for box, mag in zip(boxes, flow_magnitudes):
            cx = (box[0] + box[2])/2
            cy = (box[1] + box[3])/2
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            
            # Find existing track
            matched = False
            for tid in self.tracks:
                hist = self.tracks[tid]['history']
                last_pos = hist[-1][:2]
                
                # More flexible matching for faster moving objects
                distance = ((cx - last_pos[0])**2 + (cy - last_pos[1])**2)**0.5
                max_distance = min(100, max(50, ego_speed * 5))  # Adaptive distance threshold
                
                if distance < max_distance:
                    hist.append((cx, cy, mag, box_area))
                    if len(hist) > self.max_history:
                        hist.pop(0)
                    
                    # Only classify if we have enough history
                    if len(hist) >= self.min_detection_frames:
                        # Calculate weighted average of recent magnitudes
                        # Give more weight to higher magnitudes to detect brief movements
                        recent_mags = [h[2] for h in list(hist)[-5:]]
                        avg_mag = np.mean(recent_mags)
                        max_mag = np.max(recent_mags)
                        weighted_mag = (avg_mag * 0.7) + (max_mag * 0.3)
                        
                        # Adaptive threshold based on ego speed and object size
                        size_factor = 1.0 - min(0.5, box_area / 100000)  # Smaller objects need lower thresholds
                        threshold = max(base_threshold, ego_speed * 0.15 * size_factor)
                        
                        # Determine status with hysteresis to prevent flickering
                        if tid in self.motion_confidence:
                            # If previously moving, require lower magnitude to stay moving
                            if self.motion_confidence[tid] > 0:
                                is_moving = weighted_mag > (threshold * 0.8)
                            # If previously parked, require higher magnitude to switch to moving
                            else:
                                is_moving = weighted_mag > (threshold * 1.2)
                            
                            # Update confidence (bounded between -3 and 3)
                            if is_moving:
                                self.motion_confidence[tid] = min(3, self.motion_confidence[tid] + 1)
                            else:
                                self.motion_confidence[tid] = max(-3, self.motion_confidence[tid] - 1)
                        else:
                            # Initial classification
                            is_moving = weighted_mag > threshold
                            self.motion_confidence[tid] = 1 if is_moving else -1
                        
                        # Final status based on confidence
                        status = 'moving' if self.motion_confidence[tid] > 0 else 'parked'
                    else:
                        # Not enough history, maintain previous status or default to moving
                        status = self.tracks[tid].get('status', 'moving')
                    
                    new_tracks[tid] = {
                        'box': box,
                        'history': hist,
                        'status': status,
                        'magnitude': weighted_mag if len(hist) >= self.min_detection_frames else 0,
                        'threshold': threshold if len(hist) >= self.min_detection_frames else 0
                    }
                    matched = True
                    break
                    
            if not matched:
                # For new tracks, default to moving until we have enough history
                new_tracks[self.next_id] = {
                    'box': box,
                    'history': deque([(cx, cy, mag, box_area)], maxlen=self.max_history),
                    'status': 'moving'  # Default new tracks to moving
                }
                self.motion_confidence[self.next_id] = 1  # Start with positive confidence
                self.next_id += 1
        
        # Clean up motion confidence for tracks that no longer exist
        self.motion_confidence = {tid: conf for tid, conf in self.motion_confidence.items() if tid in new_tracks}
        
        self.tracks = new_tracks
        return self.tracks


# Main execution
if __name__ == "__main__":
    # Initialize the scene analyzer
    analyzer = ComprehensiveSceneAnalyzer()
    
    # Process video
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene12\Undist\2023-03-13_06-00-16-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene6\Undist\2023-03-03_15-31-56-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene1\Undist\2023-02-14_11-04-07-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene2\Undist\2023-03-03_10-31-11-front_undistort.mp4"
    video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene11\Undist\2023-03-11_17-19-53-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Desktop\s11.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene3\Undist\2023-02-14_11-49-54-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Desktop\scene3.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene7\Undist\2023-03-03_11-21-43-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene8\Undist\2023-03-03_11-40-47-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene9\Undist\2023-03-04_17-20-36-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene10\Undist\2023-03-06_19-48-30-front_undistort.mp4"
    
    json_output_dir = r"C:\Users\pavan\Documents\CV_P3\output\3D_OD\PM_json_11"
    
    # Create output directory if it doesn't exist
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Create window for collision threshold adjustment
    cv2.namedWindow('Collision Detection Controls')
    
    # Create trackbar for collision threshold
    cv2.createTrackbar('Collision Threshold (mm)', 'Collision Detection Controls', 20, 30,
                      lambda x: setattr(analyzer.collision_detector, 'collision_threshold_mm', x))
    
    try:
        # Process video in interactive mode
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                scene_data = analyzer.process_frame(frame, frame_number)
                
                # Draw visualization
                vis_frame = analyzer.draw_bounding_boxes_and_lanes(frame.copy(), scene_data['objects'], scene_data['lanes'])
                
                # Show frame
                cv2.imshow('Collision Detection', vis_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
        
        cv2.destroyAllWindows()
        
        
        # Process entire video in batch mode
        analyzer.process_video(video_path, json_output_dir, start_frame=0, end_frame=None, frame_skip=0)
        
        # Save visualization video
        visualization_output_path = r"C:\Users\pavan\Documents\CV_P3\output\3D_OD\PM_viz_11.mp4"
        analyzer.save_visualization(json_output_dir, video_path, visualization_output_path)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


