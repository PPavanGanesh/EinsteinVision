import cv2
import torch
import numpy as np
import argparse
import os
import time
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO


def check_and_install_dependencies():
    """Check if required packages are installed and install them if not."""
    print("Checking dependencies...")
    
    # Fix numpy version first to avoid compatibility issues
    try:
        import numpy
        numpy_version = numpy.__version__
        if numpy_version.startswith('2.'):
            print(f"Downgrading numpy from {numpy_version} to 1.26.4 to avoid compatibility issues...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"])
            print("Numpy downgraded. Please restart the script.")
            sys.exit(0)
    except ImportError:
        print("Installing numpy 1.26.4...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
        print("Numpy installed. Please restart the script.")
        sys.exit(0)
    
    required_packages = ['opencv-python', 'torch', 'torchvision', 'ultralytics', 'transformers', 'pillow']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('pillow', 'PIL'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Object Detection with Depth Estimation')
    
    # Model selection parameters
    parser.add_argument('--detection-type', type=str, choices=['bin', 'cone', 'cylinder', 'all'], default='all',
                        help='Type of detection to run: bin, cone, cylinder, or all')
    
    # Model paths
    parser.add_argument('--bin-model', type=str, 
                      default=r"C:\Users\pavan\Documents\CV_environment\.venv\obj_detection\Models\bin_detection_v1\weights\last.pt",
                      help='Path to trash bin detection model')
    parser.add_argument('--cone-model', type=str, 
                      default=r"C:\Users\pavan\Documents\CV_environment\.venv\obj_detection\Models\safty_cone_v1\best1.pt",
                      help='Path to traffic cone detection model')
    parser.add_argument('--cylinder-model', type=str, 
                      default=r'C:\Users\pavan\Documents\CV_environment\.venv\obj_detection\Models\safety_cylinders_v1\weights\best.pt',
                      help='Path to safety cylinder detection model')
    parser.add_argument('--pole-model', type=str, 
                    default=r'C:\Users\pavan\Documents\CV_environment\.venv\obj_detection\Models\safety_pole_detection3\weights\best.pt',
                    help='Path to safety pole detection model')
    

    
    # Input and output parameters
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default=r'C:\Users\pavan\Documents\CV_P3\output\Final\3D_Traffic_Obj\Scene_8',
                      help='Directory to save output videos')
    parser.add_argument('--output-name', type=str, default=None,
                      help='Custom name for output video file (without extension)')
    
    # Detection parameters with variable thresholds
    parser.add_argument('--bin-conf', type=float, default=0.9,
                      help='Confidence threshold for bin detections')
    parser.add_argument('--cone-conf', type=float, default=0.9,
                      help='Confidence threshold for cone detections')
    parser.add_argument('--cylinder-conf', type=float, default=0.85,
                      help='Confidence threshold for cylinder detections')
    parser.add_argument('--pole-conf', type=float, default=0.85,
                        help='Confidence threshold for pole detections')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                      help='Filter by class: --classes 0, or --classes 0 2 3')
    
    # Hardware options
    parser.add_argument('--device', type=str, default='',
                      help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # Display and debug options
    parser.add_argument('--view-img', action='store_true', default=False,
                      help='Show detection results in window')
    parser.add_argument('--save-txt', action='store_true',
                      help='Save results to *.txt file')
    parser.add_argument('--verbose', action='store_true',
                      help='Print verbose output')
    
    # Frame processing options
    parser.add_argument('--frame-skip', type=int, default=0,
                      help='Number of frames to skip between processing (0 = process all frames)')
    
    return parser.parse_args()

def load_models(opt):
    """
    Load the requested detection models based on the detection type.
    Args:
        opt: Command line arguments
    Returns:
        tuple: (detection_models, thresholds, depth_model)
    """
    detection_models = {}
    thresholds = {}
    
    # Set device
    device = select_device(opt.device)
    
    # Load detection models
    if opt.detection_type in ['bin', 'all']:
        try:
            from ultralytics import YOLO
            print(f"Loading bin detection model: {opt.bin_model}")
            detection_models['bin'] = YOLO(opt.bin_model)
            thresholds['bin'] = opt.bin_conf
            print(f"✓ Bin detection model loaded successfully (threshold: {opt.bin_conf})")
        except Exception as e:
            print(f"Error loading bin detection model: {e}")
    
    if opt.detection_type in ['cone', 'all']:
        try:
            print(f"Loading cone detection model: {opt.cone_model}")
            # For cone detection, we'll use PyTorch hub
            if os.path.exists(opt.cone_model):
                detection_models['cone'] = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                         path=opt.cone_model, trust_repo=True)
            else:
                detection_models['cone'] = torch.hub.load('ultralytics/yolov5', 'yolov5l', trust_repo=True)
            detection_models['cone'].to(device)
            detection_models['cone'].conf = opt.cone_conf
            thresholds['cone'] = opt.cone_conf
            print(f"✓ Cone detection model loaded successfully (threshold: {opt.cone_conf})")
        except Exception as e:
            print(f"Error loading cone detection model: {e}")
    
    if opt.detection_type in ['cylinder', 'all']:
        try:
            from ultralytics import YOLO
            print(f"Loading cylinder detection model: {opt.cylinder_model}")
            detection_models['cylinder'] = YOLO(opt.cylinder_model)
            thresholds['cylinder'] = opt.cylinder_conf
            print(f"✓ Cylinder detection model loaded successfully (threshold: {opt.cylinder_conf})")
        except Exception as e:
            print(f"Error loading cylinder detection model: {e}")
            
    if opt.detection_type in ['pole', 'all']:
        try:
            print(f"Loading safety pole detection model: {opt.pole_model}")
            detection_models['pole'] = YOLO(opt.pole_model)
            thresholds['pole'] = opt.pole_conf
            print(f"✓ Safety pole detection model loaded successfully (threshold: {opt.pole_conf})")
        except Exception as e:
            print(f"Error loading safety pole detection model: {e}")

    
    # Load depth estimation model
    print("\n" + "="*50)
    print("LOADING DEPTH ESTIMATION MODEL:")
    print("="*50)
    print("Loading MiDaS depth estimation model...")
    start_load = time.time()
    
    # Explicitly set device for depth model
    device_id = 0 if torch.cuda.is_available() else -1
    depth_model = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=device_id)
    load_time = time.time() - start_load
    print(f"✓ Depth estimation model loaded successfully in {load_time:.2f} seconds")
    
    return detection_models, thresholds, depth_model

def select_device(device=''):
    """
    Select the appropriate device (CPU/GPU).
    Args:
        device: Device string ('0', '0,1,2,3', 'cpu', etc.)
    Returns:
        torch.device: Selected device
    """
    # Check GPU availability
    print("\n" + "="*50)
    print("CHECKING GPU STATUS:")
    print("="*50)
    
    if torch.cuda.is_available():
        if device == '':
            device = 'cuda:0'
        elif device == 'cpu':
            device = 'cpu'
        else:
            try:
                torch.cuda.get_device_properties(int(device.split(',')[0]))
                device = f'cuda:{device}'
            except:
                print(f"Warning: Invalid CUDA device '{device}'. Using default CUDA device.")
                device = 'cuda:0'
        
        device_obj = torch.device(device)
        print(f"✅ GPU is available and will be used")
        print(f" - GPU Count: {torch.cuda.device_count()}")
        print(f" - Current Device: {torch.cuda.current_device()}")
        print(f" - GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Get GPU memory information if possible
        try:
            print(f" - GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f" - GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        except:
            print(" - GPU memory information not available")
    else:
        device_obj = torch.device('cpu')
        print("⚠️ GPU is not available. Using CPU instead (processing will be slow)")
    
    return device_obj

def create_output_paths(opt):
    """
    Create output file paths.
    Args:
        opt: Command line arguments
    Returns:
        tuple: (video_path, json_path)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)
        print(f"Created output directory: {opt.output_dir}")
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Get input filename without extension
    input_base = os.path.splitext(os.path.basename(opt.input))[0]
    
    # Use custom name if provided, otherwise use input filename
    base_name = opt.output_name if opt.output_name else input_base
    
    # Create output paths#############################################################################################################################
    video_path = os.path.join(opt.output_dir, f"6_3d_Traffic.mp4")
    json_path = os.path.join(opt.output_dir, f"6_3d_Traffic.json")

    print(f"Video output will be saved to: {video_path}")
    print(f"3D coordinates will be saved to: {json_path}")
    
    return video_path, json_path

def convert_to_metric(depth_map, a=None, b=None, known_depths=None, known_pixels=None):
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

def pixel_to_world_coordinates(x_pixel, y_pixel, depth_value, camera_matrix):
    """
    Convert pixel coordinates to world coordinates using depth and camera matrix.
    Args:
        x_pixel: X coordinate in pixel space
        y_pixel: Y coordinate in pixel space
        depth_value: Depth value at pixel
        camera_matrix: 3x3 camera intrinsic matrix
    Returns:
        tuple: (X_world, Y_world, Z_world)
    """
    # Camera intrinsic parameters
    fx = camera_matrix[0, 0]  # Focal length in x
    fy = camera_matrix[1, 1]  # Focal length in y
    cx = camera_matrix[0, 2]  # Principal point x
    cy = camera_matrix[1, 2]  # Principal point y
    
    # Apply the formula:
    # X_world = (X_pixel - cx) * Z_depth / fx
    # Y_world = (Y_pixel - cy) * Z_depth / fy
    # Z_world = Z_depth
    X_world = (x_pixel - cx) * depth_value / fx
    Y_world = (y_pixel - cy) * depth_value / fy
    Z_world = depth_value
    
    return X_world, Y_world, Z_world

def get_detection_center_and_size(x1, y1, x2, y2):
    """
    Calculate the center coordinates and size of a detection bounding box.
    Args:
        x1, y1: Top-left corner of bounding box
        x2, y2: Bottom-right corner of bounding box
    Returns:
        tuple: (center_x, center_y, width, height)
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    return center_x, center_y, width, height

def create_3d_bounding_box(bbox, depth_value, class_name, camera_matrix):
    """
    Create a 3D bounding box from a 2D bounding box and depth value.
    Uses camera intrinsics to project 2D points to 3D.
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    center_x, center_y = bbox['center']['x'], bbox['center']['y']
    
    # Camera intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Create 3D bounding box by projecting 2D corners
    corners_3d = []
    for corner in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        x_pixel, y_pixel = corner
        # Apply the projection formula
        X_world = (x_pixel - cx) * depth_value / fx
        Y_world = (y_pixel - cy) * depth_value / fy
        Z_world = depth_value
        corners_3d.append([float(X_world), float(Y_world), float(Z_world)])
    
    # Estimate object dimensions
    width_3d = abs(corners_3d[1][0] - corners_3d[0][0])
    height_3d = abs(corners_3d[2][1] - corners_3d[1][1])
    
    # Estimate depth based on object class (typical aspect ratios)
    if 'bin' in class_name.lower() or 'dustbin' in class_name.lower():
        depth_3d = width_3d * 1.0  # Bins are typically as deep as they are wide
    elif 'traffic cone' in class_name.lower() or 'cone' in class_name.lower():
        depth_3d = width_3d * 1.0  # Cones are typically symmetric
    elif 'traffic cylinder' in class_name.lower() or 'cylinder' in class_name.lower():
        depth_3d = width_3d * 1.0  # Cylinders are typically symmetric
    else:
        depth_3d = width_3d * 1.0  # Default for other objects
    
    # Calculate center in 3D world coordinates
    center_X = (center_x - cx) * depth_value / fx
    center_Y = (center_y - cy) * depth_value / fy
    center_3d = [
        float(center_X),
        float(center_Y),
        float(depth_value)
    ]
    
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
    
    return {
        'corners': bbox_3d,
        'center': center_3d,
        'dimensions': {
            'width': float(width_3d),
            'height': float(height_3d),
            'depth': float(depth_3d)
        }
    }

def process_video(opt, detection_models, thresholds, depth_model, video_output_path, json_output_path):
    """
    Process the video with object detection and depth estimation.
    Args:
        opt: Command line arguments
        detection_models: Dictionary of loaded detection models
        thresholds: Dictionary of confidence thresholds for each model
        depth_model: Depth estimation model
        video_output_path: Path to save output video
        json_output_path: Path to save 3D coordinates
    """
    # Camera matrix
    camera_matrix = np.array([
        [1594.7, 0, 655.2961], 
        [0, 1607.7, 414.3627], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Create a directory for frame-specific JSON files
    json_dir = os.path.splitext(json_output_path)[0]
    os.makedirs(json_dir, exist_ok=True)
    print(f"Frame-specific JSON files will be saved to: {json_dir}")
    
    # Open input video
    cap = cv2.VideoCapture(opt.input)
    if not cap.isOpened():
        print(f"Error: Could not open video {opt.input}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Data structure to store 3D coordinates of all detected objects
    all_frames_data = {
        'camera_matrix': camera_matrix.tolist(),  # Store camera matrix in JSON
        'frames': {}
    }
    
    # Process frames
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print("\n" + "="*50)
    print("PROCESSING VIDEO:")
    print("="*50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream reached")
            break
        
        frame_count += 1
        
        # Skip frames if requested (for faster processing)
        if opt.frame_skip > 0 and frame_count % (opt.frame_skip + 1) != 0:
            continue
        
        processed_count += 1
        
        # Calculate and display progress
        elapsed_time = time.time() - start_time
        fps_current = processed_count / elapsed_time if elapsed_time > 0 else 0
        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        
        if processed_count % 10 == 0 or opt.verbose:  # Update every 10 processed frames
            print(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames}), Processing speed: {fps_current:.1f} fps")
        
        # Copy original frame for visualization
        display_frame = frame.copy()
        
        # Convert frame to RGB and create PIL Image for depth estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run depth estimation
        depth_result = depth_model(pil_image)
        depth_map = np.array(depth_result["depth"])
        
        # Convert to metric depth
        metric_depth_map = convert_to_metric(depth_map)
        
        # Normalize depth map for visualization (but preserve original values for 3D reconstruction)
        depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_colored = cv2.resize(depth_colored, (frame_width, frame_height))
        
        # Store frame data
        frame_data = []
        
        # Process with each detection model
        for model_type, model in detection_models.items():
            # Use appropriate detection method based on model type
            if model_type == 'bin' or model_type == 'cylinder':
                # Process with YOLO model from ultralytics
                results = model(
                    frame,
                    conf=thresholds[model_type],
                    classes=opt.classes,
                    device=opt.device
                )
                
                # Process detections
                for box in results[0].boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    cls = int(box.cls.item())
                    
                    # Get center of detection
                    center_x, center_y, width, height = get_detection_center_and_size(x1, y1, x2, y2)
                    
                    # Get depth value at the center of the detection
                    # Resize depth map to match original frame dimensions
                    depth_h, depth_w = depth_map.shape
                    depth_scale_x = depth_w / frame_width
                    depth_scale_y = depth_h / frame_height
                    
                    # Calculate depth map coordinates
                    depth_x = int(center_x * depth_scale_x)
                    depth_y = int(center_y * depth_scale_y)
                    
                    # Ensure coordinates are within depth map bounds
                    depth_x = max(0, min(depth_w-1, depth_x))
                    depth_y = max(0, min(depth_h-1, depth_y))
                    
                    # Get depth value
                    depth_value = float(depth_map[depth_y, depth_x])
                    # Get metric depth value
                    metric_depth = float(metric_depth_map[depth_y, depth_x])
                    
                    # Convert to world coordinates
                    X_world, Y_world, Z_world = pixel_to_world_coordinates(
                        center_x, center_y, metric_depth, camera_matrix
                    )
                    
                    # Create 3D bounding box
                    bbox_3d = create_3d_bounding_box(
                        {
                            'x1': x1, 
                            'y1': y1, 
                            'x2': x2, 
                            'y2': y2,
                            'center': {
                                'x': center_x,
                                'y': center_y
                            }
                        },
                        metric_depth,
                        model_type,
                        camera_matrix
                    )
                    
                    # Choose color based on model type
                    if model_type == 'bin':
                        color = (0, 255, 0)  # Green for bins
                    else:  # cylinder
                        color = (0, 0, 255)  # Red for cylinders
                    
                    # Store detection data
                    detection_data = {
                        'type': model_type,
                        'confidence': conf,
                        'class_id': cls,
                        'pixel': {
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        },
                        'raw_depth': depth_value,
                        'metric_depth': metric_depth,
                        'world_coordinates': {
                            'x': float(X_world),
                            'y': float(Y_world),
                            'z': float(Z_world)
                        },
                        'bbox_3d': bbox_3d
                    }
                    
                    frame_data.append(detection_data)
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add 3D coordinates to visualization
                    label = f"{model_type} {conf:.2f} | X: {X_world:.2f} Y: {Y_world:.2f} Z: {Z_world:.2f}m"
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            # elif model_type == 'pole':
            #         # Process with YOLO model for safety poles
            #         results = model(frame, conf=thresholds[model_type])
                    
            #         for box in results[0].boxes:
            #             # Get box coordinates
            #             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            #             conf = float(box.conf)
            #             cls = int(box.cls.item())
                        
            #             # Get center of detection
            #             center_x, center_y, width, height = get_detection_center_and_size(x1, y1, x2, y2)
                        
            #             # Get depth and metric depth values
            #             depth_value, metric_depth = get_depth_values(depth_map, metric_depth_map, center_x, center_y, frame_width, frame_height)
                        
            #             # Convert to world coordinates
            #             X_world, Y_world, Z_world = pixel_to_world_coordinates(center_x, center_y, metric_depth, camera_matrix)
                        
            #             # Create 3D bounding box
            #             bbox_3d = create_3d_bounding_box({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'center': {'x': center_x, 'y': center_y}},
            #                                             metric_depth, model_type, camera_matrix)
                        
            #             # Choose color for safety poles
            #             color = (255, 255, 0)  # Yellow for safety poles
                        
            #             # Store detection data
            #             detection_data = {
            #                 'type': model_type,
            #                 'confidence': conf,
            #                 'class_id': cls,
            #                 'pixel': {'center_x': center_x, 'center_y': center_y, 'width': width, 'height': height, 'bbox': [int(x1), int(y1), int(x2), int(y2)]},
            #                 'raw_depth': depth_value,
            #                 'metric_depth': metric_depth,
            #                 'world_coordinates': {'x': float(X_world), 'y': float(Y_world), 'z': float(Z_world)},
            #                 'bbox_3d': bbox_3d
            #             }
                        
            #             frame_data.append(detection_data)
                        
            #             # Draw bounding box and add label
            #             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            #             label = f"Safety Pole {conf:.2f} | X: {X_world:.2f} Y: {Y_world:.2f} Z: {Z_world:.2f}m"
            #             cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
            elif model_type == 'pole':
                # Process with YOLO model for safety poles
                results = model(frame, conf=thresholds[model_type])
                
                for box in results[0].boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    cls = int(box.cls.item())
                    
                    # Get center of detection
                    center_x, center_y, width, height = get_detection_center_and_size(x1, y1, x2, y2)
                    
                    # Get depth value at the center of the detection
                    # Resize depth map to match original frame dimensions
                    depth_h, depth_w = depth_map.shape
                    depth_scale_x = depth_w / frame_width
                    depth_scale_y = depth_h / frame_height
                    
                    # Calculate depth map coordinates
                    depth_x = int(center_x * depth_scale_x)
                    depth_y = int(center_y * depth_scale_y)
                    
                    # Ensure coordinates are within depth map bounds
                    depth_x = max(0, min(depth_w-1, depth_x))
                    depth_y = max(0, min(depth_h-1, depth_y))
                    
                    # Get depth value
                    depth_value = float(depth_map[depth_y, depth_x])
                    # Get metric depth value
                    metric_depth = float(metric_depth_map[depth_y, depth_x])
                    
                    # Convert to world coordinates
                    X_world, Y_world, Z_world = pixel_to_world_coordinates(center_x, center_y, metric_depth, camera_matrix)
                    
                    # Create 3D bounding box
                    bbox_3d = create_3d_bounding_box({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'center': {'x': center_x, 'y': center_y}},
                                                    metric_depth, model_type, camera_matrix)
                    
                    # Choose color for safety poles
                    color = (255, 255, 0)  # Yellow for safety poles
                    
                    # Store detection data
                    detection_data = {
                        'type': model_type,
                        'confidence': conf,
                        'class_id': cls,
                        'pixel': {'center_x': center_x, 'center_y': center_y, 'width': width, 'height': height, 'bbox': [int(x1), int(y1), int(x2), int(y2)]},
                        'raw_depth': depth_value,
                        'metric_depth': metric_depth,
                        'world_coordinates': {'x': float(X_world), 'y': float(Y_world), 'z': float(Z_world)},
                        'bbox_3d': bbox_3d
                    }
                    
                    frame_data.append(detection_data)
                    
                    # Draw bounding box and add label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Safety Pole {conf:.2f} | X: {X_world:.2f} Y: {Y_world:.2f} Z: {Z_world:.2f}m"
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            
            elif model_type == 'cone':
                # Process with PyTorch Hub model
                results = model(frame)
                
                # Process detections
                if hasattr(results, 'pred') and len(results.pred) > 0:
                    for *xyxy, conf, cls in results.pred[0]:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(conf)
                        cls = int(cls.item())
                        
                        # Get center of detection
                        center_x, center_y, width, height = get_detection_center_and_size(x1, y1, x2, y2)
                        
                        # Get depth value at the center of the detection
                        # Resize depth map to match original frame dimensions
                        depth_h, depth_w = depth_map.shape
                        depth_scale_x = depth_w / frame_width
                        depth_scale_y = depth_h / frame_height
                        
                        # Calculate depth map coordinates
                        depth_x = int(center_x * depth_scale_x)
                        depth_y = int(center_y * depth_scale_y)
                        
                        # Ensure coordinates are within depth map bounds
                        depth_x = max(0, min(depth_w-1, depth_x))
                        depth_y = max(0, min(depth_h-1, depth_y))
                        
                        # Get depth value
                        depth_value = float(depth_map[depth_y, depth_x])
                        # Get metric depth value
                        metric_depth = float(metric_depth_map[depth_y, depth_x])
                        
                        # Convert to world coordinates
                        X_world, Y_world, Z_world = pixel_to_world_coordinates(
                            center_x, center_y, metric_depth, camera_matrix
                        )
                        
                        # Create 3D bounding box
                        bbox_3d = create_3d_bounding_box(
                            {
                                'x1': x1, 
                                'y1': y1, 
                                'x2': x2, 
                                'y2': y2,
                                'center': {
                                    'x': center_x,
                                    'y': center_y
                                }
                            },
                            metric_depth,
                            model_type,
                            camera_matrix
                        )
                        
                        # Choose color for cones
                        color = (255, 0, 0)  # Blue for cones
                        
                        # Store detection data
                        detection_data = {
                            'type': model_type,
                            'confidence': conf,
                            'class_id': cls,
                            'pixel': {
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': width,
                                'height': height,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            },
                            'raw_depth': depth_value,
                            'metric_depth': metric_depth,
                            'world_coordinates': {
                                'x': float(X_world),
                                'y': float(Y_world),
                                'z': float(Z_world)
                            },
                            'bbox_3d': bbox_3d
                        }
                        
                        frame_data.append(detection_data)
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add 3D coordinates to visualization
                        label = f"{model_type} {conf:.2f} | X: {X_world:.2f} Y: {Y_world:.2f} Z: {Z_world:.2f}m"
                        cv2.putText(display_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add depth visualization to corner of frame (small overlay)
        depth_overlay_size = (int(frame_width * 0.25), int(frame_height * 0.25))
        depth_overlay = cv2.resize(depth_colored, depth_overlay_size)
        
        # Place depth overlay in top-right corner
        roi = display_frame[10:10+depth_overlay_size[1], frame_width-10-depth_overlay_size[0]:frame_width-10]
        # Blend overlay with frame (alpha blending)
        cv2.addWeighted(depth_overlay, 0.7, roi, 0.3, 0, roi)
        display_frame[10:10+depth_overlay_size[1], frame_width-10-depth_overlay_size[0]:frame_width-10] = roi
        
        # Add depth scale bar
        cv2.rectangle(display_frame, 
                     (frame_width-10-depth_overlay_size[0], 10+depth_overlay_size[1]+5), 
                     (frame_width-10, 10+depth_overlay_size[1]+20), 
                     (0, 0, 0), -1)
        cv2.putText(display_frame, 
                   f"Depth: {depth_map.min():.1f}m - {depth_map.max():.1f}m", 
                   (frame_width-10-depth_overlay_size[0]+5, 10+depth_overlay_size[1]+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Store frame data in main data structure
        all_frames_data['frames'][frame_count] = frame_data
        
        # Save individual JSON file for this frame (even if no detections)
        frame_json_path = os.path.join(json_dir, f"frame_{frame_count-1:06d}.json")
        frame_data_with_metadata = {
            'frame_number': frame_count,
            'camera_matrix': camera_matrix.tolist(),
            'objects': frame_data
        }
        with open(frame_json_path, 'w') as f:
            json.dump(frame_data_with_metadata, f, indent=2)
        
        # Skip display to avoid cv2.imshow error
        if opt.view_img == False:
            pass  # Skip display code entirely
        
        # Write frame to output video
        out.write(display_frame)
        
        # Save intermediate JSON every 100 frames to avoid losing data if the script crashes
        if processed_count % 1 == 0:
            with open(json_output_path, 'w') as f:
                json.dump(all_frames_data, f, indent=2)
    
    # Calculate and display final statistics
    elapsed_time = time.time() - start_time
    fps_avg = processed_count / elapsed_time if elapsed_time > 0 else 0
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE:")
    print("="*50)
    print(f"✅ Processed {processed_count} of {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"✅ Average processing speed: {fps_avg:.2f} fps")
    
    # Save final JSON with all detections
    with open(json_output_path, 'w') as f:
        json.dump(all_frames_data, f, indent=2)
    
    print(f"✅ 3D coordinates saved to: {json_output_path}")
    print(f"✅ Individual frame data saved to: {json_dir}")
    print(f"✅ Output video saved to: {video_output_path}")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    opt = parse_arguments()

    # Check and install dependencies
    check_and_install_dependencies()

    # Load detection models and depth estimation model
    detection_models, thresholds, depth_model = load_models(opt)

    # Create output paths
    video_output_path, json_output_path = create_output_paths(opt)

    # Process the video
    process_video(opt, detection_models, thresholds, depth_model, video_output_path, json_output_path)

if __name__ == "__main__":
    main()

