import torch
import cv2
import numpy as np
import os
import json
import time
import sys
from pathlib import Path
from transformers import pipeline
from PIL import Image
import argparse
import math

class SpeedBreaker3DDetector:
    """
    Enhanced 3D Speed Breaker Detector with depth estimation
    """
    def __init__(self, model_path=None, conf_threshold=0.99, camera_intrinsics=None):
        """
        Initialize the 3D speed breaker detector
        
        Args:
            model_path: Path to the trained speed breaker detection model
            conf_threshold: Confidence threshold for detections
            camera_intrinsics: Camera parameters for 3D calculations
        """
        self.conf_threshold = conf_threshold
        
        # Default camera intrinsics if not provided
        if camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': 1594.7,
                'fy': 1607.7,
                'cx': 655.2961,
                'cy': 414.3627,
                'image_size': [960, 1280]
            }
        else:
            self.camera_intrinsics = camera_intrinsics
        
        # Initialize speed breaker detection model
        print("Loading speed breaker detection model...")
        try:
            # Use YOLOv5 instead of the newer Ultralytics package
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
            self.model.conf = conf_threshold  # Set confidence threshold
            print(f"Speed breaker detection model loaded successfully")
        except Exception as e:
            print(f"Error loading speed breaker detection model: {e}")
            self.model = None
            
        # Initialize depth estimation model
        print("Loading depth estimation model...")
        try:
            self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
            print("Depth model loaded successfully")
        except Exception as e:
            print(f"Error loading depth model: {e}")
            self.depth_estimator = None
    # def __init__(self, model_path=None, conf_threshold=0.3, camera_intrinsics=None):
    #     """
    #     Initialize the 3D speed breaker detector
        
    #     Args:
    #         model_path: Path to the trained speed breaker detection model
    #         conf_threshold: Confidence threshold for detections
    #         camera_intrinsics: Camera parameters for 3D calculations
    #     """
    #     self.conf_threshold = conf_threshold
        
    #     # Default camera intrinsics if not provided
    #     if camera_intrinsics is None:
    #         self.camera_intrinsics = {
    #             'fx': 1594.7,
    #             'fy': 1607.7,
    #             'cx': 655.2961,
    #             'cy': 414.3627,
    #             'image_size': [960, 1280]
    #         }
    #     else:
    #         self.camera_intrinsics = camera_intrinsics
        
    #     # Initialize speed breaker detection model
    #     print("Loading speed breaker detection model...")
    #     try:
    #         from ultralytics import YOLO
    #         self.model = YOLO(model_path)
    #         print(f"Speed breaker detection model loaded successfully")
    #     except Exception as e:
    #         print(f"Error loading speed breaker detection model: {e}")
    #         self.model = None
            
    #     # Initialize depth estimation model
    #     print("Loading depth estimation model...")
    #     try:
    #         self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    #         print("Depth model loaded successfully")
    #     except Exception as e:
    #         print(f"Error loading depth model: {e}")
    #         self.depth_estimator = None
            
    # def estimate_depth(self, frame):
    #     """
    #     Estimate depth map using MiDaS
        
    #     Args:
    #         frame: Input video frame
            
    #     Returns:
    #         depth_map: Estimated depth map
    #     """
    #     if self.depth_estimator is None:
    #         return None
            
    #     # Ensure frame is RGB (MiDaS expects RGB)
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # Get depth map
    #     result = self.depth_estimator(frame_rgb)
    #     depth_map = result["depth"]
        
    #     # Convert to numpy array if it's not already
    #     if not isinstance(depth_map, np.ndarray):
    #         depth_map = np.array(depth_map)
            
    #     return depth_map
    
    def estimate_depth(self, frame):
        """
        Estimate depth map using MiDaS
        
        Args:
            frame: Input video frame (OpenCV/NumPy format)
            
        Returns:
            depth_map: Estimated depth map
        """
        if self.depth_estimator is None:
            return None
            
        # Ensure frame is RGB (MiDaS expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Get depth map
        result = self.depth_estimator(pil_image)
        depth_map = result["depth"]
        
        # Convert to numpy array if it's not already
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map)
            
        return depth_map

        
    def convert_to_metric(self, depth_map):
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
        
    def create_3d_bounding_box(self, bbox_2d, depth_value, class_name="speed_breaker"):
        """
        Create a 3D bounding box from a 2D bounding box and depth
        
        Args:
            bbox_2d: 2D bounding box [x1, y1, x2, y2]
            depth_value: Metric depth value in millimeters
            class_name: Class name of the object
            
        Returns:
            bbox_3d: 3D bounding box information
        """
        x1, y1, x2, y2 = bbox_2d
        
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Get camera intrinsics
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # Convert pixel coordinates to world coordinates using pinhole camera model
        # X_world = (X_pixel - cx) * Z_depth / fx
        # Y_world = (Y_pixel - cy) * Z_depth / fy
        # Z_world = Z_depth
        world_x = (center_x - cx) * depth_value / fx
        world_y = (center_y - cy) * depth_value / fy
        world_z = depth_value
        
        # Calculate width and height in world coordinates
        width_pixel = x2 - x1
        height_pixel = y2 - y1
        
        # Convert pixel dimensions to world dimensions
        width_world = width_pixel * depth_value / fx
        height_world = height_pixel * depth_value / fy
        
        # Estimate depth dimension based on typical speed breaker proportions
        # Standard speed breaker: ~3-4m wide, ~0.3-0.5m deep, ~0.05-0.1m high
        width = width_world  # Use calculated width
        height = 75  # mm (0.075m) - typical height of a speed breaker
        depth = 400  # mm (0.4m) - typical depth of a speed breaker
        
        # Create 3D bounding box
        bbox_3d = {
            "center": [float(world_x), float(world_y), float(world_z)],
            "dimensions": {
                "width": float(width),
                "height": float(height),
                "depth": float(depth)
            },
            "orientation": 0.0,  # Assuming speed breakers are aligned with the road
            "corners": self.calculate_3d_corners(world_x, world_y, world_z, width, height, depth)
        }
        
        return bbox_3d
        
    def calculate_3d_corners(self, center_x, center_y, center_z, width, height, depth):
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
        
    
    def detect(self, frame):
        """
        Detect speed breakers in a single frame with 3D information
        
        Args:
            frame: Input video frame
            
        Returns:
            detections: List of dictionaries containing detection data with 3D info
            vis_frame: Visualization frame with annotations
        """
        # Make a copy for visualization
        vis_frame = frame.copy()
        
        # List to store detections
        detections = []
        
        # Estimate depth map
        depth_map = self.estimate_depth(frame)
        
        # Convert to metric depth using the inverse linear relationship
        if depth_map is not None:
            metric_depth_map = self.convert_to_metric(depth_map)
        else:
            metric_depth_map = None
        
        # Run speed breaker detection model
        if self.model is not None:
            # Set confidence threshold before inference
            self.model.conf = self.conf_threshold
            
            # Run inference without passing conf parameter
            results = self.model(frame)
            
            # Process results
            for i, det in enumerate(results.xyxy[0]):  # Process first image in batch
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                
                # Only process detections that meet the confidence threshold
                # This ensures we only include high-confidence detections
                if conf >= self.conf_threshold:
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate center of detection
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Get depth at the center of the bounding box
                    if depth_map is not None and metric_depth_map is not None:
                        if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                            # Get raw MiDaS depth value
                            raw_depth = float(depth_map[center_y, center_x])
                            # Get metric depth value
                            metric_depth = float(metric_depth_map[center_y, center_x])
                            
                            # Create 3D bounding box
                            bbox_3d = self.create_3d_bounding_box(
                                [x1, y1, x2, y2],
                                metric_depth,
                                "speed_breaker"
                            )
                        else:
                            raw_depth = None
                            metric_depth = None
                            bbox_3d = None
                    else:
                        raw_depth = None
                        metric_depth = None
                        bbox_3d = None
                    
                    # Add to detections
                    detection = {
                        'class': 'Speed Breaker',
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
                    if raw_depth is not None:
                        detection['relative_depth'] = raw_depth
                    if metric_depth is not None:
                        detection['metric_depth'] = metric_depth
                    if bbox_3d is not None:
                        detection['bbox_3d'] = bbox_3d
                    
                    detections.append(detection)
                    
                    # Draw on visualization frame
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Speed Breaker {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # If depth is available, show it
                    if metric_depth is not None:
                        depth_text = f"D: {metric_depth:.2f}mm"
                        cv2.putText(vis_frame, depth_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Show 3D coordinates
                        world_coords = bbox_3d["center"]
                        coords_text = f"X: {world_coords[0]:.2f}, Y: {world_coords[1]:.2f}, Z: {world_coords[2]:.2f}"
                        cv2.putText(vis_frame, coords_text, (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return detections, vis_frame




def process_video_for_3d_speed_breakers(video_path, output_dir, model_path=None, output_video_path=None, display=True, start_frame=0, end_frame=None):
    """
    Process a video file to detect speed breakers with 3D information
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save JSON output
        model_path: Path to the trained speed breaker detection model
        output_video_path: Path to save output video (optional)
        display: Whether to display the processing in real-time
        start_frame: Frame number to start processing from (default: 0)
        end_frame: Frame number to end processing (default: None, process until end)
    """
    # Initialize detector
    detector = SpeedBreaker3DDetector(model_path=model_path, conf_threshold=0.99)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set end frame if not specified
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    
    # Initialize video writer if output path is specified
    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables for tracking
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Starting from frame: {start_frame}")
    print(f"Ending at frame: {end_frame}")
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        print(f"Skipped to frame {start_frame}")
    
    # Process each frame
    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_count += 1
        
        # Detect speed breakers with 3D information
        detections, vis_frame = detector.detect(frame)
        
        # Create scene data
        scene_data = {
            "frame_number": frame_count,
            "frame_dimensions": {
                "width": width,
                "height": height
            },
            "camera_intrinsics": detector.camera_intrinsics,
            "objects": detections,
            "has_depth_data": detector.depth_estimator is not None
        }
        
        # Save JSON data
        json_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.json")
        with open(json_filename, 'w') as f:
            json.dump(scene_data, f, indent=2)
            
        frame_count += 1
        
        # Add frame number to visualization
        cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write to output video
        if out:
            out.write(vis_frame)
        
        # Display if requested
        if display:
            cv2.imshow("3D Speed Breaker Detection", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print progress every 10 frames
        if processed_count % 1 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = processed_count / elapsed_time if elapsed_time > 0 else 0
            progress = ((frame_count - start_frame) / (end_frame - start_frame)) * 100
            print(f"Progress: {progress:.1f}% | Frame {frame_count}/{end_frame} | {fps_processing:.2f} FPS")
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    print(f"Processing complete!")
    print(f"Processed {processed_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {processed_count / elapsed_time:.2f} FPS")
    print(f"Output saved to: {output_dir}")
    
    if output_video_path:
        print(f"Visualization saved to: {output_video_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D Speed Breaker Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save JSON output')
    parser.add_argument('--model', type=str, default=r"C:\Users\pavan\Documents\CV_environment\.venv\best_bump.pt", help='Path to trained speed breaker detection model')
    parser.add_argument('--output_video', type=str, default=r"C:\Users\pavan\Documents\CV_P3\output\3D_SpeedBump\1_SpeedBump_lastframes.mp4", help='Path to save output video (optional)')
    parser.add_argument('--display', action='store_true', help='Display processing in real-time')
    parser.add_argument('--start_frame', type=int, default=0, help='Frame number to start processing from')
    parser.add_argument('--end_frame', type=int, default=2500, help='Frame number to end processing')
    
    args = parser.parse_args()
    
    # Run standalone 3D speed breaker detection
    process_video_for_3d_speed_breakers(
        args.video, 
        args.output_dir, 
        args.model, 
        args.output_video, 
        args.display,
        args.start_frame,
        args.end_frame
    )
