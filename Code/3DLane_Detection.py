import torch
import torchvision
import cv2
import numpy as np
import os
import json
from PIL import Image
from transformers import pipeline

class LaneDetector3D:
    def __init__(self, mask_rcnn_weights, camera_matrix):
        """
        Initialize the lane detector with Mask R-CNN model and camera parameters.
        
        Args:
            mask_rcnn_weights (str): Path to the Mask R-CNN weights file
            camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
        """
        print("Initializing LaneDetector3D...")
        
        # Initialize Mask R-CNN model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            pretrained=False, num_classes=7)
        
        # Load weights
        print(f"Loading Mask R-CNN weights from: {mask_rcnn_weights}")
        ckpt = torch.load(mask_rcnn_weights, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device).eval()
        
        # Initialize depth estimator
        print("Loading depth estimation model...")
        self.depth_estimator = pipeline("depth-estimation", 
                                       model="Intel/dpt-hybrid-midas", 
                                       device=0 if torch.cuda.is_available() else -1)
        
        # Camera intrinsics
        self.camera_matrix = camera_matrix
        self.fx = camera_matrix[0, 0]  # 1594.7
        self.fy = camera_matrix[1, 1]  # 1607.7
        self.cx = camera_matrix[0, 2]  # 655.2961
        self.cy = camera_matrix[1, 2]  # 414.3627
        
        print(f"Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        # Transform for model input
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        # # Define lane class mapping
        # self.lane_types = {
        #     0: "solid-line",
        #     1: "divider-line",
        #     2: "random-line"
        # }
        
        # # Define lane colors
        # self.lane_colors = {
        #     "solid-line": (255, 0, 255),  # Magenta
        #     "divider-line": (255, 0, 0),  # Red
        #     "random-line": (255, 0, 255)  # Magenta
        # }
        
        self.lane_types = {
            0: "solid-line",
            1: "dotted-line",
            2: "divider-line",
            3: "random-line"
        }

        self.lane_colors = {
            "solid-line": (255, 0, 255),  # Magenta
            "dotted-line": (0, 255, 255),  # Yellow
            "divider-line": (255, 0, 0),   # Red
            "random-line": (255, 0, 255)   # Magenta
        }

        
        print("LaneDetector3D initialized successfully")
    
    def detect_lanes(self, frame):
        """Detect lanes using Mask R-CNN with lane type classification"""
        # Convert frame to RGB and create PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Transform image for model input
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(tensor_image)
        
        # Extract masks for lane classes
        lane_class_indices = list(range(7))  # Use all classes from your model
        masks = []
        labels = []
        scores = []
        
        for i, score in enumerate(predictions[0]['scores']):
            if score > 0.82:  # Lower threshold to detect more lanes
                class_idx = predictions[0]['labels'][i].item()
                if class_idx in lane_class_indices:
                    mask = predictions[0]['masks'][i, 0].cpu().numpy()
                    masks.append(mask)
                    labels.append(class_idx)
                    scores.append(score.item())
        
        return masks, labels, scores
    
    def extract_lane_points(self, mask):
        """Extract points from lane mask using polynomial fitting as described in Algorithm 1"""
        # Convert mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Find non-zero points in the mask
        y_indices, x_indices = np.nonzero(binary_mask)
        
        if len(y_indices) < 10:  # Not enough points for fitting
            return np.array([])
        
        try:
            # Fit a 2nd order polynomial: f(y) = ay² + by + c
            coeffs = np.polyfit(y_indices, x_indices, 2)
            
            # Calculate lane length
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            lane_length = y_max - y_min
            
            # Generate 10 equidistant points along the polynomial
            points = []
            for i in range(10):
                # Calculate y coordinate
                y_i = y_min + (i * lane_length / 9)  # 9 segments for 10 points
                
                # Calculate corresponding x using the polynomial
                x_i = coeffs[0] * (y_i ** 2) + coeffs[1] * y_i + coeffs[2]
                
                points.append([int(x_i), int(y_i)])
            
            return np.array(points)
        except:
            # Fallback to contour extraction if polynomial fitting fails
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.array([])
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Extract points
            points = approx_contour.reshape(-1, 2)
            
            return points
    
    def convert_to_metric(self, depth_map, scale_factor=3000.0, shift=0.1):
        """
        Convert MiDaS depth map to metric depth (mm)
        Using the formula: metric_depth = scale_factor * (1 / (depth_map + shift))
        """
        metric_depth = scale_factor * (1.0 / (depth_map + shift))
        return metric_depth
    
    def project_to_3d(self, points_2d, depth_map):
        """
        Project 2D points to 3D using depth map and camera intrinsics.
        Uses the formula:
        X_world = (X_pixel - cx) · Z_depth / fx
        Y_world = (Y_pixel - cy) · Z_depth / fy
        Z_world = Z_depth
        """
        points_3d = []
        
        for point in points_2d:
            x, y = point
            
            # Ensure coordinates are within image bounds
            x, y = int(min(max(x, 0), depth_map.shape[1]-1)), int(min(max(y, 0), depth_map.shape[0]-1))
            
            # Get depth at this point
            z_depth = depth_map[y, x]
            
            # Apply projection formula
            x_world = (x - self.cx) * z_depth / self.fx
            y_world = (y - self.cy) * z_depth / self.fy
            z_world = z_depth
            
            points_3d.append([float(x_world), float(y_world), float(z_world)])
        
        return points_3d
    
    # def determine_lane_type(self, mask, points_2d, position_x):
    #     """
    #     Determine lane type based on appearance and position
    #     """
    #     # Calculate mask properties
    #     binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
    #     # Check if lane is on left or right side of the image
    #     image_center = mask.shape[1] / 2
    #     is_left = position_x < image_center
        
    #     # Check if lane is dashed by looking for gaps in the mask
    #     # First, get the bounding box of the lane
    #     y_indices, x_indices = np.nonzero(binary_mask)
    #     if len(y_indices) == 0:
    #         return "random-line"
            
    #     y_min, y_max = np.min(y_indices), np.max(y_indices)
    #     x_min, x_max = np.min(x_indices), np.max(x_indices)
        
    #     # Calculate the aspect ratio of the lane
    #     width = x_max - x_min
    #     height = y_max - y_min
    #     aspect_ratio = height / (width + 1e-5)
        
    #     # Calculate the density of the mask
    #     area = width * height
    #     filled_pixels = np.sum(binary_mask > 0)
    #     density = filled_pixels / (area + 1e-5)
        
    #     # Determine lane type based on position and properties
    #     if aspect_ratio > 5 and density < 0.5:
    #         # Dashed line typically has lower density
    #         return "random-line"
    #     elif is_left and x_min < image_center - width:
    #         # Left solid line
    #         return "solid-line"
    #     elif not is_left and x_max > image_center + width:
    #         # Right solid line
    #         return "solid-line"
    #     else:
    #         # Center divider
    #         return "divider-line"
    
    # def determine_lane_type(self, mask, points_2d, position_x):
    #     """
    #     Determine lane type based on appearance and position
    #     """
    #     binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
    #     image_center = mask.shape[1] / 2
    #     is_left = position_x < image_center
        
    #     y_indices, x_indices = np.nonzero(binary_mask)
    #     if len(y_indices) == 0:
    #         return "random-line"
            
    #     y_min, y_max = np.min(y_indices), np.max(y_indices)
    #     x_min, x_max = np.min(x_indices), np.max(x_indices)
        
    #     width = x_max - x_min
    #     height = y_max - y_min
    #     aspect_ratio = height / (width + 1e-5)
        
    #     area = width * height
    #     filled_pixels = np.sum(binary_mask > 0)
    #     density = filled_pixels / (area + 1e-5)
        
    #     # Check for gaps to identify dotted lines
    #     row_sums = np.sum(binary_mask, axis=1)
    #     gaps = np.where(row_sums == 0)[0]
    #     gap_ratio = len(gaps) / height
        
    #     if gap_ratio > 0.2:  # If more than 20% of rows are empty, consider it a dotted line
    #         return "dotted-line"
    #     elif aspect_ratio > 5 and density < 0.5:
    #         return "random-line"
    #     elif is_left and x_min < image_center - width:
    #         return "solid-line"
    #     elif not is_left and x_max > image_center + width:
    #         return "solid-line"
    #     else:
    #         return "divider-line"
    
    def determine_lane_type(self, mask, points_2d, position_x):
        """Determine lane type based on appearance, position and pattern analysis"""
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Get the original image region under the mask
        if len(points_2d) == 0:
            return "random-line"
            
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        # Ensure coordinates are within bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(self.current_frame.shape[1]-1, x_max), min(self.current_frame.shape[0]-1, y_max)
        
        # Extract the region
        lane_region = self.current_frame[y_min:y_max, x_min:x_max]
        
        if lane_region.size == 0:
            return "random-line"
        
        # Convert to HSV for better color detection
        hsv_region = cv2.cvtColor(lane_region, cv2.COLOR_BGR2HSV)
        
        # Check for yellow color (typical for divider lines)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_region, yellow_lower, yellow_upper)
        yellow_ratio = np.sum(yellow_mask) / (lane_region.shape[0] * lane_region.shape[1] * 255 + 1e-6)
        
        # Check for white color (typical for solid/dotted lines)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv_region, white_lower, white_upper)
        white_ratio = np.sum(white_mask) / (lane_region.shape[0] * lane_region.shape[1] * 255 + 1e-6)
        
        # Calculate geometric properties
        image_center = self.current_frame.shape[1] / 2
        is_left = position_x < image_center
        
        # Check if the lane is horizontal (like house edges)
        y_coords = [p[1] for p in points_2d]
        y_range = max(y_coords) - min(y_coords)
        x_range = max([p[0] for p in points_2d]) - min([p[0] for p in points_2d])
        
        # If the lane is more horizontal than vertical, it's likely not a road lane
        if y_range < x_range * 0.5:
            return "random-line"
        
        # Analyze the vertical profile of the mask to detect gaps (dotted lines)
        # Sum each row to get a vertical profile
        row_sums = np.sum(binary_mask, axis=1)
        
        # Normalize the row sums
        if np.max(row_sums) > 0:
            row_sums = row_sums / np.max(row_sums)
        
        # Count transitions from high to low values (gaps)
        transitions = 0
        threshold = 0.82
        for i in range(1, len(row_sums)):
            if row_sums[i-1] > threshold and row_sums[i] <= threshold:
                transitions += 1
        
        # Calculate gap ratio (number of gaps relative to lane length)
        gap_ratio = transitions / (y_range + 1e-6)
        
        # Determine lane type based on color, position and gap analysis
        if gap_ratio > 0.05:  # Multiple gaps detected
            return "dotted-line"
        elif yellow_ratio > 0.1:
            return "divider-line"  # Yellow solid line
        elif white_ratio > 0.1:
            return "solid-line"    # White solid line
        else:
            # Check if it's on the road area
            y_mean = np.mean(y_coords)
            if y_mean < self.current_frame.shape[0] * 0.6:  # If it's in the upper part of the image
                return "random-line"
            else:
                return "solid-line" if is_left else "divider-line"


    
    def process_frame(self, frame):
        """Process a single frame for lane detection with 3D coordinates"""
        self.current_frame = frame
        # Detect lanes
        lane_masks, lane_labels, lane_scores = self.detect_lanes(frame)
        
        # Estimate depth
        depth_map = self.estimate_depth(frame)
        
        lanes_3d = []
        
        # Process each lane
        for i, mask in enumerate(lane_masks):
            # Extract 2D points from lane mask using polynomial fitting
            points_2d = self.extract_lane_points(mask)
            
            if len(points_2d) == 0:
                continue
            
            # Project to 3D
            points_3d = self.project_to_3d(points_2d, depth_map)
            
            # Determine lane type based on position and appearance
            center_x = np.mean([p[0] for p in points_2d])
            lane_type = self.determine_lane_type(mask, points_2d, center_x)
            
            # Get color based on lane type
            color = self.lane_colors.get(lane_type, (255, 255, 255))
            
            # Add to lanes list
            lanes_3d.append({
                "type": lane_type,
                "color": color,
                "points_2d": points_2d.tolist(),
                "points_3d": points_3d,
                "score": lane_scores[i]
            })
        
        return lanes_3d, depth_map
    
    def estimate_depth(self, frame):
        """Estimate depth using MiDaS model"""
        # Convert frame to RGB and create PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run depth estimation
        result = self.depth_estimator(pil_image)
        
        # Get depth map
        depth_map = np.array(result["depth"])
        
        # Convert relative depth to metric (approximate)
        metric_depth = self.convert_to_metric(depth_map)
        
        return metric_depth
    
    def visualize_lanes(self, frame, lanes_3d, depth_map):
        """Visualize detected lanes with depth information and bounding boxes"""
        vis_frame = frame.copy()
        
        # Normalize depth map for visualization
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_vis = (norm_depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
        
        # Draw lanes with bounding boxes
        for lane in lanes_3d:
            # Get color and label from lane data
            lane_type = lane["type"]
            color = lane["color"]
            
            # Draw 2D points
            points = np.array(lane["points_2d"], dtype=np.int32)
            for i in range(len(points)-1):
                cv2.line(vis_frame, tuple(points[i]), tuple(points[i+1]), color, 2)
            
            # Create bounding box around lane points
            if len(points) > 0:
                x_min = np.min(points[:, 0])
                y_min = np.min(points[:, 1])
                x_max = np.max(points[:, 0])
                y_max = np.max(points[:, 1])
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Add label above the bounding box
                cv2.putText(vis_frame, lane_type, (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Combine original frame and depth visualization
        combined = np.hstack((vis_frame, depth_color))
        
        return combined
    
    def process_video(self, video_path, output_dir):
        """Process video for lane detection with 3D coordinates"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            lanes_3d, depth_map = self.process_frame(frame)
            
            # Save results
            result = {
                "frame": frame_count,
                "lanes": lanes_3d,
                "camera_matrix": self.camera_matrix.tolist()
            }
            
            with open(os.path.join(output_dir, f"frame_{frame_count:06d}.json"), "w") as f:
                json.dump(result, f)
            
            # Visualize (optional)
            vis_frame = self.visualize_lanes(frame, lanes_3d, depth_map)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"), vis_frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}")
        
        cap.release()
        print(f"Processed {frame_count} frames")

# Example usage
if __name__ == "__main__":
    # Initialize with camera matrix
    camera_matrix = np.array([
        
        [1594.7, 0, 655.2961],
        [0, 1607.7, 414.3627],
        [0, 0, 1]
    ], dtype=np.float64)
    
    detector = LaneDetector3D(
        mask_rcnn_weights=r"C:\Users\pavan\Documents\CV_P3\model_15.pth",
        camera_matrix=camera_matrix
    )
    
    # Process a video
    detector.process_video(
        # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene9\Undist\2023-03-04_17-20-36-front_undistort.mp4",
        # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene1\Undist\2023-02-14_11-04-07-front_undistort.mp4",
        # video_path= r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene2\Undist\2023-03-03_10-31-11-front_undistort.mp4",
        
        # video_path=r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene3\Undist\2023-02-14_11-49-54-front_undistort.mp4",
        # video_path=r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene4\Undist\2023-02-14_11-51-54-front_undistort.mp4",
        # video_path=r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene5\Undist\2023-02-14_11-56-56-front_undistort.mp4",
        
        # video_path=r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene8\Undist\2023-03-03_11-40-47-front_undistort.mp4",
         video_path=r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene11\Undist\2023-03-11_17-19-53-front_undistort.mp4",
        output_dir=r"C:\Users\pavan\Documents\CV_P3\output\3D_Lanes\New_lane_CD_3d_scene11"#this to your desired output directory
    )


