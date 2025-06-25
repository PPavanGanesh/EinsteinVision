import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import json
from transformers import pipeline

class RoadMarkDetector3D:
    def __init__(self, model_path, camera_matrix, num_classes=12):
        """
        Initialize the road mark detector with Mask R-CNN model and camera parameters.
        
        Args:
            model_path (str): Path to the Mask R-CNN weights file
            camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
            num_classes (int): Number of classes in the model
        """
        print("Initializing RoadMarkDetector3D...")
        
        # Initialize the model
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # Replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        # Load model weights
        print(f"Loading model weights from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
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
        
        # Class names for visualization
        self.class_names = {
            0: "Background",
            1: "Left Arrow (LA)",
            2: "Straight Arrow (SA)",
            3: "Right Arrow (RA)",
            4: "Straight-Left Arrow (SLA)",
            5: "Straight-Right Arrow (SRA)",
            6: "Bus Lane (BL)",
            7: "Cycle Lane (CL)",
            8: "Right Arrow (DM)",
            9: "Junction Box (JB)",
            10: "Pedestrian Crossing (PC)",
            11: "Only (SL)"
        }
        
        # Colors for visualization (in BGR format for OpenCV)
        self.colors = [
            (0, 0, 0),       # Background (black)
            (0, 0, 255),     # LA (red)
            (0, 255, 0),     # SA (green)
            (255, 0, 0),     # RA (blue)
            (0, 255, 255),   # SLA (yellow)
            (255, 0, 255),   # SRA (magenta)
            (255, 255, 0),   # BL (cyan)
            (0, 0, 128),     # CL (maroon)
            (0, 128, 0),     # DM (dark green)
            (128, 0, 0),     # JB (navy)
            (0, 128, 128),   # PC (olive)
            (128, 0, 128)    # SL (purple)
        ]
        
        print("RoadMarkDetector3D initialized successfully")

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
    
    def convert_to_metric(self, depth_map, scale_factor=3000.0, shift=0.1):
        """
        Convert MiDaS depth map to metric depth (mm)
        Using the formula: metric_depth = scale_factor * (1 / (depth_map + shift))
        """
        metric_depth = scale_factor * (1.0 / (depth_map + shift))
        return metric_depth
    
    def extract_points(self, mask):
        """Extract points from road mark mask"""
        # Convert mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Find contours
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
    
    def process_frame(self, frame):
        """Process a single frame for road mark detection with 3D coordinates"""
        # Convert frame to RGB (model expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Estimate depth
        depth_map = self.estimate_depth(frame)
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        masks = predictions[0]['masks'].cpu().numpy()
        
        # Filter predictions with high confidence
        high_conf_idx = scores > 0.8
        boxes = boxes[high_conf_idx]
        labels = labels[high_conf_idx]
        scores = scores[high_conf_idx]
        masks = masks[high_conf_idx]
        
        road_marks_3d = []
        
        # Process each road mark
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            # Extract 2D points from mask
            points_2d = self.extract_points(mask[0])
            
            if len(points_2d) == 0:
                continue
            
            # Project to 3D
            points_3d = self.project_to_3d(points_2d, depth_map)
            
            # Add to road marks list
            road_marks_3d.append({
                "type": self.class_names[label],
                "score": float(score),
                "points_2d": points_2d.tolist(),
                "points_3d": points_3d,
                "box": box.tolist()
            })
        
        return road_marks_3d, depth_map
    
    def visualize_road_marks(self, frame, road_marks_3d, depth_map):
        """Visualize detected road marks with depth information"""
        vis_frame = frame.copy()
        
        # Normalize depth map for visualization
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_vis = (norm_depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
        
        # Create overlay for masks
        overlay = frame.copy()
        
        # Draw road marks
        for mark in road_marks_3d:
            # Get label and color
            label_name = mark["type"]
            label_idx = [k for k, v in self.class_names.items() if v == label_name][0]
            color = self.colors[label_idx]
            
            # Draw bounding box
            box = mark["box"]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label and score
            text = f"{label_name}: {mark['score']:.2f}"
            cv2.putText(vis_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw mask
            points = np.array(mark["points_2d"], dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
        
        # Blend the mask overlay with the original frame
        cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)
        
        # Combine original frame and depth visualization
        combined = np.hstack((vis_frame, depth_color))
        
        return combined
    
    # def process_video(self, video_path, output_dir):
    #     """Process video for road mark detection with 3D coordinates"""
    #     # Create output directory
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Open video
    #     cap = cv2.VideoCapture(video_path)
        
    #     if not cap.isOpened():
    #         print(f"Error: Could not open video {video_path}")
    #         return
        
    #     # Get video properties
    #     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = int(cap.get(cv2.CAP_PROP_FPS))
        
    #     # Define the codec and create VideoWriter object for visualization
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     vis_path = os.path.join(output_dir, "RM_11_visualization.mp4")
    #     out = cv2.VideoWriter(vis_path, fourcc, fps, (frame_width*2, frame_height))
        
    #     frame_count = 0
        
    #     while cap.isOpened():
    #         ret, frame = cap.read()
            
    #         if not ret:
    #             break
            
    #         # Process frame
    #         road_marks_3d, depth_map = self.process_frame(frame)
            
    #         # Save results as JSON
    #         result = {
    #             "frame": frame_count,
    #             "road_marks": road_marks_3d,
    #             "camera_matrix": self.camera_matrix.tolist()
    #         }
            
    #         with open(os.path.join(output_dir, f"frame_{frame_count:06d}.json"), "w") as f:
    #             json.dump(result, f)
            
    #         # Visualize
    #         vis_frame = self.visualize_road_marks(frame, road_marks_3d, depth_map)
    #         out.write(vis_frame)
            
    #         # Save visualization image
    #         cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"), vis_frame)
            
    #         frame_count += 1
    #         print(f"Processed frame {frame_count}")
        
    #     # Release everything
    #     cap.release()
    #     out.release()
        
    #     print(f"Processed {frame_count} frames. Results saved to {output_dir}")
    
    def process_video(self, video_path, output_dir):
        """Process video for road mark detection with 3D coordinates"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define the codec and create VideoWriter object for visualization
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vis_path = os.path.join(output_dir, "RM_11_visualization.mp4")
        out = cv2.VideoWriter(vis_path, fourcc, fps, (frame_width*2, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            road_marks_3d, depth_map = self.process_frame(frame)
            
            # Save results as JSON for all frames, even if no road marks are detected
            result = {
                "frame": frame_count,
                "road_marks": road_marks_3d,  # This will be an empty list if no road marks are detected
                "camera_matrix": self.camera_matrix.tolist()
            }
            
            # Save JSON for every frame
            with open(os.path.join(output_dir, f"frame_{frame_count:06d}.json"), "w") as f:
                json.dump(result, f)
            
            # Visualize
            vis_frame = self.visualize_road_marks(frame, road_marks_3d, depth_map)
            out.write(vis_frame)
            
            # Save visualization image
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"), vis_frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}")
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"Processed {frame_count} frames. Results saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Camera matrix
    camera_matrix = np.array([
        [1594.7, 0, 655.2961],
        [0, 1607.7, 414.3627],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Paths
    model_path = r"C:\Users\pavan\Documents\CV_P3\output_road_marking_model\road_marking_model_final.pth"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene7\Undist\2023-03-03_11-21-43-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene3\Undist\2023-02-14_11-49-54-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene11\Undist\2023-03-11_17-19-53-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene9\Undist\2023-03-04_17-20-36-front_undistort.mp4"
    # video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene6\Undist\2023-03-03_15-31-56-front_undistort.mp4"
    video_path = r"C:\Users\pavan\Documents\CV_P3\P3Data\Sequences\scene1\Undist\2023-02-14_11-04-07-front_undistort.mp4"
    output_dir = r"C:\Users\pavan\Documents\CV_P3\output\3D_RoadMarks\scene6"   
    # Initialize detector
    detector = RoadMarkDetector3D(model_path, camera_matrix)
    
    # Process video
    detector.process_video(video_path, output_dir)
