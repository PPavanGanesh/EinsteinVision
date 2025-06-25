# 🚗 Einstein Vision: A Safer Autonomy – Tesla-Inspired 3D Traffic Scene Visualization

This project presents a Tesla-style visualization pipeline that bridges deep learning-based perception with intuitive 3D rendering using Blender. Designed to aid human understanding and trust in autonomous systems, the pipeline detects and interprets real-world traffic scenarios and visualizes them from the ego-vehicle's perspective.

> 🔗 [📹 Pitch Video](https://drive.google.com/drive/folders/1iuleoQ68BJFc04H3BPSZ7smhzd8b1pa6?usp=sharing)

---

## 📌 Features

### ✅ **Phase 1: Basic Perception**

* Lane detection & classification (traditional + Mask R-CNN)
* Vehicle & pedestrian detection (YOLOv8x)
* Traffic light and road sign recognition
* Depth-based 3D projections of all entities

### 🚧 **Phase 2: Advanced Enhancements**

* Depth estimation using MiDaS
* Fine-grained vehicle classification & orientation
* Road markings: arrows, pedestrian crossings, speed signs
* Pedestrian pose estimation in 3D
* Detection of urban furniture (poles, cones, cylinders)

### 🧠 **Phase 3: Cognitive Abilities**

* Brake light and indicator detection
* Parked vs. moving vehicle classification (RAFT + optical flow)
* **Extra Credit**:

  * Speed bump detection (custom YOLOv8x)
  * Collision prediction & risk visualization

---

## 📂 Repository Structure

```
YourDirectoryID_p3/
├── Code/                          # Implementation scripts and visualization logic
│   ├── 3DLane_Detection.py        # Renders lane geometry in 3D
│   ├── 3D_Traffic_Detection.py    # Renders vehicles, pedestrians, lights, etc.
│   ├── 3D_RoadMarks.py            # Visualizes arrows, crossings, etc.
│   ├── 3D_Speed_breaker.py        # Speed bump visualization
│   ├── 3d_speed_limit_detect.py   # Speed limit detection and JSON export
│   ├── this_detection_never_ends.py  # Main orchestrator – integrates and renders all components
│   └── video.py                   # Video generation (if included)
├── Videos/                        # Final rendered Blender videos
│   ├── OutputVisualizationVideoSeq1.mp4
│   └── ...
├── Report.pdf                     # Technical documentation
├── ProductPitchVideo.mp4         # Local video pitch
└── README.md                     # This file
```

---

## 📷 Sample Outputs

Visualizations include:

* 3D ego-vehicle view of the traffic scene
* Annotated elements: moving/parked vehicles, brake lights, traffic arrows, pedestrian poses
* Danger zones: collision predictions (highlighted in red)
* Real-time signal interpretations

---

## 🛠 How to Run

1. **Ensure Blender is installed (≥ v3.0)**
2. Place all JSON detection outputs in the designated input directory
3. Run the master script:

   ```bash
   blender --background --python this_detection_never_ends.py
   ```
4. Output MP4 videos will be saved in the `Videos/` folder.

---

## 📑 Technical Highlights

* **YOLOv8x**: for object, vehicle, pedestrian, and road sign detection
* **Mask R-CNN**: for lane and road marking segmentation
* **MiDaS**: transformer-based monocular depth estimation
* **Blender API**: for importing, animating, and rendering scene objects
* **Optical Flow (RAFT)**: for vehicle movement classification

---

## ❗ Known Limitations

* Less robust under extreme lighting or weather conditions
* Depth estimation accuracy decreases with distance
* Pose rendering glitches in Blender due to keyframe/armature issues

---

## 🔮 Future Improvements

* Sensor fusion with LiDAR for better depth perception
* Improved pose rendering engine
* Real-time implementation with ROS2 pipeline
* Automated Blender rendering dashboard for batch processing

---

## 👨‍💻 Authors

* **Pavan Ganesh Pabbineedi** – [ppabbineedi@wpi.edu](mailto:ppabbineedi@wpi.edu)
* **Manideep Duggi** – [mduggi@wpi.edu](mailto:mduggi@wpi.edu)
