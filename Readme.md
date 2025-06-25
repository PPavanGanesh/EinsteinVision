## 3D Traffic Scene Blender Visualization
This project provides a comprehensive pipeline for 3D visualization of traffic scenes using Blender and Python. It processes detection and tracking results (vehicles, lanes, road marks, speed bumps, speed limit signs, etc.) from JSON files and renders them in a photorealistic Blender environment.

## Project code Structure
3D_RoadMarks.py – Renders road markings in 3D using Blender.

3D_Speed_breaker.py – Handles 3D placement and rendering of speed bumps.

3d_speed_limit_detect.py – Detects and saves speed limit sign data as JSON.

3D_SpeedBump.py – Loads and renders speed bump models in Blender.

3D_Traffic_Detection.py – Main code for 3D object (vehicle, pedestrian, traffic light, etc.) placement and orientation.

3DLane_Detection.py – Visualizes detected lane lines in 3D.

this_detection_never_ends.py – The main orchestrator: loads all detection JSONs, manages Blender rendering, and integrates all scene elements.

video.py – (If present) Handles video input/output or post-processing.

## Repository Structure
YourDirectoryID_p3.zip
├── Code/ # Implementation scripts and modules
├── Videos/ # Processed visualization outputs
│ ├── OutputVisualizationVideoSeq1.mp4
│ ├── ...
│ └── OutputVisualizationVideoSeq13.mp4
├── Report.pdf # Technical documentation
├── ProductPitchVideo.mp4 # Local copy of pitch video
└── README.md # This documentation

## Key Resources
- **Pitch Video**: [Google Drive Link](https://drive.google.com/drive/folders/1iuleoQ68BJFc04H3BPSZ7smhzd8b1pa6?usp=sharing)
- **Report**: See `Report.pdf` for technical specifications and implementation details
- **Visualizations**: Sample outputs in `Videos/` directory


## Usage
1. **Code Execution**: Run main scripts from `Code/` directory
2. **Output Generation**: Processed videos are in  `Videos/`


