import cv2
import os
import glob
import re

def create_video_with_opencv(input_dir, output_file, fps=3):
    """Create a video from a sequence of images using OpenCV"""
    # Get all PNG files in the directory
    image_files = glob.glob(os.path.join(input_dir, 'frame_*.png'))
    
    # Sort files numerically
    image_files.sort(key=lambda x: int(re.search(r'frame_(\d+)\.png', x).group(1)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return False
    
    # Read the first image to get dimensions
    first_img = cv2.imread(image_files[0])
    height, width, layers = first_img.shape
    
    # Create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add each image to the video
    total_frames = len(image_files)
    for i, img_file in enumerate(image_files):
        print(f"Adding {img_file} to video ({i+1}/{total_frames})")
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        video.write(img)
    
    # Release the video writer
    video.release()
    print(f"Video created at {output_file}")
    return True

# Example usage
input_directory =r"C:\Users\pavan\Documents\CV_P3\output\rendered_frames_2_Final"
output_video = r"C:\Users\pavan\Documents\CV_P3\output\Rendered_Video\2Final.mp4"
create_video_with_opencv(input_directory, output_video, fps=7)#Adjust fps as needed
