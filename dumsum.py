import cv2
import os
from tqdm import tqdm

def create_video_from_images(images_dir, output_dir):
    # Iterate over each subdirectory in images_dir
    for subdir in tqdm(os.listdir(images_dir)):
        subdir_path = os.path.join(images_dir, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
        
        # Define output video path
        output_video_path = os.path.join(output_dir, f"{subdir}.mp4")
        
        # Get list of image files in the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
        image_files.sort()  # Sort files
        
        # Get the dimensions of the first image to create the video writer object
        first_image_path = os.path.join(subdir_path, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape
        
        # Define video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
        
        # Iterate over each image file and write to video
        for image_file in image_files:
            image_path = os.path.join(subdir_path, image_file)
            frame = cv2.imread(image_path)
            out.write(frame)
        
        # Release video writer object
        out.release()

# Define input and output directories
images_dir = '/root/dataset/TikTokDress/images_all_normalized_masked'
output_dir = '/root/dataset/TikTokDress/videos_all_normalized_masked'

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create video from images
create_video_from_images(images_dir, output_dir)
