from src.utils.util import read_frames, save_videos_from_pil
import argparse
from tqdm import tqdm
from PIL import Image

def resize_video(input_path, output_path, width, height):
    try:
        # Load the video clip
        video_frames = read_frames(input_path)
        new_video_frames = []
        # Resize the video clip
        for frame in tqdm(video_frames):
            new_video_frames.append(frame.resize((width, height), Image.BICUBIC))
        
        save_videos_from_pil(new_video_frames, output_path)
        
        print("Video resized successfully.")
    except Exception as e:
        print("Error:", e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a video file.")
    parser.add_argument("--input_path", help="Path to the input video file")
    parser.add_argument("--output_path", help="Path to save the resized video file")
    parser.add_argument("--width", type=int, default=640, help="Width of the resized video (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Height of the resized video (default: 480)")
    args = parser.parse_args()

    resize_video(args.input_path, args.output_path, args.width, args.height)
