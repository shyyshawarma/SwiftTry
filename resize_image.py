from PIL import Image
import argparse
from tqdm import tqdm
def resize_image(input_path, output_path, width, height):
    try:
        # Load the video clip
        image = Image.open(input_path)
        image = image.resize((width, height))
        image.save(output_path)
        print("Image resized successfully.")
    except Exception as e:
        print("Error:", e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a video file.")
    parser.add_argument("--input_path", help="Path to the input video file")
    parser.add_argument("--output_path", help="Path to save the resized video file")
    parser.add_argument("--width", type=int, default=640, help="Width of the resized video (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Height of the resized video (default: 480)")
    args = parser.parse_args()

    resize_image(args.input_path, args.output_path, args.width, args.height)
