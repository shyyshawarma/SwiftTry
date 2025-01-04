import sys
sys.path.append('/root/Projects/Moore-AnimateAnyone')
from src.dwpose import DWposeDetector
import os
from PIL import Image
from pathlib import Path

# from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise ValueError(f"Path: {args.image_path} not exists")

    out_path = args.image_path[:-4] + "_kps.png"

    detector = DWposeDetector()
    detector = detector.to(f"cuda")

    image = Image.open(args.image_path)
    size = image.size
    result, score = detector(image)
    score = np.mean(score, axis=-1)
    result = result.resize(size)
    result.save(out_path)
