import torchvision.transforms.functional as F
import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import ToTensor
import glob
from os.path import join, exists, basename
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from einops import rearrange

def preprocess(img1_batch, img2_batch, transforms, size=[256, 192]):
    img1_batch = F.resize(img1_batch, size=size, antialias=False)
    img2_batch = F.resize(img2_batch, size=size, antialias=False)
    return transforms(img1_batch, img2_batch)

def create_overlapping_chunks(image_list, chunk_size, overlap=1):
    chunks = []
    i = 0
    while i < len(image_list):
        chunk = image_list[i:i + chunk_size]
        chunks.append(list(range(i, min(i + chunk_size, len(image_list)))))
        i += chunk_size - overlap
    return chunks





class FlowExtractor:
    def __init__(self, device):
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_large(weights=weights, progress=False).to(device)
        self.model = self.model.eval()

    @torch.no_grad()
    def extract_optical_flows(self, pil_list, device, normalize=True):
        width, height = pil_list[0].size
        frames = torch.stack([ToTensor()(pil) for pil in pil_list], dim=0)
        frames = (frames * 255).to(torch.uint8)
        width, height = frames.shape[-1], frames.shape[-2]
        clips = list(range(len(frames)))
        current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], self.transforms, size=[height, width])
        list_of_flows = self.model(current_frames.to(device), next_frames.to(device))
        predicted_flows = list_of_flows[-1]
        flow_size = torch.tensor([height, width], dtype=torch.float32).to(device)
        if normalize:
            predicted_flows = predicted_flows / flow_size.view(1, 2, 1, 1)
        return predicted_flows

    @torch.no_grad()
    def extract_optical_flows_tensor(self, video, device='cuda', normalize=True, backward=False):
        width, height = video.shape[-2], video.shape[-1]
        clips = list(range(video.shape[2]))
        batch_size = video.shape[0]
        if batch_size != 1:
            raise NotImplementedError
        video_ = rearrange(video, "b c f h w -> (b f) c h w")
        if backward:
            current_frames, next_frames = video_[clips[1:]], video_[clips[:-1]]
        else:
            current_frames, next_frames = video_[clips[:-1]], video_[clips[1:]]
        list_of_flows = self.model(current_frames.to(device), next_frames.to(device))
        predicted_flows = list_of_flows[-1]
        flow_size = torch.tensor([height, width], dtype=torch.float32).to(device)
        if normalize:
            predicted_flows = predicted_flows / flow_size.view(1, 2, 1, 1)
        return predicted_flows
        



if __name__ == '__main__':
    flow_extractor = FlowExtractor('cuda')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument(
        "--save_dir", type=str, help="Path to save extracted pose videos"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size"
    )
    args = parser.parse_args()

    video_folders = glob.glob(join(args.dataset_root, "lip_train_frames", "*"))

    for folder in tqdm(video_folders):
        all_frames = sorted(glob.glob(join(folder, "*.png")))
        if not exists(join(args.save_dir, basename(folder))):
            os.makedirs(join(args.save_dir, basename(folder)), exist_ok=True)
        
        pil_list = []
        for frame in all_frames:
            with Image.open(frame) as img:
                pil_list.append(img.resize((384, 512)).copy())
        batches = create_overlapping_chunks(pil_list, chunk_size=args.batch_size)
        for batch in batches:
            if len(batch) == 1:
                batch.insert(0, batch[0] - 1)
            predicted_flows = flow_extractor.extract_optical_flows([pil_list[b] for b in batch], 'cuda')
            predicted_flows = predicted_flows.cpu().numpy() # ([b, 2, h, w])
            for idx in range(predicted_flows.shape[0]):
                np.save(join(args.save_dir, basename(folder), f"{batch[idx]:05d}.npy"), predicted_flows[idx])