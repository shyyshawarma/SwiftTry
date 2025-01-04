import os
import argparse
import random
from datetime import datetime
from os.path import join
import gradio as gr
import numpy as np
from tqdm import tqdm
import torch
from einops import rearrange
import glob
from diffusers import AutoencoderKL, DDIMScheduler, DDIMInverseScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon_video_random import TryOnVideoPipeline
from src.utils.util import read_frames, save_videos_grid, convert_videos_to_pil
from torchvision.utils import flow_to_image



def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    mask_np = mask_np[:, :, None]
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def repaint_video(video, mask_video, result_video):
    repaint_result_video = []
    for image, mask, result in zip(video, mask_video, result_video):
        repaint_result_video.append(transforms.ToTensor()(repaint(image, mask, result)))

    return repaint_result_video


class TryOnController:
    def __init__(
        self,
        config_path="./configs/prompts/tryon.yaml",
        weight_dtype=torch.float16,
    ):
        # Read pretrained weights path from config
        self.config = OmegaConf.load(config_path)
        self.pipeline = None
        self.weight_dtype = weight_dtype
    
    def tryon_video(
        self,
        ref_cloth_image,
        video_path,
        masked_video_path,
        mask_video_path,
        pose_video_path,
        clip_length,
        num_inference_steps=25,
        cfg=3.5,
        seed=42,
        repaint=False,
        save_dir="",
        overlap_value=4
    ):  
        height, width = self.config.height, self.config.width
        video_name, cloth_name = os.path.basename(video_path), os.path.basename(ref_cloth_image)
        generator = torch.cuda.manual_seed(seed)
        if isinstance(ref_cloth_image, np.ndarray):
            ref_cloth_image = Image.fromarray(ref_cloth_image)
        elif isinstance(ref_cloth_image, str):
            ref_cloth_image = Image.open(ref_cloth_image).convert("RGB")
        elif isinstance(ref_cloth_image, Image):
            ref_cloth_image = ref_cloth_image.convert("RGB")
        if self.pipeline is None:
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_vae_path,
            ).to("cuda", dtype=self.weight_dtype)
            vae.enable_slicing()
            reference_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_referencenet_model_path,
                subfolder="unet",
            ).to(dtype=self.weight_dtype, device="cuda")

            inference_config_path = self.config.inference_config
            infer_config = OmegaConf.load(inference_config_path)
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                self.config.pretrained_base_model_path,
                self.config.motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(
                    infer_config.unet_additional_kwargs
                ),
            ).to(dtype=self.weight_dtype, device="cuda")
            
            pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=self.weight_dtype, device="cuda"
            )
            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_path
            ).to(dtype=self.weight_dtype, device="cuda")
            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)
            # inverse_scheduler = DDIMInverseScheduler(**sched_kwargs)

            # load pretrained weights
            denoising_unet.load_state_dict(
                torch.load(self.config.denoising_unet_path, map_location="cpu"),
                strict=False,
            )
            reference_unet.load_state_dict(
                torch.load(self.config.reference_unet_path, map_location="cpu"),
            )
            pose_guider.load_state_dict(
                torch.load(self.config.pose_guider_path, map_location="cpu"),
            )
            pipe = TryOnVideoPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )
            pipe = pipe.to("cuda", dtype=self.weight_dtype)
            self.pipeline = pipe

        image_list = [image_pil.resize((width, height)) for image_pil in read_frames(video_path)]
        
        
        masked_image_list = [masked_image_pil.resize((width, height)) for masked_image_pil in read_frames(masked_video_path)]
        mask_list = [mask_pil.convert('L').resize((width, height)) for mask_pil in read_frames(mask_video_path)]
        pose_list = [pose_image_pil.resize((width, height)) for pose_image_pil in read_frames(pose_video_path)]
        
        clip_length = clip_length if clip_length < len(masked_image_list) else len(masked_image_list)
        video = self.pipeline(
            ref_cloth_image,
            image_list[:clip_length],
            masked_image_list[:clip_length],
            mask_list[:clip_length],
            pose_list[:clip_length],
            width=width,
            height=height,
            video_length=clip_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=generator,
            context_frames=16,
            context_overlap=overlap_value,
        ).videos
        pixel_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        # Concat it with original video tensor
        image_tensor_list = []
        cloth_tensor_list = []
        images = image_list[:clip_length]
        for image_pil in images:
            image_tensor_list.append(pixel_transform(image_pil))
            cloth_tensor_list.append(pixel_transform(ref_cloth_image))
        image_tensor = torch.stack(image_tensor_list, dim=0)  # (f, c, h, w)
        image_tensor = image_tensor.transpose(0, 1)
        image_tensor = image_tensor.unsqueeze(0)

        cloth_tensor = torch.stack(cloth_tensor_list, dim=0)  # (f, c, h, w)
        cloth_tensor = cloth_tensor.transpose(0, 1)
        cloth_tensor = cloth_tensor.unsqueeze(0)
        if repaint:
            result_list = convert_videos_to_pil(video)
            result_video = repaint_video(image_list, mask_list, result_list)
            result_video_tensor = torch.stack(result_video, dim=0)
            video = rearrange(result_video_tensor, "t c h w -> c t h w").unsqueeze(0)

        video_full = torch.cat([image_tensor, cloth_tensor, video], dim=0)
        


        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        video_full_out_dir = os.path.join(save_dir, 'canvas')
        out_dir = os.path.join(save_dir, 'result')
        video_full_out_path = os.path.join(video_full_out_dir, f"{video_name}-{cloth_name}.mp4")
        out_path = os.path.join(out_dir, f"{video_name}-{cloth_name}.mp4")
        save_videos_grid(video_full, video_full_out_path, n_rows=3)
        save_videos_grid(video, out_path, n_rows=1)
        torch.cuda.empty_cache()
        return out_path





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/tryon_video_vivid.yaml")
    parser.add_argument("--data_dir", type=str, default="/root/dataset/ViViD/ViViD/upper_body")
    parser.add_argument("--test_pairs", type=str, default="/root/dataset/ViViD/ViViD/upper_body/test_pairs.txt")
    args = parser.parse_args()
    controller = TryOnController(args.config)
    save_dir = "./output_result/test_vivid_on_vivid_pairs_repaint"
    # read txt file
    with open(args.test_pairs, 'r') as f:
        test_pairs = f.read()
    test_pairs = test_pairs.split('\n')
    test_pairs = [pair.split() for pair in test_pairs if pair]
    for video_id, cloth_id in tqdm(test_pairs):
        cloth_path = join(args.data_dir, "images", cloth_id)
        cloth_name = os.path.basename(cloth_path)
        if os.path.exists(os.path.join(save_dir, 'canvas', f"{video_id}-{cloth_name}.mp4")):
            print(f"{video_id}-{cloth_id}.jpg.mp4 already processed...")
            continue
        controller.tryon_video(
            ref_cloth_image=cloth_path,
            video_path=join(args.data_dir, "videos", f"{video_id}"),
            masked_video_path=join(args.data_dir, "agnostic", f"{video_id}"),
            mask_video_path=join(args.data_dir, "agnostic_mask", f"{video_id}"),
            pose_video_path=join(args.data_dir, "dwpose", f"{video_id}"),
            clip_length=10000,
            repaint=True,
            save_dir=save_dir,
            overlap_value=4
        )
