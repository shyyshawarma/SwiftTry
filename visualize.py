import os
import argparse
import random
from datetime import datetime
from os.path import join
import gradio as gr
import numpy as np
from tqdm import tqdm
import torch
import glob
from diffusers import AutoencoderKL, DDIMScheduler, DDIMInverseScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon_video_debug import TryOnVideoPipeline
from src.utils.util import read_frames, save_videos_grid


def concatenate_videos(video_list, sampling_steps):
    """
    Concatenate a list of videos along the first axis, preserving intervals from each video.

    Args:
    - video_list : [N, C, F, H, W].
    - interval_length (int): Length of interval to preserve from each video.

    Returns:
    - final_video (numpy.ndarray): Concatenated video with shape [N_total, B, C, F, H, W].
    """
    interval_length = video_list.shape[2] // sampling_steps
    print(interval_length)
    final_video_frames = []
    for i, video in enumerate(video_list):
        start_frame = i * interval_length
        end_frame = start_frame + interval_length
        interval_frames = video[:, start_frame:end_frame, ...]
        final_video_frames.append(interval_frames)
    
    # Stack the extracted frames along the first axis
    final_video = np.concatenate(final_video_frames, axis=1)
    return final_video[None, ...]



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
        save_dir=""
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
            inverse_scheduler = DDIMInverseScheduler(**sched_kwargs)

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
        # self.pipeline.init_filter(
        #     width               = width,
        #     height              = height,
        #     video_length        = clip_length,
        #     filter_params       = self.config.filter_params,
        # )
        

        result = self.pipeline(
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
        )
        video, noisy_vids = result.videos, result.latents
        noisy_vids = np.concatenate(noisy_vids, axis=0)
        # visualize noisy videos
        vis_video = concatenate_videos(noisy_vids, 25)
        vis_video = torch.from_numpy(vis_video)
        pixel_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        # Concat it with original video tensor
        image_tensor_list = []
        cloth_tensor_list = []
        images = read_frames(video_path)[:clip_length]
        for image_pil in images:
            image_tensor_list.append(pixel_transform(image_pil))
            cloth_tensor_list.append(pixel_transform(ref_cloth_image))
        image_tensor = torch.stack(image_tensor_list, dim=0)  # (f, c, h, w)
        image_tensor = image_tensor.transpose(0, 1)
        image_tensor = image_tensor.unsqueeze(0)

        cloth_tensor = torch.stack(cloth_tensor_list, dim=0)  # (f, c, h, w)
        cloth_tensor = cloth_tensor.transpose(0, 1)
        cloth_tensor = cloth_tensor.unsqueeze(0)
        print(vis_video.dtype, vis_video.shape)
        breakpoint()
        video_full = torch.cat([image_tensor, cloth_tensor, video], dim=0)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        video_full_out_dir = os.path.join(save_dir, 'canvas')
        out_dir = os.path.join(save_dir, 'result')
        video_full_out_path = os.path.join(video_full_out_dir, f"{video_name}-{cloth_name}_random.mp4")
        out_path = os.path.join(out_dir, f"{video_name}-{cloth_name}_random.mp4")
        save_videos_grid(video_full, video_full_out_path, n_rows=4)
        save_videos_grid(video, out_path, n_rows=1)
        torch.cuda.empty_cache()
        return video_full


troll_vn = TryOnController("./configs/prompts/tryon_video.yaml")

ref_cloth_image = 'examples/clothes/00920_00.jpg'
video_path = 'examples/videos/00339/00339_resized.mp4'
masked_video_path = 'examples/videos/00339/00339_resized_masked.mp4'
mask_video_path = 'examples/videos/00339/00339_resized_mask.mp4'
pose_video_path = 'examples/videos/00339/00339_resized_kps.mp4'
clip_length = 300

troll_vn.tryon_video(
    ref_cloth_image,
    video_path,
    masked_video_path,
    mask_video_path,
    pose_video_path,
    clip_length,
    save_dir="examples/results"
)
# save_videos_grid(res, "examples/debug.mp4", n_rows=3)