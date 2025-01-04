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
from einops import   repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from src.models.flow_guider import FlowEncoder
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon_video_random_flow import TryOnVideoPipeline
from src.utils.util import read_frames, save_videos_grid
from tools.extract_optical_flow_from_vid import FlowExtractor, create_overlapping_chunks
from torchvision.utils import flow_to_image

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
        self.flow_extractor = None
    
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
        do_center_crop=False,
        do_resize=True,
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
            self.flow_extractor = FlowExtractor('cuda')
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
            flow_guider = FlowEncoder(guidance_embedding_channels=320, guidance_input_channels=3).to(
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
            flow_guider.load_state_dict(
                torch.load(self.config.flow_guider_path, map_location="cpu"),
            )
            pipe = TryOnVideoPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
                flow_guider=flow_guider
            )
            pipe = pipe.to("cuda", dtype=self.weight_dtype)
            self.pipeline = pipe

        image_list = [image_pil.resize((width, height)) for image_pil in read_frames(video_path)]
        
        
        masked_image_list = [masked_image_pil.resize((width, height)) for masked_image_pil in read_frames(masked_video_path)]
        mask_list = [mask_pil.convert('L').resize((width, height)) for mask_pil in read_frames(mask_video_path)]
        pose_list = [pose_image_pil.resize((width, height)) for pose_image_pil in read_frames(pose_video_path)]
        
        clip_length = clip_length if clip_length < len(masked_image_list) else len(masked_image_list)
        batches = create_overlapping_chunks(image_list[:clip_length+1], chunk_size=64)
        predicted_flows = []
        for batch in batches:
            if len(batch) == 1:
                batch.insert(0, batch[0] - 1)
            predicted_flows.append(self.flow_extractor.extract_optical_flows([image_list[b] for b in batch], 'cuda'))
        # dummy padding
        predicted_flows = torch.cat(predicted_flows + [predicted_flows[-1]], dim=0)
        predicted_flows = flow_to_image(predicted_flows)
        predicted_flows = predicted_flows / 127.5 - 1
        predicted_flows = rearrange(predicted_flows.unsqueeze(0), "b f c h w -> b c f h w").to(device='cuda', dtype=self.weight_dtype)
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
            context_frames=12,
            predicted_flows=predicted_flows
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
            mask_tensor = torch.stack([pixel_transform(mask) for mask in mask_list], dim=0) # (f, c, h, w)
            mask_tensor = mask_tensor.transpose(0, 1)
            mask_tensor = mask_tensor.unsqueeze(0)
            video = (1 - mask_tensor) * image_tensor + mask_tensor * video

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
    parser.add_argument("--config", type=str, default="./configs/prompts/tryon_video_flow.yaml")
    parser.add_argument("--data_dir", type=str, default="/root/dataset/VVT_video")
    parser.add_argument("--test_pairs", type=str, default="/root/dataset/VVT_video/test_pairs_unpaired.txt")
    args = parser.parse_args()
    controller = TryOnController(args.config)
    save_dir = "./output_result/test_vvt_vvt_flow_512_unpaired(oot)_newpairs_rectified"
    # read txt file
    with open(args.test_pairs, 'r') as f:
        test_pairs = f.read()
    test_pairs = test_pairs.split('\n')
    test_pairs = [pair.split() for pair in test_pairs if pair]
    for video_id, cloth_id in tqdm(test_pairs):
        cloth_path = glob.glob(join(args.data_dir, "lip_clothes_person", cloth_id, "*.jpg"))[0]
        cloth_name = os.path.basename(cloth_path)
        if os.path.exists(os.path.join(save_dir, 'canvas', f"{video_id}.mp4-{cloth_name}.mp4")):
            print(f"{video_id}.mp4-{cloth_id}.jpg.mp4 already processed...")
            continue
        controller.tryon_video(
            ref_cloth_image=cloth_path,
            video_path=join(args.data_dir, "test_frames", f"{video_id}.mp4"),
            masked_video_path=join(args.data_dir, "test_frames_masked", f"{video_id}.mp4"),
            mask_video_path=join(args.data_dir, "test_frames_mask", f"{video_id}.mp4"),
            pose_video_path=join(args.data_dir, "test_frames_dwpose_new", f"{video_id}.mp4"),
            clip_length=10000,
            repaint=False,
            save_dir=save_dir
        )
