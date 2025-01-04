import os
import argparse
import random
import glob
from datetime import datetime
from os.path import join
import gradio as gr
import numpy as np
from tqdm import tqdm
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon import TryOnPipeline
from src.utils.util import save_videos_from_pil



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

    def tryon_image(
        self,
        ref_cloth_image,
        image_path,
        masked_image_path,
        mask_path,
        pose_path,
        num_inference_steps=25,
        cfg=3.5,
        seed=42,
        repaint=True,
    ):  
        img_id, cloth_id = os.path.basename(image_path), os.path.basename(ref_cloth_image)
        height, width = self.config.height, self.config.width
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
                unet_additional_kwargs={
                    "use_motion_module": False,
                    "unet_use_temporal_attention": False,
                },
            ).to(dtype=self.weight_dtype, device="cuda")

            pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=self.weight_dtype, device="cuda"
            )

            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_path
            ).to(dtype=self.weight_dtype, device="cuda")
            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)

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
            # # perflow
            # delta_weights = UNet2DConditionModel.from_pretrained("hansyan/perflow-sd15-delta-weights", torch_dtype=torch.float16, variant="v0-1",).state_dict()

            pipe = TryOnPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )

            pipe = pipe.to("cuda", dtype=self.weight_dtype)
            self.pipeline = pipe
        src_image = Image.open(image_path).resize((width, height))
        masked_image = Image.open(masked_image_path).resize((width, height))
        mask = Image.open(mask_path).convert('L').resize((width, height))
        pose = Image.open(pose_path).resize((width, height))
        
        image = self.pipeline(
            ref_cloth_image,
            src_image,
            masked_image,
            mask,
            pose,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=generator,
        ).images
        
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        image = (image * 255).astype(np.uint8)
        if repaint:
            repaint_img = np.array(Image.open(image_path).resize((width, height))) # [0, 255]
            repaint_mask_img = np.array(mask)[..., None]
            repaint_mask_img = repaint_mask_img.astype(np.float32) / 255.0
            repaint_mask_img[repaint_mask_img < 0.5] = 0
            repaint_mask_img[repaint_mask_img >= 0.5] = 1
            image = image * repaint_mask_img + repaint_img * (1-repaint_mask_img)

        
        res_image_pil = Image.fromarray(image.astype(np.uint8))
        # Save ref_image, src_image and the generated_image
        w, h = res_image_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        ref_cloth_image_pil = ref_cloth_image.resize((w, h))
        orig_image_pil = Image.open(image_path).resize((w, h))

        canvas.paste(orig_image_pil, (0, 0))
        canvas.paste(ref_cloth_image_pil, (w, 0))
        canvas.paste(res_image_pil, (w * 2, 0))
        torch.cuda.empty_cache()
        return canvas, res_image_pil





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/tryon.yaml")
    parser.add_argument("--data_dir", type=str, default="/root/dataset/VVT")
    parser.add_argument("--test_pairs", type=str, default="/root/dataset/VVT/test_pairs_test.txt")
    args = parser.parse_args()
    controller = TryOnController(args.config)
    save_dir = "./output_result/test_stage1_vvt_unpaired"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'canvas'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'result'), exist_ok=True)
    # read txt file
    with open(args.test_pairs, 'r') as f:
        test_pairs = f.read()
    test_pairs = test_pairs.split('\n')
    test_pairs = [pair.split() for pair in test_pairs if pair][::-1]
    for video_id, cloth_id in tqdm(test_pairs):
        cloth_path = glob.glob(join(args.data_dir, "lip_clothes_person", cloth_id, "*.jpg"))[0]
        cloth_name = os.path.basename(cloth_path)
        if os.path.exists(os.path.join(save_dir, 'canvas', f"{video_id}.mp4-{cloth_name}.mp4")):
            print(f"{video_id}.mp4-{cloth_id}.jpg.mp4 already processed...")
            continue
        image_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test_frames', video_id,'*.png')))
        masked_image_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test_frames_masked', video_id, '*.png')))
        mask_image_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test_frames_mask', video_id, '*.png')))
        pose_image_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test_frames_dwpose_new', video_id, '*.png')))
        
        all_canvas = []
        all_result = []
        for image_path, masked_image_path, mask_image_path, pose_image_path in zip(image_paths, masked_image_paths, mask_image_paths, pose_image_paths):
            canvas, result = controller.tryon_image(
                ref_cloth_image=cloth_path,
                image_path=image_path,
                masked_image_path=masked_image_path,
                mask_path=mask_image_path,
                pose_path=pose_image_path,
                repaint=False,
                num_inference_steps=25
            )
            all_canvas.append(canvas.resize((192*3, 256)))
            all_result.append(result.resize((192, 256)))
        save_videos_from_pil(all_canvas, os.path.join(save_dir, "canvas", f"{video_id}.mp4-{cloth_name}.mp4"))
        save_videos_from_pil(all_result, os.path.join(save_dir, "result", f"{video_id}.mp4-{cloth_name}.mp4"))
