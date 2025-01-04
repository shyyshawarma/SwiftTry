import os
import argparse
import random
from datetime import datetime
from os.path import join
import gradio as gr
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon import TryOnPipeline
import glob
from src.utils.util import read_frames, save_videos_from_pil


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
    ):  
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
        src_image = Image.open(image_path).resize((width, height)) if isinstance(image_path, str) else image_path.resize((width, height))
        masked_image = Image.open(masked_image_path).resize((width, height)) if isinstance(masked_image_path, str) else masked_image_path.resize((width, height))
        mask = Image.open(mask_path).convert('L').resize((width, height)) if isinstance(mask_path, str) else mask_path.resize((width, height))
        pose = Image.open(pose_path).resize((width, height)) if isinstance(pose_path, str) else pose_path.resize((width, height))
        
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

        
        res_image_pil = Image.fromarray(image.astype(np.uint8))
        # Save ref_image, src_image and the generated_image
        w, h = res_image_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        ref_cloth_image_pil = ref_cloth_image.resize((w, h))
        orig_image_pil = src_image.resize((w, h))

        canvas.paste(orig_image_pil, (0, 0))
        canvas.paste(ref_cloth_image_pil, (w, 0))
        canvas.paste(res_image_pil, (w * 2, 0))
        torch.cuda.empty_cache()
        return canvas, res_image_pil


    def tryon_image_batch(
        self,
        ref_cloth_images,
        image_paths,
        masked_image_paths,
        mask_paths,
        pose_paths,
        save_dir,
        num_inference_steps=25,
        cfg=3.5,
        seed=42,
    ):  
        image_names = [os.path.basename(image_path) for image_path in image_paths]
        cloth_names = [os.path.basename(cloth_path) for cloth_path in ref_cloth_images]
        height, width = self.config.height, self.config.width
        generator = torch.cuda.manual_seed(seed)
        ref_cloth_images = [Image.open(ref_cloth_image).convert('RGB') for ref_cloth_image in ref_cloth_images]

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
        src_images = [Image.open(image_path).resize((width, height)) for image_path in image_paths]
        masked_images = [Image.open(masked_image_path).resize((width, height)) for masked_image_path in masked_image_paths]
        masks = [Image.open(mask_path).convert('L').resize((width, height)) for mask_path in mask_paths]
        poses = [Image.open(pose_path).resize((width, height)) for pose_path in pose_paths]
        
        images = self.pipeline(
            ref_cloth_images,
            src_images,
            masked_images,
            masks,
            poses,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=generator,
        ).images
        
        images = images.squeeze(2).permute(0, 2, 3, 1).cpu().numpy()  # (b, 3, 512, 512)
        images = (images * 255).astype(np.uint8)

        res_image_pils = [Image.fromarray(images[i].astype(np.uint8)) for i in range(images.shape[0])]
        for idx in range(len(res_image_pils)):
            # Save ref_image, src_image and the generated_image
            res_image_pil = res_image_pils[idx]
            ref_cloth_image_pil = ref_cloth_images[idx]
            image_path = image_paths[idx]

            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * 3, h), "white")
            ref_cloth_image_pil = ref_cloth_image_pil.resize((w, h))
            orig_image_pil = Image.open(image_path).resize((w, h))

            canvas.paste(orig_image_pil, (0, 0))
            canvas.paste(ref_cloth_image_pil, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            result_dir = os.path.join(save_dir, 'result')
            canvas_dir = os.path.join(save_dir, 'canvas')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)
            if not os.path.exists(canvas_dir):
                os.makedirs(canvas_dir, exist_ok=True)
            out_result_path = os.path.join(result_dir, f"{image_names[idx]}_{cloth_names[idx]}.png")
            out_canvas_path = os.path.join(canvas_dir, f"{image_names[idx]}_{cloth_names[idx]}.png")
            res_image_pil.save(out_result_path)
            canvas.save(out_canvas_path)
        torch.cuda.empty_cache()
        return True



def batchify_pairs(pairs, batch_size=16):
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    return batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/tryon.yaml")
    parser.add_argument("--data_dir", type=str, default="/root/Projects/Moore-AnimateAnyone/examples")
    parser.add_argument("--video_name", type=str, default="00333") 
    parser.add_argument("--cloth_name", type=str, default="00055_00")
    args = parser.parse_args()

    controller = TryOnController(args.config)
    ref_cloth_image = glob.glob(join(args.data_dir, 'clothes', f'{args.cloth_name}.jpg'))[0]
    image_paths = read_frames(join(args.data_dir, 'videos', args.video_name, f'{args.video_name}_resized.mp4'))
    masked_image_paths = read_frames(join(args.data_dir, 'videos', args.video_name, f'{args.video_name}_resized_masked.mp4'))
    mask_paths = read_frames(join(args.data_dir, 'videos', args.video_name, f'{args.video_name}_resized_mask.mp4'))
    pose_paths = read_frames(join(args.data_dir, 'videos', args.video_name, f'{args.video_name}_resized_kps.mp4'))
    

    save_dir = "./examples/results"
    video_full_out_dir = os.path.join(save_dir, 'canvas')
    out_dir = os.path.join(save_dir, 'result')
    video_full_out_path = os.path.join(video_full_out_dir, f"{args.video_name}-{args.cloth_name}_s1_random")
    out_path = os.path.join(out_dir, f"{args.video_name}-{args.cloth_name}_s1_random")

    if not os.path.exists(video_full_out_path):
        os.makedirs(video_full_out_path, exist_ok=True)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    canvas_list = []
    result_list = []
    no_frames = 0
    for image, masked, mask, pose in tqdm(zip(image_paths, masked_image_paths, mask_paths, pose_paths)):
        seed = random.randint(0, 10000)
        canvas, result = controller.tryon_image(ref_cloth_image, image, masked, mask, pose, seed=seed)
        canvas.save(join(video_full_out_path, "{:04d}.png".format(no_frames)))
        result.save(join(out_path, "{:04d}.png".format(no_frames)))
        canvas_list.append(canvas)
        result_list.append(result)
        if no_frames == 100:
            break
        no_frames += 1
