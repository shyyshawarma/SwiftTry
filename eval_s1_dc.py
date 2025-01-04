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
from src.data.dataset import DressCodeDataset



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
        save_dir='',
        im_name='',
        c_name='',
        repaint=False
    ):  
        image_name = im_name
        cloth_name = c_name
        height, width = self.config.height, self.config.width
        generator = torch.cuda.manual_seed(seed)
        if isinstance(ref_cloth_image, np.ndarray):
            ref_cloth_image = Image.fromarray(ref_cloth_image)
        elif isinstance(ref_cloth_image, str):
            ref_cloth_image = Image.open(ref_cloth_image).convert("RGB")
        elif isinstance(ref_cloth_image, Image.Image):
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
        mask = Image.open(mask_path).convert('L').resize((width, height)) if isinstance(mask_path, str) else mask_path.convert('L').resize((width, height))
        pose = Image.open(pose_path).resize((width, height))if isinstance(pose_path, str) else pose_path.resize((width, height))
        
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
        orig_image_pil = src_image

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
        out_result_path = os.path.join(result_dir, f"{image_name}_{cloth_name}.png")
        out_canvas_path = os.path.join(canvas_dir, f"{image_name}_{cloth_name}.png")
        res_image_pil.save(out_result_path)
        canvas.save(out_canvas_path)
        torch.cuda.empty_cache()
        return canvas


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/tryon.yaml")
    parser.add_argument("--data_dir", type=str, default="/root/dataset/dresscode")
    parser.add_argument("--type", type=str, default="unpaired")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    dc_dataset = DressCodeDataset(
        args.data_dir,
        phase='test',
        order=args.type,
        category=['upper_body'],
        size=(1024, 768)
    )
    test_dataloader = torch.utils.data.DataLoader(
        dc_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8
    )
    save_dir = f"./output_result/test_dresscode_1024x768_ft_upblocks_unpaired_200k"
    print(save_dir)
    controller = TryOnController(args.config)
    trans = transforms.ToPILImage(mode='RGB')
    trans_m = transforms.ToPILImage(mode='L')
    for idx, batch in tqdm(enumerate(test_dataloader)):
        images = trans(batch['image'][0]/2 + .5)
        masked_images = trans(batch['m'][0]/2 + .5)
        mask_images = trans_m(batch['mask'][0]*255)
        ref_cloth_images = trans(batch["cloth"][0]/2 + .5)
        pose_images = trans(batch["skeleton"][0]/2 + .5)
        im_name, c_name = batch['im_name'][0], batch['c_name'][0]
        if os.path.exists(join(save_dir, 'canvas', f'{im_name}_{c_name}.png')):
            print(f"Skipping {im_name} due to processed...")
            continue
        controller.tryon_image(
            ref_cloth_images,
            images,
            masked_images,
            mask_images,
            pose_images,
            save_dir=save_dir,
            im_name=im_name,
            c_name=c_name
        )
