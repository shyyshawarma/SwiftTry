import argparse
import copy
import logging
import math
import os
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import glob
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import lpips
import cv2
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from torch.utils.data import ConcatDataset

from src.data.dataset import VVTDataset, TikTokDressDataset
from src.models_timestep.mutual_self_attention import ReferenceAttentionControl
from src.models_timestep.pose_guider import PoseGuider
from src.models_timestep.unet_2d_condition import UNet2DConditionModel
from src.models_timestep.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon_video_random_spot import TryOnVideoPipeline
from src.utils.calculate_ssim_lpips import compute_ssim_l1_psnr, compute_lpips
from src.utils.scheduler import VideoDDIMScheduler, sample_timestep
from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    save_videos_from_pil,
    seed_everything,
)
from utils import add_caption_to_video, add_caption_to_frames, decode_latents, pred_original_video


warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")



class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps[:, 0])
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

@torch.no_grad()
def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=16,
    generator=None,
):
    logger.info(f"Running validation...")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider

    if generator is None:
        generator = torch.cuda.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=torch.float16)
    
    pipe = TryOnVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    ref_cloth_image_paths = sorted(glob.glob('./configs/inference/videos_new/ref_cloth_images/*'))
    pose_video_paths = sorted(glob.glob('./configs/inference/videos_new/pose_videos/*.mp4'))
    masked_video_paths = sorted(glob.glob('./configs/inference/videos_new/masked_videos/*.mp4'))
    mask_video_paths = sorted(glob.glob('./configs/inference/videos_new/masks/*.mp4'))
    video_paths = sorted(glob.glob('./configs/inference/videos_new/videos/*.mp4'))
    results = []
    for test_case in zip(video_paths, masked_video_paths, mask_video_paths, pose_video_paths):
        video_path, masked_path, mask_path, pose_path = test_case
        for ref_cloth_path in ref_cloth_image_paths:
            ref_name = Path(ref_cloth_path).stem
            video_name = Path(video_path).stem
            ref_cloth_pil = Image.open(ref_cloth_path).convert("RGB").resize((width, height))

            # pose
            pose_list = []
            pose_tensor_list = []
            pose_images = read_frames(pose_path)

            # calculate all 
            start = 100
            end = 180
            pose_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
            for pose_image_pil in pose_images[start:end]:
                pose_tensor_list.append(pose_transform(pose_image_pil))
                pose_list.append(pose_image_pil.resize((width, height)))

            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)

            # masked images
            masked_image_list = []
            masked_tensor_list = []
            masked_images = read_frames(masked_path)
            image_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
            for masked_image_pil in masked_images[start:end]:
                masked_tensor_list.append(image_transform(masked_image_pil))
                masked_image_list.append(masked_image_pil.resize((width, height)))

            masked_tensor = torch.stack(masked_tensor_list, dim=0)  # (f, c, h, w)
            masked_tensor = masked_tensor.transpose(0, 1)

            # masks
            masks_list = []
            masks_tensor_list = []
            masks = read_frames(mask_path)
            mask_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )
            for mask_pil in masks[start:end]:
                masks_tensor_list.append(image_transform(mask_pil.convert('L')))
                masks_list.append(mask_pil.convert('L').resize((width, height)))

            masks_tensor = torch.stack(masks_tensor_list, dim=0)  # (f, 1, h, w)
            masks_tensor = masks_tensor.transpose(0, 1)

            # source images
            image_list = []
            image_tensor_list = []
            cloth_tensor_list = []
            images = read_frames(video_path)
            for image_pil in images[start:end]:
                image_list.append(image_pil.resize((width, height)))
                image_tensor_list.append(mask_transform(image_pil))
                cloth_tensor_list.append(mask_transform(ref_cloth_pil))

            pipeline_output = pipe(
                ref_cloth_pil,
                image_list,
                masked_image_list,
                masks_list,
                pose_list,
                width,
                height,
                len(image_list),
                25,
                3.5,
                generator=generator,
                context_frames=clip_length,
                context_overlap=0
            )
            video = pipeline_output.videos

            # Concat it with original video tensor
            image_tensor = torch.stack(image_tensor_list, dim=0)  # (f, c, h, w)
            image_tensor = image_tensor.transpose(0, 1)
            image_tensor = image_tensor.unsqueeze(0)
            
            # compute ssim, lpips
            ssim_score = compute_ssim_l1_psnr(rearrange(image_tensor, "b c f h w -> (b f) c h w"), rearrange(video, "b c f h w -> (b f) c h w"), mode='ssim')
            lpips_score = compute_lpips(loss_fn_vgg, rearrange(image_tensor, "b c f h w -> (b f) c h w"), rearrange(video, "b c f h w -> (b f) c h w"))

            cloth_tensor = torch.stack(cloth_tensor_list, dim=0)  # (f, c, h, w)
            cloth_tensor = cloth_tensor.transpose(0, 1)
            cloth_tensor = cloth_tensor.unsqueeze(0)
            
            video = torch.cat([image_tensor, cloth_tensor, video], dim=0) # (b, c, f, h, w)
            video = video.transpose(1, 2) # (b, f, c, h, w)
            video = rearrange(video, "b f c h w -> 1 f c h (b w)")
            if video_name != ref_name:
                lpips_score = 1.
                ssim_score = 0.
            results.append({"name": f"{ref_name}_{video_name}", "lpips": lpips_score, "ssim": ssim_score, "vid": video})

    del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return results


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="tensorboard",
        project_dir="logs",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        inference_config_path = "./configs/inference/inference_v2.yaml"
        infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = VideoDDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = VideoDDIMScheduler(**sched_kwargs)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.reference_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda", dtype=weight_dtype)

    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"checkpoint-{stage1_ckpt_step}" , f"denoising_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"checkpoint-{stage1_ckpt_step}", f"reference_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"checkpoint-{stage1_ckpt_step}", f"pose_guider.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)

    # Set motion module learnable
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = VVTDataset(
        data_root_dir=cfg.data.meta_paths,
        img_W=cfg.data.train_width,
        img_H=cfg.data.train_height,
        sample_n_frames=cfg.data.n_sample_frames,
        sample_stride=cfg.data.sample_rate,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=8
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.tracker_name)

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos and masked_videos to latent space
                pixel_values_vid = batch["video"].to(weight_dtype)
                pixel_values_masked_vid = batch["masked_video"].to(weight_dtype)
                pixel_values_pose = batch["video_dwpose"].to(weight_dtype)
                ref_cloth_images = batch["cloth"].to(weight_dtype)
                clip_cloth_images = batch["clip_cloth"].to(weight_dtype)
                pixel_values_mask_vid = batch["mask_video"].to(weight_dtype)
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    # video
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215
                    # masked video
                    pixel_values_masked_vid = rearrange(
                        pixel_values_masked_vid, "b f c h w -> (b f) c h w"
                    )
                    latents_masked = vae.encode(pixel_values_masked_vid).latent_dist.sample()
                    latents_masked = rearrange(
                        latents_masked, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents_masked = latents_masked * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Divide a video and sample a different random timestep for a video
                timesteps = sample_timestep(
                    maximum_timestep=train_noise_scheduler.num_train_timesteps,
                    num_frames=latents.shape[2],
                    batch_size=bsz
                ).to(device=latents.device)
                timesteps = timesteps.long()
                pixel_values_pose = pixel_values_pose.transpose(
                    1, 2
                )  # (bs, c, f, H, W)

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_cloth_image_list = []
                for batch_idx, (ref_cloth_img, clip_img) in enumerate(
                    zip(
                        ref_cloth_images,
                        clip_cloth_images,
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_cloth_image_list.append(ref_cloth_img)

                with torch.no_grad():
                    ref_cloth_img = torch.stack(ref_cloth_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_cloth_image_latents = vae.encode(
                        ref_cloth_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_cloth_image_latents = ref_cloth_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

            
                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                # # DEBUG: Check if the noisy_latents is added noise correctly
                # noisy_videos = decode_latents(vae, noisy_latents)
                # # add frame idx to video
                # frame_indices = torch.range(0, latents.shape[2] - 1, dtype=torch.int)
                # captions = [str(int(timesteps[0][cap])) for cap in range(timesteps.shape[1])]
                
                # # add caption
                # noisy_videos = add_caption_to_frames(noisy_videos[0], frame_indices, captions)                
                # save_video(noisy_videos, "debug", "debug", global_step)

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                # resize mask to have same shape as latent
                pixel_values_mask_vid = rearrange(
                    pixel_values_mask_vid, "b f c h w -> (b f) c h w"
                )
                masks = torch.nn.functional.interpolate(
                    pixel_values_mask_vid,
                    size=(latents.shape[-2], latents.shape[-1])
                ) # (b f) c h' w'
                masks = rearrange(masks, "(b f) c h w -> b c f h w", f=video_length)
                
                # concat latents with masked image latents and mask
                noisy_latents = torch.cat([noisy_latents, masks, latents_masked], dim=1)


                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_cloth_image_latents,
                    clip_image_embeds,
                    pixel_values_pose,
                    uncond_fwd=uncond_fwd,
                )
                with torch.no_grad():
                    # # DEBUG: Check if the pred_original_samples is correct
                    predicted_clean_latents = pred_original_video(train_noise_scheduler, model_pred.to(latents.dtype), timesteps, latents)
                    predicted_videos = decode_latents(vae, predicted_clean_latents)
                    # add frame idx to video
                    frame_indices = torch.range(0, latents.shape[2] - 1, dtype=torch.int)
                    captions = [str(int(timesteps[0][cap])) for cap in range(timesteps.shape[1])]
                    
                    # add caption
                    predicted_videos = add_caption_to_frames(predicted_videos[0], frame_indices, captions)                
                    save_video(predicted_videos, "debugging", "debug", global_step)

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    mean_dims = list(range(1, len(loss.shape))) # c, f, h, w
                    if len(mse_loss_weights.shape) == 2:
                        mean_dims = [1, 3, 4] # c, h, w
                    
                    loss = (
                        loss.mean(dim=mean_dims)
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0 or global_step == 20:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        # delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)
                        # save motion module only
                        unwrap_net = accelerator.unwrap_model(net)
                        accelerator.save(denoising_unet.state_dict(),  os.path.join(save_path, f"denoising_unet.pth"))
                        save_checkpoint(
                            unwrap_net.denoising_unet,
                            save_dir,
                            "motion_module",
                            global_step,
                            total_limit=3,
                        )
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            clip_length=cfg.data.n_sample_frames,
                            generator=generator,
                        )

                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                all_videos = []
                                all_names = []
                                for d in sample_dicts:
                                    np_videos = d['vid'] # (N, F, C, H, W) 
                                    name = d['name']
                                    np_videos_with_caption = add_caption_to_video(np_videos.cpu().numpy(), d['lpips'], d['ssim'])
                                    all_videos.append(torch.from_numpy(np_videos_with_caption))
                                    all_names.append(name)
                                save_videos(all_videos, all_names, f"logs/{cfg.tracker_name}/videos", global_step)
                                # all_videos = torch.cat(all_videos, dim=0)
                                # tracker.writer.add_video("Sample Videos", all_videos, global_step, fps=8)
                        del sample_dicts, np_videos
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()


            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

def save_videos(all_videos, all_names, save_dir, global_step):
    os.makedirs(os.path.join(save_dir, f"step={global_step}"), exist_ok=True)
    for name, video in zip(all_names, all_videos):
        video = video.squeeze(0)
        all_frames = []
        for i in range(video.shape[0]):
            all_frames.append(transforms.ToPILImage()(video[i]))
        save_videos_from_pil(all_frames, os.path.join(save_dir, f"step={global_step}", name + '.mp4'), fps=8)


def save_video(video_tensor, name, save_dir, global_step):
    os.makedirs(os.path.join(save_dir, f"step={global_step}"), exist_ok=True)
    video = video_tensor.squeeze()
    all_frames = []
    for i in range(video.shape[1]):
        all_frames.append(transforms.ToPILImage()(torch.from_numpy(video[:, i])))
    save_videos_from_pil(all_frames, os.path.join(save_dir, f"step={global_step}", name + '.mp4'), fps=8)



def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
