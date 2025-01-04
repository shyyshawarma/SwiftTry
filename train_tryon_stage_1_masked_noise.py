import argparse
import logging
import math
import os
import os.path as osp
import gc
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
import copy
import diffusers
import mlflow

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
from src.loss import VGG19_feature_color_torchversion
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from einops import rearrange
from src.data.dataset import VITONHDDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_tryon import TryOnPipeline
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def freeze_unet_blocks(unet):
    # Freeze encoder blocks
    for param in unet.down_blocks.parameters():
        param.requires_grad = False

class MyDDIMScheduler(DDIMScheduler):

    def remove_noise(
        self,
        noisy_samples: Union[torch.FloatTensor, np.ndarray],
        noise: Union[torch.FloatTensor, np.ndarray],
        timesteps: Union[torch.IntTensor, np.ndarray],
    ) -> Union[torch.FloatTensor, np.ndarray]:
        # Make sure alphas_cumprod and timestep have same device and dtype as noisy_samples
        alphas_cumprod = self.alphas_cumprod.to(device=noisy_samples.device, dtype=noisy_samples.dtype)
        timesteps = timesteps.to(noisy_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        original_samples = (sqrt_alpha_prod * noisy_samples - sqrt_one_minus_alpha_prod * noise) 
        return original_samples

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device=self.device)
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
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


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = copy.deepcopy(ori_net.reference_unet)
    denoising_unet = copy.deepcopy(ori_net.denoising_unet)
    pose_guider = ori_net.pose_guider

    generator = torch.cuda.manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    image_enc = image_enc.to(dtype=torch.float32)

    pipe = TryOnPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    ref_cloth_image_paths = [
        "./configs/inference/ref_cloth_images/03857_00.jpg",
        "./configs/inference/ref_cloth_images/03033_00.jpg",
        "./configs/inference/ref_cloth_images/04528_00.jpg",
    ]
    pose_image_paths = [
        # "./configs/inference/poses/00006_00.jpg",
        "./configs/inference/poses/00008_00.jpg",
        "./configs/inference/poses/00013_00.jpg",
    ]
    masked_image_paths = [
        # "./configs/inference/masked_images/00006_00.jpg",
        "./configs/inference/masked_images/00008_00.jpg",
        "./configs/inference/masked_images/00013_00.jpg",
    ]
    mask_paths = [
        # "./configs/inference/masks/00006_00_mask.png",
        "./configs/inference/masks/00008_00_mask.png",
        "./configs/inference/masks/00013_00_mask.png",
    ]
    image_paths = [
        # "./configs/inference/images/00006_00.jpg",
        "./configs/inference/images/00008_00.jpg",
        "./configs/inference/images/00013_00.jpg",
    ]

    pil_images = []
    for ref_cloth_image_path in ref_cloth_image_paths:
        for pose, masked, mask, orig_image in zip(pose_image_paths, masked_image_paths, mask_paths, image_paths):
            pose_name = pose.split("/")[-1].replace(".jpg", "")
            ref_name = ref_cloth_image_path.split("/")[-1].replace(".jpg", "")

            image_pil = Image.open(orig_image).convert("RGB").resize((width, height))
            ref_cloth_image_pil = Image.open(ref_cloth_image_path).convert("RGB")
            pose_image_pil = Image.open(pose).convert("RGB").resize((width, height))
            masked_image_pil = Image.open(masked).convert("RGB").resize((width, height))
            mask_pil = Image.open(mask).convert("RGB").resize((width, height))

            image = pipe(
                ref_cloth_image_pil,
                image_pil,
                masked_image_pil,
                mask_pil,
                pose_image_pil,
                width,
                height,
                25,
                3.5,
                generator=generator,
            ).images
            image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
            # Save ref_image, src_image and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * 3, h), "white")
            ref_cloth_image_pil = ref_cloth_image_pil.resize((w, h))
            orig_image_pil = Image.open(orig_image).resize((w, h))
            canvas.paste(orig_image_pil, (0, 0))
            canvas.paste(ref_cloth_image_pil, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))

            pil_images.append({"name": f"{ref_name}_{pose_name}", "img": canvas})

    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipe, reference_unet, denoising_unet
    torch.cuda.empty_cache()

    return pil_images


def get_vgg_loss(vgg, pred, gt):
    pred_feat = vgg(pred, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
    gt_feat = vgg(gt, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
    loss_feat = 0
    weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    for i in range(len(pred_feat)):
        loss_feat += weights[i] * F.l1_loss(pred_feat[i], gt_feat[i].detach())
    return loss_feat



def decode_latents(vae, latents):
    """ Decode latents into image(s).
    
    Args:
        latents (torch.Tensor): Latent tensor of shape (b, c, f, h, w).

    Returns:
        torch.Tensor: Image(s) tensor of shape (b, 3, f, H, W).
    """
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in range(latents.shape[0]):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    # video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    return video


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )
    if not torch.cuda.is_available():
        weight_dtype = torch.float32
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = MyDDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = MyDDIMScheduler(**sched_kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        device, dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.reference_model_path,
        subfolder="unet",
    ).to(device=device)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device=device)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device=device)

    if cfg.pose_guider_pretrain:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device=device)
        # load pretrained controlnet-openpose params for pose_guider
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
        miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
        logger.info(f"Missing key for pose guider: {len(miss)}")
    else:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
        ).to(device=device)

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    freeze_unet_blocks(denoising_unet)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    pose_guider.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        batch_size=cfg.data.train_bs,
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        batch_size=cfg.data.train_bs,
        fusion_blocks="full",
    )
    # define VGG loss
    if cfg.use_vgg_perceptual_loss:
        print("VGG network initialization...")
        vgg = VGG19_feature_color_torchversion(vgg_normal_correct=False)
        vgg.load_state_dict(torch.load("pretrained_weights/vgg/vgg19_conv.pth", map_location="cpu"))
        vgg.eval()



    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention and torch.cuda.is_available():
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

    # Define datasets and dataloaders.
    train_dataset = VITONHDDataset(
        data_root_dir=cfg.data.meta_paths,
        img_H=cfg.data.train_height,
        img_W=cfg.data.train_width,
        is_paired=True,
        is_test=False,
        is_sorted=False,
        inverse_mask=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.data.train_bs, 
        shuffle=True, 
        num_workers=8
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
    if cfg.use_vgg_perceptual_loss:
        vgg = accelerator.prepare(vgg)

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
        # run_time = datetime.now().strftime("%Y%m%d-%H%M")
        # tracker_config = vars(copy.deepcopy(cfg))
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
        train_loss, train_diff_loss, train_vgg_loss = 0.0, 0.0, 0.0
        net.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert images to latent space
                images = batch["image"].to(weight_dtype)
                masked_images = batch["agn"].to(weight_dtype)
                masks = batch["agn_mask"].to(weight_dtype)
                ref_cloth_images = batch["cloth"].to(weight_dtype)
                pose_images = batch["image_dwpose"].to(weight_dtype)
                clip_cloth_images = batch['clip_cloth'].to(weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215

                    latents_masked = vae.encode(masked_images).latent_dist.sample()
                    latents_masked = latents_masked.unsqueeze(2)  # (b, c, 1, h, w)
                    latents_masked = latents_masked * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                pose_images = pose_images.unsqueeze(2)  # (bs, 3, 1, H, W)

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
                    ).latent_dist.sample()  # (bs, d, h', w')
                    ref_cloth_image_latents = ref_cloth_image_latents * 0.18215

                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_image_embeds = image_enc(
                        clip_img.to(device, dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                temp_noisy_latents = noisy_latents


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
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=(latents.shape[-2], latents.shape[-1])
                )
                masks = masks.unsqueeze(2) #  (b, c, 1, h', w')

                # mask the noisy latent
                noisy_latents = latents * (1 - masks) + noisy_latents * masks
                breakpoint()

                # concat latents with masked image latents and mask
                noisy_latents = torch.cat([noisy_latents, masks, latents_masked], dim=1)
                
                model_pred = net(
                    noisy_latents, # 9 channels
                    timesteps,
                    ref_cloth_image_latents,
                    image_prompt_embeds,
                    pose_images,
                    uncond_fwd,
                )
                if cfg.use_vgg_perceptual_loss:
                    z0_pred = train_noise_scheduler.remove_noise(
                        temp_noisy_latents, model_pred, timesteps
                    ).to(weight_dtype)
                    images_pred = decode_latents(vae, z0_pred)
                    images_gt = decode_latents(vae, latents)
                    
                    # images_pred_pil = transforms.ToPILImage()(images_pred[0])
                    # images_gt_pil = transforms.ToPILImage()(images_gt[0])
                    # canvas = Image.new('RGB', (cfg.data.train_width*2, cfg.data.train_height), "white")
                    # canvas.paste(images_pred_pil, (0, 0))
                    # canvas.paste(images_gt_pil, (cfg.data.train_width, 0))
                    # canvas.save(f'debug/{global_step}.png')
                    loss_vgg = get_vgg_loss(vgg, images_pred, images_gt)
                    loss_vgg_weight = cfg.vgg_weight_loss
                loss_l1_weight = 1.
                if cfg.snr_gamma == 0:
                    loss_diff = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    if cfg.use_vgg_perceptual_loss:
                        loss = loss_diff * loss_l1_weight + loss_vgg * loss_vgg_weight
                    else:
                        loss = loss_diff
                    loss = loss.mean()
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
                    loss_diff = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss_diff = (
                        loss_diff.mean(dim=list(range(1, len(loss_diff.shape))))
                        * mse_loss_weights
                    )
                    if cfg.use_vgg_perceptual_loss:
                        loss = loss_diff * loss_l1_weight + loss_vgg * loss_vgg_weight
                    else:
                        loss = loss_diff
                    loss = loss.mean()
                if cfg.use_vgg_perceptual_loss:
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_diff_loss = accelerator.gather(loss_diff.mean().repeat(cfg.data.train_bs)).mean()
                    train_diff_loss += avg_diff_loss.item() / cfg.solver.gradient_accumulation_steps
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_vgg_loss = accelerator.gather(loss_vgg.mean().repeat(cfg.data.train_bs)).mean()
                    train_vgg_loss += avg_vgg_loss.item() / cfg.solver.gradient_accumulation_steps

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
                if cfg.use_vgg_perceptual_loss:
                    accelerator.log({"train_loss": train_loss, "diffusion_loss": train_diff_loss, "perceptual_loss": train_vgg_loss}, step=global_step)
                else:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss, train_diff_loss, train_vgg_loss = 0.0, 0.0, 0.0
                if global_step % cfg.checkpointing_steps == 0 or global_step == 1000:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        # delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)
                        # save reference_unet and pose_guider and denoising unet as '.pth'
                        accelerator.save(reference_unet.state_dict(), os.path.join(save_path, f"reference_unet.pth"))
                        accelerator.save(pose_guider.state_dict(), os.path.join(save_path, f"pose_guider.pth"))
                        accelerator.save(denoising_unet.state_dict(), os.path.join(save_path, f"denoising_unet.pth"))
                        net.eval()
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        # validate
                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                        )
                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                np_images = np.stack([np.asarray(d['img']) for d in sample_dicts])
                                tracker.writer.add_images(f"validation", np_images, epoch, dataformats="NHWC")
                        del sample_dicts
                        gc.collect()
                    net.train()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


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

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
