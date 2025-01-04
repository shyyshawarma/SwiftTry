import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import PIL
import numpy as np
import torch
import math
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler, shift
from src.utils.scheduler import VideoDDIMScheduler
import random
from src.pipelines.utils import interpolate_features

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@dataclass
class TryOnVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    overlapped_frame_ids: Union[torch.Tensor, np.ndarray]

class TryOnVideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            VideoDDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
        )
        self.enable_vae_slicing()

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        ) # (b, c, f, h, w)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            # latents = latents.repeat(1, 1, video_length, 1, 1)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # get image latents
        if isinstance(images, PIL.Image.Image):
            images = [images]
        images = np.concatenate([np.array(i.convert("RGB"))[None, :] for i in images], axis=0)
        images = images.transpose(0, 3, 1, 2) # f, c, h, w
        images = torch.from_numpy(images).to(dtype=torch.float32) / 127.5 - 1.0
        images = images.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=images, generator=generator) # f, c, h', w'
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w", f = video_length)
        return latents, image_latents


    def prepare_condition(
        self,
        cond_image,
        width,
        height,
        device,
        dtype,
        do_classififer_free_guidance=False,
    ):
        image = self.cond_image_processor.preprocess(
            cond_image, height=height, width=width
        ).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def prepare_mask_and_masked_image(self, masked_images, masks):
        """
        Prepares a pair (masked_images, masks) to be consumed by the Try On pipeline. This means that those inputs will be
        converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
        ``masked_images`` and ``1`` for the ``masks``.

        The ``masked_images`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``masks`` will be
        binarized (``masks > 0.5``) and cast to ``torch.float32`` too.

        Args:
            masked_images (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
                It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
                ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
            masks (_type_): The mask to apply to the image, i.e. regions to inpaint.
                It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
                ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


        Raises:
            ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
            should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
            TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
                (ot the other way around).

        Returns:
            tuple[torch.Tensor]: The pair (masks, masked_images) as ``torch.Tensor`` with 4
                dimensions: ``batch x channels x height x width``.
        """
        if isinstance(masked_images, torch.Tensor):
            if not isinstance(masks, torch.Tensor):
                raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(masks)} is not")

            # Batch single image
            if masked_images.ndim == 3:
                assert masked_images.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                masked_images = masked_images.unsqueeze(0)

            # Batch and add channel dim for single mask
            if masks.ndim == 2:
                masks = masks.unsqueeze(0).unsqueeze(0)

            # Batch single mask or add channel dim
            if masks.ndim == 3:
                # Batched mask
                if masks.shape[0] == masked_images.shape[0]:
                    masks = masks.unsqueeze(1)
                else:
                    masks = masks.unsqueeze(0)

            assert masked_images.ndim == 4 and masks.ndim == 4, "Image and Mask must have 4 dimensions"
            assert masked_images.shape[-2:] == masks.shape[-2:], "Image and Mask must have the same spatial dimensions"
            assert masked_images.shape[0] == masks.shape[0], "Image and Mask must have the same batch size"
            assert masks.shape[1] == 1, "Mask image must have a single channel"

            # Check image is in [-1, 1]
            if masked_images.min() < -1 or masked_images.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")

            # Check mask is in [0, 1]
            if masks.min() < 0 or masks.max() > 1:
                raise ValueError("Mask should be in [0, 1] range")

            # # paint-by-example inverses the mask
            # mask = 1 - mask

            # Binarize mask
            masks[masks < 0.5] = 0
            masks[masks >= 0.5] = 1

            # Image as float32
            masked_images = masked_images.to(dtype=torch.float32)
        elif isinstance(masks, torch.Tensor):
            raise TypeError(f"`masks` is a torch.Tensor but `image` (type: {type(masked_images)} is not")
        else:
            if isinstance(masked_images, PIL.Image.Image):
                masked_images = [masked_images]

            masked_images = np.concatenate([np.array(i.convert("RGB"))[None, :] for i in masked_images], axis=0)
            masked_images = masked_images.transpose(0, 3, 1, 2)
            masked_images = torch.from_numpy(masked_images).to(dtype=torch.float32) / 127.5 - 1.0

            # preprocess mask
            if isinstance(masks, PIL.Image.Image):
                masks = [masks]

            masks = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in masks], axis=0)
            masks = masks.astype(np.float32) / 255.0

            # # paint-by-example inverses the mask
            # mask = 1 - mask

            masks[masks < 0.5] = 0
            masks[masks >= 0.5] = 1
            masks = torch.from_numpy(masks)

        return masks, masked_images


    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        # mask: (b, 1, f, h, w), masked_image: (b, 3, f, h, w)\
        video_length = mask.shape[2]
        mask = rearrange(mask, "b c f h w -> (b f) c h w")
        
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        ) # resize mask to (b * f, 1, h_latent, w_latent)
        mask = rearrange(mask, "(b f) c h w -> b c f h w", f=video_length) # (b, 1, f, h_latent, w_latent)
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4: # if masked image already is a latent
            masked_image_latents = masked_image
        else:
            masked_image = rearrange(masked_image, "b c f h w -> (b f) c h w")
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
            masked_image_latents = rearrange(masked_image_latents, "(b f) c h w -> b c f h w", f=video_length)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    
    @torch.no_grad()
    def __call__(
        self,
        ref_cloth_image,
        images,
        masked_images,
        masks,
        pose_images,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        cache_branch: Optional[int] = 0,
        drop_ratio=1.,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_cloth_image, return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.vae.config.latent_channels
        latents, image_latents = self.prepare_latents(
            images,         # list of source images pil
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        ) # (bs, c, f, h_latent, w_latent)
        noise = latents
        # preprocess mask and image
        # given that masked images and masks are processed
        masks, masked_images = self.prepare_mask_and_masked_image(masked_images, masks) # (f, c, h, w)

        # expand batch dim
        masks = rearrange(masks, "f c h w -> 1 c f h w")
        masked_images = rearrange(masked_images, "f c h w -> 1 c f h w")
        # prepare mask latents
        masks, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_images,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype=clip_image_embeds.dtype,
            device=device,
            generator=generator,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ) # (bs, c, f, h', w')

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_cloth_image_tensor = self.ref_image_processor.preprocess(
            ref_cloth_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_cloth_image_tensor = ref_cloth_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_cloth_image_latents = self.vae.encode(ref_cloth_image_tensor).latent_dist.mean
        ref_cloth_image_latents = ref_cloth_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = self.cond_image_processor.preprocess(
                pose_image, height=height, width=width
             )[0] # c, h, w
            pose_cond_tensor = pose_cond_tensor.unsqueeze(1) # c, 1, h, w
            pose_cond_tensor = pose_cond_tensor.to(
                device=device, dtype=self.pose_guider.dtype
            )
            pose_cond_tensor_list.append(pose_cond_tensor)
        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=1)  # (c, t, h, w)
        pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea = (
            torch.cat([pose_fea] * 2) if do_classifier_free_guidance else pose_fea
        )
        context_scheduler = get_context_scheduler(context_schedule)
        context_queue = list(
            context_scheduler(
                0,
                num_inference_steps,
                latents.shape[2],
                context_frames,
                context_stride,
                context_overlap,
            )
        )
        num_context_batches = math.ceil(len(context_queue) / context_batch_size)
        init_global_context = []
        for ctx_batch in range(num_context_batches):
            init_global_context.append(
                context_queue[
                    ctx_batch * context_batch_size : (ctx_batch + 1) * context_batch_size
                ]
            )
        
        cache_features = torch.empty_like(pose_fea)
        cache_timesteps = torch.stack([timesteps[-1]]*latents.shape[2], dim=0)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_cloth_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        # t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                # choose a shift size at each timestep t
                stride = 8 # 0, 8, 16, 24, 0, 8, 16, 24, ...
                pattern = [step * stride for step in range(context_frames//stride + 2)]
                
                shift_val = pattern[i % len(pattern)]

                # # or randomize it
                # shift_val = random.randint(0, context_frames - 1)
                # shift the context
                
                global_context = shift(init_global_context, shift_val=shift_val, context_size=context_frames, num_frames=latents.shape[2])
                prev_context = None
                for context_idx, context in enumerate(global_context):
                    if context_idx % 2 != 0 and i != 0 and i != len(timesteps) - 1:
                        cache_feat = (
                            torch.cat([cache_features[:, :, c] for c in context])
                            .to(device)
                        )
                        cache_timestep = (
                            torch.cat([cache_timesteps[c] for c in context])
                            .to(device)
                        )
                        # took from previouse computed context features
                        prev_cache_feat = (
                            torch.cat([cache_features[:, :, c] for c in prev_context])
                            .to(device)
                        )
                        # interpolate features if needed
                        cache_feat = interpolate_features(prev_cache_feat, cache_feat, cache_timestep)
                    else:
                        cache_feat = None
                        for j, c in enumerate(context):
                            cache_timesteps[c] = t

                    # 3.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    masked_image_latents_input = torch.cat([masked_image_latents[:, :, c] for c in context])
                    masks_input = torch.cat([masks[:, :, c] for c in context])
                    # concat mask and masked_image_latents to the latents
                    latent_model_input = torch.cat([latent_model_input, masks_input, masked_image_latents_input], dim=1)
                    latent_pose_input = torch.cat(
                        [pose_fea[:, :, c] for c in context]
                    )
                    # use cached latents_timestep 
                    pred, cache_feat = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        pose_cond_fea=latent_pose_input,
                        cache_features=cache_feat,
                        cache_branch=cache_branch,
                        return_dict=False,
                        **kwargs
                    )
                    
                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1
                        cache_features[:, :, c] = cache_feat
                    prev_context = context
                        

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                overlapped_frame_ids = (counter == 2).squeeze()
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()
        
        # Post-processing
        video = self.decode_latents(latents)  # (b, c, f, h, w)
        
        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video, overlapped_frame_ids

        return TryOnVideoPipelineOutput(videos=video, overlapped_frame_ids=overlapped_frame_ids)
