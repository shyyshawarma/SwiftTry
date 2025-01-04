import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import PIL
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor
from piecewise_rectified_flow.src.scheduler_perflow import PeRFlowScheduler
from src.models.mutual_self_attention import ReferenceAttentionControl



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
class TryOnPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class TryOnPipeline(DiffusionPipeline):
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
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            PeRFlowScheduler
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
            do_convert_rgb=True
        )

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
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
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
        else:
            print(":>")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # get image latents
        if isinstance(images, PIL.Image.Image):
            images = [images]
        images = np.concatenate([np.array(i.convert("RGB"))[None, :] for i in images], axis=0)
        images = images.transpose(0, 3, 1, 2) # b, c, h, w
        images = torch.from_numpy(images).to(dtype=torch.float32) / 127.5 - 1.0
        images = images.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=images, generator=generator) # b, c, h', w'
        image_latents = image_latents.unsqueeze(2) # b, c, f, h', w'
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
            # masks = 1 - masks

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
            # masks = 1 - masks

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
        # mask: (b, 1, f, h, w), masked_image: (b, 3, f, h, w)
        mask = rearrange(mask, "b c f h w -> (b f) c h w")
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        ) # resize mask to (b * f, 1, h_latent, w_latent)
        mask = rearrange(mask, "(b f) c h w -> b c f h w", f=1) # (b, 1, f, h_latent, w_latent)
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4: # if masked image already is a latent
            masked_image_latents = masked_image
        else:
            masked_image = rearrange(masked_image, "b c f h w -> (b f) c h w")
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
            masked_image_latents = rearrange(masked_image_latents, "(b f) c h w -> b c f h w", f=1)

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
        image,
        masked_image,
        mask,
        pose_image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
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

        batch_size = len(ref_cloth_image) if isinstance(ref_cloth_image, list) else 1
        
        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            [ref.resize((224, 224)) for ref in ref_cloth_image] if isinstance(ref_cloth_image, list) else ref_cloth_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
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
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            clip_image_embeds.dtype,
            device,
            generator,
            latents=kwargs.get('latents')
        )
        latents = latents.unsqueeze(2)  # (bs, c, 1, h', w')
        noise = latents
        latents_dtype = latents.dtype

        # preprocess mask and image
        masks, masked_images = self.prepare_mask_and_masked_image(masked_image, mask) # (b, c, h, w)
        # expand temporal dim
        masks = masks.unsqueeze(2) # (bs, c, 1, h', w')
        masked_images = masked_images.unsqueeze(2) # (bs, c, 1, h', w')
        # prepare mask latents
        mask, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_images,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype=clip_image_embeds.dtype,
            device=device,
            generator=generator,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ) # (bs, c, 1, h', w')

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


        # Prepare pose condition image
        pose_cond_tensor = self.cond_image_processor.preprocess(
            pose_image, height=height, width=width
        )
        pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea = (
            torch.cat([pose_fea] * 2) if do_classifier_free_guidance else pose_fea
        )

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_cloth_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        # t,
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                # concat mask and masked_image_latents to the latents
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # blend
                init_latents_proper = image_latents
                if do_classifier_free_guidance:
                    init_mask, _ = mask.chunk(2)
                else:
                    init_mask = mask

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )
                latents = (1 - init_mask) * init_latents_proper + init_mask * latents

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
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return TryOnPipelineOutput(images=images)
