import torch
import random

from typing import Optional, Union, Tuple
from einops import rearrange
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

class VideoDDIMScheduler(DDIMScheduler):
    '''Custom DDIM scheduler for video latents with different frame's timestep'''

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        is_video = len(timesteps.shape) == 2
        bsz = sample.shape[0]

        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        timesteps = rearrange(timesteps, "b f -> (b f)") if is_video else timesteps

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        sqrt_alpha_prod = rearrange(sqrt_alpha_prod, "(b f) 1 1 1 1 -> b 1 f 1 1", b=bsz) if is_video else sqrt_alpha_prod
        sqrt_one_minus_alpha_prod = rearrange(sqrt_one_minus_alpha_prod, "(b f) 1 1 1 1 -> b 1 f 1 1", b=bsz) if is_video else sqrt_one_minus_alpha_prod
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        is_video = len(timesteps.shape) == 2
        bsz = original_samples.shape[0]
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        timesteps = rearrange(timesteps, "b f -> (b f)") if is_video else timesteps
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        sqrt_alpha_prod = rearrange(sqrt_alpha_prod, "(b f) 1 1 1 1 -> b 1 f 1 1", b=bsz) if is_video else sqrt_alpha_prod
        sqrt_one_minus_alpha_prod = rearrange(sqrt_one_minus_alpha_prod, "(b f) 1 1 1 1 -> b 1 f 1 1", b=bsz) if is_video else sqrt_one_minus_alpha_prod

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


    def step(
        self,
        model_output: torch.Tensor,
        current_timestep: torch.IntTensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            current_timestep (`torch.IntTensor`):
                The current discrete timestep in the diffusion chain. For video, it should be a list of frame's timestep.
            timestep (`int`):
                The timestep t we want to step to t-1 
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alphas_cumprod = self.alphas_cumprod.to(device=model_output.device, dtype=model_output.dtype)
        alpha_prod_t = alphas_cumprod[current_timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # since the input is video latents, therefore the timestep of each frame can differ
        alpha_prod_t = alpha_prod_t[None, None, :, None, None]
        beta_prod_t = beta_prod_t[None, None, :, None, None]
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


    


def sample_timestep(maximum_timestep, num_frames, batch_size, different_values=[-80, -40, 0, 40, 80]):

    assert num_frames % 2 == 0, "num_frames must be even"
    half_num_frames = num_frames // 2
    # Generate a random base value for the first half from (0, maximum_timestep - 1)
    base_value = random.randint(0, maximum_timestep - 1)
    first_half = [base_value] * half_num_frames
    
    # Select a difference value from different_values
    difference = random.choice(different_values)
    
    # Calculate the value for the second half
    second_half_value = max(0, min(maximum_timestep - 1, base_value - difference))
    
    second_half = [second_half_value] * half_num_frames
    
    timestep = first_half + second_half

    # Combine both halves
    frames = torch.tensor([timestep]*batch_size)
    
    return frames


def sample_timestep_random(maximum_timestep, num_frames, batch_size, max_difference=80): 
    
    num_segments = random.choice([2, 3, 4])
    assert num_segments > 0, "num_segments must be greater than 0"
    assert num_frames >= num_segments, "num_frames must be at least as large as num_segments"
    
    # Randomly divide num_frames into num_segments
    lengths = sorted(random.sample(range(1, num_frames), num_segments - 1))
    lengths = [lengths[0]] + [lengths[i] - lengths[i - 1] for i in range(1, len(lengths))] + [num_frames - lengths[-1]]

    # Generate random base value for the first segment
    base_value = random.randint(0, maximum_timestep - 1)
    
    segments = []
    
    for i in range(num_segments):
        segment_length = lengths[i]
        
        if i == 0:
            # First segment has the base value
            segment = [base_value] * segment_length
        else:
            # Generate a random difference value in the range [0, max_difference]
            difference = random.randint(0, max_difference)
            segment_value = max(0, min(base_value + difference, maximum_timestep - 1))
            segment = [segment_value] * segment_length
        
        segments.extend(segment)
    
    # Create a batch by repeating the frames across the batch dimension
    all_batches_tensor = torch.tensor([segments] * batch_size)
    
    return all_batches_tensor
