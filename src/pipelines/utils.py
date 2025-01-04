import torch

tensor_interpolation = None


def get_tensor_interpolation_method():
    return tensor_interpolation


def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear


def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2


def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        # logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


import torch
import torch.nn.functional as F


def refine_noisy_latents(prev_cache_timesteps, cache_timesteps, prev_cache_latents, latents):
    min_timestep = cache_timesteps.min()
    min_timestep_indices = (cache_timesteps == min_timestep).nonzero(as_tuple=True)[0]
    higher_timestep_indices = (cache_timesteps > min_timestep).nonzero(as_tuple=True)[0]

    num_higher_frames = len(higher_timestep_indices)
    breakpoint()
    if num_higher_frames == 0:
        return latents
    
    return latents

    

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def save_caching_scheduler(caching_scheduler):
    cmap = mcolors.ListedColormap(['white', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the matrix
    plt.imshow(caching_scheduler, cmap=cmap, norm=norm)

    # Remove axis labels
    plt.axis('off')

    # Save the plot as an image
    plt.savefig('caching_scheduler.png', bbox_inches='tight', pad_inches=0)
    

    
    

    


def create_attention_mask(cache_timesteps):
    # Find the minimum timestep
    min_timestep = cache_timesteps.min()
    # Find the indices for the minimum and higher timesteps
    min_timestep_indices = (cache_timesteps == min_timestep).nonzero(as_tuple=True)[0]
    higher_timestep_indices = (cache_timesteps > min_timestep).nonzero(as_tuple=True)[0]
    
    num_higher_frames = len(higher_timestep_indices)
    if num_higher_frames == 0:
        return None
    attention_mask = torch.zeros_like(cache_timesteps).to(torch.float16)
    attention_mask[higher_timestep_indices] = float('-inf')
    return attention_mask.unsqueeze(0)



def blend_with_keyframe(prev_cache_features, cache_features, keyframe_idx=-1, initial_weight=0.5, decay_rate=0.9):
    """
    Blends a keyframe from the previous chunk with each frame of the current chunk, applying weight decay.
    
    Args:
        prev_cache_features (torch.Tensor): Features of chunk i with shape (B, C, F, H, W).
        cache_features (torch.Tensor): Features of chunk i+1 with shape (B, C, F, H, W).
        keyframe_idx (int): Index of the keyframe in the previous chunk to use for blending.
                            Defaults to -1 (last frame of the previous chunk).
        initial_weight (float): Initial weight for the keyframe from the previous chunk.
        decay_rate (float): Rate at which the weight of the keyframe decays across frames in the current chunk.
        
    Returns:
        torch.Tensor: Refined features for chunk i+1 with shape (B, C, F, H, W).
    """
    # Ensure the keyframe index is within bounds of prev_cache_features
    keyframe_idx = keyframe_idx % prev_cache_features.shape[2]  # Handle negative indices
    
    # Extract the keyframe from the previous chunk
    keyframe = prev_cache_features[:, :, keyframe_idx, :, :]

    # Initialize refined features as a copy of the current chunk's features
    refined_cache_features = cache_features.clone()
    
    # Apply weight decay as we move through frames in cache_features
    current_weight = initial_weight
    for f in range(cache_features.shape[2]):
        # Blending weights for the current frame
        weight_cache = current_weight
        weight_new = 1 - weight_cache
        
        # Blend the keyframe with the current frame
        refined_cache_features[:, :, f, :, :] = (
            weight_cache * keyframe + weight_new * cache_features[:, :, f, :, :]
        )
        
        # Update the weight with decay
        current_weight *= decay_rate  # Reduce the influence of the keyframe gradually

    return refined_cache_features
