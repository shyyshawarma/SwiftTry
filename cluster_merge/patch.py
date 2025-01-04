import math
import time
from typing import Type, Dict, Any, Tuple, Callable

import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F

from .utils import isinstance_str, init_generator, join_frame, split_frame, func_warper, join_warper, split_warper
import matplotlib.pyplot as plt
from torchmetrics.functional import pairwise_cosine_similarity
import torch_scatter
import os
from .tomesd.patch import compute_merge_single
def compute_merge(module: torch.nn.Module, x: torch.Tensor, cluster_tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    #TODO perform merging with k-means clustering
    return None



def make_cluster_tome_block(block_class, name, centroid_books, metric='cosine', threshold=0.8) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ClusterToMe to the forward function of the block.

    name: etc: down_sample_0
    """

    class ClusterToMeBlock(block_class):
        """
        Args:
            centroid_books (Tensor): Pre-computed centroids for each timestep.
            name (str): Name of the block.
            metric (str): Distance metric for clustering ('cosine' or 'l2').
            threshold (float): Threshold value for merging tokens.
        """
        _centroid_books = centroid_books
        _block_name = name
        _metric = metric
        block_name = _block_name
        threshold=0.8
        print(_block_name)
        if "up_blocks.3.attentions.0.transformer_blocks.0" in block_name or "up_blocks.3.attentions.1.transformer_blocks" in block_name:
            threshold = 0.7
        if 'up_blocks.2' in block_name:
            threshold = 0.65
        if 'up_blocks.1' in block_name or 'mid' in block_name or 'down' in block_name:
            threshold = 0.6
        _threshold = threshold
        # if 'up' in _block_name: _threshold = 0.6
        if _centroid_books is None: breakpoint()
        def split_token(self, x, centroids):
            """
            Splits tokens into merging and non-merging groups based on distance to centroids.
            
            Args:
                x (Tensor): Input tokens of shape (num_tokens, c).
                centroids (Tensor): Centroids to compare tokens against.
            
            Returns:
                Tuple: (merging_index, cluster_assign, merging_features, 
                        not_merging_index, not_merging_features)
            """
            if self._metric == 'cosine':
                dists = pairwise_cosine_similarity(x, centroids)
                scores, assign_cluster_index = torch.max(dists, dim=-1)
            elif self._metric == 'l2':
                dists = torch.cdist(x.unsqueeze(0), centroids.unsqueeze(0))[0]
                scores, assign_cluster_index = torch.min(dists, dim=-1)
            else:
                raise ValueError("Unsupported metric")

            merging_index = torch.where(scores >= self._threshold)[0]
            not_merging_index = torch.where(scores < self._threshold)[0]
            merging_features = x[merging_index]
            not_merging_features = x[not_merging_index]
            cluster_assign = assign_cluster_index[merging_index]

            return merging_index, cluster_assign, merging_features, not_merging_index, not_merging_features, scores

        # def _merge_cloth(self, x, timestep=0):
        #     """
        #     Merges tokens by clustering and averaging assigned tokens.
            
        #     Args:
        #         x (Tensor): Input tokens of shape (batch_size, num_tokens, c).
        #         timestep (int): Current timestep for selecting centroids.
            
        #     Returns:
        #         Tensor: Merged tokens of shape (1, num_merged_tokens, c).
        #     """
        #     batch_size, num_tokens, c = x.shape
        #     self.batch_size, self.num_tokens, self.c = batch_size, num_tokens, c

        #     video_tokens, cloth_tokens = x[:, :num_tokens // 2], x[:, num_tokens // 2:]
        #     video_flat = video_tokens.reshape(batch_size * num_tokens // 2, c)
        #     centroid_cur_timestep_feats = self._centroid_books[timestep].half()

        #     merging_token_index, cluster_assign, merging_token_feats, not_merging_token_index, not_merging_token_feats, scores = self.split_token(
        #         video_flat, centroid_cur_timestep_feats
        #     )

        #     used_tokens = torch.unique(cluster_assign)
        #     self.K = used_tokens.shape[0]

        #     merged = torch.zeros(self.K + not_merging_token_index.shape[0], c, device=x.device).half()
        #     merged[self.K:] = not_merging_token_feats

        #     # Use scatter_mean for efficient token aggregation per cluster
        #     merged[:self.K] = torch_scatter.scatter_mean(
        #         merging_token_feats, cluster_assign, dim=0, dim_size=self.K
        #     )

        #     # Process cloth tokens
        #     single_cloth_token = cloth_tokens[:2].reshape(-1, c)
        #     self.num_cloth_tokens = single_cloth_token.size(0)
        #     merged = torch.cat((merged, single_cloth_token), dim=0)

        #     self.not_merging_token_index = not_merging_token_index
        #     self.merging_token_index = merging_token_index
        #     self.cluster_assign = cluster_assign
        #     self.used_tokens = used_tokens 

            
        #     return merged.unsqueeze(0)

        
        def merge(self, x, timestep=0, **kwargs):
            """
            Merges tokens by clustering and averaging assigned tokens.
            
            Args:
                x (Tensor): Input tokens of shape (batch_size, num_tokens, c).
                timestep (int): Current timestep for selecting centroids.
            
            Returns:
                Tensor: Merged tokens of shape (1, num_merged_tokens, c).
            """
            # if timestep == 39 and "up_blocks.2.attentions.2.transformer" in self._block_name :
            # if  "up_blocks.2.attentions.2.transformer" in self._block_name :
            if 1==1:
                self.is_merge = True
            else:
                self.is_merge = False
                return x
            
            batch_size, num_tokens, c = x.shape
            self.batch_size, self.num_tokens, self.c = batch_size, num_tokens, c

            video_tokens = x
            video_flat = video_tokens.reshape(batch_size * num_tokens, c)
            centroid_cur_timestep_feats = self._centroid_books[timestep].half()

            merging_token_index, cluster_assign, merging_token_feats, not_merging_token_index, not_merging_token_feats, scores = self.split_token(video_flat, centroid_cur_timestep_feats)
            # print("Similarity scores", scores.max(), scores.mean())

            # used_tokens = torch.unique(cluster_assign)
            # self.K = used_tokens.shape[0]
            unique_clusters, inverse_indices = torch.unique(cluster_assign, return_inverse=True)
            self.unique_clusters = unique_clusters
            self.inverse_indices = inverse_indices
            self.K = unique_clusters.shape[0]
            
            merged = torch.zeros(self.K + not_merging_token_index.shape[0], c, device=x.device).half()
            merged[self.K:] = not_merging_token_feats

            # Use scatter_mean for efficient token aggregation per cluster
            # Use the remapped `inverse_indices` for zero-based indexing with `scatter_mean`
            merged[:self.K] = torch_scatter.scatter_mean(merging_token_feats, inverse_indices, dim=0, dim_size=self.K)

            # Process cloth tokens
            
            self.not_merging_token_index = not_merging_token_index
            self.merging_token_index = merging_token_index
            # self.debug_index = merging_token_index.clone()
            self.cluster_assign = cluster_assign
            # save_dir = "/root/Projects/Moore-AnimateAnyone/output_result/debug_cluster/"
            # files = os.listdir(save_dir)
            # saved_infor = {
            #     'cluster_assign': cluster_assign.detach().cpu(),
            #     'sim_scores': scores.detach().cpu(),
            #     'block_name': self._block_name,
            #     'timestep': timestep

            # }
            # torch.save(saved_infor, os.path.join(save_dir, f"{len(files)}.pt"))
            return merged.unsqueeze(0)

        
        def unmerge(self, merged_tokens, **kwargs):
            """
            Reverses the merging of tokens, restoring original shape.
            
            Args:
                merged_tokens (Tensor): Merged tokens from the merge function.
            
            Returns:
                Tensor: Unmerged tokens of shape (batch_size, num_tokens, c).
            """
            if not self.is_merge:
                return merged_tokens
            merged_tokens = merged_tokens[0]
            token_unmerge = torch.zeros(
                self.batch_size * self.num_tokens, self.c, device=merged_tokens.device
            ).half()
            token_unmerge[self.not_merging_token_index] = merged_tokens[self.K:]
            token_unmerge[self.merging_token_index] = merged_tokens[:self.K][self.inverse_indices] 
            token_unmerge = token_unmerge.view(self.batch_size, self.num_tokens, self.c)
            self.is_merge = False
            return token_unmerge

    return ClusterToMeBlock



def make_diffusers_tome_block(block_class, name, centroid_books, metric='cosine', threshold=0.8) -> Type[torch.nn.Module]:
    class ToMeSdBlock(block_class):
        """
        Args:
            centroid_books (Tensor): Pre-computed centroids for each timestep.
            name (str): Name of the block.
            metric (str): Distance metric for clustering ('cosine' or 'l2').
            threshold (float): Threshold value for merging tokens.
        """
        _centroid_books = centroid_books
        _block_name = name
        _metric = metric
        _threshold = threshold
        print(_block_name)
        # if 'up' in _block_name: _threshold = 0.6

        def merge(self, hidden_states, timestep=0, **kwargs):
            """
            Merges tokens by clustering and averaging assigned tokens.
            
            Args:
                x (Tensor): Input tokens of shape (batch_size, num_tokens, c).
                timestep (int): Current timestep for selecting centroids.
            
            Returns:
                Tensor: Merged tokens of shape (1, num_merged_tokens, c).
            """
            self.is_merge = True
            # if timestep == 39 and "up_blocks.2.attentions.2.transformer" in self._block_name :
            merge_tome, unmerge_tome = compute_merge_single(hidden_states, self._cluster_tome_info, **kwargs)
            self.merge_tome = merge_tome
            self.unmerge_tome = unmerge_tome
            return self.merge_tome(hidden_states)

        
        def unmerge(self, merged_tokens):
            """
            Reverses the merging of tokens, restoring original shape.
            
            Args:
                merged_tokens (Tensor): Merged tokens from the merge function.
            
            Returns:
                Tensor: Unmerged tokens of shape (batch_size, num_tokens, c).
            """
            if not self.is_merge:
                return merged_tokens
            token_unmerge = self.unmerge_tome(merged_tokens)
            self.is_merge = False
            return token_unmerge
    return ToMeSdBlock
def hook_cluster_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._cluster_tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._cluster_tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_cluster_tome_module(module: torch.nn.Module):
    """ Adds a forward pre hook to initialize random number generator.
        All modules share the same generator state to keep their randomness in VidToMe consistent in one pass.
        This hook can be removed with remove_patch. """
    def hook(module, args):
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(
                args[0].device, fallback=module.generator)
        else:
            return None

        # module.generator = module.generator.manual_seed(module._cluster_tome_info["args"]["seed"])
        return None

    module._cluster_tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        cluster_data: dict,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        token_merge_type: str = 'cluster'        

    ):

    remove_patch(model)

    diffusion_model = model.unet if hasattr(model, "unet") else model

    if isinstance_str(model, "StableDiffusionControlNetPipeline") and include_control:
        diffusion_models = [diffusion_model, model.controlnet]
    else:
        diffusion_models = [diffusion_model]

    for diffusion_model in diffusion_models:
        diffusion_model._cluster_tome_info = {
            "size": None,
            "hooks": [],
            "args": {
                "dist": "L2",
                "ratio": ratio,
                "max_downsample": max_downsample,
                "sx": sx, "sy": sy,
                "generator": None,
                "use_rand": use_rand,

            }
        }
        hook_cluster_tome_model(diffusion_model)
        if token_merge_type == 'cluster':
            make_merging_fn = make_cluster_tome_block
        elif token_merge_type == 'tomesd':
            make_merging_fn = make_diffusers_tome_block
        else:
            print("Do not support")
            breakpoint()
        
        for name, module in diffusion_model.named_modules():
            # If for some reason this has a different name, create an issue and I'll fix it
            # if isinstance_str(module, "BasicTransformerBlock") and "down_blocks" not in name:
            if isinstance_str(module, "TemporalBasicTransformerBlock"):
                make_cluster_tome_block_fn = make_merging_fn

                # Get block name and pass the codebook of that block
    
                module.__class__ = make_cluster_tome_block_fn(module.__class__, name, cluster_data[name])
                module._cluster_tome_info = diffusion_model._cluster_tome_info
                hook_cluster_tome_module(module)

    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers

    model = model.unet if hasattr(model, "unet") else model
    model_ls = [model]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_cluster_tome_info"):
                for hook in module._tome_info["hooks"]:
                    hook.remove()
                module._cluster_tome_info["hooks"].clear()

            if module.__class__.__name__ == "ClusterToMeBlock":
                module.__class__ = module._parent

    return model


def update_patch(model: torch.nn.Module, **kwargs):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_cluster_tome_info"):
                for k, v in kwargs.items():
                    setattr(module, k, v)
    return model


def collect_from_patch(model: torch.nn.Module, attr="tome"):
    """ Collect attributes in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    ret_dict = dict()
    for model in model_ls:
        for name, module in model.named_modules():
            if hasattr(module, attr):
                res = getattr(module, attr)
                ret_dict[name] = res

    return ret_dict