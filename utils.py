import cv2
import numpy as np
from einops import rearrange
from tqdm import tqdm
import torch


def add_caption_to_video(np_videos, lpips_score, ssim_score):
    caption = "LPIPS: {:.4f}, SSIM: {:.4f}".format(lpips_score, ssim_score)
    for i in range(np_videos.shape[0]):  # Loop over batch
        for j in range(np_videos.shape[1]):  # Loop over frames
            frame = np_videos[i, j].transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.putText(
                frame, 
                caption, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            np_videos[i, j] = frame.transpose(2, 0, 1) / 255. # Convert back to (C, H, W)
    return np_videos


def add_caption_to_frames(video_array, frame_indices, captions, font_scale=1.0, thickness=2, text_color=(1, 0, 0)):
    # Convert the video_array to (T, H, W, C) format and rescale to [0, 255] for PIL processing
    video_array_uint8 = (video_array * 255).astype(np.uint8)
    video_array_uint8 = video_array_uint8.transpose(1, 2, 3, 0)

    # Iterate over the frames and add the captions
    for idx, frame_idx in enumerate(frame_indices):
        caption = captions[idx]
        frame = video_array_uint8[frame_idx]

        # Calculate the position for the text (bottom-center of the frame)
        text_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - text_size[1] - 10

        # Convert text_color from [0, 1] range to [0, 255] range
        text_color_255 = (int(text_color[0] * 255), int(text_color[1] * 255), int(text_color[2] * 255))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Add text to the frame
        cv2.putText(frame, caption, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_255, thickness)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Store the modified frame back
        video_array_uint8[frame_idx] = frame

    # Convert back to original range [0, 1] and format (C, T, H, W)
    video_array_with_captions = video_array_uint8.astype(np.float32) / 255
    video_array_with_captions = video_array_with_captions.transpose(3, 0, 1, 2)

    return video_array_with_captions

def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video

