from os.path import join as opj

import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import json
from typing import List, Tuple
from .labelmap import label_map
from .dataset_util import *
from numpy.linalg import lstsq
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
import random
import torch
import pillow_avif
import albumentations as A
import glob
from src.utils.util import read_frames
from torchvision.transforms.functional import hflip

def read_image(image_path):
    # Open the image using PILLOW
    img = Image.open(image_path)

    # If the image has an alpha channel (transparency), fill the background with white
    if img.mode == 'RGBA':
        # Create a new image with a white background of the same size as the original image
        background = Image.new('RGB', img.size, (255, 255, 255))

        # Composite the original image over the white background
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel

        return background
    else:
        # If the image doesn't have an alpha channel, return it as it is
        return img






class SingleVideoDataset(Dataset):
    def __init__(
        self,
        width: int = 384,
        height: int = 512,
        n_sample_frames: int = 24,
        frame_step: int = 1,
        single_video_path: str = "",
        single_video_masked_path: str = "",
        single_video_mask_path: str = "",
        single_garment_image: str = "",
        single_video_dwpose_path: str = "",
        use_caption: bool = False,
        use_bucketing: bool = False,
        **kwargs
    ):
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.single_video_path = single_video_path
        self.single_video_masked_path = single_video_masked_path
        self.single_video_mask_path = single_video_mask_path
        self.single_video_dwpose_path = single_video_dwpose_path
        self.single_garment_image = single_garment_image
        self.clip_image_processor = CLIPImageProcessor()
        self.width = width
        self.height = height
        
    def create_video_chunks(self):
        vr = decord.VideoReader(self.single_video_path)
        vr_range = range(0, len(vr), self.frame_step)

        self.frames = list(self.chunk(vr_range, self.n_sample_frames))
        return self.frames

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def get_frame_batch(self, vr, resize=None):
        index = self.index
        frames = vr.get_batch(self.frames[self.index])
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, w, h)
        resize = transforms.Resize((height, width), antialias=True)

        return resize
    
    def process_video_wrapper(self, vid_path, is_mask=False):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        if is_mask:
            video = video[:, :1, :, :]
        return video, vr 

    def single_video_batch(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, _ = self.process_video_wrapper(train_data)
            masked_video, _ = self.process_video_wrapper(self.single_video_masked_path)
            mask_video, _ = self.process_video_wrapper(self.single_video_mask_path, is_mask=True)
            dwpose_video, _ = self.process_video_wrapper(self.single_video_dwpose_path)
            
            garment_image = Image.open(self.single_garment_image).resize((self.width, self.height))
            garment_image_clip = self.clip_image_processor(
                images=garment_image, return_tensors="pt"
            ).pixel_values[0]
            garment_image = transforms.ToTensor()(garment_image)
            return video, masked_video, mask_video, dwpose_video, garment_image, garment_image_clip
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        
        return len(self.create_video_chunks())

    def __getitem__(self, index):

        video, masked_video, mask_video, dwpose_video, garment_image, garment_image_clip = self.single_video_batch(index)

        example = {
            "video": (video / 127.5 - 1.0),
            "masked_video": (masked_video / 127.5 - 1.0),
            "mask_video": (mask_video / 255.0),
            "video_dwpose": (dwpose_video / 127.5 - 1.0),
            "cloth": (garment_image * 2.0 - 1.0),
            "clip_cloth": garment_image_clip,
            'dataset': self.__getname__()
        }

        return example


class VVTDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        img_H,
        img_W,
        sample_n_frames=24,
        sample_stride=4
    ):
        super().__init__()
        self.sample_rate = sample_stride
        self.sample_n_frames = sample_n_frames
        self.width = img_W
        self.height = img_H
        self.data_root_dir = data_root_dir
        self.video_paths = sorted(glob.glob(opj(self.data_root_dir, 'lip_train_frames', '*')))
        self.masked_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'lip_train_frames_masked', '*')))
        self.mask_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'lip_train_frames_mask', '*')))
        self.pose_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'lip_train_frames_dwpose_new', '*')))
        self.flow_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'lip_train_frames_flow512', '*')))
        self.cloth_paths = [glob.glob(opj(self.data_root_dir, 'lip_clothes_person', os.path.basename(video_path),'*.jpg'))[0] for video_path in self.video_paths]
        assert len(self.video_paths) == len(self.masked_video_paths) == len(self.mask_video_paths) == len(self.cloth_paths), "should be equal length"

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor()
            ]
        )
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        masked_video_path = self.masked_video_paths[index]
        mask_video_path = self.mask_video_paths[index]
        pose_video_path = self.pose_video_paths[index]
        cloth_path = self.cloth_paths[index]
        flow_path = self.flow_video_paths[index]

        image_paths = sorted(glob.glob(video_path + '/*.png'))
        masked_image_paths = sorted(glob.glob(masked_video_path + '/*.png'))
        mask_image_paths = sorted(glob.glob(mask_video_path + '/*.png'))
        pose_image_paths = sorted(glob.glob(pose_video_path + '/*.png'))
        flow_image_paths = sorted(glob.glob(flow_path + '/*.npy'))
        flow_image_paths += [flow_image_paths[-1]] # dummy padding
        video_length = len(image_paths)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        ).tolist()
        
        pixel_values_video = []
        pixel_values_masked_video = []
        pixel_values_mask_video = []
        pixel_values_pose_video = []
        pixel_values_flow_video = []
        for idx in batch_index:
            pixel_values_video.append(self.pixel_transform(Image.open(image_paths[idx])))
            pixel_values_masked_video.append(self.pixel_transform(Image.open(masked_image_paths[idx])))
            pixel_values_mask_video.append(self.mask_transform(Image.open(mask_image_paths[idx]).convert('L')))
            pixel_values_pose_video.append(self.pixel_transform(Image.open(pose_image_paths[idx])))
            pixel_values_flow_video.append(torch.from_numpy(np.load(flow_image_paths[idx])))
        
        pixel_values_cloth = self.pixel_transform(Image.open(cloth_path).convert("RGB"))
        cloth_pil = Image.open(cloth_path).convert("RGB")
        clip_cloth_img = self.clip_image_processor(
            images=cloth_pil, return_tensors="pt"
        ).pixel_values[0]
        
        sample = dict(
            video=torch.stack(pixel_values_video, dim=0), # f, c, h, w
            masked_video=torch.stack(pixel_values_masked_video, dim=0),
            mask_video=torch.stack(pixel_values_mask_video, dim=0),
            flow_video=torch.stack(pixel_values_flow_video, dim=0),
            video_dwpose=torch.stack(pixel_values_pose_video, dim=0),
            cloth=pixel_values_cloth,
            clip_cloth=clip_cloth_img
        )
        return sample


class ViViDDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        img_H,
        img_W,
        sample_n_frames=24,
        sample_stride=4
    ):
        super().__init__()
        self.sample_rate = sample_stride
        self.sample_n_frames = sample_n_frames
        self.width = img_W
        self.height = img_H
        self.data_root_dir = data_root_dir
        self.video_paths = sorted(glob.glob(opj(self.data_root_dir, 'videos', '*')))
        self.masked_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'agnostic', '*')))
        self.mask_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'agnostic_mask', '*')))
        self.pose_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'dwpose', '*')))
        self.cloth_paths = sorted(glob.glob(opj(self.data_root_dir, 'images', '*')))
        assert len(self.video_paths) == len(self.masked_video_paths) == len(self.mask_video_paths) == len(self.cloth_paths), "should be equal length"

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor()
            ]
        )
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        masked_video_path = self.masked_video_paths[index]
        mask_video_path = self.mask_video_paths[index]
        pose_video_path = self.pose_video_paths[index]
        cloth_path = self.cloth_paths[index]

        images = read_frames(video_path)
        masked_images = read_frames(masked_video_path)
        mask_images = read_frames(mask_video_path)
        pose_images = read_frames(pose_video_path)
        video_length = len(images)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        ).tolist()
        
        pixel_values_video = []
        pixel_values_masked_video = []
        pixel_values_mask_video = []
        pixel_values_pose_video = []
        pixel_values_flow_video = []
        for idx in batch_index:
            pixel_values_video.append(self.pixel_transform(images[idx]))
            pixel_values_masked_video.append(self.pixel_transform(masked_images[idx]))
            pixel_values_mask_video.append(self.mask_transform(mask_images[idx].convert('L')))
            pixel_values_pose_video.append(self.pixel_transform(pose_images[idx]))
        
        pixel_values_cloth = self.pixel_transform(Image.open(cloth_path).convert("RGB"))
        cloth_pil = Image.open(cloth_path).convert("RGB")
        clip_cloth_img = self.clip_image_processor(
            images=cloth_pil, return_tensors="pt"
        ).pixel_values[0]
        
        sample = dict(
            video=torch.stack(pixel_values_video, dim=0), # f, c, h, w
            masked_video=torch.stack(pixel_values_masked_video, dim=0),
            mask_video=torch.stack(pixel_values_mask_video, dim=0),
            video_dwpose=torch.stack(pixel_values_pose_video, dim=0),
            cloth=pixel_values_cloth,
            clip_cloth=clip_cloth_img
        )
        return sample


class TikTokDressDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        img_H,
        img_W,
        sample_n_frames=24,
        sample_stride=4,
        new_mask=True
    ):
        super().__init__()
        self.sample_rate = sample_stride
        self.sample_n_frames = sample_n_frames
        self.width = img_W
        self.height = img_H
        self.data_root_dir = data_root_dir
        self.train_pairs = []
        with open(os.path.join(data_root_dir, 'train_pairs_newest.txt'), 'r') as file:
            for line in file:
                self.train_pairs.append(line.strip().split(' '))
        if new_mask:
            masked_folder = 'images_all_normalized_masked_newest'
            mask_folder = 'images_all_normalized_mask_newest'
        else:
            masked_folder = 'images_all_normalized_masked'
            mask_folder = 'images_all_normalized_mask'

        self.video_paths = [opj(self.data_root_dir, 'images_all_normalized', video_name[:-4]) for video_name, garment_name in self.train_pairs]
        self.masked_video_paths = [opj(self.data_root_dir, masked_folder, video_name[:-4]) for video_name, garment_name in self.train_pairs]
        self.mask_video_paths = [opj(self.data_root_dir, mask_folder, video_name[:-4]) for video_name, garment_name in self.train_pairs]
        self.pose_video_paths = [opj(self.data_root_dir, 'images_all_normalized_dwpose', video_name[:-4]) for video_name, garment_name in self.train_pairs]
        self.cloth_paths = [opj(self.data_root_dir, 'garments_normalized_newest', garment_name) for video_name, garment_name in self.train_pairs]
        print(len(self.video_paths), len(self.masked_video_paths), len(self.mask_video_paths), len(self.cloth_paths))
        assert len(self.video_paths) == len(self.masked_video_paths) == len(self.mask_video_paths) == len(self.cloth_paths), "should be equal length"

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        video_name = os.path.basename(video_path)
        masked_video_path = self.masked_video_paths[index]
        mask_video_path = self.mask_video_paths[index]
        pose_video_path = self.pose_video_paths[index]
        cloth_path = self.cloth_paths[index]

        image_paths = sorted(glob.glob(video_path + '/*.png'))
        masked_image_paths = sorted(glob.glob(masked_video_path + '/*.png'))
        mask_image_paths = sorted(glob.glob(mask_video_path + '/*.png'))
        pose_image_paths = sorted(glob.glob(pose_video_path + '/*.png'))
        video_length = len(image_paths)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        ).tolist()
        
        pixel_values_video = []
        pixel_values_masked_video = []
        pixel_values_mask_video = []
        pixel_values_pose_video = []
        for idx in batch_index:
            pixel_values_video.append(self.pixel_transform(Image.open(image_paths[idx])))
            pixel_values_masked_video.append(self.pixel_transform(Image.open(masked_image_paths[idx])))
            pixel_values_mask_video.append(self.mask_transform(Image.open(mask_image_paths[idx]).convert('L')))
            pixel_values_pose_video.append(self.pixel_transform(Image.open(pose_image_paths[idx])))
        
        do_flip = random.random() > 0.5

        # Random flip on all images
        if do_flip:
            pixel_values_video = [hflip(img) for img in pixel_values_video]
            pixel_values_masked_video = [hflip(img) for img in pixel_values_masked_video]
            pixel_values_mask_video = [hflip(img) for img in pixel_values_mask_video]
            pixel_values_pose_video = [hflip(img) for img in pixel_values_pose_video]



        # read cloth image
        pixel_values_cloth = self.pixel_transform(read_image(cloth_path).convert("RGB"))
        cloth_pil = read_image(cloth_path).convert("RGB")
        clip_cloth_img = self.clip_image_processor(
            images=cloth_pil, return_tensors="pt"
        ).pixel_values[0]
        
        if do_flip:
            pixel_values_cloth = hflip(pixel_values_cloth)
            clip_cloth_img = hflip(clip_cloth_img)

        sample = dict(
            video=torch.stack(pixel_values_video, dim=0), # f, c, h, w
            masked_video=torch.stack(pixel_values_masked_video, dim=0),
            mask_video=torch.stack(pixel_values_mask_video, dim=0),
            video_dwpose=torch.stack(pixel_values_pose_video, dim=0),
            cloth=pixel_values_cloth,
            clip_cloth=clip_cloth_img,
            video_name=video_name
        )
        return sample

class TikTokDressTripletDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        img_H,
        img_W,
        max_window_range=24,
    ):
        super().__init__()
        self.max_window_range = max_window_range
        self.width = img_W
        self.height = img_H
        self.data_root_dir = data_root_dir
        self.video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new', '*')))
        self.masked_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_masked', '*')))
        self.mask_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_mask', '*')))
        self.pose_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_dwpose', '*')))
        self.flow_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_flow', '*')))
        self.cloth_paths = sorted([opj(self.data_root_dir, 'cloth_images', os.path.basename(video_path),f'{os.path.basename(video_path)}.png') for video_path in self.video_paths])
        print(len(self.video_paths), len(self.masked_video_paths), len(self.mask_video_paths), len(self.cloth_paths), len(self.flow_video_paths))
        assert len(self.video_paths) == len(self.masked_video_paths) == len(self.mask_video_paths) == len(self.cloth_paths) == len(self.flow_video_paths), "should be equal length"

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        masked_video_path = self.masked_video_paths[index]
        mask_video_path = self.mask_video_paths[index]
        pose_video_path = self.pose_video_paths[index]
        cloth_path = self.cloth_paths[index]
        flow_path = self.flow_video_paths[index]

        image_paths = sorted(glob.glob(video_path + '/*.png'))
        masked_image_paths = sorted(glob.glob(masked_video_path + '/*.png'))
        mask_image_paths = sorted(glob.glob(mask_video_path + '/*.png'))
        pose_image_paths = sorted(glob.glob(pose_video_path + '/*.png'))
        flow_image_paths = sorted(glob.glob(flow_path + '/*.npy'))
        flow_image_paths += [flow_image_paths[-1]] # dummy padding
        video_length = len(image_paths)

        first_frame_idx = random.randint(0, video_length - self.max_window_range)

        current_frame_idx = random.randint(first_frame_idx, first_frame_idx + self.max_window_range)
        previous_frame_idx = current_frame_idx - 1
        batch_index = [first_frame_idx, previous_frame_idx, current_frame_idx]
        pixel_values_video = []
        pixel_values_masked_video = []
        pixel_values_mask_video = []
        pixel_values_pose_video = []
        pixel_values_flow_video = []
        for idx in batch_index:
            pixel_values_video.append(self.pixel_transform(Image.open(image_paths[idx])))
            pixel_values_masked_video.append(self.pixel_transform(Image.open(masked_image_paths[idx])))
            pixel_values_mask_video.append(self.mask_transform(Image.open(mask_image_paths[idx]).convert('L')))
            pixel_values_pose_video.append(self.pixel_transform(Image.open(pose_image_paths[idx])))
            pixel_values_flow_video.append(torch.from_numpy(np.load(flow_image_paths[idx])))
        
        # read cloth image
        pixel_values_cloth = self.pixel_transform(read_image(cloth_path).convert("RGB"))
        cloth_pil = read_image(cloth_path).convert("RGB")
        clip_cloth_img = self.clip_image_processor(
            images=cloth_pil, return_tensors="pt"
        ).pixel_values[0]
        
        sample = dict(
            video=torch.stack(pixel_values_video, dim=0), # f, c, h, w
            masked_video=torch.stack(pixel_values_masked_video, dim=0),
            mask_video=torch.stack(pixel_values_mask_video, dim=0),
            video_dwpose=torch.stack(pixel_values_pose_video, dim=0),
            flow_video=torch.stack(pixel_values_flow_video, dim=0),
            cloth=pixel_values_cloth,
            clip_cloth=clip_cloth_img
        )
        return sample



class DressCodeDataset(Dataset):
    def __init__(self, dataroot_path: str,
                 phase: str,
                 order: str = 'paired',
                 category: List[str] = ['dresses', 'upper_body', 'lower_body'],
                 size: Tuple[int, int] = (256, 192)):
        """
        Initialize the PyTroch Dataset Class
        :param args: argparse parameters
        :type args: argparse
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (upper_body | lower_body | dresses)
        :type category: list(str)
        :param size: image size (height, width)
        :type size: tuple(int)
        """
        super(Dataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.height = size[0]
        self.width = size[1]
        self.radius = 5
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        self.clip_transform = CLIPImageProcessor()

        im_names = []
        c_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # Clothing image
        cloth = Image.open(os.path.join(dataroot, 'images', c_name))
        cloth = cloth.resize((self.width, self.height))
        clip_cloth = self.clip_transform(
            images=cloth, return_tensors="pt"
        ).pixel_values[0]
        cloth = self.transform(cloth)   # [-1,1]

        # Person image
        im = Image.open(os.path.join(dataroot, 'images', im_name))
        im = im.resize((self.width, self.height))
        im = self.transform(im)   # [-1,1]

        # Skeleton
        #skeleton = Image.open(os.path.join(dataroot, 'skeletons', im_name.replace("_0", "_5")))
        skeleton = Image.open(os.path.join(dataroot, 'dwpose', im_name))
        skeleton = skeleton.resize((self.width, self.height))
        skeleton = self.transform(skeleton)

        # Label Map
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(os.path.join(dataroot, 'label_maps', parse_name))
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        parse_array = np.array(im_parse)

        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 3).astype(np.float32) + \
                     (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                            (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["scarf"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        if dataroot.split('/')[-1] == 'dresses':
            label_cat = 7
            parse_cloth = (parse_array == 7).astype(np.float32)
            parse_mask = (parse_array == 7).astype(np.float32) + \
                         (parse_array == 12).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif dataroot.split('/')[-1] == 'upper_body':
            label_cat = 4
            parse_cloth = (parse_array == 4).astype(np.float32)
            parse_mask = (parse_array == 4).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                 (parse_array == label_map["pants"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        elif dataroot.split('/')[-1] == 'lower_body':
            label_cat = 6
            parse_cloth = (parse_array == 6).astype(np.float32)
            parse_mask = (parse_array == 6).astype(np.float32) + \
                         (parse_array == 12).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                 (parse_array == 14).astype(np.float32) + \
                                 (parse_array == 15).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)   # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        # Masked cloth
        im_head = im * parse_head - (1 - parse_head)
        im_cloth = im * parse_cloth + (1 - parse_cloth)

        # Shape
        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
        shape = self.transform2D(parse_shape)  # [-1,1]

        # Load pose points
        pose_name = im_name.replace('_0.jpg', '_2.json')
        with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius * (self.height/512.0)
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (self.width, self.height))
        neck_draw = ImageDraw.Draw(neck)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], self.width/384.0)
            point_y = np.multiply(pose_data[i, 1], self.height/512.0)
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r*4, point_y - r*4, point_x + r*4, point_y + r*4), 'white', 'white')
            one_map = self.transform2D(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform2D(im_pose)

        im_arms = Image.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)
        if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                data = json.load(f)
                shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                    else:
                        arms_draw.line(np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                    else:
                        arms_draw.line(np.concatenate((elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', 30, 'curve')
                else:
                    arms_draw.line(np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(),'white', 30, 'curve')

            if self.height > 512:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((10, 10), np.uint16), iterations=5)
            # elif self.args.height > 256:
            #     im_arms = cv2.dilate(np.float32(im_arms), np.ones((5, 5), np.uint16), iterations=5)
            hands = np.logical_and(np.logical_not(im_arms), arms)
            parse_mask += im_arms
            parser_mask_fixed += hands

        #delete neck
        parse_head_2 = torch.clone(parse_head)
        if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                data = json.load(f)
                points = []
                points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height/512.0))
                points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height/512.0))
                x_coords, y_coords = zip(*points)
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords, rcond=None)[0]
                for i in range(parse_array.shape[1]):
                    y = i * m + c
                    parse_head_2[int(y - 20*(self.height/512.0)):, i] = 0

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16), np.logical_not(np.array(parse_head_2, dtype=np.uint16))))

        if self.height > 512:
            parse_mask = cv2.dilate(parse_mask, np.ones((20, 20), np.uint16), iterations=5)
        # elif self.args.height > 256:
        #     parse_mask = cv2.dilate(parse_mask, np.ones((10, 10), np.uint16), iterations=5)
        else:
            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        im_mask = im * parse_mask_total
        mask = 1 - parse_mask_total
        #TODO append mask dim
        mask = torch.from_numpy(mask.numpy()[np.newaxis, ...])
        parse_mask_total = parse_mask_total.numpy()
        parse_mask_total = parse_array * parse_mask_total
        parse_mask_total = torch.from_numpy(parse_mask_total)

        uv = np.load(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5_uv.npz')))
        uv = uv['uv']
        uv = torch.from_numpy(uv)
        uv = transforms.functional.resize(uv, (self.height, self.width))

        labels = Image.open(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5.png')))
        labels = labels.resize((self.width, self.height), Image.NEAREST)
        labels = np.array(labels)

        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth
            'cloth': cloth,  # for input
            'image': im,  # for visualization
            'im_cloth': im_cloth,  # for ground truth
            'shape': shape,  # for visualization
            'im_head': im_head,  # for visualization
            'im_pose': im_pose,  # for visualization
            'pose_map': pose_map,
            'parse_array': parse_array,
            'dense_labels': labels,
            'dense_uv': uv,
            'skeleton': skeleton,
            'm': im_mask,  # for input
            'mask': mask,
            'parse_mask_total': parse_mask_total,
            'clip_cloth': clip_cloth
        }

        return result

    def __len__(self):
        return len(self.c_names)
        
        


        




class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False,
            inverse_mask=False,             
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.inverse_mask = inverse_mask
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

        # pre-process
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # pre-process
        self.size_transform = A.Compose(
            [
                A.Resize(height=img_H, width=img_W),
                A.HorizontalFlip(p=0.5)
            ], 
            additional_targets={
                "agn": "image",
                "agn_mask": "image",
                "cloth": "image",
                "image_dwpose": "image",
            }
        )
        
        self.color_transform = A.Compose(
            [
                A.HueSaturationValue(5,5,5,p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5),
            ],
            additional_targets={
                "agn": "image",
                "cloth": "image",
            }
        )
        self.clip_transform = CLIPImageProcessor()

    def __len__(self):
        return len(self.im_names)
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        agn = Image.open(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
        agn_mask = Image.open(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png"))).convert("L")
        if self.inverse_mask:
            agn_mask = 1 - agn_mask
        cloth = Image.open(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))

        image = Image.open(opj(self.drd, self.data_type, "image", self.im_names[idx]))
        image_dwpose = Image.open(opj(self.drd, self.data_type, "image-dwpose", self.im_names[idx]))
        
        # convert to numpy
        agn = np.array(agn)
        agn_mask = np.array(agn_mask)
        cloth = np.array(cloth)
        image = np.array(image)
        image_dwpose = np.array(image_dwpose)

        # transform size
        transformed_size = self.size_transform(
            image=image,
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            image_dwpose=image_dwpose            
        )        
        agn = transformed_size["agn"]
        agn_mask = transformed_size["agn_mask"]
        cloth = transformed_size["cloth"]
        image = transformed_size["image"]
        image_dwpose = transformed_size["image_dwpose"]

        # transform color
        transformed_color = self.color_transform(
            image=image,
            agn=agn,
            cloth=cloth,        
        )
        agn = transformed_color["agn"]
        cloth = transformed_color["cloth"]
        image = transformed_color["image"]
        
        clip_cloth_image = self.clip_transform(
            images=Image.fromarray(cloth), return_tensors="pt"
        ).pixel_values[0]

        # transform to tensor
        image = self.image_transform(image)
        agn = self.image_transform(agn)
        cloth = self.image_transform(cloth)
        image_dwpose = self.image_transform(image_dwpose)
        
        agn_mask = self.mask_transform(agn_mask)
        
        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            clip_cloth=clip_cloth_image,
            image=image,
            image_dwpose=image_dwpose,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )



class TikTokDressSingleDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W,         
            sample_n_frames=1,
            sample_stride=4
        ):
        super().__init__()
        self.sample_rate = sample_stride
        self.sample_n_frames = sample_n_frames
        self.width = img_W
        self.height = img_H
        self.data_root_dir = data_root_dir
        self.video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new', '*')))
        self.masked_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_masked', '*')))
        self.mask_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_mask', '*')))
        self.pose_video_paths = sorted(glob.glob(opj(self.data_root_dir, 'images_new_dwpose', '*')))
        self.cloth_paths = sorted([opj(self.data_root_dir, 'cloth_images', os.path.basename(video_path),f'{os.path.basename(video_path)}.png') for video_path in self.video_paths])
        assert len(self.video_paths) == len(self.masked_video_paths) == len(self.mask_video_paths) == len(self.cloth_paths), "should be equal length"

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.video_paths)


    def __getitem__(self, index):
        video_path = self.video_paths[index]
        masked_video_path = self.masked_video_paths[index]
        mask_video_path = self.mask_video_paths[index]
        pose_video_path = self.pose_video_paths[index]
        cloth_path = self.cloth_paths[index]

        image_paths = sorted(glob.glob(video_path + '/*.png'))
        masked_image_paths = sorted(glob.glob(masked_video_path + '/*.png'))
        mask_image_paths = sorted(glob.glob(mask_video_path + '/*.png'))
        pose_image_paths = sorted(glob.glob(pose_video_path + '/*.png'))

        video_length = len(image_paths)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        ).tolist()
        
        pixel_values_video = []
        pixel_values_masked_video = []
        pixel_values_mask_video = []
        pixel_values_pose_video = []
        for idx in batch_index:
            pixel_values_video.append(self.pixel_transform(Image.open(image_paths[idx])))
            pixel_values_masked_video.append(self.pixel_transform(Image.open(masked_image_paths[idx])))
            pixel_values_mask_video.append(self.mask_transform(Image.open(mask_image_paths[idx]).convert('L')))
            pixel_values_pose_video.append(self.pixel_transform(Image.open(pose_image_paths[idx])))
        
        # read cloth image
        pixel_values_cloth = self.pixel_transform(read_image(cloth_path).convert("RGB"))
        cloth_pil = read_image(cloth_path).convert("RGB")
        clip_cloth_img = self.clip_image_processor(
            images=cloth_pil, return_tensors="pt"
        ).pixel_values[0]
        
        sample = dict(
            image=torch.stack(pixel_values_video, dim=0)[0], # f, c, h, w
            agn=torch.stack(pixel_values_masked_video, dim=0)[0],
            agn_mask=torch.stack(pixel_values_mask_video, dim=0)[0],
            image_dwpose=torch.stack(pixel_values_pose_video, dim=0)[0],
            cloth=pixel_values_cloth,
            clip_cloth=clip_cloth_img,
            txt="",
            img_fn="",
            cloth_fn="",
        )
        return sample