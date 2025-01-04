from src.data.dataset import TikTokDressDataset
import torch
from tqdm import tqdm
train_dataset = TikTokDressDataset(
    data_root_dir='/root/dataset/TikTokDress',
    img_W=384,
    img_H=512,
    sample_n_frames=16,
    sample_stride=4,
    new_mask=True
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False, num_workers=8
)
sample = None
for step, batch in tqdm(enumerate(train_dataloader)):
    print(batch['video_name'])
