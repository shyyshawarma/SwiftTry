from skimage.metrics import structural_similarity as ssim_eval
import numpy as np
import torch
from torchvision import transforms
import lpips
import cv2

def psnr_eval(original, compressed):
    mse = torch.mean((original.float() - compressed.float()) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0  # Assuming input tensors are normalized
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def l1_eval(imageA, imageB):
    err = torch.mean(torch.abs(imageA.float() - imageB.float()))
    return err.item()

def compute_ssim_l1_psnr(gt_video, pred_video, mode):
    scores = []
    for i in range(gt_video.size(0)):
        image1 = pred_video[i].cpu().numpy().transpose(1, 2, 0)
        image2 = gt_video[i].cpu().numpy().transpose(1, 2, 0)
        image1 = (image1 * 255).astype(np.uint8)
        image2 = (image2 * 255).astype(np.uint8)
        if mode == 'L1':
            score = l1_eval(pred_video[i], gt_video[i])
        elif mode == 'PSNR':
            score = psnr_eval(pred_video[i], gt_video[i])
        else:
            # SSIM
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            score, _ = ssim_eval(image1_gray, image2_gray, full=True)
        scores.append(score)
    score_ave = np.mean(scores)
    return score_ave

def compute_lpips(loss_fn_vgg, gt_video, pred_video):
    scores = []
    for i in range(gt_video.size(0)):
        score = loss_fn_vgg(pred_video[i].unsqueeze(0).cuda(), gt_video[i].unsqueeze(0).cuda()).item()
        scores.append(score)
    score_ave = np.mean(scores)
    return score_ave

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_video', type=str, required=True)
    parser.add_argument('--gt_video', type=str, required=True)
    parser.add_argument('--mode', type=str, default='SSIM', help='SSIM, L1, PSNR, LPIPS')
    args = parser.parse_args()

    gen_video = torch.load(args.gen_video).cuda()  # Load the predicted video tensor
    gt_video = torch.load(args.gt_video).cuda()    # Load the ground truth video tensor
    mode = args.mode

    if mode == 'LPIPS':
        loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
        score_ave = compute_lpips(loss_fn_vgg, gt_video, gen_video)
    else:
        score_ave = compute_ssim_l1_psnr(gt_video, gen_video, mode)

    print(f"{mode} score: {score_ave}")
