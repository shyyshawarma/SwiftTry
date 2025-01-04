import torch
import torch.nn.functional as F


class DebiasedTemporalLoss(nn.Module):
    def __init__():
        super().__init__()
        self.beta = 1
        self.alpha = (self.beta ** 2 + 1) ** 0.5
    
    def forward(model_pred, target):
        loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")        
        ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()
        model_pred_decent = self.alpha * model_pred - self.beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
        target_decent = self.alpha * target - self.beta * target[:, :, ran_idx, :, :].unsqueeze(2)
        loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
        loss_temporal = loss_temporal + loss_ad_temporal
        return loss_temporal
