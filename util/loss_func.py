import torch
import torch.nn as nn
import torch.nn.functional as F




# multi-label下的加权MSE
class WeightedMSELoss(nn.Module):

    def __init__(self, w):
        super(WeightedMSELoss, self).__init__()
        self.num_label = len(w)
        self.w = torch.tensor(w, dtype=torch.float32).view(self.num_label, 1)

    def forward(self, out, target):
        loss = F.mse_loss(out, target, reduction='none')    # [bs, num_label]
        loss = torch.matmul(loss, self.w) / self.num_label   # [bs, 1]
        loss = loss.mean()  # 对bs求平均
        return loss





# multi-label下，根据条件削弱某个维度的loss
class FixedMSELoss(nn.Module):

    def __init__(self):
        super(FixedMSELoss, self).__init__()

    def forward(self, out, target):
        loss = F.mse_loss(out, target, reduction='none')    # [bs, num_label]
        w = torch.ones(target.size(1))
        
        # 08 EyeBlinkLeft控制 52, 53
        # 09 EyeBlinkRight控制 54, 55
        # todo: 每一个样本的bs是不同的，所以w的维度应该是[bs, num_label]
        left_eye = 1 - target[8]    # 需要denorm到[0, 1]
        right_eye = 1 - target[9]
        w[52] = left_eye
        w[53] = left_eye
        w[54] = right_eye
        w[55] = right_eye

        num_label = w.size(1)
        w = w.view(num_label, 1)
        loss = torch.matmul(loss, w) / num_label   # [bs, 1]
        loss = loss.mean()  # 对bs求平均
        return loss



