import torch
import math
import torch.nn.functional as F
class MSELoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(MSELoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target, dataset, scale=100.):
        input = dataset.denormalize(input.datach()) * scale
        target = dataset.denormalize(target) * scale
        if self.exp_weighted:
            N = input.size(0)
            return 1 / (N * (math.e - 1)) * torch.sum((torch.exp(target) - 1) * (input - target) ** 2)
        else:
            return torch.tensor(F.mse_loss(input, target), dtype=torch.float)

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class AvgPooling(torch.nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x, ptr):
        g = []
        for i in range(ptr.size(0) - 1):
            g.append(torch.mean(x[ptr[i]:ptr[i + 1]], 0, True))
        return torch.cat(g, 0)

class BPRLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(BPRLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target > target[i])
            x = torch.log(1 + torch.exp(-(input[indices] - input[i])))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1)) ** 2 * total_loss
        else:
            return 2 / N ** 2 * total_loss


def accuracy_mse(predict, target, dataset, scale=100.):
    predict = dataset.denormalize(predict.detach()) * scale
    target = dataset.denormalize(target) * scale
    return F.mse_loss(predict, target)