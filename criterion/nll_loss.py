import torch.nn.functional as F
import torch
def nll_loss(output, target):
    return F.nll_loss(output, target)

def CrossEntropyLoss():
    return torch.nn.CrossEntropyLoss()

def MSELoss():
    return torch.nn.MSELoss()