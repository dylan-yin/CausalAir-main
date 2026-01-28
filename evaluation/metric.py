import torch
import torch.nn.functional as F

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def AQI_MSE(output, target):
    return  F.mse_loss(output[:,:,0], target[:,:,0])

def AQI_RMSE(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,:,0], target[:,:,0]))
def AQI_RMSE_12(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,11,0], target[:,11,0]))
def AQI_RMSE_6(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,5,0], target[:,5,0]))
def AQI_RMSE_1(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,0,0], target[:,0,0]))

def AQI_RMSE_112(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,:12,0], target[:,:12,0]))
def AQI_RMSE_1324(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,12:24,0], target[:,12:24,0]))
def AQI_RMSE_2548(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,24:,0], target[:,24:,0]))

def AQI_MAE(output, target):# 计算MAE
    return F.l1_loss(output[:,:,0], target[:,:,0])
def AQI_RMSE(output, target): # 计算RMSE
    return torch.sqrt(F.mse_loss(output[:,:,0], target[:,:,0]))

def AQI_MAE_12(output, target):# 计算MAE
    return F.l1_loss(output[:,11,0], target[:,11,0])
def AQI_MAE_6(output, target):# 计算MAE
    return F.l1_loss(output[:,5,0], target[:,5,0])
def AQI_MAE_1(output, target):# 计算MAE
    return F.l1_loss(output[:,0,0], target[:,0,0])

def AQI_MAE_112(output, target): # 计算RMSE
    return F.l1_loss(output[:,:12,0], target[:,:12,0])
def AQI_MAE_1324(output, target): # 计算RMSE
    return F.l1_loss(output[:,12:24,0], target[:,12:24,0])
def AQI_MAE_2548(output, target): # 计算RMSE
    return F.l1_loss(output[:,24:,0], target[:,24:,0])

# L1 Sparsity metrics for matrix regularization
def Static_L1_Sparsity(output, target):
    """
    Placeholder metric for static matrix L1 sparsity.
    Actual computation is handled in trainer.
    """
    return torch.tensor(0.0)

def Dynamic_L1_Sparsity(output, target):
    """
    Placeholder metric for dynamic matrix L1 sparsity.
    Actual computation is handled in trainer.
    """
    return torch.tensor(0.0)