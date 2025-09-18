import torch
import torch.nn.functional as F
from torchvision import transforms
tp = transforms.ToTensor()#转发为张量
tt = transforms.ToPILImage()#转换为图片

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def pixelwise_euclidean_distance(x_cut, x0_cut):#不是照片，而是两个破碎数据,原，虚拟
    distance = 0
    diff = x_cut - x0_cut
    distance = torch.sum(diff ** 2)
    return torch.sqrt(distance)





# 损失函数
def loss_function_1(X_cut, X0_cut, lambda_param=0.4):

    euclidean_dist = pixelwise_euclidean_distance(X_cut, X0_cut)
    
    return euclidean_dist + lambda_param

def renew_x(X, ln, grade):
    _, channels, rows, cols = X.shape
    
    #print(X.shape)
    for a in range(channels):
        for i in range(rows):
            for j in range(cols):
                if i < rows-1 and j < cols-1:
                    X[0, a, i,j] = X[0, a, i,j] - grade * int(ln)
    return X