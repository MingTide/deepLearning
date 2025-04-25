import time

import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import  FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch.nn as nn
import torch
import  copy
import pandas as pd
from model import LeNet
# 处理训练集和验证集
def test_val_data_process():
    test_data = FashionMNIST(root='./data', train=False, download=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))
    return DataLoader(test_data, batch_size=1, shuffle=True,num_workers=0)


def test_model_process(model,test_data_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 初始化参数
    test_corrects = 0.0
    test_num = 0.0
    with torch.no_grad():
        for test_data_x,test_data_y in test_data_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_label == test_data_y.data)
            test_num += test_data_x.size(0)
    test_acc = test_corrects.double().item() / test_num
    print(" Test acc:{:.4f}".format(test_acc))



if __name__ == '__main__':
    model  = LeNet()
    model.load_state_dict(torch.load('./best_model.pth'))
    test_data = test_val_data_process()
    test_model_process(model,test_data)