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
def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True, download=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))
    train_data,val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True,num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=True,num_workers=0)
    return train_dataloader, val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    model =  model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化参数
    best_acc = 0.0
    # 训练集损失值列表
    train_loss_all = []
    # 验证集损失值列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y =b_y.to(device)
            model.train()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x =b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print("{} train Loss:{:.4f} Train Acc:{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} Val Loss:{:.4f} Val Acc:{:.4f}".format(epoch,val_loss_all[-1],val_acc_all[-1]))
        # 寻找的最高准确值的权重
        if val_loss_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_loss_all[-1]
            # 保存参数
            best_model_wts = copy.deepcopy(model.state_dict())
        # 计算耗时
        time_use = time.time() - since
        print("训练耗时:{:.0f}m{:.0f}s".format(time_use/60, time_use%60))
        # 选择最优参数加载最高准确率

    torch.save( best_model_wts,"./best_model.pth")

    train_process = pd.DataFrame(data={
            "epoch":range(num_epochs),
            "train_loss_all":train_loss_all,
            "val_loss_all":val_loss_all,
            "train_acc_all":train_acc_all,
            "val_acc_all":val_acc_all
        })
    return train_process

def matlap_acc_loss(tran_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(tran_process['epoch'],tran_process.train_loss_all,'ro-',label='train_loss')
    plt.plot(tran_process['epoch'],tran_process.val_loss_all,'bs-',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(tran_process['epoch'], tran_process.train_acc_all, 'ro-', label='train_acc')
    plt.plot(tran_process['epoch'], tran_process.val_acc_all, 'cs-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()

if __name__ == '__main__':
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet,train_dataloader,train_dataloader,num_epochs=20)
    matlap_acc_loss(train_process)
