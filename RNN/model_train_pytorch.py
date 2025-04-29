import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch

from RNN.LSTM_model import LSTMModel

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv('./data/Load.csv',index_col=0)
dataset = dataset.fillna(method='PAD')

dataset = np.array(dataset)

a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)

train = dataset.iloc[0:int(len(a)*0.8),[0]]
val = dataset.iloc[int(len(a)*0.8):int(len(a)*0.9),[0]]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)


# 初始化模型
model = LSTMModel()

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=0.01)



x_train= []
y_train = []
for i in np.arange(96,len(train)):
    x_train.append(train[i-96:i,:])
    y_train.append(train[i])

x_train = np.array(x_train)
y_train = np.array(y_train)


x_val= []
y_val = []
for i in np.arange(96,len(val)):
    x_val.append(val[i-96:i,:])
    y_val.append(val[i])

x_val,y_val = np.array(x_val),np.array(y_val)

# 将数据转换为 PyTorch 张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# 训练模型
train_losses = []
val_losses = []
for epoch in range(1000):
    model.to(device)
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
    # 打印训练过程信息
    print(f'Epoch {epoch + 1}/{30}, Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}')
# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')

# 为了后续绘图，模拟一个包含训练和验证损失的 history 对象
class History:
    def __init__(self, train_losses, val_losses):
        self.history = {'loss': train_losses, 'val_loss': val_losses}

history = History(train_losses, val_losses)

plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')

plt.title('LSTM神经网络loss值',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()