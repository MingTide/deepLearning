import torch
import torch.nn as nn
import torch.optim as optim

# 定义 LSTM 模型类
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=10, hidden_size=15, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(15, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x