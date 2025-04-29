import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM ,SimpleRNN
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


dataset = pd.read_csv('./data/Load.csv',index_col=0)
dataset = dataset.fillna(method='PAD')

dataset = np.array(dataset)

a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)

test = dataset.iloc[int(len(a)*0.9):,[0]]


# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
test = scaler.fit_transform(test)

x_test = []
y_test = []
for i in np.arange(96,len(test)):
    x_test.append(test[i-96:i,:])
    y_test.append(test[i])

x_test,y_test = np.array(x_test),np.array(y_test)

model = load_model('LSTM_model.h5')
# 模型测试
predict = model.predict(x_test)

# 反归一化
predict = scaler.inverse_transform(predict)
real = scaler.inverse_transform(y_test)
#
# plt.figure(figsize=(12,8))
# plt.plot(predict, label='predict')
# plt.plot(real, label='real')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(loc='best',fontsize=15)
# plt.ylabel('负荷值', fontsize=15)
# plt.xlabel('采样点',fontsize=15)
# plt.title('基于LSTM神经网络负荷预测',fontsize=15)
# plt.show()
#
R2 = r2_score(real, predict)
MAE = mean_absolute_error(real, predict)
RMSE = np.sqrt(mean_squared_error(real, predict))
MAPE = np.mean(np.abs(real - predict) / real)
print("R2",R2)
print("MAE",MAE)
print("RMSE",RMSE)
print("MAPE",MAPE)



