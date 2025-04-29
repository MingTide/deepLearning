import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM ,SimpleRNN
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

train = dataset.iloc[0:int(len(a)*0.8),[0]]
val = dataset.iloc[int(len(a)*0.8):int(len(a)*0.9),[0]]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
val = scaler.fit_transform(val)

x_train= []
y_train = []
for i in np.arange(96,len(train)):
    x_train.append(train[i-96:i,:])
    y_train.append(train[i])

x_train,y_train = np.array(x_train),np.array(y_train)

x_val= []
y_val = []
for i in np.arange(96,len(val)):
    x_val.append(val[i-96:i,:])
    y_val.append(val[i])

x_val,y_val = np.array(x_val),np.array(y_val)


model = Sequential()
model.add(LSTM(units=10,return_sequences=True,activation='relu'))
model.add(LSTM(units=15,return_sequences=False,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.compile(optimizer=keras.optimizers.Adam(0.01),loss='mse')
history = model.fit(x_train,y_train,batch_size=512,epochs=30,validation_data=(x_val,y_val))

model.save('lstm_model.h5')
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')

plt.title('LSTM神经网络loss值',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()