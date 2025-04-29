import matplotlib.pyplot as plt
import  numpy as np
import  pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
dataset = pd.read_csv('./data/Load.csv',index_col=0)
dataset = dataset.fillna(method='pad')
dataset = np.array(dataset)


a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)
# real = np.array(dataset)
# plt.figure(figsize = (20,8))
#
# plt.plot(real)
#
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# labels = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']
# plt.xticks(range(0,35040,2920), labels=labels)
# plt.ylabel('负荷值',fontsize=15)
# plt.xlabel("月份",fontsize=15)
# plt.show()

weak_data = dataset.iloc[96*6:96*12,:]
weak_data = np.array(weak_data)
plt.figure(figsize = (20,8))
plt.plot(weak_data)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周七']
plt.xticks(range(0,96*7,96), labels=labels)
plt.ylabel('负荷值',fontsize=15)
plt.xlabel("日期",fontsize=15)
plt.show()

