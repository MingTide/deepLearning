from matplotlib import pyplot as plt
from torchvision.datasets import  FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np

train_data = FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))
# 60000  一个Batch为64张照片
train_loader =  Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True,num_workers=0)

# 获得一个Batch的数据 b_x是 (batch_size, channels, height, width)，移除第1维度是 (batch_size, height, width)
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16*2,ii+1)
    plt.imshow(batch_x[ii,:,:],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()



