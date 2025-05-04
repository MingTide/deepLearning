import  torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,strides=1,use_1x1 = False):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1:
            # 填充为0
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=strides,padding=0)
        else:
            self.conv3 = None
    def forward(self, x):
        y =self.ReLU( self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 :
            x = self.conv3(x)
            self.ReLU(y+x)
        else:
            self.ReLU(y + x)
        return y

class ResNet18(nn.Module):
    def __init__(self,Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(in_channels=64, out_channels=64, use_1x1 = False),
            Residual(in_channels=64, out_channels=64, use_1x1=False),
        )

        self.b3 = nn.Sequential(
            Residual(in_channels=64, out_channels=128, strides=2,use_1x1=True),
            Residual(in_channels=128, out_channels=128, use_1x1=False),
        )

        self.b4 = nn.Sequential(
            Residual(in_channels=128, out_channels=256,strides=2, use_1x1=True),
            Residual(in_channels=256, out_channels=256, use_1x1=False),
        )

        self.b5 = nn.Sequential(
            Residual(in_channels=256, out_channels=512, strides=2, use_1x1=True),
            Residual(in_channels=512, out_channels=512, use_1x1=False),
        )

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    print(summary(model, (1, 224, 224)))

