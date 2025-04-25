import numpy as np
import torch
def print_hi():
    ngpu = 1
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3,3).cuda())
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
if __name__ == '__main__':
    print_hi()
