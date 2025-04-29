import torch
import torch.nn as nn
def test_loss_function():
    # 模拟模型输出，假设 batch_size=2，num_classes=3
    output = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]], requires_grad=True)
    # 模拟真实标签
    b_y = torch.tensor([2, 0])

    # 使用 nn.CrossEntropyLoss() 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, b_y)
    print(f"nn.CrossEntropyLoss() 计算的损失: {loss.item()}")

    # 手动计算交叉熵损失
    # 1. 应用 LogSoftmax 函数
    log_softmax = nn.LogSoftmax(dim=1)
    log_probs = log_softmax(output)

    # 2. 计算负对数似然损失
    batch_size = output.size(0)
    manual_loss = 0
    for i in range(batch_size):
        manual_loss -= log_probs[i, b_y[i]]
    manual_loss /= batch_size

    print(f"手动计算的损失: {manual_loss.item()}")


def testItem():
    # 创建一个单元素张量 item()函数实验
    single_element_tensor = torch.tensor([42.0])
    print(f"单元素张量: {single_element_tensor}")
    print(f"张量类型: {type(single_element_tensor)}")

    # 使用 item() 方法将单元素张量转换为 Python 标量
    scalar_value = single_element_tensor.item()
    print(f"转换后的 Python 标量: {scalar_value}")
    print(f"标量类型: {type(scalar_value)}")

    # 在模型训练中使用 item() 获取损失值
    criterion = torch.nn.CrossEntropyLoss()
    output = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    loss = criterion(output, target)
    print(f"损失值张量: {loss}")
    print(f"转换后的损失值标量: {loss.item()}")

