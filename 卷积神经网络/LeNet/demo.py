import numpy as np

# 创建一个有单维度的数组
arr = np.array([
    [
        [1],
        [2],
        [3]
    ]
])
print("原始数组形状:", arr.shape)

# 使用 squeeze 函数移除单维度
squeezed_arr = np.squeeze(arr)
print("压缩后数组形状:", squeezed_arr.shape)
a = np.array([[1,2,3],[2,3,4]])
print(a.shape)