import numpy as np
from PIL import Image

def pca_image_augmentation(image_path):
    # 读取图像
    image = Image.open(image_path)
    image_array = np.array(image).astype(np.float32)

    # 1. RGB 三通道归一化处理
    mean = np.mean(image_array, axis=(0, 1))
    std = np.std(image_array, axis=(0, 1))
    normalized_image = (image_array - mean) / std

    # 2. 图片按通道展平
    h, w, c = normalized_image.shape
    flattened_image = normalized_image.reshape(-1, c)

    # 3. 计算协方差矩阵
    cov_matrix = np.cov(flattened_image, rowvar=False)

    # 4. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 5. 引入抖动系数
    alpha = np.random.normal(0, 0.1, 3)
    delta = np.dot(eigenvectors, alpha * eigenvalues)

    # 6. 图像增强
    augmented_image = image_array + delta
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    return Image.fromarray(augmented_image)

# 使用示例
image_path = './AlexNet/img_1.png'
augmented_image = pca_image_augmentation(image_path)
augmented_image.show()
