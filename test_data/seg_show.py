import os
import numpy as np
import matplotlib.pyplot as plt

# 定义目录路径
input_directory = './'

# 遍历目录下的所有.npy文件
for file_name in os.listdir(input_directory):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_directory, file_name)

        # 读取.npy文件
        data = np.load(file_path, allow_pickle=True).item()

        # 处理并绘制RGB图像
        if 'rgb' in data:
            rgb = data['rgb']
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(rgb)
            plt.title(f"RGB Image - {file_name}")
            plt.axis('off')

        # 处理并绘制深度图像
        if 'depth' in data:
            depth = data['depth']
            plt.subplot(1, 3, 2)
            plt.imshow(depth, cmap='gray')
            plt.title(f"Depth Map - {file_name}")
            plt.colorbar()
            plt.axis('off')

        # 处理并绘制分割图
        if 'seg' in data:
            seg = data['seg']
            unique_labels = np.unique(seg)

            print(f"File: {file_name}")
            print(f"Unique labels in seg: {unique_labels}")

            plt.subplot(1, 3, 3)
            plt.imshow(seg, cmap='tab20')  # 使用 tab20 colormap 展示不同的标签
            plt.title(f"Segmentation Map - {file_name}")
            plt.colorbar()
            plt.axis('off')

        # 显示绘图结果
        plt.tight_layout()
        plt.show()
