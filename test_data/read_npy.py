import os
import numpy as np
import pandas as pd

# 定义目录路径和输出Excel文件名
input_directory = './'
output_excel = 'output_data.xlsx'

# 初始化一个空的DataFrame列表
data_frames = []

# 遍历目录下的所有.npy文件
for file_name in os.listdir(input_directory):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_directory, file_name)

        # 读取.npy文件
        data = np.load(file_path, allow_pickle=True).item()

        # 检查数据结构并转换为DataFrame
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        # 1D数组
                        df = pd.DataFrame(value, columns=[key])
                    elif value.ndim == 2:
                        # 2D数组
                        df = pd.DataFrame(value)
                    elif value.ndim == 3:
                        # 3D数组，需要展平或以其他方式处理
                        # 展平3D数组到2D，每个通道作为单独的一列
                        flat_array = value.reshape(-1, value.shape[-1])
                        df = pd.DataFrame(flat_array, columns=[f"{key}_{i}" for i in range(value.shape[-1])])
                    else:
                        raise ValueError(f"Unsupported array dimension: {value.ndim}")
                elif isinstance(value, list):
                    # 列表
                    df = pd.DataFrame(value, columns=[key])
                else:
                    # 标量值
                    df = pd.DataFrame([value], columns=[key])

                # 添加文件名和键信息
                df['source_file'] = file_name
                df['key'] = key
                data_frames.append(df)
        else:
            # 如果不是字典，直接尝试转换为DataFrame
            df = pd.DataFrame(data)
            df['source_file'] = file_name
            data_frames.append(df)

# 将所有DataFrame合并成一个
combined_df = pd.concat(data_frames, ignore_index=True)

# 写入到Excel文件
combined_df.to_excel(output_excel, index=False)

print(f"Data has been written to {output_excel}")
