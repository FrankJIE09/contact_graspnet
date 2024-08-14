import numpy as np
import open3d as o3d
import os


def create_rgbd_image(rgb, depth):
    """
    从RGB和深度图像创建RGBD图像
    :param rgb: RGB图像，类型为np.uint8
    :param depth: 深度图像，类型为np.float32或np.uint16
    :return: Open3D RGBD图像对象
    """
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
    return rgbd_image


# 获取 test_data 目录下的所有 .npy 文件
data_dir = './'
files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

for file in files:
    # 加载 .npy 文件，并允许使用 pickle 反序列化
    data_path = os.path.join(data_dir, file)
    data = np.load(data_path, allow_pickle=True).item()

    # 提取RGB图像和深度图
    rgb = data['rgb']
    depth = data['depth']

    # 确保深度图为np.uint16类型，如果原始数据为float32可以通过乘以1000转换为毫米单位
    if depth.dtype == np.float32:
        depth = (depth * 1000).astype(np.uint16)

    # 创建RGBD图像
    rgbd_image = create_rgbd_image(rgb, depth)

    # 从RGBD图像和内参矩阵创建点云
    K = data['K']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=rgb.shape[1], height=rgb.shape[0],
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # 翻转点云，使其在Open3D中显示得当
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud from {file}")
