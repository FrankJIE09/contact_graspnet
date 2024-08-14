import open3d as o3d
import numpy as np


def load_intrinsics_from_npy(npy_path):
    """
    从 .npy 文件中加载相机内参矩阵
    :param npy_path: .npy 文件路径
    :return: 相机内参矩阵 K
    """
    K = np.load(npy_path)
    return K


def point_cloud_to_depth(pcd, intrinsic, width, height):
    """
    将点云投影到图像平面生成深度图像
    :param pcd: Open3D 点云对象
    :param intrinsic: Open3D 相机内参对象
    :param width: 图像宽度
    :param height: 图像高度
    :return: 深度图像
    """
    # 创建深度图像
    depth_image = np.zeros((height, width))

    # 投影点云
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    for i in range(pcd_points.shape[0]):
        x, y, z = pcd_points[i]
        u = int((x * fx) / z + cx)
        v = int((y * fy) / z + cy)

        if 0 <= u < width and 0 <= v < height:
            depth_image[v, u] = z

    return depth_image


def point_cloud_to_rgb_image(pcd, intrinsic, width, height):
    """
    从点云颜色生成RGB图像
    :param pcd: Open3D 点云对象
    :param intrinsic: Open3D 相机内参对象
    :param width: 图像宽度
    :param height: 图像高度
    :return: RGB图像
    """
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 获取点云颜色信息
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    for i in range(pcd_points.shape[0]):
        r, g, b = pcd_colors[i] * 255
        u = int((pcd_points[i][0] * fx) / pcd_points[i][2] + cx)
        v = int((pcd_points[i][1] * fy) / pcd_points[i][2] + cy)

        if 0 <= u < width and 0 <= v < height:
            rgb_image[v, u] = [r, g, b]

    return rgb_image


def convert_ply_to_struct(ply_file, npy_file, image_width, image_height):
    """
    将PLY文件转换为目标结构
    :param ply_file: 输入的PLY文件路径
    :param npy_file: 相机内参的.npy文件路径
    :param image_width: 生成图像的宽度
    :param image_height: 生成图像的高度
    :return: 包含RGB、深度图、K矩阵和seg的字典
    """
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(ply_file)

    # 从.npy文件加载内参矩阵
    K = load_intrinsics_from_npy(npy_file)

    # 创建内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_width, height=image_height,
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )

    # 将点云转换为深度图像
    depth = point_cloud_to_depth(pcd, intrinsic, image_width, image_height)

    # 从点云生成RGB图像
    rgb = point_cloud_to_rgb_image(pcd, intrinsic,image_width, image_height)

    # 创建目标结构
    result = {
        'rgb': rgb,
        'depth': depth,
        'K': K,
        'seg': []
    }

    return result


def save_struct_to_npy(result_struct, save_path):
    """
    将结果结构保存为 .npy 文件
    :param result_struct: 要保存的结构
    :param save_path: 保存 .npy 文件的路径
    """
    np.save(save_path, result_struct)


# 使用示例
ply_file_path = "1.ply"
npy_file_path = "intrinsics_matrix.npy"
image_width = 640  # 设置图像宽度
image_height = 480  # 设置图像高度
save_file_path = "result_struct.npy"  # 保存路径

# 转换为目标结构
result_struct = convert_ply_to_struct(ply_file_path, npy_file_path, image_width, image_height)
print(result_struct)
# 保存为 .npy 文件
save_struct_to_npy(result_struct, save_file_path)

print(f"Result saved to {save_file_path}")
