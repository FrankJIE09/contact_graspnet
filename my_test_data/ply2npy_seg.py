import open3d as o3d
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

def load_intrinsics_from_npy(npy_path):
    K = np.load(npy_path)
    return K

def point_cloud_to_depth(pcd, intrinsic, width, height):
    depth_image = np.zeros((height, width))
    pcd_points = np.asarray(pcd.points)
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
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
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

def segment_image(rgb_image):
    model = deeplabv3_resnet50(pretrained=True).eval()
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(rgb_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return output_predictions

def convert_ply_to_struct(ply_file, npy_file, image_width, image_height):
    pcd = o3d.io.read_point_cloud(ply_file)
    K = load_intrinsics_from_npy(npy_file)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_width, height=image_height,
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    depth = point_cloud_to_depth(pcd, intrinsic, image_width, image_height)
    rgb = point_cloud_to_rgb_image(pcd, intrinsic, image_width, image_height)
    seg = segment_image(rgb)
    result = {
        'rgb': rgb,
        'depth': depth,
        'K': K,
        'seg': seg
    }
    return result

def save_struct_to_npy(result_struct, save_path):
    np.save(save_path, result_struct)

# 使用示例
ply_file_path = "1.ply"
npy_file_path = "intrinsics_matrix.npy"
image_width = 640
image_height = 480
save_file_path = "result_struct.npy"

# 转换为目标结构
result_struct = convert_ply_to_struct(ply_file_path, npy_file_path, image_width, image_height)

# 保存为 .npy 文件
save_struct_to_npy(result_struct, save_file_path)

print(f"Result saved to {save_file_path}")
