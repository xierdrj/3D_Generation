import os
import open3d as o3d
import numpy as np

def convert_npy_to_pcd(npy_path, save_dir="test", prefix="point_cloud"):
    """
    将.npy文件转换为PCD格式，支持两种数据形状：
    1. (num_points, 3) - 单个点云
    2. (num_samples, num_points, 3) - 多个点云
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载点云数据
    point_clouds = np.load(npy_path)
    print(f"数据形状: {point_clouds.shape}")
    print(f"数据类型: {point_clouds.dtype}")
    
    # 确保数据类型为float64
    if point_clouds.dtype != np.float64:
        point_clouds = point_clouds.astype(np.float64)
        print("已将数据类型转换为float64")
    
    try:
        if point_clouds.ndim == 2 and point_clouds.shape[1] == 3:
            # 情况1：单个点云 (num_points, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_clouds)
            
            filename = os.path.join(save_dir, f"{prefix}.pcd")
            o3d.io.write_point_cloud(filename, pcd)
            print(f"已保存单个点云: {filename}（包含 {point_clouds.shape[0]} 个点）")
            
        elif point_clouds.ndim == 3 and point_clouds.shape[2] == 3:
            # 情况2：多个点云 (num_samples, num_points, 3)
            num_samples = point_clouds.shape[0]
            for i in range(num_samples):
                single_cloud = point_clouds[i]
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(single_cloud)
                
                filename = os.path.join(save_dir, f"{prefix}_{i}.pcd")
                o3d.io.write_point_cloud(filename, pcd)
                
                if i % 100 == 0 or i == num_samples - 1:
                    print(f"已保存 {filename}（样本 {i+1}/{num_samples}，包含 {single_cloud.shape[0]} 个点）")
            
            print(f"共保存 {num_samples} 个点云到目录: {save_dir}")
            
        else:
            raise ValueError(f"不支持的形状: {point_clouds.shape}。"
                            "仅支持 (num_points, 3) 或 (num_samples, num_points, 3)。")
    
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # 配置路径
    NPY_PATH = "/root/autodl-tmp/CanonicalVAE/our_data_gen.npy"
    SAVE_DIR = "our_data_pcd"
    FILENAME_PREFIX = "point_cloud"  # 输出文件名前缀
    
    # 执行转换
    success = convert_npy_to_pcd(NPY_PATH, SAVE_DIR, FILENAME_PREFIX)
    
    if success:
        print("转换完成!")
    else:
        print("转换失败!")