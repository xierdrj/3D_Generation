import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import time

# 添加 Pointnet2_PyTorch 库到路径
sys.path.append('Pointnet2_PyTorch/pointnet2_ops_lib')
from pointnet2_ops import pointnet2_utils

def farthest_pts_sampling_tensor(pts, num_samples):
    '''
    使用 pointnet2_utils.furthest_point_sample 实现最远点采样
    
    参数:
        pts: 点云张量，形状为 [batch_size, num_points, 3]
        num_samples: 要采样的点数
        
    返回:
        采样后的点云张量，形状为 [batch_size, num_samples, 3]
    '''
    # 确保输入是浮点张量
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts, dtype=torch.float32)
    
    # 添加批次维度（如果是单点云）
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)  # [1, N, 3]
    
    # 移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = pts.to(device)
    
    # 执行最远点采样
    sampled_pts_idx = pointnet2_utils.furthest_point_sample(pts, num_samples)  # [B, num_samples]
    
    # 收集采样点
    sampled_pts = pointnet2_utils.gather_operation(
        pts.transpose(1, 2).contiguous(),  # [B, 3, N]
        sampled_pts_idx  # [B, num_samples]
    ).transpose(1, 2).contiguous()  # [B, num_samples, 3]
    
    return sampled_pts

def downsample_pointclouds(folder_path, target_points=15000):
    """
    对指定文件夹中的所有.npy文件进行下采样
    
    参数:
        folder_path: 包含.npy文件的文件夹路径
        target_points: 目标点数(默认15000)
    """
    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    print(f"在文件夹 {folder_path} 中找到 {len(npy_files)} 个.npy文件")
    print(f"开始下采样到 {target_points} 个点...")
    
    total_time = 0
    processed_files = 0
    
    for file_name in tqdm(npy_files, desc="处理进度"):
        try:
            file_path = os.path.join(folder_path, file_name)
            
            # 加载点云数据
            point_cloud = np.load(file_path)
            
            # 检查点云形状
            if point_cloud.shape[0] < target_points:
                print(f"警告: 文件 {file_name} 只有 {point_cloud.shape[0]} 个点，跳过")
                continue
                
            # 执行下采样
            start_time = time.time()
            
            # 使用GPU加速的FPS采样
            downsampled_cloud = farthest_pts_sampling_tensor(
                point_cloud, 
                target_points
            )
            
            # 转换回numpy数组并移除批次维度
            downsampled_cloud = downsampled_cloud.cpu().numpy().squeeze(0)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # 保存替换原始文件
            np.save(file_path, downsampled_cloud)
            processed_files += 1
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
    
    avg_time = total_time / processed_files if processed_files > 0 else 0
    print(f"完成! 成功处理 {processed_files} 个文件")
    print(f"平均处理时间: {avg_time:.4f} 秒/文件")
    print(f"总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    # 设置数据集路径
    base_dir = "00000001"
    
    # 要处理的子文件夹
    sub_folders = ["train", "test", "val"]
    
    # 目标点数
    TARGET_POINTS = 15000
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 对每个子文件夹进行处理
    for folder in sub_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            print(f"\n处理文件夹: {folder_path}")
            downsample_pointclouds(folder_path, TARGET_POINTS)
        else:
            print(f"文件夹不存在: {folder_path}")