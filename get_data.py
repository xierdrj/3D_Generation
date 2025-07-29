import os
import numpy as np
import shutil
from tqdm import tqdm  # 用于显示进度条（可选，需安装）

def extract_specific_npy_files(source_dir, target_dir, npz_filter=None, npy_key_filter=None, overwrite=False):
    """
    递归遍历源文件夹，筛选指定名称的.npz文件，提取其中特定的.npy文件到目标文件夹
    
    参数:
        source_dir: 源文件夹路径（包含.npz文件的根目录）
        target_dir: 目标文件夹路径（保存提取的.npy文件）
        npz_filter: .npz文件名筛选条件（如"points.npz"，支持部分匹配）
        npy_key_filter: .npy文件键名筛选条件（如"verts"，提取npz中键名包含此字符串的.npy）
        overwrite: 是否覆盖已存在的文件（默认False）
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 递归遍历源文件夹，收集符合条件的.npz文件
    npz_file_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.npz') and (npz_filter is None or npz_filter in file):
                npz_file_list.append(os.path.join(root, file))
    
    # 显示总文件数量
    print(f"共发现 {len(npz_file_list)} 个符合条件的.npz文件，开始提取...")
    
    # 用于统计成功提取的文件数
    success_count = 0
    
    # 遍历所有符合条件的.npz文件并提取
    for npz_path in tqdm(npz_file_list, desc="提取进度"):
        try:
            with np.load(npz_path) as npz_data:
                # 获取所有符合条件的.npy键名
                valid_keys = [
                    key for key in npz_data.files 
                    if npy_key_filter is None or npy_key_filter in key
                ]
                
                if not valid_keys:
                    print(f"警告: 文件 {npz_path} 中未找到符合条件的.npy文件")
                    continue
                
                # 提取并保存符合条件的.npy文件
                for key in valid_keys:
                    # 修改1: 使用原始文件路径创建唯一文件名
                    # 生成相对路径并替换路径分隔符
                    rel_path = os.path.relpath(npz_path, source_dir)
                    safe_filename = rel_path.replace(os.sep, "__")
                    
                    # 修改2: 去掉原始扩展名，添加键名和新扩展名
                    npy_filename = f"{os.path.splitext(safe_filename)[0]}__{key}.npy"
                    npy_path = os.path.join(target_dir, npy_filename)
                    
                    # 检查是否需要覆盖
                    if os.path.exists(npy_path) and not overwrite:
                        continue
                    
                    # 保存.npy文件
                    np.save(npy_path, npz_data[key])
                    success_count += 1
        except Exception as e:
            print(f"处理文件 {npz_path} 时出错: {str(e)}")
    
    print(f"提取完成！共成功保存 {success_count} 个.npy文件到 {target_dir}")

if __name__ == "__main__":
    # 配置路径和筛选条件（请根据实际情况修改）
    SOURCE_DIRECTORY = "dataset_shape_as_point"  # 源文件夹路径
    TARGET_DIRECTORY = "extracted_npy_files"     # 目标文件夹路径
    NPZ_FILTER = "pointcloud.npz"                # .npz文件名筛选条件
    NPY_KEY_FILTER = "points"                    # .npy文件键名筛选条件
    OVERWRITE_EXISTING = True                    # 修改3: 建议设置为True以确保覆盖测试
    
    # 执行提取
    extract_specific_npy_files(
        SOURCE_DIRECTORY, 
        TARGET_DIRECTORY, 
        NPZ_FILTER, 
        NPY_KEY_FILTER, 
        OVERWRITE_EXISTING
    )