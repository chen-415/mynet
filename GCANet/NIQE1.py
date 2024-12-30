import os
import cv2
import numpy as np
from scipy.stats import norm

def compute_mscn_coefficients(image, C=1):
    """
    计算图像的 MSCN 系数。
    MSCN: Mean Subtracted Contrast Normalized
    参数：
        image (numpy.ndarray): 输入灰度图像。
        C (float): 防止除零的小常数。
    返回：
        numpy.ndarray: MSCN 系数矩阵。
    """
    mean_local = cv2.GaussianBlur(image, (7, 7), 1.166)  # 局部均值
    sigma_local = cv2.GaussianBlur(np.square(image), (7, 7), 1.166) - np.square(mean_local)  # 局部方差
    sigma_local = np.sqrt(np.maximum(sigma_local, 0))  # 避免出现负值

    mscn_coefficients = (image - mean_local) / (sigma_local + C)
    return mscn_coefficients

def calculate_niqe(image):
    """
    基于公式实现 NIQE 的核心部分。
    参数：
        image (numpy.ndarray): 输入灰度图像。
    返回：
        float: 图像的 NIQE 值。
    """
    if len(image.shape) == 3:  # 转为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32) / 255.0

    # 计算 MSCN 系数
    mscn_coefficients = compute_mscn_coefficients(image)

    # 提取 MSCN 的统计特性 (如均值、方差等)
    mean_mscn = np.mean(mscn_coefficients)
    var_mscn = np.var(mscn_coefficients)

    # 模拟自然场景统计（此处为简化版，可换为模型参数）
    natural_mean = 0  # 假设为预定义自然场景均值
    natural_std = 1   # 假设为预定义自然场景标准差

    # 计算 NIQE 距离
    niqe_score = np.sqrt((mean_mscn - natural_mean) ** 2 + (var_mscn - natural_std) ** 2)
    return niqe_score

def calculate_folder_niqe_average(folder_path):
    """
    计算文件夹中所有图像的平均 NIQE 值。
    参数：
        folder_path (str): 文件夹路径。
    返回：
        float: 文件夹中图像的平均 NIQE 值。
    """
    niqe_values = []
    total_files = 0

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            total_files += 1
            try:
                # 读取图像
                image = cv2.imread(file_path)
                if image is not None:
                    niqe_score = calculate_niqe(image)
                    niqe_values.append(niqe_score)
                    print(f"Processed {total_files}: {filename} - NIQE: {niqe_score:.4f}")
                else:
                    print(f"Warning: Could not read {file_path}. Skipping...")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # 检查是否有有效图像
    if not niqe_values:
        raise ValueError("No valid images found in the specified folder.")

    # 计算平均 NIQE 值
    return np.mean(niqe_values)

if __name__ == "__main__":
    folder_path = input("enter").strip()
    try:
        print("Starting NIQE calculation...")
        average_niqe = calculate_folder_niqe_average(folder_path)
        print(f"Average NIQE of images in folder: {average_niqe:.4f}")
    except Exception as e:
        print(f"Error: {e}")
