import os
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import laplace

def calculate_fade(image):
    """
    计算单张图像的 FADE 值。
    参数：
        image (numpy.ndarray): 输入的图像（BGR 格式）。
    返回：
        float: 图像的 FADE 值。
    """
    if len(image.shape) == 3:  # 转换为灰度图
        gray_image = rgb2gray(image)
    else:
        gray_image = image

    # 计算 Laplacian 作为清晰度评估（雾感知密度的基础）
    laplacian_variance = laplace(gray_image).var()

    # FADE 的计算（伪实现，可替换为实际公式或模型）
    fade_value = 1 / (1 + laplacian_variance)
    return fade_value

def calculate_folder_fade_average(folder_path):
    """
    计算文件夹中所有图像的平均 FADE 值。
    参数：
        folder_path (str): 文件夹路径。
    返回：
        float: 文件夹中图像的平均 FADE 值。
    """
    fade_values = []
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
                    fade = calculate_fade(image)
                    fade_values.append(fade)
                    print(f"Processed {total_files}: {filename} - FADE: {fade:.4f}")
                else:
                    print(f"Warning: Could not read {file_path}. Skipping...")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # 检查是否有有效图像
    if not fade_values:
        raise ValueError("No valid images found in the specified folder.")

    # 计算平均 FADE 值
    return np.mean(fade_values)

if __name__ == "__main__":
    folder_path = input("enter").strip()
    try:
        print("Starting FADE calculation...")
        average_fade = calculate_folder_fade_average(folder_path)
        print(f"Average FADE of images in folder: {average_fade:.4f}")
    except Exception as e:
        print(f"Error: {e}")
